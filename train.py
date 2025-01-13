import logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.WARNING)
import copy
import random
from gpt.loss import ce_loss
import sys
from PIL import Image
import time
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from utils.log_utils import clean_checkpoints, plot_spectrogram_to_numpy, summarize
from typing import List, Optional, Tuple, Union
import torch
import os
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
from torch import nn
from dac.nn import loss as losses
from torch.optim import AdamW
from accelerate import Accelerator
from dac.model.dac import DAC
from gpt.llama3.model import Transformer as GPT
from gpt.llama3.model import ModelArgs
# from bark.model import GPT
import numpy as np
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
# from gpt.llama3.model import ModelArgs
from dataset import TextAudioCollate, TextAudioSpeakerLoader
from utils.data_utils import spec_to_mel_torch, mel_spectrogram_torch, HParams, spectrogram_torch
from utils import utils
import torchaudio
from audiotools.core import util as atutil
from dac.nn import loss as losses
from dac.model.discriminator import Discriminator
from gpt.generation import ras_sampling

from torchaudio.functional import phase_vocoder, resample, spectrogram
from audiotools import AudioSignal
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
from einops import rearrange, repeat
from gpt.utils import setup_cache

from utils import cnhubert

cnhubert.cnhubert_base_path='pretrain_model/cnhubert'

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

np.set_printoptions(threshold=sys.maxsize)
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
def get_grad_norm(model):
    total_norm = 0
    for name,p in model.named_parameters():
        try:
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            else:
                # print(name)
                continue
        except Exception as e:
            print(e)
            print(name)
    total_norm = total_norm ** (1. / 2) 
    return total_norm
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def cycle(dl):
    while True:
        for data in dl:
            yield data
def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
class TransformerWrapper(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.transformer_temporal = GPT(ModelArgs(**hps.t2s))
        self.transformer_depth = GPT(ModelArgs(**hps.s2a))
        self.out_proj = nn.ModuleList([
            nn.Linear(hps.t2s.dim, hps.vq.codebook_size+1) for _ in range(hps.vq.n_codebooks)
            ])
        self.codebook_num = hps.vq.n_codebooks
        self.text_embeddings = nn.Embedding(
                hps.data.text_vocab_size+10, hps.t2s.dim
            )
        self.acoustic_embeddings = nn.ModuleList([nn.Embedding(
                hps.vq.codebook_size+10, hps.t2s.dim
            ) for _ in range(hps.vq.n_codebooks)])
        self.hps = hps

    def forward(self, text_block: torch.Tensor, acoustic_block: torch.Tensor):
        # text B 129 
        # acoustic B K T
        device = text_block.device
        text_emb = self.text_embeddings(text_block)#B 129 C
        acoustic_emb = torch.stack([self.acoustic_embeddings[i](acoustic_block[:,i,:]) for i in range(self.codebook_num)],dim=1)
        h_temporal_in = torch.cat([text_emb,acoustic_emb.sum(dim=1)],dim=1).to(device) #B T C
        h_temporal_out = self.transformer_temporal(None, 0, h=h_temporal_in) #B T C
        h_temporal_out = h_temporal_out[:,128:-1,:]
        acoustic_depth_in = rearrange(acoustic_emb, 'b k t c -> (b t) k c')
        h_temporal_out = rearrange(h_temporal_out, 'b t c -> (b t) 1 c')
        h_depth_in = torch.cat([h_temporal_out, h_temporal_out+acoustic_depth_in[:,:-1,:]], dim=1)
        depth_output = self.transformer_depth(None, 0, h=h_depth_in)# BT K C
        outputs = []
        for i in range(self.codebook_num):
            outputs.append(self.out_proj[i](depth_output[:,i,:]))
        output = torch.stack(outputs).transpose(0,1)
        target = rearrange(acoustic_block, 'b k t -> (b t) k')
        #output BT K C target BT K
        
        # indices = torch.randperm(target.shape[0])[:1024]
        # output = output[indices,:,:]
        # target = target[indices,:]
        return output, target
    def infer(self, text_block, acoustic_block):
        #B T     B K T
        TEXT_VOCAB_SIZE = 128256
        ACOUSTIC_VOCAB_SIZE = 2048
        CODEBOOK_SIZE = 2048
        N_CODEBOOKS = 8
        TEXT_BLOCK_LENGTH = 128
        ACOUSTIC_BLOCK_LENGTH = 750
        temperature = 0.9
        device = text_block.device
        
        assert text_block.shape[0]==1 and acoustic_block.shape[0]==1
        total_len = 129+750
        text_emb = self.text_embeddings(text_block)#B 129 C
        prompt_length = text_block.shape[-1]+acoustic_block.shape[-1]
        acoustic_emb = torch.stack([self.acoustic_embeddings[i](acoustic_block[:,i,:]) for i in range(self.codebook_num)],dim=1)
        h_temporal_in = torch.full((1, total_len, 1024), 0, dtype=torch.float, device=device)
        h_temporal_in[:,:prompt_length,:] = torch.cat([text_emb,acoustic_emb.sum(dim=1)],dim=1).to(device) #B T C
        h_depth_in = torch.full((1, N_CODEBOOKS+1, 1024), 0, dtype=torch.float, device=device)

        delays = self.hps.train.delay
        max_delay = max(delays)
        prev_pos = 0
        code_gen = [[] for _ in range(N_CODEBOOKS)]
        end_delay = 0
        for cur_pos in tqdm(range(prompt_length, total_len)):
            h_temporal_out = self.transformer_temporal.forward(None, prev_pos, h=h_temporal_in[:, prev_pos:cur_pos,:], infer_mode=True)
            h_temporal_out = h_temporal_out[:,-1:,:]
            prev_pos_depth = 0
            h_depth_in[:,0,:] = h_temporal_out
            for cur_pos_depth in range(1, N_CODEBOOKS+1):
                depth_out = self.transformer_depth.forward(None, prev_pos_depth, h=h_depth_in[:,prev_pos_depth:cur_pos_depth,:], infer_mode=True)
                depth_out = depth_out[:,-1:,:]
                logits = self.out_proj[cur_pos_depth - 1](depth_out)
                if cur_pos_depth > 1:
                    logits = logits[:,:,:-1]
                next_token = ras_sampling(logits[:,-1].squeeze(0)/ temperature, code_gen[cur_pos_depth-1])
                if end_delay!=0 and cur_pos_depth==1:
                    next_token = torch.ones_like(next_token)*CODEBOOK_SIZE
                if next_token == CODEBOOK_SIZE:
                    end_delay+=1
                if end_delay>max_delay:
                    break
                code_gen[cur_pos_depth-1].append(next_token.item())
                prev_pos_depth = cur_pos_depth
                h_depth_in[:,cur_pos_depth,:] = self.acoustic_embeddings[cur_pos_depth-1](next_token) + h_temporal_out
                h_temporal_in[:,cur_pos,:] += self.acoustic_embeddings[cur_pos_depth-1](next_token)
            if end_delay>max_delay:
                break
            prev_pos = cur_pos
        gen_tokens = torch.from_numpy(np.array(code_gen)).to(device)
        gen_tokens = torch.cat([acoustic_block, gen_tokens.unsqueeze(0)],dim=-1)
        for n in range(1, N_CODEBOOKS):
            gen_tokens[:,n, :] = torch.roll(gen_tokens[:,n, :],-delays[n],1)
        gen_tokens = gen_tokens[:,:,:-max_delay]
        return gen_tokens
from accelerate import DistributedDataParallelKwargs
class Trainer(object):
    def __init__(self, cfg_path):

        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        # Create the custom configuration
        # process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours
        # self.accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
        self.accelerator = Accelerator()
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        # self.cfg = json.load(open(cfg_path))
        hps = HParams(**self.cfg)
        self.hps = hps
        self.config = hps
        dataset = TextAudioSpeakerLoader(hps)
        collate_fn = TextAudioCollate()
        self.dataloader = DataLoader(
            dataset,
            batch_size=hps.train.batch_size,
            num_workers=hps.train.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            drop_last=True,
            prefetch_factor=16,)
        self.train_steps = self.cfg['train']['train_steps']
        self.val_freq = self.cfg['train']['val_freq']
        if self.accelerator.is_main_process:
            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)
        
        self.vq = DAC(**self.hps.vq)
        self.lm = TransformerWrapper(hps)
        self.discriminator = Discriminator(**self.hps.discriminator)
        
        # 初始化优化器
        self.vq_optimizer = AdamW(self.vq.parameters(), self.hps.train.learning_rate, betas=self.hps.train.betas, eps=self.hps.train.eps)
        self.lm_optimizer = AdamW(self.lm.parameters(), self.hps.train.learning_rate, betas=self.hps.train.betas, eps=self.hps.train.eps)
        self.discriminator_optimizer = AdamW(self.discriminator.parameters(), self.hps.train.learning_rate, betas=self.hps.train.betas, eps=self.hps.train.eps)  # discriminator 优化器

        # 初始化调度器
        self.scheduler_vq = torch.optim.lr_scheduler.ExponentialLR(self.vq_optimizer, gamma=self.hps.train.lr_decay, last_epoch=-1)
        self.scheduler_lm = torch.optim.lr_scheduler.ExponentialLR(self.lm_optimizer, gamma=self.hps.train.lr_decay, last_epoch=-1)
        self.scheduler_discriminator = torch.optim.lr_scheduler.ExponentialLR(self.discriminator_optimizer, gamma=self.hps.train.lr_decay, last_epoch=-1)  # discriminator 调度器

        # 准备模型和优化器
        self.vq, self.vq_optimizer,\
        self.lm, self.lm_optimizer, self.discriminator, self.discriminator_optimizer, self.dataloader = self.accelerator.prepare(
            self.vq, self.vq_optimizer, 
            self.lm, self.lm_optimizer, 
            self.discriminator, self.discriminator_optimizer,  # 添加的 discriminator
            self.dataloader
        )
    
        self.dataloader = cycle(self.dataloader)
        self.step=0
        self.epoch=1
        self.gradient_accumulate_every=self.hps.train.gradient_accumulate_every
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data_discriminator = {
            'step': self.step,
            'epoch': self.epoch,
            'discriminator': self.accelerator.get_state_dict(self.discriminator),
            # 'discriminator_opt': self.accelerator.get_state_dict(self.discriminator_optimizer)
        }

        data_vq = {
            'step': self.step,
            'epoch': self.epoch,
            'vq': self.accelerator.get_state_dict(self.vq),
            # 'vq_opt': self.accelerator.get_state_dict(self.vq_optimizer)
        }

        data_lm = {
            'step': self.step,
            'epoch': self.epoch,
            'lm': self.accelerator.get_state_dict(self.lm),
            # 'lm_opt': self.accelerator.get_state_dict(self.lm_optimizer)
        }

        if self.hps.target == 'vq':
            torch.save(data_vq, str(self.logs_folder / f'vq-{milestone}.pt'))
            torch.save(data_discriminator, str(self.logs_folder / f'discriminator-{milestone}.pt'))
        if self.hps.target == 'lm':
            torch.save(data_lm, str(self.logs_folder / f'lm-{milestone}.pt'))

    def load(self, model_paths):
        accelerator = self.accelerator
        device = accelerator.device

        if model_paths['vq'] is not None:
            # 加载 VQ
            vq_data = torch.load(model_paths['vq'], map_location=device)
            vq_state_dict = vq_data['vq']
            # vq_opt_state_dict = vq_data['vq_opt']

            vq = accelerator.unwrap_model(self.vq)
            current_vq_dict = vq.state_dict()
            vq_state_dict = {
                k: v if v.size() == current_vq_dict[k].size() else current_vq_dict[k]
                for k, v in zip(current_vq_dict.keys(), vq_state_dict.values())
            }
            vq.load_state_dict(vq_state_dict, strict=False)

        if model_paths['lm'] is not None:
            # 加载 LM
            lm_data = torch.load(model_paths['lm'], map_location=device)
            lm_state_dict = lm_data['lm']
            # lm_opt_state_dict = lm_data['lm_opt']

            lm = accelerator.unwrap_model(self.lm)
            current_lm_dict = lm.state_dict()
            lm_state_dict = {
                k: v if v.size() == current_lm_dict[k].size() else current_lm_dict[k]
                for k, v in zip(current_lm_dict.keys(), lm_state_dict.values())
            }
            lm.load_state_dict(lm_state_dict, strict=False)

        if model_paths['discriminator'] is not None:
            # 加载 Discriminator
            discriminator_data = torch.load(model_paths['discriminator'], map_location=device)
            discriminator_state_dict = discriminator_data['discriminator']
            # discriminator_opt_state_dict = discriminator_data['discriminator_opt']

            discriminator = accelerator.unwrap_model(self.discriminator)
            current_discriminator_dict = discriminator.state_dict()
            discriminator_state_dict = {
                k: v if v.size() == current_discriminator_dict[k].size() else current_discriminator_dict[k]
                for k, v in zip(current_discriminator_dict.keys(), discriminator_state_dict.values())
            }
            discriminator.load_state_dict(discriminator_state_dict, strict=False)
        # 加载优化器状态
        # try:
        #     self.vq_optimizer.load_state_dict(vq_opt_state_dict)
        # except:
        #     print('Fail to load vq_opt')
        # try:
        #     self.lm_optimizer.load_state_dict(lm_opt_state_dict)
        # except:
        #     print('Fail to load lm_opt')
        # try:
        #     self.discriminator_optimizer.load_state_dict(discriminator_opt_state_dict)
        # except:
        #     print('Fail to load discriminator_opt')

    def preprocess(self, signal):
        signal = signal.ensure_max_of_audio(1.0)
        signal = signal.normalize(-16)
        return signal

    def vq_step(self,pbar, waveform_loss, stft_loss, mel_loss, gan_loss,hubert=None,w2vbert=None):
        accel = self.accelerator
        device = accel.device
        batch = next(self.dataloader)
        data = atutil.prepare_batch(batch, accel.device)
        lambdas = self.hps.lambdas
        self.vq.train()
        self.discriminator.train()
        output = {}
        with torch.no_grad():
            signal = self.preprocess(AudioSignal(data['wav'],self.hps.data.sampling_rate))

        with accel.autocast():
            out = self.vq(signal.audio_data, self.hps.data.sampling_rate)
            recons = AudioSignal(out["audio"], signal.sample_rate)
            commitment_loss = out["vq/commitment_loss"]
            codebook_loss = out["vq/codebook_loss"]

        with accel.autocast():
            output["adv/disc_loss"] = gan_loss.discriminator_loss(recons, signal)

        self.discriminator_optimizer.zero_grad()
        accel.backward(output["adv/disc_loss"])
        # accel.scaler.unscale_(self.discriminator_optimizer)
        output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(), 10.0
        )
        self.discriminator_optimizer.step()
        self.scheduler_discriminator.step()
        
        vq = self.accelerator.unwrap_model(self.vq)
        semantic_emb = vq.semantic_proj(out['zs'][:,0,:,:].transpose(-1,-2))
        
        audio_16k = torchaudio.functional.resample(signal.audio_data, 24000, 16000)
        hubert_emb = hubert(audio_16k.squeeze(1)).transpose(1,2)
        hubert_emb = torch.nn.functional.pad(hubert_emb, (0,int(semantic_emb.shape[1]*vq.scale_size-hubert_emb.shape[-1])), "constant", 0)
        hubert_emb = vq.ssl_proj(hubert_emb).transpose(1,2)
        
        with accel.autocast():
            signal = signal[:,:,:recons.shape[-1]]
            output['semantic/loss'] = (1-nn.CosineSimilarity(dim=-1)(semantic_emb, hubert_emb)).mean()
            output["stft/loss"] = stft_loss(recons, signal)
            output["mel/loss"] = mel_loss(recons, signal)
            output["waveform/loss"] = waveform_loss(recons, signal)
            (
                output["adv/gen_loss"],
                output["adv/feat_loss"],
            ) = gan_loss.generator_loss(recons, signal)
            output["vq/commitment_loss"] = commitment_loss
            output["vq/codebook_loss"] = codebook_loss
            # for k in output:
            #     print(k,output[k])
            output["loss"] = sum([v * output[k] for k, v in lambdas.items() if k in output])

        self.vq_optimizer.zero_grad()
        accel.backward(output["loss"])
        grad_norm = get_grad_norm(self.vq)
        # accel.scaler.unscale_(optimizer_vq)
        output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
            self.vq.parameters(), 1e3
        )
        self.vq_optimizer.step()
        self.scheduler_vq.step()
        # accel.update()
        output["other/learning_rate"] = self.vq_optimizer.param_groups[0]["lr"]
        pbar.set_description(f'G_loss: {output["loss"]:.4f} D_loss: {output["adv/disc_loss"]:.4f} lr: {output["other/learning_rate"]:.5f}')
        if self.accelerator.is_main_process and self.step % self.val_freq == 0:
            hps = self.hps
            eval_model = self.accelerator.unwrap_model(self.vq)
            eval_model.eval()
            with torch.no_grad():
                wav_eval = eval_model(
                    data['raw_wav'][:,:,:48000], self.hps.data.sampling_rate
                )['audio']
            eval_model.train()
            milestone = self.step // self.cfg['train']['save_freq']
            torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), wav_eval[0].detach().cpu(), hps.data.sampling_rate)
            torchaudio.save(str(self.logs_folder / f'gt-{milestone}.wav'), data['raw_wav'][:,:,:48000][0].detach().cpu(), hps.data.sampling_rate)
        return {k: v for k, v in sorted(output.items())}

    def lm_step(self, pbar):
        Herz = 24000/np.prod(self.hps.vq.encoder_rates)
        TEXT_VOCAB_SIZE = 128256
        ACOUSTIC_VOCAB_SIZE = 2048
        CODEBOOK_SIZE = 2048
        N_CODEBOOKS = 8
        TEXT_BLOCK_LENGTH = 128
        ACOUSTIC_BLOCK_LENGTH = Herz*30

        ACOUSTIC_PAD_TOKEN = ACOUSTIC_VOCAB_SIZE
        TEXT_PAD_TOKEN = TEXT_VOCAB_SIZE
        ACOUSTIC_INFER_TOKEN = TEXT_VOCAB_SIZE + 1

        total_loss = 0.
        device = self.accelerator.device
        for _ in range(self.gradient_accumulate_every):
            data = next(self.dataloader)
            for d in data:
                data[d] = data[d].to(device)
            text_token = data['text']
            text_length = data['text_length']
            signal = self.preprocess(AudioSignal(data['raw_wav'],self.hps.data.sampling_rate)).audio_data
            vq = self.accelerator.unwrap_model(self.vq)
            vq.eval()
            with torch.no_grad():
                z, zs, codes, latents, commitment_loss, codebook_loss = vq.encode(signal)
                acoustic_token = codes.clone().squeeze(1)
                acoustic_length = (data['raw_wav_length']/(24000/Herz)).long()
            
            ACOUSTIC_BLOCK_LENGTH = 750
            ACOUSTIC_BLOCK_LENGTH = min(ACOUSTIC_BLOCK_LENGTH, torch.max(acoustic_length)+5)
            delays = self.hps.train.delay
            B = acoustic_token.shape[0]
            text_block = torch.full((B, TEXT_BLOCK_LENGTH), TEXT_PAD_TOKEN, dtype=torch.long).to(device)
            acoustic_block = torch.full((B, N_CODEBOOKS, ACOUSTIC_BLOCK_LENGTH), ACOUSTIC_PAD_TOKEN, dtype=torch.long).to(device)
            for i in range(B):
                text_block_length = min(text_length[i], TEXT_BLOCK_LENGTH)
                text_block[i,:text_block_length] = text_token[i,:text_block_length]
                acoustic_block_length = min(acoustic_length[i], ACOUSTIC_BLOCK_LENGTH)
                acoustic_block[i,:N_CODEBOOKS,:acoustic_block_length] = acoustic_token[i,:N_CODEBOOKS,:acoustic_block_length]
            acoustic_block[i,:N_CODEBOOKS,-2:] = ACOUSTIC_PAD_TOKEN
            for i in range(B):
                for j, delay in enumerate(delays):
                    acoustic_block[i,j,:] = torch.roll(acoustic_block[i,j,:],delay,0)
            # print(text_block.shape, acoustic_block.shape)
            text_block = torch.cat((text_block, torch.full((B, 1), ACOUSTIC_INFER_TOKEN).to(device)),dim=-1)
            # labels = torch.cat((text_block, acoustic_block),dim=-1)
            with self.accelerator.autocast():
                logits, target = self.lm(text_block, acoustic_block)
                loss, right_cnt, total_cnt,\
                    top1_rate = ce_loss(logits[:,:,:].transpose(1,2),
                                        target[:,:],
                                        torch.LongTensor([N_CODEBOOKS]*logits.shape[0]).to(device),device)
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()
            self.accelerator.backward(loss)
        grad_norm = get_grad_norm(self.lm)
        self.accelerator.clip_grad_norm_(self.lm.parameters(), 1.0)
        pbar.set_description(f'{self.hps.target}_loss: {total_loss:.4f}')
        self.accelerator.wait_for_everyone()
        self.lm_optimizer.step()
        if self.step%100==0:
            self.scheduler_lm.step()
        self.lm_optimizer.zero_grad()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process and self.step % self.val_freq == 0:
            if acoustic_length[0]<101:
                return
            eval_model = self.accelerator.unwrap_model(self.lm)
            eval_model.eval()
            # print(text_token)
            setup_cache(eval_model.transformer_temporal, 1, device)
            setup_cache(eval_model.transformer_depth, 1, device)
            with torch.no_grad():
                code_eval = eval_model.infer(
                    text_block[0].unsqueeze(0),
                    acoustic_block[0,:,:50].unsqueeze(0)
                )
            eval_model.train()
            if code_eval.shape[-1]==0:
                return
            # print(code_eval.shape)
            # code_eval = code_eval.unsqueeze(0)
            # print(code_eval.shape, code_eval)
            with torch.no_grad():
                z = vq.quantizer.from_codes(code_eval)[0]
                wav_eval = vq.decode(z)
                wav_gt = data['raw_wav']
            milestone = self.step // self.cfg['train']['save_freq']
            torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), wav_eval[0].detach().cpu(), self.hps.data.sampling_rate)
            torchaudio.save(str(self.logs_folder / f'gt-{milestone}.wav'), wav_gt[0].detach().cpu(), self.hps.data.sampling_rate)
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        hps = self.hps
        cnt=0

        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.logs_folder)
        epoch=self.epoch
        # self.hubert = cnhubert.get_model().to(device)
        
        waveform_loss = losses.L1Loss()
        stft_loss = losses.MultiScaleSTFTLoss(**self.hps.MultiScaleSTFTLoss)
        mel_loss = losses.MelSpectrogramLoss(**self.hps.MelSpectrogramLoss)
        gan_loss = losses.GANLoss(self.discriminator)
        hubert = cnhubert.get_model().to(device)
        # self.hubert = None
        # self.w2vbert = w2vbert.get_model('pretrain_model/w2vbert/').to(device)
        with tqdm(initial = self.step, total = self.train_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_steps:
                # self.dataloader.batch_sampler.epoch=epoch
                # with torch.autograd.detect_anomaly():
                if self.hps.target == 'vq':
                    self.vq_step(pbar, waveform_loss, stft_loss, mel_loss, gan_loss,hubert=hubert)
                elif self.hps.target == 'lm':
                    self.lm_step(pbar)
                else:
                    print("Not support train target!")
                if accelerator.is_main_process and self.step % self.cfg['train']['save_freq']==0:
                    keep_ckpts = self.cfg['train']['keep_ckpts']
                    if keep_ckpts > 0:
                        clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                    self.save(self.step//1000)
                self.step += 1
                pbar.update(1)
                # if self.step%50000==0:
                    # self.scheduler_g.step()
                    # self.scheduler_d.step()
                    # self.scheduler_gpt.step()
                    # epoch = epoch + 1
        accelerator.print('training complete')


if __name__ == '__main__':
    config_path = 'config/config.yaml'
    trainer = Trainer(cfg_path=config_path)
    trainer.load({
                'discriminator':'logs/2024-12-11-17-21-00/discriminator-697.pt',#disc
                'vq':'logs/2024-12-11-17-21-00/vq-697.pt',#vq
                'lm':'logs/2025-01-08-11-50-13/lm-4.pt',#lm
                })
    trainer.train()