from gpt.generation import generate_t2s
from audiotools import AudioSignal
from gpt.llama3.tokenizer import ChatFormat, Tokenizer
from train import TransformerWrapper
import math
import yaml
import torchaudio
from utils.data_utils import HParams
import torchaudio.functional as F
import torch
from dac.model.dac import DAC
from gpt.llama3.model import Transformer as GPT
from gpt.llama3.model import ModelArgs
import numpy as np
import random
from gpt.utils import setup_cache

def preprocess(signal):
    signal = signal.ensure_max_of_audio(1.0)
    signal = signal.normalize(-16)
    return signal
def load(hps, model_paths, device):
    # dac = DAC(**hps.dac).to(device)
    dac = None
    vq = DAC(**hps.vq).to(device)
    lm = TransformerWrapper(hps).to(device)

    # 加载 VQ
    vq_data = torch.load(model_paths['vq'], map_location=device)
    vq.load_state_dict(vq_data['vq'], strict=False)

    # 加载 LM
    lm_data = torch.load(model_paths['lm'], map_location=device)
    lm.load_state_dict(lm_data['lm'], strict=False)
    
    return vq, lm
MODELS_PATH = {
                'vq':'dptts/logs/2024-12-11-17-21-00/vq-697.pt',#vq
                'lm':'dptts/logs/2024-12-17-14-36-22/lm-40.pt'
            }
with open('config/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
hps = HParams(**cfg)
device = 'cuda:1'

Herz = 24000/np.prod(hps.vq.encoder_rates)
TEXT_VOCAB_SIZE = 128256
ACOUSTIC_VOCAB_SIZE = 2048
CODEBOOK_SIZE = 2048
N_CODEBOOKS = 8
TEXT_BLOCK_LENGTH = 128
ACOUSTIC_BLOCK_LENGTH = 750

ACOUSTIC_PAD_TOKEN = ACOUSTIC_VOCAB_SIZE
TEXT_PAD_TOKEN = TEXT_VOCAB_SIZE
ACOUSTIC_INFER_TOKEN = TEXT_VOCAB_SIZE + 1

vq,lm = load(hps, MODELS_PATH, device)
vq, lm = vq.eval(), lm.eval()

# refer_text = "仔细看，"
# refer_audio_path = 'prompt1.wav'
refer_text = "仔细看，这些文件上有些词汇跟花花琼脂的配方很像呢，仪器残骸也和玉质宝珠的一些地方有相似之处。"
refer_audio_path = '4.wav'
# refer_text = "而在我们追上他们之前，"
# refer_audio_path = '2.wav'
refer_audio,sr = torchaudio.load(refer_audio_path)
refer_audio = F.resample(refer_audio, sr, hps.data.sampling_rate)
# text = "红鲤鱼与绿鲤鱼与驴。"
text = "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"
# text = "大家好，今天来点大家想看的东西。"
text = refer_text + text
tokenizer = Tokenizer('llama_tokenizer/tokenizer.model')
text_token = tokenizer.encode(text,bos=False,eos=False)
text_token = torch.LongTensor(text_token).unsqueeze(0)

TEXT_BLOCK_LENGTH = 128

text_length = torch.LongTensor([text_token.shape[-1]])
signal = preprocess(AudioSignal(refer_audio,hps.data.sampling_rate)).audio_data.to(device)
with torch.no_grad():
    z, zs, codes, latents, commitment_loss, codebook_loss = vq.encode(signal)
    acoustic_token = codes.clone().squeeze(1)
    acoustic_length = torch.LongTensor([acoustic_token.shape[-1]])

ACOUSTIC_BLOCK_LENGTH = 750
ACOUSTIC_BLOCK_LENGTH = min(ACOUSTIC_BLOCK_LENGTH, torch.max(acoustic_length)+5)
delays = hps.train.delay

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

setup_cache(lm.transformer_temporal, 1, device)
setup_cache(lm.transformer_depth, 1, device)
with torch.no_grad():
        code_eval = lm.infer(
        text_block[0].unsqueeze(0),
        acoustic_block[0,:,:acoustic_length[0]].unsqueeze(0)
    )
with torch.no_grad():
    z = vq.quantizer.from_codes(code_eval)[0]
    wav_eval = vq.decode(z)
torchaudio.save('lm.wav', wav_eval[0].detach().cpu(), hps.data.sampling_rate)
