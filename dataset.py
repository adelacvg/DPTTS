import os
import random
import numpy as np
import torch
import torch.utils.data
from utils import utils
from utils.mel_processing import spectrogram_torch, spec_to_mel_torch, mel_spectrogram_torch
from utils.utils import load_filepaths_and_text, load_wav_to_torch
import torchaudio.functional as F
from gpt.llama3.tokenizer import ChatFormat, Tokenizer
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import glob
import json
from tqdm import tqdm
import h5py
import re
# import h5py
def read_jsonl(path):
    with open(path, 'r') as f:
        json_str = f.read()
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list
def read_hdf5(path):
    with h5py.File(path, 'r') as hdf:
        paths = hdf['paths'][:]
        texts = hdf['texts'][:]
    return paths,texts
def write_jsonl(path, all_paths):
    with open(path,'w', encoding='utf-8') as file:
        for item in all_paths:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')
"""Multi speaker version"""
def find_audio_files(folder_path, suffixes):
    files = []
    for suffix in suffixes:
        files.extend(glob.glob(os.path.join(folder_path, '**', f'*{suffix}'),recursive=True))
    return files
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, hparams, all_in_mem: bool = False, vol_aug: bool = True):
        self.tokenizer = Tokenizer('llama_tokenizer/tokenizer.model')
        self.audiopaths_and_text = []
        
        self.train_paths = hparams.data.training_files_gpt
        if self.train_paths.endswith(".jsonl"):
            self.audiopaths_and_text = read_jsonl(self.train_paths)
        else:
            self.hdf = h5py.File(self.train_paths, 'r')
            self.audiopaths = self.hdf['paths']
            self.texts = self.hdf['texts']
            self.length = len(self.texts)
            self.indices = list(range(self.length))
            random.shuffle(self.indices)
            # self.audiopaths,self.texts = read_hdf5(self.train_paths)
        self.hparams = hparams
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.n_mel_channels = hparams.data.n_mel_channels
        self.mel_fmin = hparams.data.mel_fmin
        self.mel_fmax = hparams.data.mel_fmax
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.unit_interpolate_mode = hparams.data.unit_interpolate_mode
        self.sampling_rate = hparams.data.sampling_rate
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        
        self.all_in_mem = all_in_mem
    def get_audio(self, path_and_text):
        try:
            audiopath, text = path_and_text['path'], path_and_text['text']
            raw_text = text
            raw_text = text.replace(',','，')
            if raw_text[-1] not in '。！？“':
                raw_text = raw_text+'。'
            # raw_text = re.sub(r'[,.!?;:[](){}|，。！？；：「」『』（）《》【】]', '', raw_text)
            # print(raw_text)
            text = self.tokenizer.encode(text,bos=False,eos=False)
            text = torch.Tensor(text)
            wav, sr = torchaudio.load(audiopath)
            # wav = torch.clamp(wav, min=-0.99, max=0.99)
            # if wav.shape[-1]/sr < 0.61 or wav.shape[-1]/sr > 30.1:
            #     print(audiopath)
            #     return None,None,None,None
            if wav.shape[0] > 1:
                wav = wav[0].unsqueeze(0)
            wav = F.resample(wav, sr, self.sampling_rate)
            audio_norm = wav
            mel = mel_spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.n_mel_channels,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                self.mel_fmin,
                self.mel_fmax,
            )
            mel = torch.squeeze(mel,0)
            return audio_norm, text, raw_text, audiopath, mel
        except Exception as e:
            print(e)
            return None,None,None,None
    def random_slice(self, audio_norm, text, raw_text, audiopath, mel):
        l = min(mel.shape[1], audio_norm.shape[-1]//self.hop_length//8*8)
        audio_norm = audio_norm[:, :l * self.hop_length]
        mel = mel[:, :l]
        raw_wav = audio_norm
        raw_mel =  mel
        segment_size = 48
        if audio_norm.shape[-1] > segment_size*self.hop_length:
            start = random.randint(0, mel.shape[1]-segment_size)
            end = start + segment_size
            mel = mel[:, start:end]
            audio_norm = audio_norm[:, start * self.hop_length : end * self.hop_length]
        else:
            return None
        return audio_norm, raw_wav, mel, raw_mel, text,  raw_text, audiopath
    def __getitem__(self, index):
        try:
            if self.train_paths.endswith(".jsonl"):
                ret = self.random_slice(*self.get_audio(self.audiopaths_and_text[index]))
            else:
                actual_index = self.indices[index]
                data = {
                    'path': self.audiopaths[actual_index].decode('utf-8'),
                    'text': self.texts[actual_index].decode('utf-8')
                }
                ret = self.random_slice(*self.get_audio(data))
        except Exception as e:
            if self.train_paths.endswith(".jsonl"):
                print(self.audiopaths_and_text[index])
            else:
                print({'path':self.audiopaths[index].decode('utf-8'),'text':self.texts[index].decode('utf-8')})
            print(e)
            return None
        return ret
    def __len__(self):
        if self.train_paths.endswith(".jsonl"):
            return len(self.audiopaths_and_text)
        else:
            return len(self.texts)
class TextAudioCollate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        # batch = batch[8:12]
        if len(batch) == 0:
            return None
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].shape[1] for x in batch]),
            dim=0, descending=True)
        max_wav_len = max([x[0].size(1) for x in batch])
        max_raw_wav_len = max([x[1].size(1) for x in batch])
        max_mel_len = max([x[2].size(1) for x in batch])
        max_raw_mel_len = max([x[3].size(1) for x in batch])
        max_text_len = max([len(x[4]) for x in batch])
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        raw_wav_padded = torch.FloatTensor(len(batch), 1, max_raw_wav_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_mel_len)
        raw_mel_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_raw_mel_len)
        text_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        raw_wav_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))
        raw_mel_lengths = torch.LongTensor(len(batch))
        wav_padded.zero_()
        text_padded.zero_()
        raw_wav_padded.zero_()
        mel_padded.zero_()
        raw_mel_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            wav = row[0]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
                       
            raw_wav = row[1]
            raw_wav_padded[i, :, :raw_wav.size(1)] = raw_wav
            raw_wav_lengths[i] = raw_wav.size(1)
            
            mel = row[2]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] =mel.size(1)
            
            raw_mel = row[3]
            raw_mel_padded[i, :, :raw_mel.size(1)] = raw_mel
            raw_mel_lengths[i] = raw_mel.size(1)
            
            text = row[4]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)
 
        raw_text = [x[5] for x in batch]
        audiopath = [x[6] for x in batch]
        # print(raw_text)
        # print(audiopath)
        # print(torch.max(spec_padded),torch.min(spec_padded),torch.max(wav_padded),torch.min(wav_padded))
        return {
            "wav":wav_padded,
            "wav_length":wav_lengths,
            "raw_wav":raw_wav_padded,
            "raw_wav_length":raw_wav_lengths,
            "mel":mel_padded,
            "mel_length":mel_lengths,
            "raw_mel":raw_mel_padded,
            "raw_mel_length":raw_mel_lengths,
            "text":text_padded,
            "text_length":text_lengths,
        }