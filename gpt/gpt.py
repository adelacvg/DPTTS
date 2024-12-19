from gpt.llama3.model import Transformer, ModelArgs
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
import torch
import random
from gpt.utils import make_pad_mask
from gpt.llama3.tokenizer import ChatFormat, Dialog, Message, Tokenizer
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

def sample_top_p_top_k(probs, p, k):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    if k > 0:
        probs_sort[:, k:] = 0
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class GPT(nn.Module):
    def __init__(self, hps, stage, infer=False):
        super().__init__()
        self.stage = stage
        self.hps = hps
        if stage == 't2s':
            self.init_t2s(hps, infer)
            #TODO
        elif stage == 's2a':
            self.vocab_a_size = hps.s2a.vocab_a_size
            self.vocab_s_size = hps.s2a.vocab_s_size
            self.n_q = hps.s2a.n_q
            self.init_s2a(hps, infer)
            self.vocab_s = nn.Embedding(hps.s2a.vocab_s_size, hps.s2a.dim)
            self.vocab_a = nn.ModuleList([nn.Embedding(hps.s2a.vocab_a_size, hps.s2a.dim) for _ in range(hps.s2a.n_q)])
        
    def init_t2s(self,hps, infer):
        self.model = Transformer(ModelArgs(**hps.t2s,inference_mode = infer))
    def init_s2a(self,hps, infer):
        self.model = Transformer(ModelArgs(**hps.s2a,inference_mode = infer))
    def preprocess_t2s(self, text_token, text_token_length, semantic_token, semantic_token_length):
        pass
    def preprocess_s2a(self, semantic_token, semantic_token_length, acoustic_token, acoustic_token_length):
        # <sos> prompt_semantic <st> text <et> semantic <eos>
        #semantic_token (B,T)
        #acoustic_token (B,K,T)
        # print(semantic_token.shape, semantic_token_length, acoustic_token.shape, acoustic_token_length)
        device = semantic_token.device
        semantic_list = unpad_sequence(semantic_token, semantic_token_length, batch_first=True)
        acoustic_list = unpad_sequence(acoustic_token, acoustic_token_length, batch_first=True)
        all_list = []
        prompt_lengths = []
        for i in range(len(semantic_list)):
            prompt_length = random.randint(10, int(2/3*acoustic_list[i].shape[-1]))
    
            # 随机选择起始位置
            st = random.randint(0, acoustic_list[i].shape[-1] - prompt_length)
            ed = st + prompt_length
            prompt_list = acoustic_list[i][:,st:ed] #rand (10, 500) length from b
            prompt_lengths.append(prompt_length)
            prompt_emb = sum([self.vocab_a[k](prompt_list[k]) for k in range(self.n_q)])
            semantic_emb = self.vocab_s(semantic_list[i])
            acoustic_emb = sum([self.vocab_a[k](acoustic_list[i][k]) for k in range(self.n_q)])
            all_emb = torch.cat((self.vocab_s(torch.LongTensor([self.vocab_s_size-2]).to(device)),
                                   prompt_emb,
                                   self.vocab_s(torch.LongTensor([self.vocab_s_size-1]).to(device)),
                                   semantic_emb,
                                   self.vocab_a[0](torch.LongTensor([self.vocab_a_size-2]).to(device)),
                                   acoustic_emb,
                                   self.vocab_a[0](torch.LongTensor([self.vocab_a_size-1]).to(device))))
            all_list.append(all_emb)
        all_token = pad_sequence(all_list, batch_first=True).to(device)
        prompt_lengths = torch.LongTensor(prompt_lengths).to(device)
        return all_token, prompt_lengths
    def infer_preprocess_s2a(self, semantic_token, semantic_token_length, acoustic_token, acoustic_token_length):
        device = semantic_token.device
        semantic_list = unpad_sequence(semantic_token, semantic_token_length, batch_first=True)
        acoustic_list = unpad_sequence(acoustic_token, acoustic_token_length, batch_first=True)
        all_list = []
        prompt_lengths = []
        for i in range(len(semantic_list)):
            prompt_length = random.randint(10, int(2/3*acoustic_list[i].shape[-1]))
    
            # 随机选择起始位置
            st = random.randint(0, acoustic_list[i].shape[-1] - prompt_length)
            ed = st + prompt_length
            prompt_list = acoustic_list[i][:,st:ed] #rand (10, 500) length from b
            prompt_lengths.append(prompt_length)
            prompt_emb = sum([self.vocab_a[k](prompt_list[k]) for k in range(self.n_q)])
            semantic_emb = self.vocab_s(semantic_list[i])
            acoustic_emb = sum([self.vocab_a[k](acoustic_list[i][k]) for k in range(self.n_q)])
            all_emb = torch.cat((self.vocab_s(torch.LongTensor([self.vocab_s_size-2]).to(device)),
                                   prompt_emb,
                                   self.vocab_s(torch.LongTensor([self.vocab_s_size-1]).to(device)),
                                   semantic_emb,
                                   self.vocab_a[0](torch.LongTensor([self.vocab_a_size-2]).to(device))
                                   ))
            all_list.append(all_emb)
        all_token = pad_sequence(all_list, batch_first=True).to(device)
        prompt_lengths = torch.LongTensor(prompt_lengths).to(device)
        return all_token, prompt_lengths
    def forward(self, token_a, token_a_length, token_b, token_b_length):
        if self.stage=='s2a':
            all_emb, prompt_lengths = self.preprocess_s2a(token_a, token_a_length, token_b, token_b_length)
            out = self.model(all_emb,0)[:, :-1, :]
            out = [self.output[i](out).float() for i in range(self.n_q)]
        elif self.stage == 't2s':
            target = all_token[:, 1:]
            out = self.output(out).float()
            
        return out, prompt_lengths
    @torch.inference_mode()
    def generate_s2a(
            self,
            prompt_embs,
            prompt_lengths,
            top_k=30,
            temperature=1.0,
            top_p=0.8,
            max_gen_len=4096
    ):
        params = self.model.params
        bsz = len(prompt_embs)
        device = prompt_embs.device
        assert bsz==1

        min_prompt_len = torch.min(prompt_lengths).cpu().item()
        max_prompt_len = torch.max(prompt_lengths).cpu().item()
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        embs = torch.full((bsz, total_len, prompt_embs.shape[-1]), 0, dtype=torch.long, device=device)
        embs[:, :prompt_embs.shape[1],:] = prompt_embs

        prev_pos = 0
        # eos_reached = torch.tensor([False] * bsz, device=device)
        token_length = torch.tensor([min_prompt_len] * bsz, device=device)
        input_text_mask = ~make_pad_mask(prompt_lengths, maxlen=total_len)
        out_tokens = torch.zeros(bsz, max_gen_len, self.n_q, dtype = torch.long).to(device)
        
        
        for cur_pos in range(min_prompt_len, total_len):
            latent = self.model.forward(embs[:, prev_pos:cur_pos,:], prev_pos)
            logits = [self.output[i](latent).float() for i in range(self.n_q)]
            next_tokens = []
            for i in range(len(logits)):
                if i>0:
                    logits[i][:, -1, self.vocab_a_size-1] = float("-inf")
                logits[i][:, -1, self.vocab_a_size-2] = float("-inf")
                if temperature > 0:
                    probs = torch.softmax(logits[i][:, -1] / temperature, dim=-1)
                    next_token = sample_top_p_top_k(probs, top_p, top_k)
                else:
                    next_token = torch.argmax(logits[i][:, -1], dim=-1)
                next_tokens.append(next_token)
            for i in range(self.n_q):
                out_tokens[:,cur_pos,i] = next_tokens[i]
            next_emb = sum([self.vocab_a[k](next_tokens[k]) for k in range(self.n_q)])
            # next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            # next_token = torch.where(
            #     input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            # )
            embs[:, cur_pos, :] = next_emb
            eos_reached = (next_tokens[0] == self.vocab_a_size-1)
            # eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == vocab_b_size-1)
            if eos_reached:
                break
            token_length += 1
            prev_pos = cur_pos
            # if all(eos_reached):
                # break

        return out_tokens, token_length
    @torch.inference_mode()
    def generate_t2s(
            self,
            prompt_tokens,
            prompt_lengths,
            vocab_b_size,
            top_k=30,
            temperature=1.0,
            top_p=0.8,
            max_gen_len=4096
    ):
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = torch.min(prompt_lengths).cpu().item()
        max_prompt_len = torch.max(prompt_lengths).cpu().item()
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        tokens = torch.full((bsz, total_len), 0, dtype=torch.long, device=prompt_tokens.device)
        tokens[:, :prompt_tokens.shape[1]] = prompt_tokens

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=prompt_tokens.device)
        token_length = torch.tensor([min_prompt_len] * bsz, device=prompt_tokens.device)
        input_text_mask = ~make_pad_mask(prompt_lengths, maxlen=total_len)
        
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits = logits[:, :, :vocab_b_size]
            logits[:, -1, vocab_b_size-2] = float("-inf")
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p_top_k(probs, top_p, top_k)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == vocab_b_size-1)
            token_length += 1 - eos_reached.long()
            prev_pos = cur_pos
            if all(eos_reached):
                break

        return tokens, token_length
    # def generate(
    #     self,
    #     prompt_tokens: List[List[int]],
    #     max_gen_len: int,
    #     temperature: float = 0.6,
    #     top_p: float = 0.9,
    #     logprobs: bool = False,
    #     echo: bool = False,
    # ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    #     """
    #     Generate text sequences based on provided prompts using the language generation model.

    #     Args:
    #         prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
    #         max_gen_len (int): Maximum length of the generated text sequence.
    #         temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
    #         top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
    #         logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
    #         echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    #     Returns:
    #         Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

    #     Note:
    #         This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
    #         If logprobs is True, token log probabilities are computed for each generated token.

    #     """
    #     params = self.model.params
    #     bsz = len(prompt_tokens)
    #     assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    #     min_prompt_len = min(len(t) for t in prompt_tokens)
    #     max_prompt_len = max(len(t) for t in prompt_tokens)
    #     assert max_prompt_len <= params.max_seq_len
    #     total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    #     pad_id = self.tokenizer.pad_id
    #     tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    #     for k, t in enumerate(prompt_tokens):
    #         tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    #     if logprobs:
    #         token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

    #     prev_pos = 0
    #     eos_reached = torch.tensor([False] * bsz, device="cuda")
    #     input_text_mask = tokens != pad_id
    #     if min_prompt_len == total_len:
    #         logits = self.model.forward(tokens, prev_pos)
    #         token_logprobs = -F.cross_entropy(
    #             input=logits.transpose(1, 2),
    #             target=tokens,
    #             reduction="none",
    #             ignore_index=pad_id,
    #         )

    #     stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

    #     for cur_pos in range(min_prompt_len, total_len):
    #         logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
    #         if temperature > 0:
    #             probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
    #             next_token = sample_top_p(probs, top_p)
    #         else:
    #             next_token = torch.argmax(logits[:, -1], dim=-1)

    #         next_token = next_token.reshape(-1)
    #         # only replace token if prompt has already been generated
    #         next_token = torch.where(
    #             input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
    #         )
    #         tokens[:, cur_pos] = next_token
    #         if logprobs:
    #             token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
    #                 input=logits.transpose(1, 2),
    #                 target=tokens[:, prev_pos + 1 : cur_pos + 1],
    #                 reduction="none",
    #                 ignore_index=pad_id,
    #             )
    #         eos_reached |= (~input_text_mask[:, cur_pos]) & (
    #             torch.isin(next_token, stop_tokens)
    #         )
    #         prev_pos = cur_pos
    #         if all(eos_reached):
    #             break

    #     if logprobs:
    #         token_logprobs = token_logprobs.tolist()
    #     out_tokens, out_logprobs = [], []
    #     for i, toks in enumerate(tokens.tolist()):
    #         # cut to max gen len
    #         start = 0 if echo else len(prompt_tokens[i])
    #         toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
    #         probs = None
    #         if logprobs:
    #             probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
    #         # cut to after eos tok if any
    #         for stop_token in self.tokenizer.stop_tokens:
    #             try:
    #                 eos_idx = toks.index(stop_token)
    #                 toks = toks[:eos_idx]
    #                 probs = probs[:eos_idx] if logprobs else None
    #             except ValueError:
    #                 pass
    #         out_tokens.append(toks)
    #         out_logprobs.append(probs)
    #     return (out_tokens, out_logprobs if logprobs else None)
