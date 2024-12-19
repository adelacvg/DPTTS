import torch
import random
from tqdm import tqdm
from gpt.utils import make_pad_mask
from gpt.llama3.tokenizer import ChatFormat, Dialog, Message, Tokenizer
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores, decoded_tokens, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids


def random_sampling(weighted_scores, decoded_tokens):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids

def setup_cache(llama_model, max_batch_size, device="cpu"):
    for layer in llama_model.layers:
        layer.attention.cache_k = torch.zeros(
            (
                max_batch_size,
                llama_model.params.max_seq_len,
                layer.attention.n_local_kv_heads,
                layer.attention.head_dim,
            ),
            device=device
        )
        layer.attention.cache_v = torch.zeros(
            (
                max_batch_size,
                llama_model.params.max_seq_len,
                layer.attention.n_local_kv_heads,
                layer.attention.head_dim,
            ),
            device=device
        )

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

def generate_t2s(
        model,
        prompt_tokens,
        prompt_lengths,
        top_k=30,
        temperature=1.0,
        top_p=0.8,
        max_gen_len=4096
):
    TEXT_VOCAB_SIZE = 128256
    SEMANTIC_VOCAB_SIZE = 2048
    TEXT_PRESERVE = 25
    SEMANTIC_PRESERVE = 25
    CODEBOOK_SIZE=1024
    N_CODEBOOKS = 8
    SEMANTIC_PAD_TOKEN = SEMANTIC_VOCAB_SIZE
    assert prompt_tokens.shape[0]==1
    total_len = 128+1500#1628
    tokens = torch.full((1, total_len), 0, dtype=torch.long, device=prompt_tokens.device)
    tokens[:, :prompt_tokens.shape[1]] = prompt_tokens
    prev_pos = 0
    code_gen = {}
    for i in range(N_CODEBOOKS):
        code_gen[i] = []
    for cur_pos in tqdm(range(prompt_lengths[0], total_len)):
        all_logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos, infer_mode=True)
        for i in range(CODEBOOK_SIZE):
            logits = all_logits[i]
        code_step = (cur_pos-129)%N_CODEBOOKS
        logits_st = code_step*CODEBOOK_SIZE
        logits_ed = (code_step+1)*CODEBOOK_SIZE
        eos_logits = logits[:, :, [SEMANTIC_PAD_TOKEN]]
        logits = logits[:, :, logits_st:logits_ed]
        if code_step==0:
            logits = torch.cat((logits, eos_logits), dim=-1)
        next_token = ras_sampling(logits[:,-1].squeeze(0), code_gen[code_step])
        # print(cur_pos, next_token)
        if code_step==0 and next_token == CODEBOOK_SIZE:
            break
        next_token = next_token.reshape(-1)+logits_st
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos
    gen_tokens = tokens[:,129:cur_pos]
    # print(gen_tokens)
    gen_tokens = gen_tokens[:,:gen_tokens.shape[1]]
    gen_tokens = gen_tokens.reshape(-1,N_CODEBOOKS).T
    # print(gen_tokens)
    for n in range(1, N_CODEBOOKS):
        gen_tokens[n, :] -= n * CODEBOOK_SIZE
    return gen_tokens