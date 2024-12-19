import torch
from torch.nn.utils.rnn import pad_sequence
from gpt.utils import make_pad_mask
from torchmetrics.classification import MulticlassAccuracy

def ce_loss(outputs, real, lengths, device, plen=None):
    E = 1e-8
    with torch.no_grad():
        mask = make_pad_mask(lengths).to(outputs.device)
        mask = 1-mask.type(torch.int32)
    loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs, real.long())
    loss = (loss * mask).sum()/(mask.sum() + E)
    
    total_cnt = mask.sum()
    
    real_1 = real * mask - (1-mask.type(torch.int32))
    top1_rate = MulticlassAccuracy(
        outputs.shape[1],
        top_k=1,
        average="micro",
        multidim_average="global",
        ignore_index=-1
    ).to(outputs.device)(outputs, real_1)
    
    right_cnt = top1_rate * total_cnt
    
    return loss, right_cnt.type(torch.int64).item(), total_cnt.type(torch.int64).item(), top1_rate.item()