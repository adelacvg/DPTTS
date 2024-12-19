from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import torch
import torchaudio
from torch import nn

class W2VBert(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.processor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)
        self.w2vbert = Wav2Vec2BertModel.from_pretrained(model_name, local_files_only=True)

    def forward(self, wav):
        device = wav.device
        inputs = self.processor(wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        input_value = inputs.input_features.to(device)
        
        with torch.no_grad():
            outputs = self.w2vbert(input_value)
        return outputs.last_hidden_state.transpose(1,2)

def get_model(model_path):
    model = W2VBert(model_path)
    model.eval()
    return model