from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
from torch import nn

class Encodec(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.encodec = EncodecModel.from_pretrained(model_path)

    def forward(self, wav):
        device = wav.device
        inputs = self.processor(wav.tolist(), sampling_rate=self.processor.sampling_rate, return_tensors="pt")
        audio_codes = self.encodec(inputs["input_values"].to(device), inputs["padding_mask"].to(device)).audio_codes[0]
        return audio_codes
    def decode(self, audio_codes):
        audio_values = self.encodec.decode(audio_codes,[None])
        return audio_values
        

def get_model(model_path):
    model = Encodec(model_path)
    model.eval()
    return model

# # load the model + processor (for pre-processing the audio)
# model = EncodecModel.from_pretrained("facebook/encodec_24khz")
# processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# # cast the audio data to the correct sampling rate for the model
# librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
# audio_sample = librispeech_dummy[0]["audio"]["array"]

# # pre-process the inputs
# inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

# # explicitly encode then decode the audio inputs
# encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
# audio_values = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]

# # or the equivalent with a forward pass
# audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values

# # you can also extract the discrete codebook representation for LM tasks
# # output: concatenated tensor of all the representations
# audio_codes = model(inputs["input_values"], inputs["padding_mask"]).audio_codes