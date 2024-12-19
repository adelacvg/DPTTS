import torch


def get_model(path, device):
    import whisper

    model = whisper.load_model(path, device=device)

    return model.encoder


def get_content(model=None, wav_16k_tensor=None):
    from whisper import log_mel_spectrogram, pad_or_trim

    dev = next(model.parameters()).device
    mel = log_mel_spectrogram(wav_16k_tensor).to(dev)[:, :3000]
    # if torch.cuda.is_available():
    #     mel = mel.to(torch.float16)
    feature_len = mel.shape[-1] // 2
    assert mel.shape[-1] < 3000, "输入音频过长，只允许输入30以内音频"
    with torch.no_grad():
        in_feat = pad_or_trim(mel, 3000)
        feature = model(in_feat)[
            :mel.shape[0], :feature_len, :
        ].transpose(1, 2)
    return feature