import gradio as gr
import librosa
import tempfile
import numpy as np
import torch
import sox
import scipy.io.wavfile as wavfile

from data import load_wav, log_mel_spectrogram, plot_mel, plot_attn
from models import load_pretrained_wav2vec
from models import FragmentVC
from copy import deepcopy

# Load the pre-trained voice conversion model
sample_rate = 16000
preemph = 0.97
hop_len = 326
win_len = 1304
n_fft = 1304
n_mels = 80
f_min = 80
decrease_factor = 0.7
wav2vec_path = "./pretrained/wav2vec_small.pt"
ckpt_path = "./pretrained/fragment.pt"
vocoder_path = "./pretrained/vocoder.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


wav2vec = load_pretrained_wav2vec(wav2vec_path).to(device)
print("[INFO] Wav2Vec is loaded from", wav2vec_path)

checkpoint = torch.load(ckpt_path, map_location=device)
model = FragmentVC()
model.load_state_dict(checkpoint)
model.eval()
model.to(device)
print("[INFO] FragmentVC is loaded from", ckpt_path)

vocoder = torch.jit.load(vocoder_path).to(device).eval()
print("[INFO] Vocoder is loaded from", vocoder_path)

tfm = sox.Transformer()
tfm.vad(location=1)
tfm.vad(location=-1)


def voice_conversion(source_audio, target_audio):
    # Convert the input audio to the desired output voice
    sr, wav = source_audio
    wav = librosa.core.resample(np.float64(np.squeeze(wav)), orig_sr=sr, target_sr=sample_rate)
    src_wav = wav / (np.abs(wav).max() + 1e-6)
    src_wav = deepcopy(tfm.build_array(input_array=src_wav, sample_rate_in=sample_rate))
    src_wav = torch.FloatTensor(src_wav).unsqueeze(0).to(device)

    sr, wav = target_audio
    wav = librosa.core.resample(np.float64(np.squeeze(wav)), orig_sr=sr, target_sr=sample_rate)
    tgt_wav = wav / (np.abs(wav).max() + 1e-6)
    tgt_wav = tfm.build_array(input_array=tgt_wav, sample_rate_in=sample_rate)
    tgt_wav = deepcopy(tgt_wav)
    tgt_mel = log_mel_spectrogram(
        tgt_wav, preemph, sample_rate, n_mels, n_fft, hop_len, win_len, f_min
    )
    tgt_mels = [tgt_mel]
    tgt_mel = np.concatenate(tgt_mels, axis=0)
    tgt_mel = torch.FloatTensor(tgt_mel.T).unsqueeze(0).to(device)

    with torch.no_grad():
        src_feat = wav2vec.extract_features(src_wav, None)['x']
        out_mel, attns = model(src_feat, tgt_mel)
        out_mel = out_mel.transpose(1, 2).squeeze(0)
        out_wav = vocoder.generate([out_mel])[0]
        out_wav = out_wav.cpu().numpy()

    # Save the converted audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, sample_rate, out_wav * decrease_factor)
        print(f.name)
        audio_file = f.name

    return audio_file

# Create the input and output interfaces for Gradio
source_audio_interface = gr.inputs.Audio(label='Source-Speaker Audio')
target_audio_interface = gr.inputs.Audio(label='Target-Speaker Audio')
output_audio_interface = gr.outputs.Audio(label='Output Audio', type='filepath')

# Create the Gradio interface
gr.Interface(
    fn=voice_conversion,
    inputs=[source_audio_interface, target_audio_interface],
    outputs=output_audio_interface,
    title='Voice Conversion for Japanese Demo').queue().launch(share=True)
