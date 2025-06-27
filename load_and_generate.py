from params import params, AttrDict
from model import WaveGrad_light
import numpy as np
import torch
from glob import glob
import matplotlib.pyplot as plt
import torchaudio

# Used for spectrogram and baseline
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN

text = "mary had a little lamb"

#"Sometimes rugs were urgently required and not forthcoming" LJ002-0332

# load LJspeech 002-0332 audio and mel spectrogram (baseline)
SPECTROGRAM_BASELINE = False
if SPECTROGRAM_BASELINE:
    audio_path = f'LJSpeech-1.1/LJ002-0332.wav'
    mel_path = f'LJSpeech-1.1/LJ002-0332.wav.spec.npy'
    audio_LJ002_0332, sample_rate_LJ002_0332 = torchaudio.load(audio_path)
    spectrogram_LJ002_0332 = np.load(mel_path)
    print(f"Audio shape: {audio_LJ002_0332.shape}, Sample rate: {sample_rate_LJ002_0332}")
if __name__ == '__main__':
    # torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load WaveGrad_light from state_dict .pt file
    checkpoint_path = 'weights/weights-40000.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WaveGrad_light(AttrDict(params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.params.override(params)

    # predict mel spectrogram from text
    # Intialize TTS (tacotron2) 
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")

    # Running the TTS
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    TTS_baseline = hifi_gan.decode_batch(mel_output)
    mel_output = mel_output.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

    if SPECTROGRAM_BASELINE: # Use the actual spectrogram from the dataset
        spectrogram = torch.from_numpy(spectrogram_LJ002_0332)
    else: # use the TTS baseline mel spectrogram
        # upsample mel_output to match mel_spectrogram shape
        mel_output = torch.from_numpy(mel_output).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        mel_output = torch.nn.functional.interpolate(mel_output, size=(128,mel_output.shape[3]), mode='bilinear', align_corners=False)
        spectrogram = mel_output.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    print(f"Spectrogram shape: {spectrogram.shape}")

    # Generate using wavegrad_light
    with torch.no_grad():
        beta = np.array(model.params.noise_schedule)
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)
        
        # Expand rank 2 tensors by adding a batch dimension.
        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.to(device)
        audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
        noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
        for n in range(len(alpha) - 1, -1, -1):
            print(n, len(alpha) - 1)
            c1 = 1 / alpha[n]**0.5
            c2 = (1 - alpha[n]) / (1 - alpha_cum[n])**0.5

            audio = c1 * (audio - c2 * model(audio, spectrogram, noise_scale[n]).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)
    
    torchaudio.save("output.wav", audio.cpu(), sample_rate=params.sample_rate)

    # Tacotron2 and HiFi-GAN for TTS baseline
    torchaudio.save('TTS_baseline.wav',TTS_baseline.squeeze(1), 22050)

    # plot the audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(audio.cpu().numpy()[0], color='blue')
    plt.title('Generated Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

    # plot the TTS baseline audio waveform
    TTS_baseline = TTS_baseline.squeeze(1).cpu().numpy()[0]  # Remove batch dimension and convert to numpy
    plt.figure(figsize=(10, 4))
    plt.plot(TTS_baseline, color='blue')
    plt.title('TTS baseline Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()