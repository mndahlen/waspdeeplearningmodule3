from params import params, AttrDict
from model import WaveGrad_light
import numpy as np
import torch
from glob import glob
import matplotlib.pyplot as plt
import torchaudio
from speechbrain.inference.TTS import Tacotron2

text = "Mary had a little lamb"

if __name__ == '__main__':
    # torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load WaveGrad_light from state_dict .pt file
    checkpoint_path = 'weights/weights-25000.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WaveGrad_light(AttrDict(params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.params.override(params)

    # predict mel spectrogram from text
    # Intialize TTS (tacotron2) 
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
    # Running the TTS
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    mel_output = mel_output.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

    # load and plot mel spectrogram from file in LJSpeech-1.1
    mel_files = glob('LJSpeech-1.1/*.spec.npy')
    mel_file = mel_files[0]  # Use the first mel spectrogram file
    mel_spectrogram = np.load(mel_file)

    # upsample mel_output to match mel_spectrogram shape
    mel_output = torch.from_numpy(mel_output).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    mel_spectrogram = torch.from_numpy(mel_spectrogram).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    mel_output = torch.nn.functional.interpolate(mel_output, size=(128,710), mode='bilinear', align_corners=False)
    spectrogram = mel_output.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

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

    # plot the audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(audio.cpu().numpy()[0], color='blue')
    plt.title('Generated Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()