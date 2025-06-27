import torch
import matplotlib.pyplot as plt
import numpy as np
import torchaudio as T
import torchaudio.transforms as TT
import torchaudio.functional as F
from model import WaveGrad_light
from params import params, AttrDict
# Used for spectrogram and baseline
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN

# set noise schedule to 10 steps for this
params.noise_schedule=np.linspace(1e-6, 0.01, 10).tolist()

def get_energy_score(audio, spectrogram):
    # torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # unsqueeze audio to add a channel dimension
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    # load WaveGrad_light from state_dict .pt file
    checkpoint_path = 'weights/weights-40000.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WaveGrad_light(AttrDict(params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.params.override(params)

    energy_score = 0.0
    with torch.no_grad():
        beta = np.array(model.params.noise_schedule)
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        # Expand rank 2 tensors by adding a batch dimension.
        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.to(device)

        noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

        for n in range(len(alpha) - 1, -1, -1):
            print(n, len(alpha) - 1)
            
            # torch zeros of shape 1 model.params.hop_samples * spectrogram.shape[-1]
            noise = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1])
            audio_noisy = torch.zeros_like(noise)
            audio_noisy[0,:audio.shape[-1]] = noise_scale[n] * audio + (1.0 - noise_scale[n]**2)**0.5 *audio_noisy[0,:audio.shape[-1]]

            # print shape of audio_noisy and spectrogram
            if n > 0:
                diff = model(audio_noisy, spectrogram, noise_scale[n]).squeeze(1) - noise
                energy_score += torch.mean(diff**2).item()
    return energy_score

if __name__ == '__main__':
    # load in distribution (ID)
    path_in_distribution_wav = 'LJSpeech-1.1/LJ002-0332.wav'
    path_in_distribution_spectrogram = 'LJSpeech-1.1/LJ002-0332.wav.spec.npy'
    audio_ID, sr = T.load(path_in_distribution_wav)
    audio_ID = torch.clamp(audio_ID[0], -1.0, 1.0)
    spectrogram_ID = torch.from_numpy(np.load(path_in_distribution_spectrogram))

    # load out of distribution (OD), preprocess it
    path_out_of_distribution_wav = 'death-black-metal-suspenseful-guitar-riff_177bpm_B_minor.wav'
    audio_OD, sr = T.load(path_out_of_distribution_wav)
    audio_OD = torch.clamp(audio_OD[0], -1.0, 1.0)
    audio_OD = F.resample(audio_OD, orig_freq=sr, new_freq=params.sample_rate)
    # limit audio length to same as ID
    audio_OD = audio_OD[:audio_ID.shape[0]]
    hop = params.hop_samples
    win = hop * 4
    n_fft = 2**((win-1).bit_length())
    f_max = sr / 2.0
    mel_spec_transform = TT.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win, hop_length=hop, f_min=20.0, f_max=f_max, power=1.0, normalized=True)
    with torch.no_grad():
        spectrogram_OD = mel_spec_transform(audio_OD)
        spectrogram_OD = 20 * torch.log10(torch.clamp(spectrogram_OD, min=1e-5)) - 20
        spectrogram_OD = torch.clamp((spectrogram_OD + 100) / 100, 0.0, 1.0)

    # Kind of in disribution generated w speechbrain (KID)
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")
    spectrogram_KID, mel_length, alignment = tacotron2.encode_text("mary had a little lamb but it was not white as snow")
    audio_KID = hifi_gan.decode_batch(spectrogram_KID)
    # limit audio length to same as ID
    audio_KID = audio_KID[:audio_ID.shape[0]]
    # limit mel spectrogram length to same as ID
    spectrogram_KID = spectrogram_KID[:, :spectrogram_ID.shape[1]]
    spectrogram_KID = spectrogram_KID.squeeze(0).cpu().numpy()
    spectrogram_KID = torch.from_numpy(spectrogram_KID).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    spectrogram_KID = torch.nn.functional.interpolate(spectrogram_KID, size=(128,spectrogram_KID.shape[3]), mode='bilinear', align_corners=False)
    spectrogram_KID = spectrogram_KID.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

    # Kind of in disribution generated w speechbrain 2 (KID2) 
    spectrogram_KID2, mel_length, alignment = tacotron2.encode_text("WASP deep learning is a game changer and will revolutionize the world")
    audio_KID2 = hifi_gan.decode_batch(spectrogram_KID2)
    # limit audio length to same as ID
    audio_KID2 = audio_KID2[:audio_ID.shape[0]]
    # limit mel spectrogram length to same as ID
    spectrogram_KID2 = spectrogram_KID2[:, :spectrogram_ID.shape[1]]
    spectrogram_KID2 = spectrogram_KID2.squeeze(0).cpu().numpy()
    spectrogram_KID2 = torch.from_numpy(spectrogram_KID2).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    spectrogram_KID2 = torch.nn.functional.interpolate(spectrogram_KID2, size=(128,spectrogram_KID2.shape[3]), mode='bilinear', align_corners=False)
    spectrogram_KID2 = spectrogram_KID2.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

    # print energy of audio for ID and ODD
    print(f'Energy of in-distribution sample: {torch.mean(audio_ID**2).item():.4f}')
    print(f'Energy of kind of in-distribution sample: {torch.mean(spectrogram_KID**2).item():.4f}')
    print(f'Energy of kind of in-distribution sample 2: {torch.mean(spectrogram_KID2**2).item():.4f}')
    print(f'Energy of out-of-distribution sample: {torch.mean(audio_OD**2).item():.4f}')
    # calculate energy score for ID and ODD
    energy_score_ID = get_energy_score(audio_ID, spectrogram_ID)
    energy_score_KID = get_energy_score(audio_KID, spectrogram_KID)
    energy_score_KID2 = get_energy_score(audio_KID2, spectrogram_KID2)
    energy_score_OD = get_energy_score(audio_OD, spectrogram_OD)
    print(f'Energy score for in-distribution sample: {energy_score_ID:.4f}')
    print(f'Energy score for kind of in-distribution sample: {energy_score_KID:.4f}')
    print(f'Energy score for kind of in-distribution sample 2: {energy_score_KID2:.4f}')
    print(f'Energy score for out-of-distribution sample: {energy_score_OD:.4f}')

