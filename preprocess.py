# Edited for WASP course by Martin Dahl at Linköping University
#
#
#
# ===============================================================================
# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from tqdm import tqdm

from params import params


def transform(filename):
  audio, sr = T.load(filename)
  if params.sample_rate != sr:
    raise ValueError(f'Invalid sample rate {sr}.')
  audio = torch.clamp(audio[0], -1.0, 1.0)

  hop = params.hop_samples
  win = hop * 4
  n_fft = 2**((win-1).bit_length())
  f_max = sr / 2.0
  mel_spec_transform = TT.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win, hop_length=hop, f_min=20.0, f_max=f_max, power=1.0, normalized=True)

  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())
  
def preprocess_dataset(filenames):
  for filename in tqdm(filenames, desc='Preprocessing'):
    try:
      transform(filename)
    except Exception as e:
      print(f'Error processing {filename}: {e}')
      continue
