from preprocess import preprocess_dataset
from dataset import from_path 
from params import params
from model import WaveGrad_light
import numpy as np
import torch
from glob import glob
import os
import torch.nn as nn

def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)

def state_dict(model, optimizer, step, params):
    if hasattr(model, 'module') and isinstance(model.module, nn.Module):
      model_state = model.module.state_dict()
    else:
      model_state = model.state_dict()
    return {
        'step': step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in optimizer.state_dict().items() },
        'params': dict(params)
    }

def save_to_checkpoint(model, optimizer, step, params, filename='weights'):
  save_basename = f'{filename}-{step}.pt'
  save_name = f'{save_basename}'
  link_name = f'{filename}.pt'
  torch.save(state_dict(model, optimizer, step, params), save_name)
  if os.name == 'nt':
    torch.save(state_dict(model, optimizer, step, params), link_name)
  else:
    if os.path.islink(link_name):
      os.unlink(link_name)
    os.symlink(save_basename, link_name)

if __name__ == '__main__':
    # torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get filenames
    filenames = []
    filenames += glob('LJSpeech-1.1/*.wav', recursive=True)

    # pre-process dataset (only need to run once)
    if 1:
      preprocess_dataset(filenames)

    # setup dataset
    dataset = from_path("LJSpeech-1.1", params)

    # model
    model = WaveGrad_light(params).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    # loss function
    loss_fn = torch.nn.L1Loss()

    # noise schedule
    beta = np.array(params.noise_schedule)
    noise_level = np.cumprod(1 - beta)**0.5
    noise_level = np.concatenate([[1.0], noise_level], axis=0)
    noise_level = torch.tensor(noise_level.astype(np.float32))

    # train
    step = 0
    device = next(model.parameters()).device
    losses = []
    while True:
        for features in dataset:
            print(step)
            features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            for param in model.parameters():
                param.grad = None

            audio = features['audio']
            spectrogram = features['spectrogram']

            N, T = audio.shape
            S = 1000
            device = audio.device
            noise_level = noise_level.to(device)

            s = torch.randint(1, S + 1, [N], device=audio.device)
            l_a, l_b = noise_level[s-1], noise_level[s]
            noise_scale = l_a + torch.rand(N, device=audio.device) * (l_b - l_a)
            noise_scale = noise_scale.unsqueeze(1)
            noise = torch.randn_like(audio)
            noisy_audio = noise_scale * audio + (1.0 - noise_scale**2)**0.5 * noise

            predicted = model(noisy_audio, spectrogram, noise_scale.squeeze(1))
            loss = loss_fn(noise, predicted.squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

            # Check for NaN loss
            if torch.isnan(loss).any():
                raise RuntimeError(f'Detected NaN loss at step {step}.')
            if step % 100 == 0:
                print("step: ", step, "loss: ", loss)
            if step % 1000 == 0:
                save_to_checkpoint(model, optimizer, step, params)
                # save losses to npy
                np.save(f'losses_{step}.npy', np.array(losses))
            step += 1
