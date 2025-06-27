# WASP Deep Learning Course module 3 assignment

## References
+ The code in this repository is partly reused from https://github.com/lmnt-com/wavegrad/tree/master which implements *WaveGrad: Estimating Gradients For Waveform Generation* by Nanxin Chen et al (2020). Compared to the original repository, I use a smaller version of the model (modified by myself) and custom scripts for training and sample generation.
+ The dataset used is LJSpeech-1.1. 
+ A wrapped Tacotron2 from Speechbrain is used to predict a mel-spectrogram from input text.
+ The death-black-metal *death-black-metal-suspenseful-guitar-riff_177bpm_B_minor.wav* sample out of distribution wav is taken from https://samplefocus.com/.

## Requirements
+ All dependencies are specified in *requirements.py*

## Usage
+ **1. Dataset setup (If want to train)**: Download LJSpeech-1.1 from https://keithito.com/LJ-Speech-Dataset/, extract all .wav files and put them directly into a *"LJSpeech-1.1"* directory in the root of this repo.
+ **2. Train**: If you have setup the dataset properly, you can train the diffusion model by running *train_and_save.py*. Break the process when you are satisfied with the number of iterations. Or just use the weights trained by me for 40000 iterations. When you run the training script, make sure you have no other WAV files anywhere in the repo but the dataset files.
+ **3. Generate**: To generate samples of "mary had a little lamb" using my weights trained for 40000 iterations, simply run *load_and_generate.py*. You can change the model by editing the variable *checkpoint_path* in *load_and_generate.py*. There you can also change the text to generate speech for.  
+ **4. Plot learning convergence**: Use *plot_learning_convergence.py* to plot 40000 iterations of learning in a log-log plot, demonstrating the approximate log-log linear convergence. 
+ **5. Out-of-distribution detection(OOD)**: I have implemented OOD by computing the MSE of the denoiser on a sample of data. In the report I compare an ID sample and a OD sample. You can test this by running *out_of_distribution_detection.py*.
