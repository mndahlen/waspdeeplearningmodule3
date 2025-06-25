from datasets import load_dataset

# load LJ speech dataset
dataset = load_dataset("lj_speech", split="train", trust_remote_code=True)

# print the first sample    
print(dataset[0])

# plot the first audio sample
import matplotlib.pyplot as plt
plt.plot(dataset[0]['audio']['array'])
plt.show()
# print(f"Audio: {dataset[0]['audio']['path']}")
# print(f"Text: {dataset[0]['text']}\n")