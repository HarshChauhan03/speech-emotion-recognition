import librosa
import numpy as np

# Load Audio File
audio, sample_rate = librosa.load(
    "speech.wav",
    sr=None
)

# Extract MFCC Features
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

print("MFCC Shape:", mfcc.shape)
print("Audio Preprocessing Completed")