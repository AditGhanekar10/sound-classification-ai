import librosa
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/sound_classifier.pkl")

# Path to test audio file
audio_path = "data/UrbanSound8K/audio/fold5/100032-3-0-0.wav"

print("Analyzing audio:", audio_path)

# Load audio
signal, sample_rate = librosa.load(audio_path)

# Extract MFCC features
mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sample_rate,
    n_mfcc=40
)

mfcc_scaled = np.mean(mfcc.T, axis=0)

# Reshape for model
mfcc_scaled = mfcc_scaled.reshape(1, -1)

# Predict
prediction = model.predict(mfcc_scaled)

print("Predicted sound:", prediction[0])