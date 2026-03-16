import streamlit as st
import librosa
import numpy as np
import joblib
import soundfile as sf

# Load trained model
model = joblib.load("models/sound_classifier.pkl")

st.title("Sound Classification AI")

st.write("Upload an audio file and the AI will predict the sound.")

# Upload audio file
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    # Read audio
    signal, sample_rate = sf.read(uploaded_file)

    # Convert to mono if stereo
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_scaled = np.mean(mfcc.T, axis=0)

    mfcc_scaled = mfcc_scaled.reshape(1, -1)

    # Predict
    prediction = model.predict(mfcc_scaled)

    st.success(f"Predicted Sound: {prediction[0]}")