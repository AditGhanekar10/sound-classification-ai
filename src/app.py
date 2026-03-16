import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier

st.title("Sound Classification AI")

# Load dataset
df = pd.read_csv("audio_features.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

st.success("Model ready")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    signal, sample_rate = sf.read(uploaded_file)

    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_scaled = np.mean(mfcc.T, axis=0)
    mfcc_scaled = mfcc_scaled.reshape(1, -1)

    prediction = model.predict(mfcc_scaled)

    st.success(f"Predicted Sound: {prediction[0]}")
