import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
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

    # Play audio
    st.audio(uploaded_file)

    # Load audio
    signal, sample_rate = sf.read(uploaded_file)

    # Convert stereo → mono
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Waveform visualization
    st.subheader("Audio Waveform")

    fig, ax = plt.subplots()
    librosa.display.waveshow(signal, sr=sample_rate, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # Loading spinner
    with st.spinner("Analyzing audio..."):

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_scaled = mfcc_scaled.reshape(1, -1)

        # Prediction
        prediction = model.predict(mfcc_scaled)
        probabilities = model.predict_proba(mfcc_scaled)

        confidence = np.max(probabilities) * 100

    # Results
    st.success(f"Predicted Sound: {prediction[0]}")
    st.info(f"Confidence: {confidence:.2f}%")
