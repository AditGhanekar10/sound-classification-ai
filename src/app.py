import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(
    page_title="Sound Classification AI",
    page_icon="🎧",
    layout="centered"
)

# Header
st.title("🎧 Sound Classification AI")
st.markdown(
"""
Upload an **audio (.wav) file** and the AI will analyze the sound and predict what it is.

Examples you can try:
- dog barking  
- car horn  
- siren  
- drilling
"""
)

# Load dataset
df = pd.read_csv("audio_features.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

st.success("Model ready for predictions")

uploaded_file = st.file_uploader(
    "Upload your audio file",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    signal, sample_rate = sf.read(uploaded_file)

    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    st.subheader("📈 Audio Waveform")

    fig, ax = plt.subplots()
    librosa.display.waveshow(signal, sr=sample_rate, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    with st.spinner("🔍 Analyzing sound..."):

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_scaled = mfcc_scaled.reshape(1, -1)

        prediction = model.predict(mfcc_scaled)
        probabilities = model.predict_proba(mfcc_scaled)

        confidence = np.max(probabilities)

    st.subheader("🔎 Prediction Result")

    st.success(f"Predicted Sound: **{prediction[0]}**")

    st.progress(float(confidence))

    st.write(f"Confidence: **{confidence*100:.2f}%**")
