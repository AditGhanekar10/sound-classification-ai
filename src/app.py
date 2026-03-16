import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Page settings
st.set_page_config(
    page_title="Sound Classification AI",
    page_icon="🎧",
    layout="centered"
)

# Header
st.title("🎧 Sound Classification AI")

st.markdown("""
Upload a **.wav audio file** and the AI will analyze the sound.

Example sounds you can try:
- dog barking
- car horn
- siren
- drilling
""")

# Load dataset
df = pd.read_csv("audio_features.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

st.success("Model ready for predictions")

# Upload file
uploaded_file = st.file_uploader(
    "Upload an audio file (.wav)",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    signal, sample_rate = sf.read(uploaded_file)

    # Convert stereo to mono
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Waveform
    st.subheader("📈 Audio Waveform")

    fig, ax = plt.subplots()
    librosa.display.waveshow(signal, sr=sample_rate, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # Prediction
    with st.spinner("🔍 Analyzing sound..."):

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_scaled = mfcc_scaled.reshape(1, -1)

        probabilities = model.predict_proba(mfcc_scaled)[0]

        classes = model.classes_

        results = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

    st.subheader("🔎 Prediction Results")

    top3 = results[:3]

    for label, prob in top3:

        st.write(f"**{label}**")

        st.progress(float(prob))

        st.write(f"{prob*100:.2f}% confidence")
