# Sound Classification AI 🎧

An AI-powered environmental sound classification system that identifies urban sounds from audio files.

The model analyzes audio signals using **MFCC (Mel-Frequency Cepstral Coefficients)** and predicts the sound category using a **Random Forest machine learning model**.

A web application built with Streamlit allows users to upload audio files and instantly get predictions.

---

## Features

* Environmental sound recognition
* MFCC-based audio feature extraction
* Machine learning classification
* Web interface for audio upload and prediction
* Fast predictions for new audio files

---

## Sound Classes

The model can recognize the following sounds:

* dog_bark
* children_playing
* car_horn
* siren
* drilling
* street_music
* engine_idling
* air_conditioner
* jackhammer
* gun_shot

---

## Dataset

The model is trained on the **UrbanSound8K dataset**, which contains **8,732 labeled audio clips** across 10 environmental sound categories.

Dataset structure:

```
UrbanSound8K
│
├── audio
│   ├── fold1
│   ├── fold2
│   └── ...
│
└── metadata
    └── UrbanSound8K.csv
```

---

## Project Structure

```
sound-classification-ai
│
├── src
│   ├── app.py
│   ├── predict_sound.py
│   └── train_model.py
│
├── models
│   └── sound_classifier.pkl
│
├── data
│   └── audio_features.csv
│
└── README.md
```

---

## Machine Learning Pipeline

```
Audio File
     ↓
Load Audio (Librosa)
     ↓
MFCC Feature Extraction
     ↓
Feature Vector (40 features)
     ↓
Random Forest Classifier
     ↓
Sound Prediction
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* Librosa
* Scikit-learn
* Streamlit

---

## Running the Web App

Install dependencies:

```
pip install streamlit librosa numpy pandas scikit-learn joblib soundfile
```

Run the application:

```
streamlit run src/app.py
```

Then open:

```
http://localhost:8501
```

---

## Example Workflow

1. Upload a `.wav` audio file
2. The system extracts MFCC features
3. The trained model analyzes the sound
4. The predicted class is displayed

---

## Future Improvements

* Deep learning CNN model for higher accuracy
* Real-time microphone sound detection
* Live spectrogram visualization
* Cloud deployment

---

## Author

Adit Ghanekar
