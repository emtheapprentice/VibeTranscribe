from sklearn.ensemble import BaggingClassifier
import pickle
import soundfile
import librosa
import numpy as np

"""Extracts the necessary features for prediction from an audio file"""

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result

"""Unpickle model and predict emotions from each finished stream's wav file"""

def vibecheck(file_name):
    model = pickle.load(open("Models/Bagging_classifier.model", "rb"))
    features = extract_feature(file_name, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    result = model.predict(features)[0]
    return result

def discard_iftooshort(file_name):
    length = librosa.get_duration(filename=file_name)
    if length > 1:
        return 1
    else:
        return 0

def vibetocolor(vibe):
    if vibe == 'neutral':
        color = 'black'
    #current version only deals with high, neutral and low moods
    elif vibe == 'happy':
        color = 'gold4'
    #elif vibe == 'sad':
    #   color = 'dark slate gray'
    elif vibe == 'sad':
        color = 'AntiqueWhite4'
    #elif vibe == 'angry':
    #    color = 'OrangeRed4'
    #elif vibe == 'fear':
    #    color = 'midnight blue'
    return color