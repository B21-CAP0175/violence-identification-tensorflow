import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "training_mix_forthesecondtime.h5"
panjang_sampel = 22050


def preprocess(file_path, jumlah=13, fft=2048, hop=512):
        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= panjang_sampel:
            # ensure consistency of the length of the signal
            signal = signal[:panjang_sampel]

            # extract MFCCs
            mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=jumlah, n_fft=fft,
                                         hop_length=hop)
        return mfcc.T


loaded=tf.keras.models.load_model(SAVED_MODEL_PATH)

def predict(file_path):
       
        # extract MFCC
        mfcc = preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = loaded.predict(mfcc)

        return predictions


res=predict("./dataset/testing.wav")
res

print("Prediksi Kekerasan: \n")
print("Domestik: "+str(res[0,0]*100)+"%")
print("Fisik: "+str(res[0,1]*100)+"%")
print("Seksual: "+str(res[0,2]*100)+"%")
print("Penguntitan: "+str(res[0,3]*100)+"%")