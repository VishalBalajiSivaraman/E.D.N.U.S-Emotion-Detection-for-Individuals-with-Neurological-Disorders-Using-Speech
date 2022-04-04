import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
import pickle


emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

 #DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'sad' ,'angry']
# define the base models
models = list()
models.append(('LogR',LogisticRegression(solver='liblinear')))
models.append(('SVC',SVC(gamma='scale', probability=True)))
models.append(('KNN',KNeighborsClassifier()))
models.append(('BC',BaggingClassifier(n_estimators=250)))
models.append(('RC',RandomForestClassifier(n_estimators=250)))
models.append(('EXTC',ExtraTreesClassifier(n_estimators=250)))
models.append(('XGBC',XGBClassifier()))
models.append(('MLPC',MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=750)))

# meta model
meta = LogisticRegression(solver='liblinear')

def label_encoder(y):
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    y=le.fit_transform(y)
    return y

def load_data(ts):
    x,y=[],[]
    for file in glob.glob("F:\\python\\dl programs\\SP\\DATA\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        print(emotion)
        if emotion in observed_emotions:
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
    y=label_encoder(y)
    return train_test_split(np.array(x), y, test_size=ts ,random_state=9)

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

model = StackingClassifier(estimators=models, final_estimator=meta, cv=25)
encoder = load_model('AE-5-3-encoder.h5')
ts=0.25
X_train, X_test, y_train, y_test=load_data(ts)
print('Train', X_train.shape, y_train.shape, 'Test', X_test.shape, y_test.shape)
X_train_encode = encoder.predict(X_train)
X_test_encode = encoder.predict(X_test)
model.fit(X_train_encode,y_train)
#score=model.score(X_test_encode,y_test)
filename = 'SLM.h5'
pickle.dump(model, open(filename, 'wb'))
print("Model saved succesfully!!!")
loaded_model = pickle.load(open(filename, 'rb'))
print("Loaded Model Sucessfully")
result = loaded_model.score(X_test_encode, y_test)
print('Super learner model score: %.3f'%(result*100))
yhat = loaded_model.predict(X_test_encode)
print('Super Learner: %.3f' % (accuracy_score(y_test, yhat) * 100))






 
