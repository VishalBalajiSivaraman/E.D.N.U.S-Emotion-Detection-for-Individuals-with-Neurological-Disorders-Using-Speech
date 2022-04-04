import streamlit as st
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import librosa
import soundfile
import random
import pickle
from twilio.rest import Client
import numpy as np
import smtplib 
from tensorflow.keras.models import load_model
from scipy.io import wavfile
import time
import xlwt
from xlwt import Workbook
import datetime 
df=pd.read_csv('F:/python/projects/Ednus/CSV/Info.csv')
ds = pd.read_csv('F:/python/projects/Ednus/CSV/options.csv')
dv =pd.read_csv('F:/python/projects/Ednus/CSV/LV.csv') 
symbols = df['Info'].tolist()
sb = ds['OPT'].tolist()
sv = dv['OPT'].tolist()
tk = st.sidebar.selectbox('Choose your desired option from the available maintopics ',sb)
tc = st.sidebar.selectbox('Choose your desired choice from the available subtopics under Technical Information',sv)
ticker = st.sidebar.selectbox('Choose your desired choice from the available subtopics under Project Information',symbols)
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

observed_emotions=['calm', 'happy', 'sad','angry']

encoder = load_model('F:/python/projects/Ednus/Models/AE-5-3-encoder.h5')
filename = 'F:/python/projects/Ednus/Models/SLM.h5'
loaded_model = pickle.load(open(filename, 'rb'))
def coupons():
    coupons=['SUDDENLY10','FREEFORALL','DOUPGRADE25','enjoyurself45','20045067','klm234der','PLATINUM257','IOWACAPT909','ImPaSse100','BKNOW24905']
    fin=len(coupons)-1
    cpn = random.randint(0,fin)
    cn = str(coupons[cpn])
    return cn
def verification(name,age,pn,en,cp):
    if len(str(name))!=0 and len(str(age))!=0 and len(str(pn))!=0 and len(str(en))!=0 and len(str(cp))!=0:
        return 1
    else:
        return -1
    
def verify(name,age,pn,en):
    if len(str(name))!=0 and len(str(age))!=0 and len(str(pn))!=0 and len(str(en))!=0:
        return 1
    else:
        return -1
    
def sfile(uploadedFile):
    if uploadFile is not None:
        samplerate, data = wavfile.read(uploadedFile)
        if samplerate!=0:
            return 1

def load_data(file):
    x=[]
    feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    X=np.array(x)
    return X

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

def process(uploadFile):
    X_test=load_data(uploadFile)
    X = encoder.predict(X_test)
    yhat = loaded_model.predict(X)
    yhat =np.ravel(yhat)
    y=sum(yhat)
    emotion=observed_emotions[y]
    return emotion
    
    

if (tk == 'Project Information'):
    if (ticker == 'Abstract & Introduction'):
        st.title('  E.D.N.U.S: Emotion Detection for Individuals with Neurological Disorders Using Speech  ')
        st.header('**Abstract**')
        st.markdown('Human emotion recognition plays an important role in the interpersonal relationship. Emotions are reflected from speech, hand and gestures of the body and through facial expressions.')
        st.markdown('Hence extracting and understanding of emotion is crucial step which has to be undertaken in order to safeguard the interests of the general public as well as the user.')
        st.markdown('Since, emotions are prone to be exhibited from facial expressions, body movement and gestures, and speech. As a result, the technology is said to contribute in the emergence of the so-called emotional or emotive Internet of all the available systems available, we the team propose a novel Speech Emotion Recognition (SER) system where the task is to recognize the emotion from speech irrespective of the semantic contents.')
        st.markdown('Although; emotions are subjective and even for humans it is hard to notate them in natural speech communication, the proposed system outweighs the odds in any given situation.')
        st.markdown('The reason for our choosing would be due to the drawbacks present in the Traditional Facial Recognition system, where the machine makes an erroneous decision with regard to classification of an emotion based on the body language exhibited to the similarity of body language exhibited by another emotion.')
        st.markdown('As disclosed earlier we the team propose a novel hybrid SER system titled E.D.N.U.S in order to outweigh the odds.')
        st.markdown('The proposed system **E.D.N.U.S** stands for **Emotion Detection for Individuals with Neurological Disorders Using Speech**.')
        st.markdown('The proposed versatile system incorporates a user interface which gathers the inputs from the user especially the person"s audio file, followed by which the concept of feature extraction is performed on the input audio signal features by the proposed autoencoder model, upon which the features are sent to the proposed Super learner model as input for identifying calm, happy, sad and angry emotions, which could be customized based on consumer needs.')
        st.header('**Introduction**')
        st.markdown('Emotion is a psychological condition that is linked to the neurological system. It is what a person feels on the inside as a result of the environment in his immediate surroundings.')
        st.markdown('A person"s emotions can be sensed in a variety of ways. Tonal characteristics, face expression, and body gesture are some of the predominant ways in which one can conclude the person’ emotions.')
        st.markdown('Human information processing includes the computation or categorization of emotion based on speech or facial expression, out of which computation / categorization of emotion based on speech signal shows more accurate results due to the fact that the facial expression would be the same for certain emotions like guilt, fear, lie, etc. Hence as a result during categorization of the emotion, the model would result in erroneous decisions.')
        st.markdown('So based on the above validated conclusion we the team have devised an ideal, cost effective, applicable solution which we term as')
        st.markdown('**E.D.N.U.S: Emotion Detection for Individuals with Neurological Disorders Using Speech**')
        st.markdown('From the above acronym one can conclude that the proposed system would be categorizing the emotion of any individual based on his speech sample (these include talking, short sentences, etc.), Based on the results appropriate actions could be taken for maintaining the wellbeing of the person.')
    elif (ticker == 'Literature Survey & Related Work'):
        st.header('**Literature Survey**')
        st.markdown('**Topic:**')
        st.markdown('Emotion recognition from speech: a review')
        st.markdown('**Summary:**')
        st.markdown('This research article looks at speech emotion detection, with a particular focus on subjects like emotional speech corpora, distinct types of speech features, and models for identifying emotions from speech.')           
        st.markdown('**Link:**https://link.springer.com/article/10.1007/s10772-011-9125-1')
        st.markdown('**Topic:**')
        st.markdown('Speech based human emotion recognition using MFCC')
        st.markdown('**Summary:**')            
        st.markdown('The speaker"s emotions are identified using data collected from the speaker speech signal in this research study. The Mel Frequency Cepstral Coefficient (MFCC) method is used to detect a speaker"s mood through their voice. The developed method was evaluated for the emotions of happiness, sadness, and rage, and the efficiency was determined to be about 80%.')
        st.markdown('**Link:**https://ieeexplore.ieee.org/abstract/document/8300161')
        st.markdown('**Topic:**')
        st.markdown('Hidden Markov model-based speech emotion recognition.')
        st.markdown('**Summary:**')            
        st.markdown('The introduction of speech emotion identification using continuous hidden Markov models is the subject of this research article. Throughout the article, two techniques are propagated and compared. The first technique uses Gaussian mixture models to classify an utterance"s global statistical framework, which is generated from the raw pitch and energy contour of the speech signal. A second technique adds temporal complexity by employing continuous hidden Markov models that take into account many states and use low-level instantaneous characteristics rather than global statistics.')
        st.markdown('**Link:**https://ieeexplore.ieee.org/abstract/document/1202279')
        st.markdown('**Topic:**')
        st.markdown('Emotion Recognition in Speech Using Neural Networks')
        st.markdown('**Summary:**')            
        st.markdown('This study examines why emotion detection in speech is an important and practical research issue, as well as a system for emotion recognition that employs one-class-in-one neural networks.')
        st.markdown('**Link:**https://link.springer.com/article/10.1007/s005210070006')
        st.markdown('**Topic:**')
        st.markdown('Multimodal Speech Emotion Recognition Using Audio and Text.')
        st.markdown('**Summary:**')            
        st.markdown('This study proposes a unique deep dual recurrent encoder model that employs both text and audio signals at the same time to gain a better comprehension of speech data. Because emotional conversation is made up of both sound and spoken content, our model uses dual recurrent neural networks (RNNs) to store the information from audio and text sequences and then integrates it to predict the emotion class. This architecture analyses voice data from the signal to the language level, allowing it to make better use of the data"s information than models that only focus on audio characteristics.')
        st.markdown('**Link:**https://ieeexplore.ieee.org/abstract/document/8639583')
        st.header('**Related Work**')
        st.markdown('The project has been presented in the form of a journal & an article earlier, we the team have taken the liberty of attaching the same.')
        st.markdown('**Journal:** Real Time Emotion Detection from Speech Using Raspberry Pi 3')
        st.markdown('In this journal the team have followed the approach of using few models and comparing the same individually accuracy wise, the team had also mentioned that the working environment for implementing their version of the project would be MATLAB, followed by which the team have presented a brief introduction (inclusion of Autoencoder model) with regard to their version of the four-class emotion classification model using Raspberry pi 3 (which currently lacks the processing power of supporting our version of the project).')
        st.markdown('**Article:**Speech Emotion Recognition using Python')
        st.markdown('In this article the programmer has followed the approach of using a single classification model (Multi-Layer Perceptron (MLP)Classification Model) for the classification of four different emotions (i.e., four class emotion classification model). We the team would also like to point out that the programmer has performed hyper tuning for the model which resulted in an accuracy score of 72 percent, at this stage we the team would also like to conclude that no autoencoders were used for the same.')            
    elif (ticker == 'Project Overview'):
        st.header('**Project Overview**')
        st.markdown('**Artificial Intelligence** is the theory and development of computer systems able to perform tasks normally requiring human intelligence, such as visual perception, speech recognition, decision-making, and translation between languages.')
        st.markdown('**Machine Learning** is defined as the use and development of computer systems that are able to learn and adapt without following explicit instructions,by using algorithms and statistical models to analyse and draw inferences from patterns in data.')
        st.markdown('**Deep Learning** is a type of machine learning based on artificial neural networks in which multiple layers of processing are used to extract progressively higher-level features from data.')
        st.markdown('**Speech processing** is the study of speech signals and the processing methods of signals. The signals are usually processed in a digital representation, so speech processing can be regarded as a special case of digital signal processing, applied to speech signals')
        st.header('**Theme**')
        st.markdown('Machine Learning/Artificial Intelligence /Data Science/Speech Processing')
        st.header('**Technology Stack Used**')
        st.markdown('**1.)Python IDE:** Since Python would be the preferred language for designing/implementation of the algorithm due to wide variety of libraries and support.')
        st.markdown('**2.)Pytorch:** Vital Library required to process and boost the efficiency of the algorithm(Note GPU support is also available based on the host configuration).')
        st.markdown('**3.)TensorFlow:** Vital Library required to construct and validate the efficacy of the model (Note GPU support is also available based on the host configuration).')
        st.markdown('**4.)Streamlit:** Vital Library required to host the proposed project as a interactive functional website using the same platform Python.')
        st.markdown('**5.)SMS API services(optional):** For alerting the user with respect to the results of the Phycological Analysis Tool (P.A.T).')
        st.header('**Our Objective**')
        st.markdown('**Short term Objective (Project)**')
        st.markdown('We envision to construct a fully functional system using  software ,which would be compatible with any host system.')
        st.markdown('**Long term Objective (Product)**')
        st.markdown('We envision to construct a unique, interactive Health Bot which would be embedded into a cloud service, followed by which the users could access the same using a dedicated Interactive website.')
        st.header('**Existing Systems**')
        st.markdown('The existing system in general refer to the current techniques/ products employed in real time or available in the market which aims at tackling the designated problem statement.')
        st.markdown('The existing systems for the proposed deep learning-based speech emotion recognition system, are the manual interrogation or the chatting process which may be time consuming, often leads to wrong conclusions and violates privacy.')
        st.markdown('The other alternative system would be the polygraph machine, which is prone to tampering of data.')
        st.markdown('**Manual Interrogation Process**')
        image1 = Image.open('Pictures/MIP.png')
        st.image(image1, caption = 'Manual Interrogation Process', use_column_width = True)
        st.markdown('**Drawbacks**')
        st.markdown('1)Time Consuming')
        st.markdown('2)Often leads to wrong Conclusion')
        st.markdown('3)Violates Suspect Rights')
        st.markdown('**Polygraph Machine**')
        image2 = Image.open('Pictures/PM.png')
        st.image(image2, caption = 'Polygraph Machine', use_column_width = True)
        st.markdown('**Drawbacks**')
        st.markdown('1)Suspects bypass the system')
        st.markdown('2)Tampering of Data is possible')
    elif (ticker == 'Procedure& Project Novelty'):
        st.header('**Project Procedure**')
        st.markdown('The proposed deep learning based speech recognition system works by the following steps:')
        st.markdown('**Step1:**Initially the subject would upload the audio file recording of his/ her speech.')
        st.markdown('**Step2:**The speech signal present in the file would be processed into an audio signal that would be sent to the model which is embedded in the User Interface for validation.')
        st.markdown('**Step3:**Upon completion of input validation, the proposed autoencoder model would extract the best speech features from the input speech features such as Mel, Chroma & Mel Frequency Cepstral Coefficients (MFCC), followed by which the best features are supplied as input to the proposed Super Learner Model, which would classify the emotion corresponding to the audio signal. Appropriate action may be taken by user for their wellbeing, based on the emotion classification report.')
        st.header('**Process Overview**')
        st.markdown('As disclosed earlier, the proposed system would require a speech sample from the user in order to perform the thorough review, thus once when the audio file which consists of the speech sample is uploaded through the User Interface (UI) to the system.')
        st.markdown('The system would then execute the process of feature extraction on the input audio file where key speech characteristics encoded in the audio file are detected and retrieved by the speech feature extraction code which is incorporated in the driver code.')
        st.markdown('These speech features are then supplied as input to the proposed Deep Autoencoder which leverages the function of Feature engineering i.e. the best features are discovered and extracted from the given set of speech features.')
        st.markdown('These best characteristics are subsequently offered asinput to the proposed super learner model for emotion analysis. Once the processof emotion analysis is performed by the super learner model i.e., when the appropriate emotion for the specified input audio file is effectively concluded by the model.')
        st.markdown('The ensuing feeling is eventually conveyed back to the user, upon which measures might be done for the person’s wellness.')            
        st.header('**Workflow Chart**')
        image = Image.open('Pictures/FC.gif')
        st.image(image, caption = 'Project Workflow Chart', use_column_width = True)
        st.header('**Project Novelty**')
        st.markdown('We the team have proposed a unique approach to the project with the following specifications')
        st.markdown('**1) Multi Label Classification :** The proposed classification model can classify up to 4 unique/similar emotions with high accuracy.')
        st.markdown('**2) Feature Engineering : ** Speech Features extracted by a customized neural network framework, termed as the autoencoder is incorporated')
        st.markdown('**3) A Super learner Classification Model : ** The proposed learner model comprises of a stack of eight different unique hyper tuned models. The model as a whole facilitates precise classification as a result yield’s high accuracy.')

        
        
    elif (ticker == 'Features & Target Audience'):
        st.header('**Project Features**')
        st.markdown('The Project offers several features namely')
        st.markdown('**1)Accesbility**: The proposed system incorporates a unique interactive website,which could be easily be accessed by users across the globe with the aid of internet.')
        st.markdown('**2)Security**: The proposed system restricts access to the designated user alone,data provided by the user feel safe.')
        st.markdown('**3)Precision**: The proposed system as a whole yield better results with high efficiency when compared to results yielded by other models')
        st.markdown('**4)Compatability**: The proposed system is highly compatible and can be accessed across a wide range of devices.')
        st.markdown('**5)Low Time / Memory Consumption**: The proposed system would consume low time and space while processing results with high efficiency.')
        st.markdown('**6)Customizable**: The proposed system can be customized based on consumer requirements')
        st.header('**Target Audience**')
        st.markdown('**1)Police/Military Police (MP) Interrogations**')
        st.markdown('**2)Psychiatrist Therapy Sessions**')
        st.markdown('**3)Company Recruitment process and much more**')
        st.markdown('**4) Employee Stress Test**')
        st.markdown('**5) Psychological Evaluation Tool (PET) for individuals**')
            
    elif (ticker == 'Conclusion & References'):
        st.header ('**Conclusion**')
        st.markdown('Therefore, it can be concluded that the proposed Super learner model incorporated with a Deep Autoencoder model outperforms other models with its ground breaking results on a dense dataset like speech. As a result, based on the above set of conclusive results it can be positively concluded that the proposed project has been implemented based on the features rendered earlier Furthermore, it is proposed to take this project forward as a product by incorporating certain features elaborated under the future scope section.')
        st.header('**Goal:**')
        st.markdown('We envision to construct a fully functional system using Raspberry pi which would be compatible with any host system.')
        st.header('**Proposed Product Name:**')
        st.markdown('E.D.N.U.S: Emotion Detection for Individuals with Neurological Disorders Using Speech')
        st.header('**References**')
        st.markdown('**Topic:**')
        st.markdown('1)Stuti Juyal, Chirag Killa, Gurvinder Pal Singh, Nishant Gupta, Vedika Gupta ‘Emotion Recognition from Speech Using Deep Neural Network')
        st.markdown('**Link:**https://link.springer.com/chapter/10.1007/978-3-030-76167-7_1')
        st.markdown('**Topic:**')
        st.markdown('2)Huang, F., Zhang, J., Zhou, C. et al. A deep learning algorithm using a fully connected sparse autoencoder neural network for landslide susceptibility prediction. Landslides 17, 217–229 (2020). DOI: 10.1007/s10346-019-01274-9 ')
        st.markdown('**Link:**https://link.springer.com/article/10.1007%2Fs10346-019-01274-9')
        st.markdown('**Topic:**')
        st.markdown('3)R. A. Khalil, E. Jones, M. I. Babar, T. Jan, M. H. Zafar, and T. Alhussain, "Speech Emotion Recognition Using Deep Learning Techniques: A Review," in IEEE Access, vol. 7, pp. 117327-117345, 2019, DOI: 10.1109/ACCESS.2019.2936124 ')
        st.markdown('**Link:**https://ieeexplore.ieee.org/document/8805181')
        st.markdown('**Topic:**')
        st.markdown('4)Mehebub Sahana, Binh Thai Pham, Manas Shukla, Romulus Costache, Do Xuan Thu, Rabin Chakrabortty, Neelima Satyam, Huu Duy Nguyen, Tran Van Phong, Hiep Van Le, Subodh Chandra Pal, G. Areendran, Kashif Imdad & Indra Prakash (2020) Rainfall induced landslide susceptibility mapping using novel hybrid soft computing methods based on multi-layer perceptron neural network classifier, Geocarto International')
        st.markdown('**Link:**DOI: 10.1080/10106049.2020.1837262')
        st.markdown('**Topic:**')
        st.markdown('5)Sheena Christabel Pravin, Palanivelan, M, ‘A Hybrid Deep Ensemble for Speech Disfluency Classification’, Circuits, Systems, and Signal Processing, Springer, vol. 40, no.8, pp. 3968-3995, July 2021')
        st.markdown('**Link:**DOI: 10.1080/10106049.2020.1837262')
        st.markdown('**Topic:**')
        st.markdown('5)Sheena Christabel Pravin, Palanivelan, M, ‘A Hybrid Deep Ensemble for Speech Disfluency Classification’, Circuits, Systems, and Signal Processing, Springer, vol. 40, no.8, pp. 3968-3995, July 2021')
        st.markdown('**Link:**https://www.researchgate.net/publication/349228170_A_Hybrid_Deep_Ensemble_for_Speech_Disfluency_Classification')
        st.markdown('**Topic:**')
        st.markdown('6)Krishnan, P.T., Joseph Raj, A.N. & Rajangam, V. Emotion classification from speech signal based on empirical mode decomposition and non-linear features. Complex Intell. Syst. 7, 1919–1934 (2021). DOI:10.1007/s40747-021-00295-z')
        st.markdown('**Link:**https://link.springer.com/article/10.1007%2Fs40747-021-00295-z')
        st.markdown('**Topic:**')
        st.markdown('7)Mohamad Nezami, O., Jamshid Lou, P. & Karami, M. ShEMO: a large-scale validated database for Persian speech emotion detection. Lang Resources & Evaluation 53, 1–16 (2019).DOI: 10.1007/s10579-018-9427-x ')
        st.markdown('**Link:**https://link.springer.com/article/10.1007%2Fs10579-018-9427-x')
        st.markdown('**Topic:**')
        st.markdown('8)M. Deshpande and V. Rao, "Depression detection using emotion artificial intelligence," 2017 International Conference on Intelligent Sustainable Systems (ICISS), 2017, pp. 858-862, DOI: 10.1109/ISS1.2017.8389299 ')
        st.markdown('**Link:**https://ieeexplore.ieee.org/document/8389299')
        st.markdown('**Topic:**')
        st.markdown('9)M. N. Stolar, M. Lech, R. S. Bolia and M. Skinner, "Real-time speech emotion recognition using RGB image classification and transfer learning," 2017 11th International Conference on Signal Processing and Communication Systems (ICSPCS), 2017, pp. 1-8, DOI: 10.1109/ICSPCS.2017.8270472')
        st.markdown('**Link:**https://link.springer.com/article/10.1007%2Fs40747-021-00295-z')
        st.markdown('**Topic:**')
        st.markdown('10)J. D. Arias-Londono, J. I. Godino-Llorente, M. Markaki, and Y. Stylianou, On combining information from modulation spectra and Mel-frequency cepstral coefficients for automatic detection of pathological voices, Logoped. Phoniatr. Vocol. 36(2) (2011) 60–69.')
        st.markdown('**Link:**https://pubmed.ncbi.nlm.nih.gov/21073260')
    elif (ticker == 'Project Team'):
         st.header('**Project Team**')
         st.markdown('**Team Guide:**')
         st.markdown('**Dr. Sheena Christabel Pravin**')
         st.markdown('Assistant Professor')
         st.markdown('Department of Electronics and Communication Engineering')
         st.markdown('**Institution:Rajalakshmi Engineering College**')
         st.markdown('**Team Members:**')
         st.markdown('**Mr. Surendaranath.K**')
         st.markdown('Final year student,180801201')
         st.markdown('Department of Electronics and Communication Engineering')
         st.markdown('**Institution:Rajalakshmi Engineering College**')
         st.markdown('**Mr. Vishal.B**')
         st.markdown('Final year student,180801223')
         st.markdown('Department of Electronics and Communication Engineering')
         st.markdown('**Institution:Rajalakshmi Engineering College**')
         st.markdown('**Mr. Vishal Balaji Sivaraman**')
         st.markdown('Final year student,180801224')
         st.markdown('Department of Electronics and Communication Engineering')
         st.markdown('**Institution:Rajalakshmi Engineering College**')
         
elif (tk == 'Technical Information'):
    if (tc == 'Proposed Flow'):
        st.header('**Working Principle**')
        st.markdown('**Two pivotal components namely Autoencoder and Super Learner Model (S.L.M) contribute as a backbone in the proposed project.**')
        st.header('**Autoencoder**')
        st.markdown('An Autoencoder is a customizable feed forward neural network which comprises of an encoder and decoder model. Now the primary function of the autoencoder model is to reduce complexity by performing dimensionality reduction (selecting the best features), which in turn would contribute to a boost in accuracy during training of the parent model. In our project we propose a **Deep Autoencoder**.')
        st.header('**Super learner Model**')
        st.markdown('The parent model chosen for training would be a super learner model which is more or less a cascaded structure of a number of machine learning models , so since this project focuses on Emotion classification based on speech signal hence we propose a **Super Learner Classification Algorithm**.')
        st.header('**Project Algorithm WorkFlowchart**')
        st.markdown('The proposed algorithm flowchart incorporated in this project. Initially the audio features Mel, Chroma and MFCC, are extracted from the audio data provided. The features are then fed to the Autoencoder to select the best features using dimensionality reduction. These selected features are fed to the Super learner model which uses a combination of machine learning and Deep Learning algorithm to finally predict the emotion felt by the provider.')
        image3 = Image.open('Pictures/paw.png')
        st.image(image3, caption = 'Algorithm WorkFlowChart', use_column_width = True)
    elif (tc == 'Model Architecture'):
        st.header('**Super Learner Model Architecture**')
        st.markdown('Super Learner is an algorithm that uses cross-validation to estimate the performance of multiple machine learning models, or the same model with different settings. It then creates an optimal weighted average of those models, aka an "ensemble", using the test data performance. The below diagram explains the architecture of the super learner model equipped.')
        image6 = Image.open('Pictures/SLM.png')
        st.image(image6, caption = 'Super Learner Model Architecture', use_column_width = True)
        st.header('**Autoencoder Architecture**')
        st.markdown('An autoencoder is a neural network architecture capable of discovering structure within data in order to develop a compressed representation of the input. The autoencoder in general comprises of three sections namely encoder, bottle neck and the decoder. The encoder section focuses on shrinking the number of input features fed to the model. The bottle neck section provides the best set of features engineered from the model. Finally, the decoder section aims at reconstructing the features.')
        image4 = Image.open('Pictures/AE.png')
        st.image(image4, caption = 'Autoencoder General Architecture', use_column_width = True)
        st.header('**Proposed Deep Autoencoder: Complete Autoencoder Flowchart**')
        image11 = Image.open('Pictures/FAE.png')
        st.image(image11, caption = 'Complete Autoencoder Flowchart', use_column_width = True)
        st.header('**Proposed Deep Autoencoder: Encoder  Section**')
        image5 = Image.open('Pictures/DAE.png')
        st.image(image5, caption = 'Encoder Section Flowchart', use_column_width = True)
        st.markdown('Furthermore, the first figure  under this section renders the complete flowchart of a seven-layer Deep autoencoder network where the first three layers are dedicated to serve the functions of an encoder. Similarly, the last three layers are dedicated to serve the functions of a decoder. The middle layer which consists of 30 neurons is dedicated to serve as the Bottle Neck layer. In general, the neurons in the first layer of the encoder section are adjusted to the number of features (180).Similarly, the neurons in the final layer of the decoding portion are adjusted to the number of features (180). In addition, all the layers’ activation functions have been modified to Rectified Linear Unit (ReLU) with an addition of special functions such as Batch Normalization and Leaky ReLu respectively, while the secon figure under this section represents the flowchart of the proposed autoencoder’s encoder section.')
    elif (tc == 'Results'):
        st.header('Model Results')
        st.markdown('**Autoencoder results**')
        st.markdown('the proposed Deep Autoencoder model achieves a training score of 0.9948 and a validation score of 0.9558, with minimal training and validation loss of 0.05 for a total of 10,000 epochs respectively.')
        image7 = Image.open('Pictures/AR.png')
        st.image(image7, caption = 'Autoencoder Results', use_column_width = True)
        st.markdown('**Super learner results**')
        image8 = Image.open('Pictures/SR.png')
        st.image(image8, caption = 'Super learner  Results', use_column_width = True)
        st.markdown('**Graphical Comparision between the models**')
        st.markdown('The images rendered below depict the various Super Learner metrics of the model with and without using Deep Autoencoder.')
        image11 = Image.open('Pictures/ACM.png')
        st.image(image11, caption = 'Accuracy Chart', use_column_width = True)
        image12 = Image.open('Pictures/F1M.png')
        st.image(image12, caption = 'F1- Chart', use_column_width = True)
        image13 = Image.open('Pictures/JCM.png')
        st.image(image13, caption = 'Jaccard Score Chart', use_column_width = True)
        image14 = Image.open('Pictures/HLM.png')
        st.image(image14, caption = 'Hamming Loss Chart', use_column_width = True)
        st.markdown('**Confusion Matrix**')
        st.markdown('A confusion matrix is a table that is often used to describe the performanceof a classification model (or "classifier") on a set of test data for which the truevalues are known. The confusion matrix itself is relatively simple to understand,but the related terminology can be confusing.')
        st.markdown('**Confusion Matrix : SLM model with the inclusion of Autoencoder**')
        image9 = Image.open('Pictures/CFMS.png')
        st.image(image9, caption = 'Confusion Matrix for SLM model with the inclusion of Autoencoder', use_column_width = True)
        st.markdown('**Confusion Matrix : SLM model without the inclusion of Autoencoder**')
        image10 = Image.open('Pictures/CFMY.png')
        st.image(image10, caption = 'Confusion Matrix for SLM model without the inclusion of Autoencoder', use_column_width = True)
        st.markdown('**Project outcome**')
        st.markdown('The Figures rendered below portray the comprehensive functioning of the suggested system. Concurrently, the figures also demonstrate the User interface (UI) characteristics incorporated in the system in order to promote improved user experience.')
        image15 = Image.open('Pictures/OP1.png')
        st.image(image15, caption = 'Project Outcome1', use_column_width = True)
        image16 = Image.open('Pictures/OP2.png')
        st.image(image15, caption = 'Project Outcome2', use_column_width = True)
            
elif (tk == 'Product Demo'):
    st.title('**Product Demo Form**')
    st.header('Kindly enter the details with care')
    form = st.form(key='my_form')
    name = form.text_input("Enter your full name",max_chars=30)  
    age = form.number_input("Enter your age", 0, 100, 18, 1)
    pn = form.text_input("Enter your phonenumber", max_chars=10)
    en = form.text_input("Enter your email-id",max_chars=50)
    cn = coupons()
    form.markdown('**{}**'.format(cn))
    cp = form.text_input("Enter the coupon code")
    if cp != cn:
        form.markdown('**Please enter the  coupon code properly**')
    elif len(str(cp))!=0:
        form.markdown('**Please enter the  coupon code**')
    else:
        form.markdown('**Coupon Code Sucessfully accepted**')
    form.header('**Audio File Upload**')
    uploadFile = form.file_uploader(label="Upload image", type=['wav', 'mp4','mp3'])
    submit = form.form_submit_button('Submit')
    if submit:
            flg = verification(name,age,pn,en,cp)
            if flg ==1:
                form.write('**Details have been verified sucessfully**')
                cin = sfile(uploadFile)
                if cin:
                    form.write("Audio File Uploaded Successfully")
                form.write('**Details have been sucessfuly saved**')
                emotion=process(uploadFile)
                time.sleep(10)
                form.write("**The predicted emotion is {}**".format(emotion))
                time.sleep(15)
                form.write("**Thank you {} for using our demo session , hope the product has satisfied your needs, we look forward in registering and purchasing our liscensed version of the same**".format(str(name)))
            else:
                time.sleep(5)
                form.write('**Kindly fill the details**')


                
            
            
    
    
            
            
            
            
            
                
                
                
                
            
            
            
            
            
            
            
            

        
        
        
        
        
        
        
    
    
    
    
    

