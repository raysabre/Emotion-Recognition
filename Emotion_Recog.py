import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import librosa,librosa.display
import seaborn as sns
import os
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv1D,Conv2D,MaxPooling2D,MaxPooling1D
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from sklearn import metrics

def emotionfun(x):
    y=int(x)
    if(y==1):
        return ("neutral")
    elif(y==2):
        return ("calm")
    elif(y==3):
        return ("happy")
    elif(y==4):
        return ("sad")
    elif(y==5):
        return ("angry")
    elif(y==6):
        return ("fearful")
    elif(y==7):
        return ("disgust")
    elif(y==8):
        return ("surprised")


def intensityfun(x):
    y=int(x)
    if(y==1):
        return ("normal")
    elif(y==2):
        return ("strong")

def statementfun(x):
    y=int(x)
    if(y==1):
        return ("Kids are talking by the door")
    elif(y==2):
        return ("Dogs are sitting by the door")

def repetitionfun(x):
    y=int(x)
    if(y==1):
        return ("First Repetition")
    elif(y==2):
        return ("Second Repetition")

def maleorfemalefun(x):
    y=int(x)
    if(y%2==0):
        return ("Female")
    elif(y%2!=0):
        return ("Male")

def traindf_function():
    
    emotion=[]
    intensity=[]
    statement=[]
    repetition=[]
    maleorfemale=[]

    count=0
    Ids=[]

    for i in range(1,49):
        
        if(i<=9):
            directory='/datadrivee/intern/ram/Audio_Speech/Actor_0'+str(i)+'/'
        elif(i>9 and i<25):
            directory='/datadrivee/intern/ram/Audio_Speech/Actor_'+str(i)+'/'
        elif(i>=25 and i<34):
            directory='/datadrivee/intern/ram/Audio_Song/Actor_0'+str((i-24))+'/'
        else:
            directory='/datadrivee/intern/ram/Audio_Song/Actor_'+str((i-24))+'/'
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                count+=1
                x=filename.split("-")
                emotion.append(x[2])
                intensity.append(x[3])
                statement.append(x[4])
                repetition.append(x[5])
                y=x[6].split('.')
                maleorfemale.append(y[0])
                Ids.append(filename)    
                
    for i in range(count):
        emotion[i]=emotionfun(emotion[i])
        intensity[i]=intensityfun(intensity[i])
        statement[i]=statementfun(statement[i])
        repetition[i]=repetitionfun(repetition[i])
        maleorfemale[i]=maleorfemalefun(maleorfemale[i])
        
    df=pd.DataFrame({'Emotion':emotion,'Intensity':intensity,'statement':statement,'Repetition':repetition,'Gender':maleorfemale},index=Ids)
    return df

def windows(data,window_size):
    start=0
    while (start<len(data)):
        yield start,start+window_size
        start+=int(window_size/2)

def specgramfunction(frames,bands):
    x=[]
    sr=[]
    log_specgrams=[]
    j=0
    labels=[]
    clip_id=[]
    window_size=512*(frames-1) # frames=81,bands=60
    for i in range(1,49):
        
        if(i<=9):
            directory='/datadrivee/intern/ram/Audio_Speech/Actor_0'+str(i)+'/'
        elif(i>9 and i<25):
            directory='/datadrivee/intern/ram/Audio_Speech/Actor_'+str(i)+'/'
        elif(i>=25 and i<34):
            directory='/datadrivee/intern/ram/Audio_Song/Actor_0'+str((i-24))+'/'
        else:
            directory='/datadrivee/intern/ram/Audio_Song/Actor_'+str((i-24))+'/'
            
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                x.append(librosa.load(directory+filename)[0])
                sound_clip=x[j]
                sr.append(librosa.load(directory+filename)[1])
                for (start,end) in windows(x[j],window_size):
                    if(len(x[j][start:end])==window_size):
                        signal=sound_clip[start:end]
                        melspec=librosa.feature.melspectrogram(signal,n_mels=bands)
                        logspec=librosa.amplitude_to_db(melspec)
                        logspec=logspec.T.flatten()[:,np.newaxis].T
                        log_specgrams.append(logspec)
                        clip_id.append(filename)
                    
                j+=1
    
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)       
    return features,clip_id

def mfccLSTM(frames,bands):
    x=[]
    sr=[]
    mfccs=[]
    j=0
    labels=[]
    clip_id=[]
    window_size=512*(frames-1)
    for i in range(1,49):
        
        if(i<=9):
            directory='/datadrivee/intern/ram/Audio_Speech/Actor_0'+str(i)+'/'
            
        elif(i>9 and i<25):
            directory='/datadrivee/intern/ram/Audio_Speech/Actor_'+str(i)+'/'

        elif(i>=25 and i<34):
            directory='/datadrivee/intern/ram/Audio_Song/Actor_0'+str((i-24))+'/'
            
        else:
            directory='/datadrivee/intern/ram/Audio_Song/Actor_'+str((i-24))+'/'
        
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                x.append(librosa.load(directory+filename)[0])
                sound_clip=x[j]
                sr.append(librosa.load(directory+filename)[1])
                for (start,end) in windows(x[j],window_size):
                    if(len(x[j][start:end])==window_size):
                        signal=sound_clip[start:end]
                        mfcc=librosa.feature.mfcc(y=signal, sr=sr[j],n_mfcc = bands).T.flatten()[:, np.newaxis].T
                        mfccs.append(mfcc)
                        clip_id.append(filename)
                    
                j+=1
    features=np.asarray(mfccs).reshape(len(mfccs),bands,frames)
    return features,clip_id

def mfccslogmean(bands):
    mfccs=[]
    labels=[]
    clip_id=[]
    for i in range(1,49): # bands=40
        
        if(i<=9):
            directory='/datadrivee/intern/ram/Audio_Speech/Actor_0'+str(i)+'/'
            
        elif(i>9 and i<25):
            directory='/datadrivee/intern/ram/Audio_Speech/Actor_'+str(i)+'/'

        elif(i>=25 and i<34):
            directory='/datadrivee/intern/ram/Audio_Song/Actor_0'+str((i-24))+'/'
            
        else:
            directory='/datadrivee/intern/ram/Audio_Song/Actor_'+str((i-24))+'/'
        
        
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                sound_clip,sample_rate=librosa.load(directory+filename,res_type='kaiser_fast',duration=3,offset=0.5)
                mfcc=np.mean(librosa.feature.mfcc(y=sound_clip,sr=sample_rate,n_mfcc=bands).T,axis=0)
                mfccs.append(mfcc)
                clip_id.append(filename)
    return mfccs,clip_id
            

def labelfunction(df,clip_id):
    labels=[]
    for i in clip_id:
        emotion=df.index ==i
        gender=df.index ==i
        lol=df[emotion]['Emotion']
        lolaf=df[gender]['Gender']
        lol1=str(lol).split("   ")
        lolaf1=str(lolaf).split("   ")
        lol2=str(lol1[1]).split("\n")
        lolaf2=str(lolaf1[1]).split("\n")
        labels.append(lolaf2[0]+"_"+lol2[0])
    return labels

def choosefeaturetype(df,check,bands,frames):
    if(check==1):
        features,clip_id=specgramfunction(frames,bands)
        labels=labelfunction(df,clip_id)
        return features,labels
    if(check==2):
        features,clip_id=mfccLSTM(frames,bands)
        labels=labelfunction(df,clip_id)
        return features,labels
    if(check==3):
        features,clip_id=mfccslogmean(bands)
        labels=labelfunction(df,clip_id)
        return features,labels
        

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def get_train_test(df,check,bands,frames):
    
    features,labels=choosefeaturetype(df,check,bands,frames)
    X_train,X_test,y_train,y_test=train_test_split(features,labels,test_size=0.2,random_state=30)
    
    y_train=np.array(y_train)
    lb=LabelEncoder()
    y_train=np_utils.to_categorical(lb.fit_transform(y_train))
    
    y_test=np.array(y_test)
    lb=LabelEncoder()
    y_test=np_utils.to_categorical(lb.fit_transform(y_test))
    
    num_labels=y_train.shape[1]
    
    return X_train,X_test,y_train,y_test,num_labels

def specgramCNNmodel(bands,frames,num_labels):
    model=Sequential()
    
    model.add(Conv2D(128,kernel_size=3, border_mode='same', input_shape=(bands, frames, 2)))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(256,kernel_size=3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(256, kernel_size=3, border_mode='valid'))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    
    model.add(Dense(256, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation("softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
    return model

def lstmModel(bands,frames,num_labels):
    lstm_model=Sequential()
    lstm_model.add(LSTM(1024,return_sequences=True,input_shape=(bands,frames)))
    lstm_model.add(Dropout(0.4))

    lstm_model.add(LSTM(512,return_sequences=True))
    lstm_model.add(Dropout(0.4))

    lstm_model.add(LSTM(256))
    lstm_model.add(Dropout(0.4))

    lstm_model.add(Dense(512))
    lstm_model.add(Activation("relu"))
    lstm_model.add(Dropout(0.5))

    lstm_model.add(Dense(256))
    lstm_model.add(Activation("relu"))
    lstm_model.add(Dropout(0.4))

    lstm_model.add(Dense(num_labels))
    lstm_model.add(Activation("softmax"))
    lstm_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return lstm_model

def conv1Dmodel(bands,num_labels):
    model=Sequential()
    model.add(Conv1D(128,kernel_size=5,padding='same',input_shape=(bands,1)))
    model.add(Activation('relu'))

    model.add(Conv1D(128,kernel_size=5,padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(128,kernel_size=5,padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128,kernel_size=5,padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128,kernel_size=5,padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
    return model

def lstm_train(X_train,y_train,bands,frames,num_labels,filepath,load_existing=False,
               batch_size=16,epochs=250,val_split=0.2):
    if(load_existing):
        lstm_model=load_model(filepath)
    else:    
        lstm_model=lstmModel(bands,frames,num_labels)
    checkpoint = ModelCheckpoint(filepath,save_best_only=True)
    callbacks_list=[checkpoint]
    
    lstm_model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=val_split,callbacks=callbacks_list)
    
    return lstm_model

def specgramCNN_train(X_train,y_train,bands,frames,num_labels,filepath,load_existing=False,
               batch_size=16,epochs=250,val_split=0.2):
    if(load_existing):
        specnn_model=load_model(filepath)
    else:    
        specnn_model=specgramCNNmodel(bands,frames,num_labels)
    checkpoint = ModelCheckpoint(filepath,save_best_only=True)
    callbacks_list=[checkpoint]
    
    specnn_model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=val_split,callbacks=callbacks_list)
    
    return specnn_model

def conv1D_train(X_train,y_train,bands,frames,num_labels,filepath,load_existing=False,
               batch_size=16,epochs=250,val_split=0.2):
    if(load_existing):
        conv1D_model=load_model(filepath)
    else:    
        conv1D_model=conv1Dmodel(bands,num_labels)
    checkpoint = ModelCheckpoint(filepath,save_best_only=True)
    callbacks_list=[checkpoint]
    
    conv1D_model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=val_split,callbacks=callbacks_list)
    
    return conv1D_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def model_metrics(model,X_test,y_test):
    y_predict=model.predict(X_test)
    y_test_non_category = [ np.argmax(t) for t in y_test ]
    y_predict_non_category = [ np.argmax(t) for t in y_predict ]

    
    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
    sns.heatmap(conf_mat,annot=True,annot_kws={"size": 16})
    plt.show()
    print(classification_report(y_test_non_category,y_predict_non_category))
if __name__ == '__main__':
	df=traindf_function()
	X_train,X_test,y_train,y_test,num_labels=get_train_test(df,2,40,81)
	lstm_train(X_train,y_train,40,81,num_labels,'ram/lstm_model_gpu.h5',epochs=1000)


