#Latest last result is this file. I should implement CBAM


from asyncore import loop
from operator import countOf
#from tkinter.ttk import _Padding
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Normalization,Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
from keras.utils import to_categorical 
from tensorflow import keras as ks
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from CbamLayer import CBAM
from CbamLayer2 import CBAM2
from CbamLayer4 import CBAM4
import os
import time
 

 
labels = ['Walking', 
          'Jogging', 
          'Stairs', 
          'Sitting', 
          'Standing', 
          'Typing',
          'Brushing Teeth',
          'Eating Soup',
          'Eating Chips',
          'Eating Pasta',
          'Drinking from Cup',
          'Eating Sandwich',
          'Kicking (Soccer Ball)',
          'Playing Catch w/Tennis Ball',
          'Dribbling (Basketball)',
          'Writing',
          'Clapping',
          'Folding Clothes']
 

rawData = []
trainData = []
trainLabels = []
testData = []
testLabels = []

counter = 0 

with open("D:\PythonTest\Test4Cnn\WorkingDatasets\\2\Preprocessed\PreprocessedDataSet.txt", "r") as filestream:  
#with open("D:\PythonTest\Test4Cnn\FakedDataSet.txt", "r") as filestream:    
    for line in filestream:
        
        #if counter>=100000:
        #    break

        #To debug reading data:
        #if counter>8000000:
        #    print("line is: " + line)
        #    currentLineValues = line.split(",")
        #    print("currentLineValues: " + str(currentLineValues))
        #else:
        #    currentLineValues = line.split(",")

        currentLineValues = line.split(",")
        userId = currentLineValues[0]
        label = currentLineValues[1]
        labelIndex = labels.index(label)
         
        timeStamp = int(currentLineValues[2])
        xAcc =float(currentLineValues[5])
        yAcc = float(currentLineValues[6])
        zAcc = float(currentLineValues[7][:-1])
        
        rawData.append([userId,timeStamp, label, xAcc, yAcc, zAcc])
        counter+=1
         

columns = ['UserId','TimeStamp', 'Activity', 'X', 'Y', 'Z']
rawDataFrame = pd.DataFrame(data=rawData, columns=columns) 


walkingActivityData = rawDataFrame[rawDataFrame.Activity == 'Walking'].sort_values(by=['UserId', 'TimeStamp'])
joggingActivityData = rawDataFrame[rawDataFrame.Activity == 'Jogging'].sort_values(by=['UserId', 'TimeStamp'])
stairsActivityData = rawDataFrame[rawDataFrame.Activity == 'Stairs'].sort_values(by=['UserId', 'TimeStamp'])
sittingActivityData = rawDataFrame[rawDataFrame.Activity == 'Sitting'].sort_values(by=['UserId', 'TimeStamp'])
standingActivityData = rawDataFrame[rawDataFrame.Activity == 'Standing'].sort_values(by=['UserId', 'TimeStamp'])
typingActivityData = rawDataFrame[rawDataFrame.Activity == 'Typing'].sort_values(by=['UserId', 'TimeStamp'])
brushingTeethActivityData = rawDataFrame[rawDataFrame.Activity == 'Brushing Teeth'].sort_values(by=['UserId', 'TimeStamp'])
eatingSoupActivityData = rawDataFrame[rawDataFrame.Activity == 'Eating Soup'].sort_values(by=['UserId', 'TimeStamp'])
eatingChipsActivityData = rawDataFrame[rawDataFrame.Activity == 'Eating Chips'].sort_values(by=['UserId', 'TimeStamp'])
eatingPastaActivityData = rawDataFrame[rawDataFrame.Activity == 'Eating Pasta'].sort_values(by=['UserId', 'TimeStamp'])
drinkingFromCupActivityData = rawDataFrame[rawDataFrame.Activity == 'Drinking from Cup'].sort_values(by=['UserId', 'TimeStamp'])
eatingSandwichActivityData = rawDataFrame[rawDataFrame.Activity == 'Eating Sandwich'].sort_values(by=['UserId', 'TimeStamp'])
kickingSoccerBallActivityData = rawDataFrame[rawDataFrame.Activity == 'Kicking (Soccer Ball)'].sort_values(by=['UserId', 'TimeStamp'])
playingCatchTennisBallActivityData = rawDataFrame[rawDataFrame.Activity == 'Playing Catch w/Tennis Ball'].sort_values(by=['UserId', 'TimeStamp'])
dribblinlgBasketballActivityData = rawDataFrame[rawDataFrame.Activity == 'Dribbling (Basketball)'].sort_values(by=['UserId', 'TimeStamp'])
writingActivityData = rawDataFrame[rawDataFrame.Activity == 'Writing'].sort_values(by=['UserId', 'TimeStamp']) 
clappingActivityData = rawDataFrame[rawDataFrame.Activity == 'Clapping'].sort_values(by=['UserId', 'TimeStamp']) 
foldingClothesActivityData = rawDataFrame[rawDataFrame.Activity == 'Folding Clothes'].sort_values(by=['UserId', 'TimeStamp']) 
 
 




minCount = min(
    walkingActivityData.shape[0], 
    joggingActivityData.shape[0], 
    stairsActivityData.shape[0], 
    sittingActivityData.shape[0], 
    standingActivityData.shape[0], 
    typingActivityData.shape[0], 
    brushingTeethActivityData.shape[0], 
    eatingSoupActivityData.shape[0], 
    eatingChipsActivityData.shape[0], 
    eatingPastaActivityData.shape[0], 
    drinkingFromCupActivityData.shape[0], 
    eatingSandwichActivityData.shape[0], 
    kickingSoccerBallActivityData.shape[0], 
    playingCatchTennisBallActivityData.shape[0], 
    dribblinlgBasketballActivityData.shape[0], 
    writingActivityData.shape[0], 
    clappingActivityData.shape[0], 
    foldingClothesActivityData.shape[0]
    )


samplingRate = 20
frameSize =samplingRate*4
overlapSize = samplingRate*2

frameSize =90
overlapSize = 45

countOfFramesOfMinActivity = int(minCount/float(overlapSize))
countOfNeededRecordsForEachActivity = countOfFramesOfMinActivity * overlapSize



walkingActivityData = walkingActivityData.head(countOfNeededRecordsForEachActivity).copy()
joggingActivityData = joggingActivityData.head(countOfNeededRecordsForEachActivity).copy()
stairsActivityData = stairsActivityData.head(countOfNeededRecordsForEachActivity).copy()
sittingActivityData = sittingActivityData.head(countOfNeededRecordsForEachActivity).copy()
standingActivityData = standingActivityData.head(countOfNeededRecordsForEachActivity).copy()
typingActivityData = typingActivityData.head(countOfNeededRecordsForEachActivity).copy()
brushingTeethActivityData = brushingTeethActivityData.head(countOfNeededRecordsForEachActivity).copy()
eatingSoupActivityData = eatingSoupActivityData.head(countOfNeededRecordsForEachActivity).copy()
eatingChipsActivityData = eatingChipsActivityData.head(countOfNeededRecordsForEachActivity).copy()
eatingPastaActivityData = eatingPastaActivityData.head(countOfNeededRecordsForEachActivity).copy()
drinkingFromCupActivityData = drinkingFromCupActivityData.head(countOfNeededRecordsForEachActivity).copy()
eatingSandwichActivityData = eatingSandwichActivityData.head(countOfNeededRecordsForEachActivity).copy()
kickingSoccerBallActivityData = kickingSoccerBallActivityData.head(countOfNeededRecordsForEachActivity).copy()
playingCatchTennisBallActivityData = playingCatchTennisBallActivityData.head(countOfNeededRecordsForEachActivity).copy()
dribblinlgBasketballActivityData = dribblinlgBasketballActivityData.head(countOfNeededRecordsForEachActivity).copy()
writingActivityData = writingActivityData.head(countOfNeededRecordsForEachActivity).copy()
clappingActivityData = clappingActivityData.head(countOfNeededRecordsForEachActivity).copy()
foldingClothesActivityData = foldingClothesActivityData.head(countOfNeededRecordsForEachActivity).copy()
 



columns = ['UserId', 'TimeStamp', 'Activity', 'X', 'Y', 'Z']
balancedDataFrame = pd.concat([walkingActivityData,
                               joggingActivityData, 
                               stairsActivityData, 
                               sittingActivityData, 
                               standingActivityData, 
                               typingActivityData, 
                               brushingTeethActivityData, 
                               eatingSoupActivityData, 
                               eatingChipsActivityData, 
                               eatingPastaActivityData, 
                               drinkingFromCupActivityData, 
                               eatingSandwichActivityData, 
                               kickingSoccerBallActivityData, 
                               playingCatchTennisBallActivityData, 
                               dribblinlgBasketballActivityData, 
                               writingActivityData, 
                               clappingActivityData, 
                               foldingClothesActivityData 
                               ], ignore_index=True)
   

del walkingActivityData
del joggingActivityData
del stairsActivityData
del sittingActivityData
del standingActivityData
del typingActivityData
del brushingTeethActivityData
del eatingSoupActivityData
del eatingChipsActivityData
del eatingPastaActivityData
del drinkingFromCupActivityData
del eatingSandwichActivityData
del kickingSoccerBallActivityData
del playingCatchTennisBallActivityData
del dribblinlgBasketballActivityData,
del writingActivityData
del clappingActivityData
del foldingClothesActivityData  



le = LabelEncoder()
balancedDataFrame['Label'] = le.fit_transform(balancedDataFrame['Activity']) 


X = balancedDataFrame[['X', 'Y', 'Z']]


scaler = StandardScaler()
X = scaler.fit_transform(X)
balancedDataFrame['X'] = [row[0] for row in X]
balancedDataFrame['Y'] = [row[1] for row in X]
balancedDataFrame['Z'] = [row[2] for row in X]
 
del X


briefData = balancedDataFrame.drop(['UserId', 'TimeStamp', 'Activity'], axis=1)

frames = []
labels = []

del balancedDataFrame


for activityNumber in range(0,18):
    frameCounterBase = (activityNumber*countOfFramesOfMinActivity)
    for counter in range(frameCounterBase+1,frameCounterBase+countOfFramesOfMinActivity): 
        fromNumber=(counter-1)*overlapSize
        toNumber = ((counter+1)*overlapSize)
        frames.append(briefData.iloc[fromNumber:toNumber].drop(['Label'],axis=1).values.tolist())
        labels.append(briefData.Label[fromNumber])
         

frames = np.asarray(frames).reshape(-1, frameSize, 3)
labels = np.asarray(labels)

del briefData


X_train, X_test, y_train, y_test = train_test_split(frames, labels, test_size = 0.2, random_state = 0, stratify = labels)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)



#this was the latest serius method
#def createModel():
#    model = Sequential()
#    model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
#    model.add(Dropout(0.1))

#    model.add(Conv2D(32, (2, 2), activation='relu'))
#    model.add(Dropout(0.2))

#    model.add(Flatten())

#    model.add(Dense(64, activation = 'relu'))
#    model.add(Dropout(0.5))

#    model.add(Dense(6, activation='softmax'))
#    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#    return model



#def createModel2():
#    model = Sequential()
#    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape))
#    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))

#    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
#    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

#    model.add(Flatten())

#    model.add(Dense(64, activation = 'relu'))
#    #model.add(Dropout(0.5))
    
#    model.add(Dense(32, activation = 'relu'))

#    model.add(Dense(6, activation='softmax'))
#    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#    return model




#def createModel3():
#    inputLayer = Input(shape=X_train[0].shape)
#    conv2d1=Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape)(inputLayer)


#    model = Sequential()
#    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape))
#    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))

#    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
#    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

#    model.add(Flatten())

#    model.add(Dense(64, activation = 'relu'))
#    #model.add(Dropout(0.5))
    
#    model.add(Dense(32, activation = 'relu'))

#    model.add(Dense(6, activation='softmax'))
#    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#    return model


#def createModel4():
#    model = Sequential()
#    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape))
#    #model.add(Dropout(0.1))

#    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
#    #model.add(Dropout(0.2))
     
#    #model.add(CBAM2())
#    model.add(CBAM4())

#    model.add(Flatten())

#    model.add(Dense(64, activation = 'relu'))
#    #model.add(Dropout(0.5))

#    model.add(Dense(6, activation='softmax'))
#    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#    return model



def CNN_Method1():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu',input_shape = X_train[0].shape))
    #model.add(Dropout(0.1))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu')) 
    #model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu')) 
    #model.add(Dropout(0.5))
    model.add(Dense(18, activation='softmax')) 
    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model


def CNN_BN_Method2():
    model = Sequential()
    #model.add(BatchNormalization(input_shape = X_train[0].shape))
    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape))
    #model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu')) 
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(64, activation = 'relu')) 
    #model.add(BatchNormalization())

    model.add(Dense(18, activation='softmax')) 
    #model.add(BatchNormalization())
    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model

def CNN_BN_CBAM_Method3():
    model = Sequential()
    #model.add(BatchNormalization(input_shape = X_train[0].shape))
    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu')) 
    model.add(BatchNormalization())
     
    model.add(CBAM4()) 
    #model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(64, activation = 'relu')) 
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Dense(18, activation='softmax')) 
    #model.add(BatchNormalization())
    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model


def plot_learningCurve(history, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show(block=True)
  
  # os.system("pause")

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show(block=True)
  
model = CNN_BN_CBAM_Method3()
# model = CNN_Method1()
model.summary()
start = time.time() 
#history = model.fit(X_train, y_train, epochs = 17, validation_data= (X_test, y_test), verbose=1)
history = model.fit(X_train, y_train, epochs = 25, validation_split=0.2, verbose=1)
end = time.time()
print(f'Training duration: {end - start}')
plot_learningCurve(history, 25)

predict_x=model.predict(X_test) 
y_pred=np.argmax(predict_x,axis=1)

mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=le.classes_, show_normed=True, figsize=(7,7)) 
plt.show(block=True)

os.system("pause")