#Latest last result is this file. I should implement CBAM
#Latest last result is this file. I should implement CBAM
from tkinter import FIRST
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.decomposition import PCA
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC, SVC
# from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score
import os

import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Normalization,Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow import keras as ks
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from CbamLayer4 import CBAM4





warnings.filterwarnings('ignore')
#rows * 128 * 9 * 1
#Train & Test

filesPath = 'C:\\D Drive\\PythonTest\\Test4Cnn\\WorkingDatasets\\3\\UCI HAR Dataset'


def LoadGroup(filesPath, groupTitle):
    inputFileNames = [
                 'total_acc_x_'+groupTitle+'.txt', 
                 'total_acc_y_'+groupTitle+'.txt', 
                 'total_acc_z_'+groupTitle+'.txt',
                 'body_acc_x_'+groupTitle+'.txt', 
                 'body_acc_y_'+groupTitle+'.txt', 
                 'body_acc_z_'+groupTitle+'.txt',
                 'body_gyro_x_'+groupTitle+'.txt', 
                 'body_gyro_y_'+groupTitle+'.txt', 
                 'body_gyro_z_'+groupTitle+'.txt'
                 ]
    
    groupfilesPath=f'{filesPath}\\{groupTitle}'
    
    inputfilesPath=f'{groupfilesPath}\\Inertial Signals\\'
    
    inputValues = np.loadtxt(f'{inputfilesPath}{inputFileNames[0]}')
    for counter in range(1,len(inputFileNames)):
        newData = np.loadtxt(f'{inputfilesPath}{inputFileNames[counter]}')
        inputValues = np.dstack((inputValues, newData))
    
    
    inputValues= inputValues.reshape(inputValues.shape[0], inputValues.shape[1], inputValues.shape[2], 1)
    outputValues =np.loadtxt(f'{groupfilesPath}\\y_{groupTitle}.txt')
    return inputValues, outputValues

 

X_train, y_train = LoadGroup(filesPath,'train')
X_test, y_test = LoadGroup(filesPath,'test')




labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
#1 WALKING
#2 WALKING_UPSTAIRS
#3 WALKING_DOWNSTAIRS
#4 SITTING
#5 STANDING
#6 LAYING

y_train-=1
y_test-=1

   
# from asyncore import loop
# from operator import countOf
# #from tkinter.ttk import _Padding
# import matplotlib.pyplot as plt
 

 
# labels = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
# labels2 = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']


# rawData = []
# trainData = []
# trainLabels = []
# testData = []
# testLabels = []

# counter = 0 

# # importing required module
# import csv
 









# with open("C:\\D Drive\\PythonTest\\Test4Cnn\\WorkingDatasets\\1\\CorrectedDataSet.txt", "r") as filestream:  
# #with open("D:\PythonTest\Test4Cnn\FakedDataSet.txt", "r") as filestream:    
#     for line in filestream:
        
#         #if counter>=100000:
#         #    break

#         #To debug reading data:
#         #if counter>8000000:
#         #    print("line is: " + line)
#         #    currentLineValues = line.split(",")
#         #    print("currentLineValues: " + str(currentLineValues))
#         #else:
#         #    currentLineValues = line.split(",")

#         currentLineValues = line.split(",")
#         userId = currentLineValues[0]
#         label = currentLineValues[1]
#         labelIndex = labels.index(label)
         
#         timeStamp = int(currentLineValues[2])
#         xAcc =float(currentLineValues[5])
#         yAcc = float(currentLineValues[6])
#         zAcc = float(currentLineValues[7][:-1])
        
#         rawData.append([userId,timeStamp, label, xAcc, yAcc, zAcc])
#         counter+=1
         

# columns = ['UserId','TimeStamp', 'Activity', 'X', 'Y', 'Z']
# rawDataFrame = pd.DataFrame(data=rawData, columns=columns) 


# walkingActivityData = rawDataFrame[rawDataFrame.Activity == 'Walking'].sort_values(by=['UserId', 'TimeStamp'])
# joggingActivityData = rawDataFrame[rawDataFrame.Activity == 'Jogging'].sort_values(by=['UserId', 'TimeStamp'])
# upstairsActivityData = rawDataFrame[rawDataFrame.Activity == 'Upstairs'].sort_values(by=['UserId', 'TimeStamp'])
# downstairsActivityData = rawDataFrame[rawDataFrame.Activity == 'Downstairs'].sort_values(by=['UserId', 'TimeStamp'])
# sittingActivityData = rawDataFrame[rawDataFrame.Activity == 'Sitting'].sort_values(by=['UserId', 'TimeStamp'])
# standingActivityData = rawDataFrame[rawDataFrame.Activity == 'Standing'].sort_values(by=['UserId', 'TimeStamp'])


# minCount = min(
#     walkingActivityData.shape[0],
#     joggingActivityData.shape[0],
#     upstairsActivityData.shape[0],
#     downstairsActivityData.shape[0],
#     sittingActivityData.shape[0],
#     standingActivityData.shape[0])




# samplingRate = 20
# frameSize =samplingRate*4
# overlapSize = samplingRate*2

# frameSize =90
# overlapSize = 45

# countOfFramesOfMinActivity = int(minCount/float(overlapSize))
# countOfNeededRecordsForEachActivity = countOfFramesOfMinActivity * overlapSize


# walkingActivityData = walkingActivityData.head(countOfNeededRecordsForEachActivity).copy()
# joggingActivityData = joggingActivityData.head(countOfNeededRecordsForEachActivity).copy()
# upstairsActivityData = upstairsActivityData.head(countOfNeededRecordsForEachActivity).copy()
# downstairsActivityData = downstairsActivityData.head(countOfNeededRecordsForEachActivity).copy()
# sittingActivityData = sittingActivityData.head(countOfNeededRecordsForEachActivity).copy()
# standingActivityData = standingActivityData.head(countOfNeededRecordsForEachActivity).copy()



# columns = ['UserId', 'TimeStamp', 'Activity', 'X', 'Y', 'Z']
# balancedDataFrame = pd.concat([walkingActivityData,
#                                joggingActivityData,
#                                upstairsActivityData,
#                                downstairsActivityData,
#                                sittingActivityData,
#                                standingActivityData], ignore_index=True)

# del walkingActivityData
# del joggingActivityData
# del upstairsActivityData
# del downstairsActivityData
# del sittingActivityData
# del standingActivityData


# le = LabelEncoder()
# balancedDataFrame['Label'] = le.fit_transform(balancedDataFrame['Activity']) 


# X = balancedDataFrame[['X', 'Y', 'Z']]


# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# balancedDataFrame['X'] = [row[0] for row in X]
# balancedDataFrame['Y'] = [row[1] for row in X]
# balancedDataFrame['Z'] = [row[2] for row in X]
 
# del X


# briefData = balancedDataFrame.drop(['UserId', 'TimeStamp', 'Activity'], axis=1)

# frames = []
# labels = []

# del balancedDataFrame


# for activityNumber in range(0,6):
#     frameCounterBase = (activityNumber*countOfFramesOfMinActivity)
#     for counter in range(frameCounterBase+1,frameCounterBase+countOfFramesOfMinActivity): 
#         fromNumber=(counter-1)*overlapSize
#         toNumber = ((counter+1)*overlapSize)
#         frames.append(briefData.iloc[fromNumber:toNumber].drop(['Label'],axis=1).values.tolist())
#         labels.append(briefData.Label[fromNumber])
         

# frames = np.asarray(frames).reshape(-1, frameSize, 3)
# labels = np.asarray(labels)

# del briefData


# X_train, X_test, y_train, y_test = train_test_split(frames, labels, test_size = 0.2, random_state = 0, stratify = labels)

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)



# #this was the latest serius method
# #def createModel():
# #    model = Sequential()
# #    model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
# #    model.add(Dropout(0.1))

# #    model.add(Conv2D(32, (2, 2), activation='relu'))
# #    model.add(Dropout(0.2))

# #    model.add(Flatten())

# #    model.add(Dense(64, activation = 'relu'))
# #    model.add(Dropout(0.5))

# #    model.add(Dense(6, activation='softmax'))
# #    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# #    return model



# #def createModel2():
# #    model = Sequential()
# #    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape))
# #    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))

# #    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# #    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

# #    model.add(Flatten())

# #    model.add(Dense(64, activation = 'relu'))
# #    #model.add(Dropout(0.5))
    
# #    model.add(Dense(32, activation = 'relu'))

# #    model.add(Dense(6, activation='softmax'))
# #    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# #    return model




# #def createModel3():
# #    inputLayer = Input(shape=X_train[0].shape)
# #    conv2d1=Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape)(inputLayer)


# #    model = Sequential()
# #    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape))
# #    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))

# #    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# #    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

# #    model.add(Flatten())

# #    model.add(Dense(64, activation = 'relu'))
# #    #model.add(Dropout(0.5))
    
# #    model.add(Dense(32, activation = 'relu'))

# #    model.add(Dense(6, activation='softmax'))
# #    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# #    return model


# #def createModel4():
# #    model = Sequential()
# #    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape))
# #    #model.add(Dropout(0.1))

# #    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# #    #model.add(Dropout(0.2))
     
# #    #model.add(CBAM2())
# #    model.add(CBAM4())

# #    model.add(Flatten())

# #    model.add(Dense(64, activation = 'relu'))
# #    #model.add(Dropout(0.5))

# #    model.add(Dense(6, activation='softmax'))
# #    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# #    return model



def CNN_Method1():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu',input_shape = X_train[0].shape))
    #model.add(Dropout(0.1))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu')) 
    #model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu')) 
    #model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax')) 
    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model


def CNN_BN_Method2():
    model = Sequential()
    #model.add(BatchNormalization(input_shape = X_train[0].shape))
    model.add(Conv2D(16, (3, 3), padding='same', activation = 'relu', input_shape = X_train[0].shape))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu')) 
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(64, activation = 'relu')) 
    #model.add(BatchNormalization())

    model.add(Dense(6, activation='softmax')) 
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

    model.add(Dense(6, activation='softmax')) 
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
  plt.show()
  
  os.system("pause")

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

model = CNN_BN_CBAM_Method3()
model.summary()

# results = []

# # This is to measure time duration of running algorithm
# # opening the file
# # with open("D:\Temp\CNN_BN_Method2.csv", "w", newline="") as f:
# #     # creating the writer
# #     writer = csv.writer(f)
# #     # using writerows, all rows at once
# #     for runCounter in range(0,50):
# #         start = time.time() 
# #         #history = model.fit(X_train, y_train, epochs = 17, validation_data= (X_test, y_test), verbose=1)
# #         history = model.fit(X_train, y_train, epochs = 25, validation_split=0.2, verbose=1)
# #         end = time.time()
# #         results.append(end-start)
# #         writer.writerow([end-start])


     
# #history = model.fit(X_train, y_train, epochs = 17, validation_data= (X_test, y_test), verbose=1)
history = model.fit(X_train, y_train, epochs = 25, validation_split=0.2, verbose=1)



# #print(f'Training duration: {end - start}')
plot_learningCurve(history, 25)

predict_x=model.predict(X_test) 
y_pred=np.argmax(predict_x,axis=1)

# # This is to print failed tests
# # opening the file
# with open("D:\Temp\CBAMFails.csv", "w", newline="") as f:     
# #     # creating the writer     
#     writer = csv.writer(f)
# #     # using writerows, all rows at once
#     for counter in range(0,y_test.shape[0]):
#         if y_test[counter]!=y_pred[counter]:
#             writer.writerow([f'Real: {labels2[y_test[counter]]} - Predicted as {labels2[y_pred[counter]]}'])
#             for rowCounter in range(0,90):
#                 writer.writerow([X_test[counter][rowCounter][0][0],X_test[counter][rowCounter][1][0],X_test[counter][rowCounter][2][0]])
#             writer.writerow([])
        




mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=labels, show_normed=True, figsize=(7,7)) 
plt.show()

os.system("pause")