from asyncore import loop
import tensorflow as tf
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import random
import os
from keras.utils import to_categorical 
from tensorflow import keras as ks
import math


x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]])

print("Numpy 2D-array is: ", x)

x = tf.reshape(x, [1, 3, 3, 1])

print("Numpy 2D-array is: ", x)

max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),   strides=(1, 1), padding='valid')
max_pool_2d(x)
#<tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
#  array([[[[5.],
#           [6.]],
#          [[8.],
#           [9.]]]], dtype=float32)>



 
 
labels = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
labelsArrays = [
    [1.0,0.0,0.0,0.0,0.0,0.0], 
    [0.0,1.0,0.0,0.0,0.0,0.0], 
    [0.0,0.0,1.0,0.0,0.0,0.0], 
    [0.0,0.0,0.0,1.0,0.0,0.0], 
    [0.0,0.0,0.0,0.0,1.0,0.0], 
    [0.0,0.0,0.0,0.0,0.0,1.0], 
   ]

rawData = []
trainData = []
trainLabels = []
testData = []
testLabels = []

counter = 0
trainSize = 0.7
testSize = 0.3

with open("D:\PythonTest\Test4Cnn\CorrectedDataSet.txt", "r") as filestream:  
#with open("D:\PythonTest\Test4Cnn\FakedDataSet.txt", "r") as filestream:    
    for line in filestream:
        
        if counter>=100000:
            break

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
        xAcc =float(currentLineValues[3])
        yAcc = float(currentLineValues[4])
        zAcc = float(currentLineValues[5][:-1])
        
        rawData.append([labelsArrays[labelIndex],xAcc,yAcc,zAcc])
        counter+=1

     
        
         
         
trainCount =int(counter*0.7)




print(f'Count of all rows: {counter}')
packs = math.floor(counter/45)
print(f'count of floor rows: {packs}')


#0-90
#45-135
#90-180 
raw2dData =[]
for loopCounter in range(0,packs-1):
    fromIndex = loopCounter * 45
    toIndex = (loopCounter+2) * 45

    raw2dData.append(rawData[fromIndex:toIndex])



trainCount =int(packs*0.7)



trainRawData =raw2dData[0:trainCount] 



for loopCounter in range(trainCount):
    rawPack = trainRawData[loopCounter]
    dataPack = []
    labelPack = []

    for loopCounter2 in range(90):
        dataPack.append(rawPack[loopCounter2][1:]) 
        labelPack.append(rawPack[loopCounter2][0])

    trainData.append(dataPack)
    trainLabels.append(labelPack)

     
 
trainData = np.array(trainData)
trainLabels =np.array(trainLabels)

 
print('TrainData[0].shape: ',trainData[0].shape) 
print('TrainData[0]: ',trainData[0]) 
 









testRawData =raw2dData[trainCount:] 

for loopCounter in range(packs-trainCount-1):
    rawPack = testRawData[loopCounter]
    dataPack = []
    labelPack = []

    for loopCounter2 in range(90):
        dataPack.append(rawPack[loopCounter2][1:]) 
        labelPack.append(rawPack[loopCounter2][0])

    testData.append(dataPack)
    testLabels.append(labelPack)

     
 
 
testData = np.array(testData)
testLabels = np.array(testLabels)

 
print('TestData[0].shape: ',testData[0].shape) 
print('TestData[0]: ',testData[0]) 
 

 
print('\n\n')
 
 

def createModel():
    model = ks.Sequential()

    model.add(ks.layers.Conv2D(16, (3, 3),
                               activation=ks.activations.relu,
                               kernel_initializer = 'he_uniform',
                               padding= 'same',
                               input_shape=(90, 3, 1)))

    model.add(ks.layers.Conv2D(16, (3, 3),
                               activation=ks.activations.relu,
                               kernel_initializer='he_uniform',
                               padding='same'))
    
    model.summary()

    #model.add(ks.layers.MaxPooling2D((2, 2)))

    model.add(ks.layers.Conv2D(32, (3, 3),
                               activation=ks.activations.relu,
                               kernel_initializer='he_uniform',
                               padding='same'))

    model.add(ks.layers.Conv2D(32, (3, 3),
                               activation=ks.activations.relu,
                               kernel_initializer='he_uniform',
                               padding='same'))
    
    model.summary()
    #model.add(ks.layers.MaxPooling2D((3, 3)))

    model.add(ks.layers.Conv2D(64, (3, 3),
                               activation=ks.activations.relu,
                               kernel_initializer='he_uniform',
                               padding='same'))

    model.add(ks.layers.Conv2D(64, (3, 3),
                               activation=ks.activations.relu,
                               kernel_initializer='he_uniform',
                               padding='same'))

    #model.add(ks.layers.MaxPooling2D((3, 3)))

    model.add(ks.layers.Flatten())

    model.add(ks.layers.Dense(128,
                              activation=ks.activations.relu,
                              kernel_initializer='he_uniform'))

    model.add(ks.layers.Dense(10, activation=ks.activations.softmax))

    optimizer = ks.optimizers.SGD(learning_rate=0.01, momentum= 0.9)

    model.compile(optimizer=optimizer, loss=ks.losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model



cnn = createModel()



history = cnn.fit(
    trainData, 
    trainLabels, 
    #batch_size=1000,
    epochs=10, 
    validation_split=0.2, 
    verbose=1, 
    #callbacks=None,
    #shuffle=True,
    #class_weight=None,
    #sample_weight=None,
    #initial_epoch=0,
    #steps_per_epoch=None,
    #validation_steps=None,
    #validation_batch_size=None,
    #validation_freq=1,
    #max_queue_size=10,
    #workers=1,
    #use_multiprocessing=False 
    )

loss, acc = mlp_model.evaluate(testData, testLabels)
print("Testing set loss: ",loss)
print("Testing set acc: ",acc)