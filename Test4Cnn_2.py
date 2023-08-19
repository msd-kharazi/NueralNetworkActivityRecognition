import tensorflow as tf
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import random
import os
from keras.utils import to_categorical 






 
 
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
        
        #if counter>=20000:
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
        xAcc =float(currentLineValues[3])
        yAcc = float(currentLineValues[4])
        zAcc = float(currentLineValues[5][:-1])
        
        rawData.append([labelsArrays[labelIndex],xAcc,yAcc,zAcc])
        counter+=1

     
        
         

#normalizer = Normalization(axis=-1)
#normalizer.adapt(training_data)

#normalized_data = normalizer(training_data)
#print("var: %.4f" % np.var(normalized_data))
#print("mean: %.4f" % np.mean(normalized_data))



#random.shuffle(rawData)
trainCount =int(counter*0.7)


trainRawData =rawData[0:trainCount] 
trainData = np.array(list(item[1:] for item in trainRawData))
#trainData = trainData/15.0
#trainData = tf.keras.utils.normalize(trainData, axis=-1) 
trainLabels =np.array(list(item[0] for item in trainRawData))
#trainLabels = trainLabels/5.0
#trainLabels = to_categorical(trainLabels,6)

 
print('TrainData[0].shape: ',trainData[0].shape) 
print('TrainData[0]: ',trainData[0]) 
 


testRawData = rawData[trainCount:]
testData = np.array(list(item[1:] for item in testRawData))
#testData = testData/15.0
#testData = tf.keras.utils.normalize(testData, axis=-1)
testLabels =np.array(list(item[0] for item in testRawData)) 
#testLabels = tf.keras.utils.normalize(testLabels, axis=-1)
#testLabels = testLabels/5.0
#testLabels = to_categorical(testLabels)

print('Test data shape before reshape: ', testData.shape) 
#testData = testData.reshape(testData.shape[0],3,1) 
print('Test data shape after reshape: ', testData.shape) 
print('TestData[0].shape: ',testData[0].shape) 
print('TestData[0]: ',testData[0]) 
 


print('rawData count:' + str(len(rawData)))
print('counter:' + str(counter))
print('trainCount:' + str(trainCount) )
print('trainData count:' + str(len(trainData)) )
print('trainLabels count:' + str(len(trainLabels)) )
print('testData count:' + str(len(testData)) )
print('testLabels count:' + str(len(testLabels)) )
print('\n\n')
 
 


def build_model(): 

    model = tf.keras.models.Sequential()  # a basic feed-forward model
    model.add(tf.keras.Input(shape=(3,))) 
    model.add(tf.keras.layers.Dense(
        32,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None))
    
    model.add(tf.keras.layers.Dense(
        32,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None))

    model.add(tf.keras.layers.Dense(
        6,
        activation=tf.nn.softmax,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None))

        
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # Good default optimizer to start with
        loss='categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
        metrics=['accuracy'],  # what to track         
        #loss_weights=None,
        #weighted_metrics=None,
        #run_eagerly=None,
        #steps_per_execution=None,
        #jit_compile=None
        )

    return model

mlp_model = build_model()


# Store training stats
history = mlp_model.fit(
    trainData, 
    trainLabels, 
    #batch_size=1000,
    epochs=5, 
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

test_predictions = mlp_model.predict(testData[301400:301410])

for counter in range(10):
    print(labels[list(testLabels[301400+trainCount+counter]).index(testLabels[301400+trainCount+counter].max())],'==========>',labels[list(test_predictions[counter]).index(test_predictions[counter].max())])
        



#plot_prediction(test_labels, test_predictions)



#model = Sequential() 
#model.add(ConvD(16, (3, 3),padding = 'same', activation='sigmoid', input_shape=(3)))
#model.add(Conv2D(32, (3, 3), activation='sigmoid'))
#model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#model.add(Dense(64))
#model.add(Dense(6))
#model.add(Activation('softmax'))

#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

#model.fit(trainData, trainLabels, epochs=3, validation_split=0.3)


#val_loss, val_acc = model.evaluate(testData, testLabels)  # evaluate the out of sample data with model
#print(val_loss)  # model's loss (error)
#print(val_acc)  # model's accuracy


os.system("pause")
#model.add(Activation('relu'))


#model = tf.keras.models.Sequential()   
#model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
#model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
#model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
#model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
#model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
#model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

#model.compile(optimizer='adam',  # Good default optimizer to start with
#              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
#              metrics=['accuracy'])  # what to track

#model.fit(x_train, y_train, epochs=3)  # train the model

#val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
#print(val_loss)  # model's loss (error)
#print(val_acc)  # model's accuracy





#mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
#(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test


#x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
#x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1


#model = tf.keras.models.Sequential()  # a basic feed-forward model
#model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
#model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
#model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
#model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
#model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
#model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
#model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

#model.compile(optimizer='adam',  # Good default optimizer to start with
#              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
#              metrics=['accuracy'])  # what to track

#model.fit(x_train, y_train, epochs=3)  # train the model

#val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
#print(val_loss)  # model's loss (error)
#print(val_acc)  # model's accuracy