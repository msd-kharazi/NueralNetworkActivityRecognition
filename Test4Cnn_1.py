import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import random
import os


 
labels = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
rawData = []
trainData = []
trainLabels = []
testData = []
testLabels = []

counter = 0
trainSize = 0.7
testSize = 0.3

with open("D:\PythonTest\Test4Cnn\WISDM_ar_latest\WISDM_ar_v1.1_raw.txt", "r") as filestream:    
    for line in filestream:
        
        if counter>=20000:
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
        if labelIndex<0 or labelIndex>=len(labels): 
            raise Exception('Error: ',label, ' does not exist in valid labels list!')

        timeStamp = int(currentLineValues[2])
        xAcc =float(currentLineValues[3])
        yAcc = float(currentLineValues[4])
        zAcc = float(currentLineValues[5][:-2])
        
        rawData.append([labelIndex,xAcc,yAcc,zAcc])
        counter+=1

         
random.shuffle(rawData)
trainCount =int(counter*0.7)


trainRawData =rawData[0:trainCount] 
trainData = np.array(list(item[1:] for item in trainRawData))
trainData = tf.keras.utils.normalize(trainData, axis=1) 
trainLabels =np.array(list(item[0] for item in trainRawData))


print('Train data shape before reshape: ', trainData.shape) 
trainData = trainData.reshape(trainData.shape[0],3,1) 
print('Train data shape after reshape: ', trainData.shape) 
print('TrainData[0].shape: ',trainData[0].shape) 
print('TrainData[0]: ',trainData[0]) 
 


testRawData = rawData[trainCount:]
testData = np.array(list(item[1:] for item in testRawData))
testData = tf.keras.utils.normalize(testData, axis=1)
testLabels =np.array(list(item[0] for item in testRawData))
 

print('Test data shape before reshape: ', testData.shape) 
testData = testData.reshape(testData.shape[0],3,1) 
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
 

def build_conv1D_model1():

  n_timesteps = trainData.shape[1] #13
  n_features  = trainData.shape[2] #1 
  model = tf.keras.Sequential(name="model_conv1D")
  model.add(tf.keras.layers.Input(shape=(n_timesteps,n_features)))
  model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', name="Conv1D_1"))
  #model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', name="Conv1D_2"))
  #model.add(tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(32, activation='relu', name="Dense_1"))
  model.add(tf.keras.layers.Dense(n_features, name="Dense_2"))

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])
  return model


def build_conv1D_model2(): 

    model = tf.keras.models.Sequential()  # a basic feed-forward model
    model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

    model.compile(optimizer='adam',  # Good default optimizer to start with
                  loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
                  metrics=['accuracy'])  # what to track
    #model.build()
    return model

#model_conv1D = build_conv1D_model1()
model_conv1D = build_conv1D_model2()
#model_conv1D = build_conv1D_model3()
#model_conv1D = build_conv1D_model4()
#model_conv1D = build_conv1D_model5()
#model_conv1D.summary()


# Store training stats
history = model_conv1D.fit(trainData, trainLabels, epochs=10, validation_split=0.2, verbose=1)

loss, acc = model_conv1D.evaluate(testData, testLabels)
print("Testing set loss: ",loss)
print("Testing set acc: ",acc)

#test_predictions = model_conv1D.predict(test_data_reshaped).flatten()
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