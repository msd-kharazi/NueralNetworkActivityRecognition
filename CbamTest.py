import tensorflow as tf
from CbamLayer import CBAM
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

# تعریف مدل
model = Sequential()
model.add(Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=(90, 3, 1)))
#model.add(MaxPooling2D((2, 2)))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D((2, 2)))
# اضافه کردن لایه CBAM به لایه‌های کانولوشن

model.summary()
model.add(CBAM())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# کامپایل مدل
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
    
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

loss, acc = cnn.evaluate(testData, testLabels)
print("Testing set loss: ",loss)
print("Testing set acc: ",acc)
