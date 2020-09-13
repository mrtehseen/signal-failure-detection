import pandas as pd
import numpy as np
import tensorflow as tf
import signal_data
import keras
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt
import pandas as pd

train, test = signal_data.load_data()
features, labels = train
featuresT, labelsT =test

batch_size=50
dataset=signal_data.train_input_fn(features, labels, batch_size)
datasetT=signal_data.train_input_fn(featuresT,labelsT,batch_size)

train_path, test_path = signal_data.maybe_download()
df = pd.read_csv (r'InputData/BTestLabeled.csv')

median1 = df['a'].median()
std1 = df['a'].std()
var1 = df['a'].var()

m=str(median1)
s=str(std1)
v=str(var1)

print("Median:"+m)
print("Standard Deviation:"+s)
print("Variance:"+v)




# All the inputs are numeric
feature_columns = [
    tf.feature_column.numeric_column(name)
    for name in signal_data.CSV_COLUMN_NAMES[:-1]]



model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(7, activation='sigmoid'))
model.add(layers.Dense(5, activation='sigmoid'))
model.add(layers.Dense(3, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile( loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
fittedModel=model.fit(dataset, epochs=100,steps_per_epoch=30,validation_data=datasetT,validation_steps=25)

history=fittedModel

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


print("\nSaving Mode:")
# Save entire model to a HDF5 file
#model.save('my_model2.h5')
keras_file = "keras_modelv3.h5"
tf.keras.models.save_model(model, keras_file)
print("Saving Done.")



print("\nThis is evaluation:")
model.evaluate(datasetT, steps=100)

print("\nThis is prediction:")
result = model.predict(featuresT, batch_size=1)
print("\nThis is result:")
print(result.shape)