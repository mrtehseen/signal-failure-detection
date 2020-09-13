from keras.models import load_model
import signal_data
import tensorflow as tf

train, test = signal_data.load_data()
features, labels = train
featuresT, labelsT =test
featuresarr=featuresT.__array__(dtype=int);
model = tf.keras.models.load_model("keras_modelv3.h5")
pred = model.predict(featuresarr)
print(pred)