import pandas as pd
import numpy as np
import tensorflow as tf
import signal_data

# train = pd.read_csv("InputData/BTest.csv").values

#train_path = "InputData/BTestLabeled.csv"

# Creates a dataset that reads all of the records from two CSV files, each with
# eight float columns
# filename = ["InputData/BTest.csv"]
# record_defaults = [tf.string] * 7   # Eight required float columns
# dataset = tf.data.experimental.CsvDataset(filename, record_defaults, header=True, select_cols=[1,2,3,4,5,6,7])

train, test = signal_data.load_data()
features, labels = train
featuresT, labelsT =test

batch_size=10
dataset=signal_data.train_input_fn(features, labels, batch_size)


train_path, test_path = signal_data.maybe_download()

model_dir="ModelSavingT"

# All the inputs are numeric
feature_columnss = [
    tf.feature_column.numeric_column(name)
    for name in signal_data.CSV_COLUMN_NAMES[:-1]]

# Build the estimator
est = tf.estimator.DNNClassifier(feature_columns=feature_columnss,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=2,model_dir=model_dir)
# Train the estimator
batch_size = 100
est.train(
    steps=1000,
    input_fn=lambda : signal_data.train_input_fn(features, labels, batch_size),)


# Evaluate the model.
eval_result = est.evaluate(
    input_fn=lambda:signal_data.eval_input_fn(featuresT,labelsT,batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Generate Prediction from model
predictions = est.predict(
    input_fn=lambda:signal_data.eval_input_fn(featuresT,labels=None,batch_size=10))

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

for pred_dict, expec in zip(predictions, labelsT):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(signal_data.SPECIES[class_id],
                         100 * probability, expec))


dir="ModelSavingT\\PBModels"
#print(eval_result)


feature_spec = tf.feature_column.make_parse_example_spec(feature_columnss)
#print(feature_spec)

#Build receiver function, and export.
serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
export_dir = est.export_savedmodel(dir, serving_input_receiver_fn)
print("Model Saved")
print("Train Path"+train_path)
#print(serving_input_receiver_fn)
#tf.lite.Interpreter

#converter = tf.lite.TFLiteConverter.from_saved_model("SavedModel/test3/1554927432/")
#tflite_model = converter.convert()
#open("linf_model.tflite", "wb").write(tflite_model)

#help(tf.contrib.lite.TFLiteConverter)
