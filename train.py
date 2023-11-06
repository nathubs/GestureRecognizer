import os
import tensorflow as tf

assert tf.__version__.startswith("2")

from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt


dataset_path = "./data/gestrue"

print(dataset_path)
labels = []
for i in os.listdir(dataset_path):
    if os.path.isdir(os.path.join(dataset_path, i)):
        labels.append(i)
print("labels:", labels)

data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path, hparams=gesture_recognizer.HandDataPreprocessingParams()
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

hparams = gesture_recognizer.HParams(export_dir="exported_model",epochs=20)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data, validation_data=validation_data, options=options
)
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")
model.export_model()

#hparams = gesture_recognizer.HParams(learning_rate=0.003, export_dir="exported_model_2")
#model_options = gesture_recognizer.ModelOptions(dropout_rate=0.2)
#options = gesture_recognizer.GestureRecognizerOptions(model_options=model_options, hparams=hparams)
#model_2 = gesture_recognizer.GestureRecognizer.create(
#    train_data=train_data,
#    validation_data=validation_data,
#    options=options
#)

#loss, accuracy = model_2.evaluate(test_data)
#print(f"Test loss:{loss}, Test accuracy:{accuracy}")
