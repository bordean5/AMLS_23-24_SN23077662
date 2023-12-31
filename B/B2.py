import numpy as np
import matplotlib.pyplot as plt
import random
import keras

from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras import layers
from keras.regularizers import l2
from keras.models import load_model

random.seed(42)

np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)

global file_path
file_path="./Datasets/pathmnist.npz"

def load_data(file_path:str):
  """
    load the data with file path and extract subsets
  Args:
        file_path:str
  Returns:
        train_images, val_images, test_images, train_labels, val_labels, test_labels
  """
  data = np.load(file_path)

  train_images=data["train_images"]
  val_images=data["val_images"]
  test_images=data["test_images"]
  train_labels=data["train_labels"].ravel()  #flatten the labels
  val_labels=data["val_labels"].ravel()
  test_labels=data["test_labels"].ravel()

  data.close()
  return train_images, val_images, test_images, train_labels, val_labels, test_labels

def normalization(train_images, val_images,test_images):
  train=train_images/255.0
  val=val_images/255.0
  test=test_images/255.0
  return train, val ,test

def model_training(train,train_labels,val,val_labels):
  class_weights = compute_class_weight(class_weight='balanced', 
                                     classes=np.unique(train_labels), 
                                     y=train_labels.flatten())

  class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}

  model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding="same",input_shape=(28, 28, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu',padding="same"),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu',padding="same",kernel_regularizer=l2(0.001)),
    layers.GlobalAveragePooling2D(),

    Flatten(),
    Dense(64, activation='relu'),
    layers.Dropout(0.2),
    Dense(9, activation='softmax')
  ])

  model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  early_stopping = EarlyStopping(monitor='val_loss', patience=5, 
                               verbose=1, mode='min', restore_best_weights=True)

  history = model.fit(train, train_labels,epochs=60, class_weight=class_weights_dict,
                    validation_data=(val, val_labels),callbacks=[early_stopping])

  model.summary()

  return history, model

def testing(model, test, test_labels):
  test_loss, test_acc = model.evaluate(test,  test_labels, verbose=1)
  predictions = model.predict(test)
  predicted_labels = np.argmax(predictions, axis=1)
  print(classification_report(test_labels,predicted_labels)) 

def load_best_model():
  path="./B/model862.h5"
  saved_model = load_model(path)
  saved_model.summary()
  return saved_model

def demonstration():
  train_images, val_images, test_images, train_labels, val_labels, test_labels=load_data(file_path)
  train, val ,test=normalization(train_images,val_images,test_images)
  history,model=model_training(train,train_labels,val,val_labels)
  testing(model,test, test_labels)
  print("\n")
  print("Load the model:")
  best_model=load_best_model()
  testing(best_model, test, test_labels)

if __name__ == "__main__":
  demonstration() 

