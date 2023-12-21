import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import layers
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.callbacks import EarlyStopping
from keras.models import load_model



#np.random.seed(42)
#tf.random.set_seed(42)
#keras.utils.set_random_seed(42)

global file_path
file_path="./Datasets/pneumoniamnist.npz"



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
  train_labels=data["train_labels"]  #flatten the labels
  val_labels=data["val_labels"]
  test_labels=data["test_labels"]

  data.close()

  return train_images, val_images, test_images, train_labels, val_labels, test_labels

def data_augmentation(images):
    images = layers.RandomFlip("horizontal",seed=42)(images)
    images = layers.RandomRotation(0.1)(images)
    return images
  
def gaussian_noise(images):
    images = layers.GaussianNoise(0.1)(images)
    return images

def image_augmentation(train_images,train_labels):
  images_0 = train_images[train_labels.flatten() == 0]
  images_1 = train_images[train_labels.flatten() == 1]

  augmented_images = layers.RandomFlip("horizontal")(images_0[0:len(images_0)])
  augmented_images = layers.RandomRotation(0.1)(augmented_images)

  augmented_images2 = layers.GaussianNoise(0.1)(images_0[0:len(images_0)])

  train_images2=np.concatenate((train_images,augmented_images,augmented_images2),axis=0)
  train_labels2=np.concatenate((train_labels.flatten(),[0]*len(augmented_images),[0]*len(augmented_images2)),axis=0)

  indices = np.arange(len(train_labels2))
  np.random.shuffle(indices)
  train_images3 = train_images2[indices]
  train_labels3 = train_labels2[indices]

  return train_images3, train_labels3

def normalization(train_images, val_images,test_images):
  train=train_images/255.0
  val=val_images/255.0
  test=test_images/255.0
  return train, val ,test

def model_training(train,train_labels,val,val_labels):
  count_0=np.sum(train_labels==0)
  count_1=np.sum(train_labels==1)
  count_all=np.sum(train_labels)
  weight_for_0=(1 / count_0) * (count_all) / 2.0
  weight_for_1=(1 / count_1) * (count_all) / 2.0
  print(weight_for_0,weight_for_1)

  class_weight={
    0:weight_for_0,
    1:weight_for_1
  }

  model = Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', padding="same",input_shape=(28, 28, 1)),
  layers.MaxPooling2D(2, 2),
  layers.Conv2D(64, (3, 3), activation='relu',padding="same"),
  layers.MaxPooling2D(2, 2),
    
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

  early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.004, patience=5, verbose=1, mode='min')

  history = model.fit(train, train_labels,epochs=40, class_weight=class_weight,
                    validation_data=(val, val_labels),callbacks=[early_stopping])

  model.summary()

  return history, model

def testing(model, test, test_labels):
  test_loss, test_acc = model.evaluate(test,  test_labels, verbose=1)
  pred=model.predict(test)
  pred=(pred >= (1/ 2)).astype(int)
  conf_matrix = confusion_matrix(test_labels, pred)
  precision = precision_score(test_labels, pred)
  recall = recall_score(test_labels, pred)
  f1 = f1_score(test_labels, pred)

  print("condusion matrix:\n", conf_matrix)
  print("Precision:", precision)
  print("Recall:", recall)
  print("F1 Score:", f1)
  print(classification_report(test_labels, pred))  

def load_best_model():
  path="./A/best_model.h5"
  saved_model = load_model(path)
  saved_model.summary()
  return saved_model


def demonstration():
  train_images1, val_images, test_images, train_labels1, val_labels, test_labels=load_data(file_path)
  train_images, train_labels=image_augmentation(train_images1,train_labels1) 
  train, val ,test=normalization(train_images,val_images,test_images)
  history,model=model_training(train,train_labels,val,val_labels)
  testing(model,test, test_labels)
  print("\n")
  print("Load the best model:")
  best_model=load_best_model()
  testing(best_model, test, test_labels)


if __name__ == "__main__":
  demonstration()

