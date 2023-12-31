"""
this file contains SVM, random forest, knn and logistic regression
for 

"""


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

global file_path
file_path="./Datasets/pathmnist.npz"

def load_data(path:str):
    """
      load the data with file path and extract subsets
    Args:
        path:str
    Returns:
        train_images, val_images, test_images, train_labels, val_labels, test_labels
    """
    data = np.load(path)

    train_images=data["train_images"]
    val_images=data["val_images"]
    test_images=data["test_images"]
    train_labels=data["train_labels"].ravel()  #flatten the labels
    val_labels=data["val_labels"].ravel()
    test_labels=data["test_labels"].ravel()

    data.close()

    return train_images, val_images, test_images, train_labels, val_labels, test_labels

def pre_processing(train_images, val_images, test_images):
    """
      preprocess the data with standardscaler, pca and normalization
    Args:
        all data images subsets
    Returns:
        train_images, val_images, test_images after preprocessing
    """
    #flatten image features
    train_flatten_images = train_images.reshape(train_images.shape[0], -1)
    val_flatten_images = val_images.reshape(val_images.shape[0], -1)
    test_flatten_images = test_images.reshape(test_images.shape[0], -1)
    # scale the input features of the data
    scaler = StandardScaler()
    train_standard_images = scaler.fit_transform(train_flatten_images) #only fit the train data
    val_standard_images = scaler.transform(val_flatten_images)  
    test_standard_images = scaler.transform(test_flatten_images)
    # creating the PCA model
    pca = PCA(n_components=200)

    # using the PCA model on our standardized data
    pca.fit(train_standard_images)

    train_pca=pca.transform(train_standard_images)
    val_pca=pca.transform(val_standard_images)
    test_pca=pca.transform(test_standard_images)
    # contains the proportion of the total variance in the data explained by each principal component
    #normalize
    min_max_scaler = preprocessing.MinMaxScaler()  #normalize
    min_max_scaler.fit(train_pca)
    train = min_max_scaler.transform(train_pca)
    val = min_max_scaler.transform(val_pca)
    test = min_max_scaler.transform(test_pca)
  
    return train, val, test

def randomf_training(train, train_labels):
    """
      train the random forest model 
    Args:
        train images and labels
    Returns:
        the rf model
    """
    print("training random forest model for pathmnist...")
    rf = RandomForestClassifier(n_estimators=250,n_jobs=4,class_weight="balanced")
    rf =rf.fit(train, train_labels)

    return rf

def testing(model, test, test_labels):
    """
      give the report of the testing
    Args:
        the model and test data
    Returns:
       print the report
    """
    pred = model.predict(test)
    conf_matrix = confusion_matrix(test_labels, pred)
    precision = precision_score(test_labels, pred,average="weighted")
    recall = recall_score(test_labels, pred,average="weighted")
    f1 = f1_score(test_labels, pred,average="weighted")

    print("condusion matrix:\n", conf_matrix)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Test Accuracy:", accuracy_score(test_labels, pred))
    print(classification_report(test_labels, pred))

def demonstration():
    """
      run the whole process use this function
    
    """
    train_images, val_images, test_images, train_labels, val_labels, test_labels=load_data(file_path)
    train, val, test= pre_processing(train_images, val_images, test_images)
    rf_model=randomf_training(train, train_labels)
    print("\n"+"The test for the randomforest model: ")
    testing(rf_model, test, test_labels)



if __name__ == "__main__":
    demonstration()
# End-of-file (EOF)