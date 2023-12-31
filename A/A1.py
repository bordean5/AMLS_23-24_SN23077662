"""
this file contains SVM, random forest, knn and logistic regression
for pneumoniamnist classification

"""
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


global file_path
file_path="./Datasets/pneumoniamnist.npz"



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
    pca = PCA(n_components=0.97)

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

def svm_hyperparameter_tuning(train,val, train_labels,val_labels):
    """
      use validation dataset to do hyperparameter tuning
    Args:
        train validation images and labels
    Returns:
        best prarmeter
    """
    param_grid = {
        'C': [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],  
        'kernel': ['linear', 'rbf',"poly"],  
        'gamma': ['scale', 'auto'],  
        'class_weight':['balanced']
    }
    svm_model = svm.SVC()

    grid_search = GridSearchCV(svm_model, param_grid, cv=4, scoring='accuracy', verbose=1)

    grid_search.fit(train, train_labels)

    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(val)  
    val_accuracy = accuracy_score(val_labels, val_predictions)

    print("SVM Best Parameters:", grid_search.best_params_)
    print("SVM Validation Accuracy:", val_accuracy)
    print("\n")
    best_params = grid_search.best_params_

    return best_params

def knn_hyperparameter_tuning(train,val, train_labels,val_labels):
    """
      use validation dataset to do hyperparameter tuning
    Args:
        train validation images and labels
    Returns:
        best prarmeter for knn
    """
    param_grid = {
        'n_neighbors': [3,4,5,6,7,8,9,10,11], 
        "weights":["distance"],
    }
    kNN = KNeighborsClassifier()

    grid_search = GridSearchCV(kNN, param_grid, cv=4, scoring='accuracy', verbose=1)
    grid_search.fit(train, train_labels)

    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(val) 
    val_accuracy = accuracy_score(val_labels, val_predictions)

    print("KNN Best Parameters:", grid_search.best_params_)
    print("KNN Validation Accuracy:", val_accuracy)
    print("\n")
    best_params = grid_search.best_params_

    return best_params

def log_hyperparameter_tuning(train,val, train_labels,val_labels):
    """
      use validation dataset to do hyperparameter tuning
    Args:
        train validation images and labels
    Returns:
        best prarmeter for logistic regression
    """
    param_grid = {
        'penalty': ["l2"],  
        "C":[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'solver':['lbfgs'],
        'class_weight':['balanced']
    }
    log=LogisticRegression(max_iter=400)

    grid_search = GridSearchCV(log, param_grid, cv=4, scoring='accuracy', verbose=1)
    grid_search.fit(train, train_labels)

    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(val) 
    val_accuracy = accuracy_score(val_labels, val_predictions)

    print("Log Best Parameters:", grid_search.best_params_)
    print("Log Validation Accuracy:", val_accuracy)
    print("\n")
    best_params = grid_search.best_params_

    return best_params

def randomf_hyperparameter_tuning(train,val, train_labels,val_labels):
    """
      use validation dataset to do hyperparameter tuning
    Args:
        train validation images and labels
    Returns:
        best prarmeter for random forest
    """
    param_grid = {
        "n_estimators":[50,100,150],
        'class_weight':["balanced"]
    }
    rf= RandomForestClassifier()

    grid_search = GridSearchCV(rf, param_grid, cv=4, scoring='accuracy', verbose=1)
    grid_search.fit(train, train_labels)

    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(val) 
    val_accuracy = accuracy_score(val_labels, val_predictions)

    print("RandomForest Best Parameters:", grid_search.best_params_)
    print("RandomForest Validation Accuracy:", val_accuracy)
    print("\n")
    best_params = grid_search.best_params_

    return best_params

def svm_training(train, train_labels, best_params):
    """
      train the svm model with best parameters
    Args:
        train  images and labels
    Returns:
        the svm model
    """
    best_svm = svm.SVC(**best_params)
    best_svm =best_svm.fit(train, train_labels)

    return best_svm

def knn_training(train, train_labels, best_params):
    """
      train the knn model with best parameters
    Args:
        train images and labels
    Returns:
        the knn model
    """

    best_knn = KNeighborsClassifier(**best_params)
    best_knn =best_knn.fit(train, train_labels)

    return best_knn

def log_training(train, train_labels, best_params):
    """
      train the logistic regression model with best parameters
    Args:
        train images and labels
    Returns:
        the log model
    """

    best_log = LogisticRegression(**best_params)
    best_log =best_log.fit(train, train_labels)

    return best_log

def randomf_training(train, train_labels, best_params):
    """
      train the random forest model with best parameters
    Args:
        train images and labels
    Returns:
        the rf model
    """

    best_rf = RandomForestClassifier(**best_params)
    best_rf =best_rf.fit(train, train_labels)

    return best_rf

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
    precision = precision_score(test_labels, pred)
    recall = recall_score(test_labels, pred)
    f1 = f1_score(test_labels, pred)

    print("Test Accuracy:", accuracy_score(test_labels, pred))
    print("condusion matrix:\n", conf_matrix)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(classification_report(test_labels, pred))

def demonstration():
    """
      run the whole process use this function
    
    """ 
    train_images, val_images, test_images, train_labels, val_labels, test_labels=load_data(file_path)
    train, val, test= pre_processing(train_images, val_images, test_images)
    svm_params=svm_hyperparameter_tuning(train,val,train_labels,val_labels)
    svm_model=svm_training(train, train_labels, svm_params)

    knn_params=knn_hyperparameter_tuning(train,val,train_labels,val_labels)
    knn_model=knn_training(train, train_labels, knn_params)

    log_params=log_hyperparameter_tuning(train,val,train_labels,val_labels)
    log_model=log_training(train, train_labels, log_params)

    rf_params=randomf_hyperparameter_tuning(train,val,train_labels,val_labels)
    rf_model=randomf_training(train, train_labels, rf_params)


    print("\n"+"The test for the SVM model: ")
    testing(svm_model, test, test_labels)

    print("\n"+"The test for the KNN model: ")
    testing(knn_model, test, test_labels)

    print("\n"+"The test for the logistic regression model: ")
    testing(log_model, test, test_labels)

    print("\n"+"The test for the randomforest model: ")
    testing(rf_model, test, test_labels)


if __name__ == "__main__":
    demonstration()
# End-of-file (EOF)