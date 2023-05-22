import joblib
import numpy as np
from Knn import KNN
from FaceReco import *

trPath = 'Train'
tePath = 'Test'
kernel = np.array(
                [
                [0, -1, 0],
                [-1, 5,-1],
                [0, -1, 0]
                ])

def manAccuracy(pred, test):
    right = 0
    for p,t in zip(test,pred):
        if p == t:
            right+=1
    return (right/len(test))*100

def report(y_pred,test):
    print ("Predictions: ",y_pred)
    print(f"Manula Accuracy: {manAccuracy(y_pred , test)}")
  

if __name__ == "__main__":
    
    train_data_loaded = joblib.load('train_data.joblib')
    train_label_loaded = joblib.load('train_labels.joblib')
    test_data,test_label = load_data(tePath)
    
    model = KNN()
    y_pred = model.Classifier(9,train_data_loaded,test_data,train_label_loaded)

    report(y_pred,test_label)

    #camModel(train_data,train_label) 
     




