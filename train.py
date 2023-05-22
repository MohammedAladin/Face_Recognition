import numpy as np
import joblib
import numpy as np
from FaceReco import *

trPath = 'Train'
tePath = 'Test'
kernel = np.array(
                [
                [0, -1, 0],
                [-1, 5,-1],
                [0, -1, 0]
                ])



if __name__ == "__main__":
    
    train_data,train_label=load_data(trPath)

    joblib.dump(train_data, 'train_data.joblib')
    joblib.dump(train_label, 'train_labels.joblib')


    #camModel(train_data,train_label) 
     

