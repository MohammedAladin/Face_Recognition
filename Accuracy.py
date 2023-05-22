import os
import numpy as np
import face_recognition
from datetime import datetime
import joblib

trPath = 'Train'
tePath = 'Test'
model_path = 'face_recognition_model.tflite'

class Performance_Measures:
    TP = 0
    TN = 0
    FP = 0
    FN = 0

def get_TP_TN_FP_FN(cm, test_num):
    Measures_obj = Performance_Measures()
    Measures_obj.TP = cm[test_num][test_num]
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != test_num and j != test_num:
                Measures_obj.TN += cm[i][j]
            if i != test_num and j == test_num:
                Measures_obj.FN += cm[i][j]
            if i == test_num and j != test_num:
                Measures_obj.FP += cm[i][j]
    return Measures_obj

def precision(TP, FP):
    return TP / (TP + FP)

def recall(TP, FN):
    return TP / (TP + FN)

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def manAccuracy(pred, test):
    right = 0
    for p, t in zip(test, pred):
        if p == t:
            right += 1
    return (right / len(test)) * 100

def report(confusionMatrix, y_pred, test):
    print("Confusion matrix:")
    print(confusionMatrix)
    print("Predictions:", y_pred)
    for i in range(len(confusionMatrix)):
        Measures_obj = get_TP_TN_FP_FN(confusionMatrix, i)
        print(f"Class {i + 1}:")
        print(f"TP: {Measures_obj.TP}, TN: {Measures_obj.TN}, FP: {Measures_obj.FP}, FN: {Measures_obj.FN}")
        print(f"Precision: {precision(Measures_obj.TP, Measures_obj.FP)}")
        print(f"Recall: {recall(Measures_obj.TP, Measures_obj.FN)}")
        print(f"F1 Score: {f1_score(precision(Measures_obj.TP, Measures_obj.FP), recall(Measures_obj.TP, Measures_obj.FN))}")
        print()
    print(f"Manual Accuracy: {manAccuracy(y_pred, test)}")

class KNN:
    distances = [[] * 2]
    final_label = []

    def getDistance(self, test_vector, train_feature_vectors, train_labels):
        for i in range(len(train_feature_vectors)):
            distance = np.sqrt(np.sum((test_vector - train_feature_vectors[i]) ** 2))
            self.distances.append([distance, train_labels[i]])
        self.distances = sorted(self.distances, key=lambda x: x[0])
        return self.distances

    def getLabel(self, k):
        labels = []
        for i in range(k):
            if self.distances[i][0] < 0.55:
                labels.append(self.distances[i][1])
            else:
                labels.append("UNKNOWN PERSON!")
        return labels

    def getNearestNeighbor(self, k):
        labels = self.getLabel(k)
        return max(set(labels), key=labels.count)

    def Classifier(self, k, train_features, test_features, train_labels):
        for i in range(len(test_features)):
            self.distances = []
            self.getDistance(test_features[i], train_features, train_labels)
            self.final_label.append(self.getNearestNeighbor(k))
        return self.final_label

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'{name}, Time: {time}, Date: {date}')
            f.writelines("\n")

def findEncoding(img):
    encodings = []
    names = []

    image = face_recognition.load_image_file(img)
    face_locations = face_recognition.face_locations(image, model="hog")
    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

    for encoding in face_encodings:
        encoding = np.array(encoding).ravel()
        encodings.append(encoding)
        names.append(os.path.basename(img).split('.')[0])

    return encodings, names

def load_data(path, flag):
    Xdata = []
    yLabel = []
    j = 0

    for img in os.listdir(path):
        j += 1
        cnt = 0

        if flag is False:
            print("Training.... Person:", j)
        else:
            print("Testing.... Person:", j)

        data, name = findEncoding(f'{path}/{img}')

        for d, n in zip(data, name):
            cnt += 1
            Xdata.append(d)
            yLabel.append(n)

            if flag is True and cnt == 4:
                break

    return Xdata, yLabel

if __name__ == "__main__":
    train_data, train_label = load_data(trPath, False)
    test_data, test_label = load_data(tePath, True)

    model = KNN()
    y_pred = model.Classifier(9, train_data, test_data, train_label)

    joblib.dump(train_data, 'train_data.joblib')
    joblib.dump(train_label, 'train_labels.joblib')

