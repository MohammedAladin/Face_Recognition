import numpy as np
class KNN:
    distances = [[]*2]
    final_label = []
    def getDistance(self,test_vector,train_feature_vectors,train_labels): 
        for i in range(len(train_feature_vectors)):
            distance = np.sqrt(np.sum((test_vector-train_feature_vectors[i])**2)) #calculate the distance between test vector and train vectors
            self.distances.append([distance,train_labels[i]]) #append the distance and label to the distances array
        self.distances = sorted(self.distances, key=lambda x:x[0]) #sort the distances array by the distance
        return self.distances
    def getLabel(self,k):
        labels = []
        for i in range(k):
            if self.distances[i][0] < 0.55:
                labels.append(self.distances[i][1])
            else:
                labels.append("UNKNOWN PERSON!")
        return labels

    def getNearestNeighbor(self,k):
        labels = self.getLabel(k)
        return max(set(labels), key=labels.count)

    def Classifier(self,k,train_features, test_features, Trainlabels):

        for i in range(len(test_features)):
            self.distances = []
            self.getDistance(test_features[i],train_features,Trainlabels)
            self.final_label.append(self.getNearestNeighbor(k))
        return self.final_label

