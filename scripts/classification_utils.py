import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score



def flatten_dim12(data):
    num_epochs = data.shape[0]
    num_channels = data.shape[1]
    num_samples = data.shape[2]

    data_flattened = data.reshape(num_epochs, num_channels * num_samples)
    return data_flattened


def windowed_means(data, num_points):
    
    num_epochs = data.shape[0]
    num_channels = data.shape[1]
    num_samples = data.shape[2]
    
    window_size = num_samples / num_points
    data_downsampled = np.zeros((num_epochs, num_channels, num_points))
    
    for i in range(num_points):
        start = int(window_size * i)
        end = int(window_size * (i + 1))
        data_downsampled[:, :, i] = np.mean(data[:, :, start : end], axis=-1)
        
    return data_downsampled


class LDA_classifier:
    
    def __init__(self, X_train, y_train, cross_val=None):
        self.X_train = X_train
        self.y_train = y_train
        self.cross_val = cross_val
        

    def train_binary(self):
        self.classifier = LinearDiscriminantAnalysis(solver='lsqr',  shrinkage='auto')
        self.classifier.fit(self.X_train, self.y_train)
        
        if self.cross_val is not None:
            cross_val_score_avg = np.mean(cross_val_score(self.classifier, self.X_train, self.y_train, cv=self.cross_val))

            print('Cross validation score average: ', cross_val_score_avg)            
            return cross_val_score_avg
        
        else:
            return 0                              
            

    def test_binary(self, X_test=None, y_test=None):

        if X_test is not None and y_test is not None:
            predictions = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            print('Testing accuracy: ', accuracy)
            return predictions, accuracy
            
        else:
            predictions = self.classifier.predict(self.X_train)
            accuracy = accuracy_score(self.y_train, predictions)

            print('Training accuracy: ', accuracy)
            return predictions, accuracy
        

    def train_1_vs_1(self):

        ## Separate data by labels
        unique_labels = np.unique(self.y_train)
        self.num_unique_labels = len(unique_labels)
        self.label_offset = unique_labels[0]
        
        ## Train classifiers
        self.classifiers = []
        cross_val_scores = []
        
        for i in range(self.num_unique_labels):
            for j in range(i + 1, self.num_unique_labels):
                
                ## Extract data and labels for two classes
                data_extracted = self.X_train[np.any([self.y_train == unique_labels[i], 
                                                      self.y_train == unique_labels[j]], axis=0)]
                labels_extracted = self.y_train[np.any([self.y_train == unique_labels[i], 
                                                        self.y_train == unique_labels[j]], axis=0)]

                lda = LinearDiscriminantAnalysis(solver='lsqr',  shrinkage='auto')
                lda.fit(data_extracted, labels_extracted)
                self.classifiers.append(lda)

                if self.cross_val is not None:

                    cross_val_score_avg = np.mean(cross_val_score(lda, data_extracted, labels_extracted, cv=self.cross_val))

                    print('Cross validation scores for labels ', 
                          unique_labels[i], ' and ', 
                          unique_labels[j], ': ', cross_val_score_avg)

                    cross_val_scores.append(cross_val_score_avg)

        return cross_val_scores


    def test_1_vs_1(self, X_test=None, y_test=None):

        if X_test is not None and y_test is not None:
    
            votes = np.zeros((len(y_test), self.num_unique_labels))
            
            for classifier in self.classifiers:    
                predictions = classifier.predict(X_test) 
                
                for i, prediction in enumerate(predictions):
                    votes[i, prediction - self.label_offset] += 1
                      
            final_predictions = np.argmax(votes, axis=1) + self.label_offset
            accuracy = accuracy_score(y_test, final_predictions)
            
            print('Testing accuracy: ', accuracy)
            return final_predictions, accuracy
    
        else:
            votes = np.zeros((len(self.y_train), self.num_unique_labels))
            
            for classifier in self.classifiers:    
                predictions = classifier.predict(self.X_train) 
                
                for i, prediction in enumerate(predictions):
                    votes[i, prediction - self.label_offset] += 1
                      
            final_predictions = np.argmax(votes, axis=1) + self.label_offset
            accuracy = accuracy_score(self.y_train, final_predictions)
            
            print('Training accuracy: ', accuracy)
            return final_predictions, accuracy
        