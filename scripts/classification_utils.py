import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



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


def one_vs_one_classifier(X_train, y_train, X_test=None, y_test=None, cross_val=5):
    
    ## Separate data by labels
    unique_labels = np.unique(y_train)
    num_unique_labels = len(unique_labels)
    label_offset = unique_labels[0]
    

    ## Train classifiers
    classifiers = []
    
    for i in range(num_unique_labels):
        for j in range(i + 1, num_unique_labels):
            
            ## Select data from two classes and concatenate them
            data_c1 = X_train[y_train == unique_labels[i]]
            data_c2 = X_train[y_train == unique_labels[j]]
            data_concat = np.concatenate((data_c1, data_c2), axis=0)
            
            labels_c1 = y_train[y_train == unique_labels[i]]
            labels_c2 = y_train[y_train == unique_labels[j]]
            labels_concat = np.concatenate((labels_c1, labels_c2), axis=0)
            
            data_concat, labels_concat = shuffle(data_concat, labels_concat, random_state=42)
            
            lda = LinearDiscriminantAnalysis(solver='lsqr',  shrinkage='auto')
            lda.fit(data_concat, labels_concat)
            
            classifiers.append(lda)
            
            if cross_val > 0:
                print('Cross validation scores for labels ', 
                      unique_labels[i], ' and ', 
                      unique_labels[j], ': ', cross_val_score(lda, data_concat, labels_concat, cv=cross_val))
                
                    
    ## Test classifiers
    if X_test is not None and y_test is not None:

        votes = np.zeros((len(y_test), num_unique_labels))
        
        for classifier in classifiers:    
            predictions = classifier.predict(X_test) 
            
            for i, prediction in enumerate(predictions):
                votes[i, prediction - label_offset] += 1
                  
        final_predictions = np.argmax(votes, axis=1) + label_offset
        
        return final_predictions

    else:
        votes = np.zeros((len(y_train), num_unique_labels))
        
        for classifier in classifiers:    
            predictions = classifier.predict(X_train) 
            
            for i, prediction in enumerate(predictions):
                votes[i, prediction - label_offset] += 1
                  
        final_predictions = np.argmax(votes, axis=1) + label_offset

        return final_predictions