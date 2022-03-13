import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import classification_utils as utils



def CSP(c1_data, c2_data, n_top=3, n_bot=3):
    
    num_c1_trials = c1_data.shape[0]
    num_c2_trials = c2_data.shape[0]
    num_channels = c1_data.shape[1]
    
    ## Calculate normalized spatial covariance of each trial
    c1_trial_covs = np.zeros((num_c1_trials, num_channels, num_channels))
    c2_trial_covs = np.zeros((num_c2_trials, num_channels, num_channels))
    
    for i in range(num_c1_trials):
        c1_trial = c1_data[i]
        c1_trial_prod = c1_trial @ c1_trial.T
        c1_trial_cov = c1_trial_prod / np.trace(c1_trial_prod)
        c1_trial_covs[i] = c1_trial_cov
    
    for i in range(num_c2_trials):
        c2_trial = c2_data[i]
        c2_trial_prod = c2_trial @ c2_trial.T
        c2_trial_cov = c2_trial_prod / np.trace(c2_trial_prod)
        c2_trial_covs[i] = c2_trial_cov
    
    
    ## Calculate averaged normalized spatial covariance
    c1_trial_covs_avg = np.mean(c1_trial_covs, axis=0)
    c2_trial_covs_avg = np.mean(c2_trial_covs, axis=0)
    
    ## Calculate composite spatial covariance
    R12 = c1_trial_covs_avg + c2_trial_covs_avg
    
    ## Eigen-decompose composite spatial covariance
    R12_eigval, R12_eigvec = np.linalg.eig(R12)
    
    ## Create diagonal matrix of eigenvalues        
    R12_eigval_diag = np.diag(R12_eigval)
    
    ## Calculate Whitening transformation matrix
    P12 = np.linalg.inv(np.sqrt(R12_eigval_diag)) @ R12_eigvec.T
    
    ## Whitening Transform average covariance
    S12_1 = P12 @ c1_trial_covs_avg @ P12.T
    S12_2 = P12 @ c2_trial_covs_avg @ P12.T
    
    ## Eigen-decompose whitening transformed average covariance
    S12_1_eigval, S12_1_eigvec = np.linalg.eig(S12_1)
    S12_2_eigval, S12_2_eigvec = np.linalg.eig(S12_2)
    
    #print(S12_1_eigval + S12_2_eigval)
    
    ## Take the top and bottom eigenvectors to contruct projection matrix W
    sort_indices = np.argsort(S12_1_eigval)
    top_n_indices = list(sort_indices[-n_top:])
    bot_n_indices = list(sort_indices[:n_bot])
    S12_1_eigvec_extracted = S12_1_eigvec[:, top_n_indices + bot_n_indices]
    W12 = S12_1_eigvec_extracted.T @ P12
    
    return W12


def apply_CSP(W, data):
    num_epochs = data.shape[0]
    num_channels = data.shape[1]
    num_channels_transformed = W.shape[0]
    num_samples = data.shape[2]
    
    data_transformed = np.zeros((num_epochs, num_channels_transformed, num_samples))
    
    for i, epoch in enumerate(data):
        epoch_transformed = W @ epoch
        data_transformed[i, :, :] = epoch_transformed
    
    return data_transformed


def one_vs_one_CSP_classifier(X_train, y_train, X_test=None, y_test=None, cross_val=5):
    
    ## Separate data by labels
    unique_labels = np.unique(y_train)
    num_unique_labels = len(unique_labels)
    label_offset = unique_labels[0]
    

    ## Train classifiers
    classifiers = []
    Ws = []
    
    for i in range(num_unique_labels):
        for j in range(i + 1, num_unique_labels):
            
            ## Select data from two classes
            data_c1 = X_train[y_train == unique_labels[i]]
            data_c2 = X_train[y_train == unique_labels[j]]
            
            ## Apply CSP to transform these data
            W = CSP(data_c1, data_c2)
            Ws.append(W)
            data_c1 = apply_CSP(W, data_c1)
            data_c2 = apply_CSP(W, data_c2)
            
            ## Concatenate data and labels
            data_concat = np.concatenate((data_c1, data_c2), axis=0)
            labels_c1 = y_train[y_train == unique_labels[i]]
            labels_c2 = y_train[y_train == unique_labels[j]]
            labels_concat = np.concatenate((labels_c1, labels_c2), axis=0)
            
            ## Flatten data
            data_concat = utils.flatten_dim12(data_concat)
            
            ## Shuffle data
            data_concat, labels_concat = shuffle(data_concat, labels_concat, random_state=42)
            
            lda = LinearDiscriminantAnalysis(solver='lsqr',  shrinkage='auto')
            lda.fit(data_concat, labels_concat)
            
            classifiers.append(lda)
            
            if cross_val > 0:
                print('Mean cross validation score for labels ', 
                      unique_labels[i], ' and ', 
                      unique_labels[j], ': ', 
                      np.mean(cross_val_score(lda, data_concat, labels_concat, cv=cross_val)))
                
                    
    ## Test classifiers
    if X_test is not None and y_test is not None:

        votes = np.zeros((len(y_test), num_unique_labels))
        
        for classifier, W in zip(classifiers, Ws): 
            X_test_ = apply_CSP(W, X_test)
            X_test_ = utils.flatten_dim12(X_test_)
            predictions = classifier.predict(X_test_) 
            
            for i, prediction in enumerate(predictions):
                votes[i, prediction - label_offset] += 1
                  
        final_predictions = np.argmax(votes, axis=1) + label_offset
        
        return final_predictions

    else:
        votes = np.zeros((len(y_train), num_unique_labels))
        
        for classifier, W in zip(classifiers, Ws): 
            X_train_ = apply_CSP(W, X_train)
            X_train_ = utils.flatten_dim12(X_train_)
            predictions = classifier.predict(X_train_) 
            
            for i, prediction in enumerate(predictions):
                votes[i, prediction - label_offset] += 1
                  
        final_predictions = np.argmax(votes, axis=1) + label_offset

        return final_predictions
    