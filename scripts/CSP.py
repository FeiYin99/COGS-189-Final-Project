import sys
import numpy as np
import scipy as sp
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

sys.path.append('/home/inffzy/Desktop/cogs189/cogs189_final_project/scripts')
import classification_utils as utils



def CSP(c1_data, c2_data, n_top):
    
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
    bot_n_indices = list(sort_indices[:n_top])
    S12_1_eigvec_extracted = S12_1_eigvec[:, top_n_indices + bot_n_indices]
    W12 = S12_1_eigvec_extracted.T @ P12
    
    return W12


def CSP2(c1_data, c2_data, n_top):

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

    avg_cov_sum = c1_trial_covs_avg + c2_trial_covs_avg
    eigval, eigvec = sp.linalg.eigh(c1_trial_covs_avg, avg_cov_sum)

    ## Take the top and bottom eigenvectors to contruct projection matrix W
    sort_indices = np.argsort(eigval)
    top_n_indices = list(sort_indices[-n_top:])
    bot_n_indices = list(sort_indices[:n_top])
    eigvec_extracted = eigvec[:, top_n_indices + bot_n_indices]
    W = eigvec_extracted.T

    return W


def apply_CSP_transform(all_data, W):

    num_epochs = all_data.shape[0]
    num_channels = all_data.shape[1]
    num_channels_transformed = W.shape[0]
    num_samples = all_data.shape[2]
    
    data_transformed = np.zeros((num_epochs, num_channels_transformed, num_samples))
    
    for i, epoch in enumerate(all_data):
        epoch_transformed = W @ epoch
        data_transformed[i, :, :] = epoch_transformed
    
    return data_transformed


def CSP_extract_features(all_data, W, n_top):

    extracted_features = np.zeros((all_data.shape[0], 2 * n_top))

    for i, epoch in enumerate(all_data):

        Z = W @ epoch

        #print(np.var(Z, axis=-1).shape)

        var_sum = np.sum(np.var(Z, axis=-1))

        for k in range(n_top):

            Z_k = Z[k]
            f_k = np.log10(np.var(Z_k) / var_sum)

            extracted_features[i, k] = f_k

    return extracted_features

    


class CSP_LDA_classifier:
    
    def __init__(self, X_train, y_train, cross_val=None, n_top=3, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.cross_val = cross_val
        self.n_top = n_top
        self.random_state = random_state
        

    def train_binary(self):
        
        ## Separate data by labels
        unique_labels = np.unique(self.y_train)
        num_unique_labels = len(unique_labels)
        label_offset = unique_labels[0]

        assert num_unique_labels == 2
        
        data_c1 = self.X_train[self.y_train == unique_labels[0]]
        data_c2 = self.X_train[self.y_train == unique_labels[1]]
        labels_c1 = self.y_train[self.y_train == unique_labels[0]]
        labels_c2 = self.y_train[self.y_train == unique_labels[1]]
            
        ## Apply CSP to transform data
        #self.CSP_transform = CSP(data_c1, data_c2, self.n_top)
        self.CSP_transform = CSP2(data_c1, data_c2, self.n_top)
        extracted_features = CSP_extract_features(self.X_train, self.CSP_transform, self.n_top)
        
        self.classifier = LinearDiscriminantAnalysis(solver='lsqr',  shrinkage='auto')
        self.classifier.fit(extracted_features, self.y_train)
        
        if self.cross_val is not None:
            cross_val_score_avg = np.mean(cross_val_score(self.classifier, extracted_features, self.y_train, cv=self.cross_val))

            print('Cross validation score average: ', cross_val_score_avg)            
            return cross_val_score_avg
        
        else:
            return 0                              
            

    def test_binary(self, X_test=None, y_test=None):

        if X_test is not None and y_test is not None:

            X_test_ = CSP_extract_features(X_test, self.CSP_transform, self.n_top)
            predictions = self.classifier.predict(X_test_)
            accuracy = accuracy_score(y_test, predictions)

            print('Testing accuracy: ', accuracy)
            return predictions, accuracy
            
        else:

            X_train_ = CSP_extract_features(self.X_train, self.CSP_transform, self.n_top)
            predictions = self.classifier.predict(X_train_)
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
        self.CSP_transforms = []
        cross_val_scores = []
        
        for i in range(self.num_unique_labels):
            for j in range(i + 1, self.num_unique_labels):
                
                ## Extract data and labels for two classes
                data_c1 = self.X_train[self.y_train == unique_labels[i]]
                data_c2 = self.X_train[self.y_train == unique_labels[j]]
                labels_c1 = self.y_train[self.y_train == unique_labels[i]]
                labels_c2 = self.y_train[self.y_train == unique_labels[j]]

                ## Apply CSP to transform data
                CSP_transform = CSP2(data_c1, data_c2, self.n_top)
                self.CSP_transforms.append(CSP_transform)
                data_c1 = CSP_extract_features(data_c1, CSP_transform, self.n_top)
                data_c2 = CSP_extract_features(data_c2, CSP_transform, self.n_top)
        
                ## Concatenate data and labels
                data_concat = np.concatenate((data_c1, data_c2), axis=0)
                labels_concat = np.concatenate((labels_c1, labels_c2), axis=0)
        
                ## Shuffle data
                data_concat, labels_concat = shuffle(data_concat, labels_concat, random_state=self.random_state)

                lda = LinearDiscriminantAnalysis(solver='lsqr',  shrinkage='auto')
                lda.fit(data_concat, labels_concat)
                self.classifiers.append(lda)

                if self.cross_val is not None:

                    cross_val_score_avg = np.mean(cross_val_score(lda, data_concat, labels_concat, cv=self.cross_val))

                    print('Cross validation scores for labels ', 
                          unique_labels[i], ' and ', 
                          unique_labels[j], ': ', cross_val_score_avg)

                    cross_val_scores.append(cross_val_score_avg)

        return cross_val_scores


    def test_1_vs_1(self, X_test=None, y_test=None):

        if X_test is not None and y_test is not None:
    
            votes = np.zeros((len(y_test), self.num_unique_labels))
            
            for classifier, CSP_transform in zip(self.classifiers, self.CSP_transforms):   

                X_test_ = CSP_extract_features(X_test, CSP_transform, self.n_top)
                predictions = classifier.predict(X_test_) 
                
                for i, prediction in enumerate(predictions):
                    votes[i, prediction - self.label_offset] += 1
                      
            final_predictions = np.argmax(votes, axis=1) + self.label_offset
            accuracy = accuracy_score(y_test, final_predictions)
            
            print('Testing accuracy: ', accuracy)
            return final_predictions, accuracy
    
        else:
            votes = np.zeros((len(self.y_train), self.num_unique_labels))
            
            for classifier, CSP_transform in zip(self.classifiers, self.CSP_transforms):   
                
                X_train_ = CSP_extract_features(self.X_train, CSP_transform, self.n_top)
                predictions = classifier.predict(X_train_) 

                for i, prediction in enumerate(predictions):
                    votes[i, prediction - self.label_offset] += 1
                      
            final_predictions = np.argmax(votes, axis=1) + self.label_offset
            accuracy = accuracy_score(self.y_train, final_predictions)
            
            print('Training accuracy: ', accuracy)
            return final_predictions, accuracy
