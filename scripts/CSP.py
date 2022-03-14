import sys
import numpy as np
import scipy as sp
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

sys.path.append('/home/inffzy/Desktop/cogs189/cogs189_final_project/scripts')
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


def CSP2(c1_data, c2_data, n_top=3, n_bot=3):

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
    bot_n_indices = list(sort_indices[:n_bot])
    eigvec_extracted = eigvec[:, top_n_indices + bot_n_indices]
    W = eigvec_extracted.T

    return W


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


class CSP_LDA_classifier:
    
    def __init__(self, X_train, y_train, cross_val=None, num_samples=10):
        self.X_train = X_train
        self.y_train = y_train
        self.cross_val = cross_val
        self.num_samples = num_samples
        

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
        #self.CSP_transform = CSP(data_c1, data_c2)
        self.CSP_transform = CSP2(data_c1, data_c2)
        data_transformed = apply_CSP(self.CSP_transform, self.X_train)

        ## Downsample with windowed means
        #data_transformed = utils.windowed_means(data_transformed, self.num_samples)
        
        ## Flatten data
        #data_transformed = utils.flatten_dim12(data_transformed)

        data_transformed = np.mean(data_transformed, axis=-1)


        
        self.classifier = LinearDiscriminantAnalysis(solver='lsqr',  shrinkage='auto')
        self.classifier.fit(data_transformed, self.y_train)
        
        if self.cross_val is not None:
            cross_val_score_avg = np.mean(cross_val_score(self.classifier, data_transformed, self.y_train, cv=self.cross_val))

            print('Cross validation score average: ', cross_val_score_avg)            
            return cross_val_score_avg
        
        else:
            return 0                              
            

    def test_binary(self, X_test=None, y_test=None):

        if X_test is not None and y_test is not None:

            X_test_ = apply_CSP(self.CSP_transform, X_test)

            X_test_ = np.mean(X_test_, axis=-1)

            #X_test_ = utils.windowed_means(X_test_, self.num_samples)
            #X_test_ = utils.flatten_dim12(X_test_)

            predictions = self.classifier.predict(X_test_)
            accuracy = accuracy_score(y_test, predictions)

            print('Testing accuracy: ', accuracy)
            return predictions, accuracy
            
        else:

            X_train_ = apply_CSP(self.CSP_transform, self.X_train)

            X_train_ = np.mean(X_train_, axis=-1)

            #X_train_ = utils.windowed_means(X_train_, self.num_samples)
            #X_train_ = utils.flatten_dim12(X_train_)

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
                CSP_transform = CSP2(data_c1, data_c2)
                self.CSP_transforms.append(CSP_transform)
                data_c1 = apply_CSP(CSP_transform, data_c1)
                data_c2 = apply_CSP(CSP_transform, data_c2)
        
                ## Concatenate data and labels
                data_concat = np.concatenate((data_c1, data_c2), axis=0)
                labels_concat = np.concatenate((labels_c1, labels_c2), axis=0)

                data_concat = np.mean(data_concat, axis=-1)
        
                ## Downsample with windowed means
                #data_concat = utils.windowed_means(data_concat, self.num_samples)
                
                ## Flatten data
                #data_concat = utils.flatten_dim12(data_concat)
                
                ## Shuffle data
                data_concat, labels_concat = shuffle(data_concat, labels_concat, random_state=42)

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

                X_test_ = apply_CSP(CSP_transform, X_test)
                X_test_ = np.mean(X_test_, axis=-1)

                #X_test_ = utils.windowed_means(X_test_, self.num_samples)
                #X_test_ = utils.flatten_dim12(X_test_)

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
                
                X_train_ = apply_CSP(CSP_transform, self.X_train)
                X_train_ = np.mean(X_train_, axis=-1)

                #X_train_ = utils.windowed_means(X_train_, self.num_samples)
                #X_train_ = utils.flatten_dim12(X_train_)
                
                predictions = classifier.predict(X_train_) 

                for i, prediction in enumerate(predictions):
                    votes[i, prediction - self.label_offset] += 1
                      
            final_predictions = np.argmax(votes, axis=1) + self.label_offset
            accuracy = accuracy_score(self.y_train, final_predictions)
            
            print('Training accuracy: ', accuracy)
            return final_predictions, accuracy
