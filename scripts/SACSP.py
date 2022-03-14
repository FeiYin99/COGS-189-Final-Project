import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

sys.path.append('/home/inffzy/Desktop/cogs189/cogs189_final_project/scripts')
import classification_utils as utils



def SACSP_cost_function(data_fourier, h_or_l, w_or_v, avg_cov_sum):
    ## Equation 4 and 5 in paper
    
    numerator = (w_or_v.T @ data_fourier @ np.diag(h_or_l) @ np.matrix(data_fourier).H @ w_or_v)[0, 0]
    denominator = (w_or_v.T @ avg_cov_sum @ w_or_v)[0, 0]
    
    return numerator / denominator


def SACSP_update_spatial_filters(data_fourier, h_or_l, avg_cov_sum, top_n):
	## Equation 6 and 7 in paper

    E = data_fourier @ np.diag(h_or_l) @ np.matrix(data_fourier).H
    
    ## Generalized eigen-decomposition of Eq6 and Eq7 along with sum of class' spatial covariances
    eigval, eigvec = sp.linalg.eigh(E, avg_cov_sum)
    
    ## Extract top n eigen vectors
    sort_indices = np.argsort(eigval)
    top_n_indices = list(sort_indices[-top_n:])
    top_n_eigvec = eigvec[:, top_n_indices]
    
    return top_n_eigvec


def SACSP(c1_data, c2_data, R=3, M=1, e=1e-6):
    
    ## R = number of spectral/spatial filters for each class
    ## M = number of initializations of spectral filters
    
    t = c1_data.shape[-1]  ## Number of time samples
    #F = sp.linalg.dft(t)  ## Fourier matrix with shape t x t
    
    H = np.ones((M, t))  ## Initialize M number of h vectors
    L = np.ones((M, t))  ## Initialize M number of l vectors
 
    c1_data_avg = np.mean(c1_data, axis=0)
    c2_data_avg = np.mean(c2_data, axis=0)

    c1_data_avg_cov = np.cov(c1_data_avg)
    c2_data_avg_cov = np.cov(c2_data_avg)
    avg_cov_sum = c1_data_avg_cov + c2_data_avg_cov
    
    c1_data_avg_fourier = np.fft.fft(c1_data_avg)
    c2_data_avg_fourier = np.fft.fft(c2_data_avg)
    
    c1_filter_pairs = []
    c2_filter_pairs = []
    
    for m in range(M):
        
        ## h and l are spectral filters
        h = H[m, :]
        l = L[m, :]
        
        ## w and v are spatial filters
        W = SACSP_update_spatial_filters(c1_data_avg_fourier, h, avg_cov_sum, top_n=R)
        V = SACSP_update_spatial_filters(c2_data_avg_fourier, l, avg_cov_sum, top_n=R)
        
        for r in range(R):  
            
            ### Class 1 optimization ###
            
            w = np.atleast_2d(W[:, r]).T  ## From W, get w as a column vector
            
            c1_cost = 0
            c1_cost_increase = e + 1
            
            while c1_cost_increase > e:
            
                ## Update spectral filter h_r_m from Eq9    
                E = np.matrix(c1_data_avg_fourier).H @ w @ w.T @ c1_data_avg_fourier
                
                for k in range(t):
                    h[k] = E[k, k] / np.linalg.norm(np.diag(E))
                
                H[m, :] = h
                
                ## Update spatial filter w_r by selecting the eigenvector 
                ##   corresponding to the largest eigenvalue from Eq6
                w = SACSP_update_spatial_filters(c1_data_avg_fourier, h, avg_cov_sum, top_n=1)
                W[:, r] = w.flatten()

                ## Calculate cost
                c1_cost_prev = c1_cost
                c1_cost = SACSP_cost_function(c1_data_avg_fourier, h, w, avg_cov_sum)
                c1_cost_increase = c1_cost - c1_cost_prev
   
            c1_filter_pairs.append([h, w])
            
            
            ### Class 2 optimization ###
            
            v = np.atleast_2d(V[:, r]).T  ## From V, get v as a column vector
            
            c2_cost = 0
            c2_cost_increase = e + 1
            
            while c2_cost_increase > e:
            
                ## Update spectral filter l_r_m from Eq10    
                E = np.matrix(c2_data_avg_fourier).H @ v @ v.T @ c2_data_avg_fourier
                
                for k in range(t):
                    l[k] = E[k, k] / np.linalg.norm(np.diag(E))
                
                L[m, :] = l
                
                ## Update spatial filter v_r by selecting the eigenvector 
                ##   corresponding to the largest eigenvalue from Eq7
                v = SACSP_update_spatial_filters(c2_data_avg_fourier, l, avg_cov_sum, top_n=1)
                V[:, r] = v.flatten()

                ## Calculate cost
                c2_cost_prev = c2_cost
                c2_cost = SACSP_cost_function(c2_data_avg_fourier, l, v, avg_cov_sum)
                c2_cost_increase = c2_cost - c2_cost_prev

            c2_filter_pairs.append([l, v])
    
    ## From the M x R pairs of spatial and spectral filters, select R pairs that maximize the cost function

    ## Not implemented since choosing M = 1
    
    return c1_filter_pairs, c2_filter_pairs


def SACSP_extract_features(all_data, c1_filter_pairs, c2_filter_pairs, R=3):
    
    t = all_data.shape[-1]  ## Number of time samples
    F = sp.linalg.dft(t)  ## Fourier matrix with shape t x t
    
    extracted_features = np.zeros((all_data.shape[0], 2 * R))
    
    for i, epoch in enumerate(all_data):    
        epoch_fourier = epoch @ F
        
        for j, filter_pair in enumerate(c1_filter_pairs + c2_filter_pairs):
            spectral_filter = filter_pair[0]  
            spatial_filter = filter_pair[1]
            extracted_feature = np.log(spatial_filter.T @ 
                                       epoch_fourier @ 
                                       np.diag(spectral_filter) @ 
                                       np.matrix(epoch_fourier).H @ 
                                       spatial_filter)
            
            extracted_features[i, j] = extracted_feature[0, 0]
            
    return extracted_features


def SACSP2_cost_function(data, h_or_l, w_or_v, avg_cov_sum):
    ## Equation 4 and 5 in paper

    num_trials = data.shape[0]
    num_channels = data.shape[1]

    E = np.zeros((num_trials, num_channels, num_channels))

    for i in range(num_trials):
        trial = data[i]
        trial_fourier = np.fft.fft(trial)
        E[i] = trial_fourier @ np.diag(h_or_l) @ np.matrix(trial_fourier).H

    E_avg = np.mean(E, axis=0)

    numerator = (w_or_v.T @ E_avg @ w_or_v)[0, 0]
    denominator = (w_or_v.T @ avg_cov_sum @ w_or_v)[0, 0]
    
    return numerator / denominator


def SACSP2_update_spatial_filters(data, h_or_l, avg_cov_sum, top_n):
    ## Equation 6 and 7 in paper

    num_trials = data.shape[0]
    num_channels = data.shape[1]

    E = np.zeros((num_trials, num_channels, num_channels))

    for i in range(num_trials):
        trial = data[i]
        trial_fourier = np.fft.fft(trial)
        E[i] = trial_fourier @ np.diag(h_or_l) @ np.matrix(trial_fourier).H

    E_avg = np.mean(E, axis=0)
    
    ## Generalized eigen-decomposition of Eq6 and Eq7 along with sum of class' spatial covariances
    eigval, eigvec = sp.linalg.eigh(E_avg, avg_cov_sum)
    
    ## Extract top n eigen vectors
    sort_indices = np.argsort(eigval)
    top_n_indices = list(sort_indices[-top_n:])
    #top_n_indices = list(sort_indices[:top_n])
    top_n_eigvec = eigvec[:, top_n_indices]
    
    return top_n_eigvec


def SACSP2_update_spectral_filters(data, w_or_v):
    ## Equation 9 and 10 in paper
    
    num_trials = data.shape[0]
    num_samples = data.shape[-1]

    E = np.zeros((num_trials, num_samples, num_samples))

    for i in range(num_trials):
        trial = data[i]
        trial_fourier = np.fft.fft(trial)
        E[i] = np.matrix(trial_fourier).H @ w_or_v @ w_or_v.T @ trial_fourier

    E_avg = np.mean(E, axis=0)
    E_avg_diag_norm = np.linalg.norm(np.diag(E_avg))

    h = np.zeros(num_samples)

    for k in range(num_samples):
        h[k] = E_avg[k, k] / E_avg_diag_norm

    return h


def SACSP2(c1_data, c2_data, R=3, M=3, e=1e-6, sampling_f=250):
    
    ## R = number of spectral/spatial filters for each class
    ## M = number of initializations of spectral filters
    
    t = c1_data.shape[-1]  ## Number of time samples
    
    H = np.zeros((M, t))  ## Initialize M number of h vectors
    L = np.zeros((M, t))  ## Initialize M number of l vectors
    freq = np.fft.fftfreq(t) * sampling_f

    ## 1st set of spectral filter initialization: all ones
    H[0, :] = np.ones(t)
    L[0, :] = np.ones(t)

    if M > 1:
        ## 2nd set of spectral filter initialization: ones only in 7-15 Hz band
        h2 = np.zeros(t)
        h2[np.all([freq >= 7, freq <= 15], axis=0)] = 1
        h2[np.all([freq <= -7, freq >= -15], axis=0)] = 1
    
        l2 = np.zeros(t)
        l2[np.all([freq >= 7, freq <= 15], axis=0)] = 1
        l2[np.all([freq <= -7, freq >= -15], axis=0)] = 1
    
        H[1, :] = h2
        L[1, :] = l2

    if M > 2:
        ## 3rd set of spectral filter initialization: ones only in 15-30 Hz band
        h3 = np.zeros(t)
        h3[np.all([freq >= 15, freq <= 30], axis=0)] = 1
        h3[np.all([freq <= -15, freq >= -30], axis=0)] = 1
    
        l3 = np.zeros(t)
        l3[np.all([freq >= 15, freq <= 30], axis=0)] = 1
        l3[np.all([freq <= -15, freq >= -30], axis=0)] = 1
    
        H[2, :] = h3
        L[2, :] = l3


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
    
    c1_filter_pairs = []
    c2_filter_pairs = []

    c1_costs = []
    c2_costs = []
    
    for m in range(M):
        
        ## h and l are spectral filters
        h = H[m, :]
        l = L[m, :]
        
        ## w and v are spatial filters
        W = SACSP2_update_spatial_filters(c1_data, h, avg_cov_sum, top_n=R)
        V = SACSP2_update_spatial_filters(c2_data, l, avg_cov_sum, top_n=R)
        
        for r in range(R):  
            
            ### Class 1 optimization ###
            
            w = np.atleast_2d(W[:, r]).T  ## From W, get w as a column vector
            
            c1_cost = 0
            c1_cost_increase = e + 1
            
            while c1_cost_increase > e:
            
                ## Update spectral filter h_r_m from Eq9    
                h = SACSP2_update_spectral_filters(c1_data, w)
                H[m, :] = h
                
                ## Update spatial filter w_r by selecting the eigenvector 
                ##   corresponding to the largest eigenvalue from Eq6
                w = SACSP2_update_spatial_filters(c1_data, h, avg_cov_sum, top_n=1)
                W[:, r] = w.flatten()

                ## Calculate cost
                c1_cost_prev = c1_cost
                c1_cost = SACSP2_cost_function(c1_data, h, w, avg_cov_sum)
                c1_cost_increase = c1_cost - c1_cost_prev
   
            c1_filter_pairs.append([h, w])
            c1_costs.append(c1_cost)
            
            
            ### Class 2 optimization ###
            
            v = np.atleast_2d(V[:, r]).T  ## From V, get v as a column vector
            
            c2_cost = 0
            c2_cost_increase = e + 1
            
            while c2_cost_increase > e:
            
                ## Update spectral filter l_r_m from Eq10    
                l = SACSP2_update_spectral_filters(c2_data, v)
                L[m, :] = l
                
                ## Update spatial filter v_r by selecting the eigenvector 
                ##   corresponding to the largest eigenvalue from Eq7
                v = SACSP2_update_spatial_filters(c2_data, l, avg_cov_sum, top_n=1)
                V[:, r] = v.flatten()

                ## Calculate cost
                c2_cost_prev = c2_cost
                c2_cost = SACSP2_cost_function(c2_data, l, v, avg_cov_sum)
                c2_cost_increase = c2_cost - c2_cost_prev

            c2_filter_pairs.append([l, v])
            c2_costs.append(c2_cost)


    ## From the M x R pairs of spatial and spectral filters, select R pairs (for each class) that maximize the cost function
    c1_costs_sort_indices = np.argsort(c1_costs)
    #c1_top_R_indices = list(c1_costs_sort_indices[-R:])
    c1_top_R_indices = list(c1_costs_sort_indices[:R])
    best_c1_filter_pairs = [c1_filter_pairs[i] for i in c1_top_R_indices]

    c2_costs_sort_indices = np.argsort(c2_costs)
    #c2_top_R_indices = list(c2_costs_sort_indices[-R:])
    c2_top_R_indices = list(c2_costs_sort_indices[:R])
    best_c2_filter_pairs = [c2_filter_pairs[i] for i in c2_top_R_indices]
    
    return best_c1_filter_pairs, best_c2_filter_pairs


def SACSP2_extract_features(all_data, c1_filter_pairs, c2_filter_pairs, R=3):
    
    t = all_data.shape[-1]  ## Number of time samples
    F = sp.linalg.dft(t)  ## Fourier matrix with shape t x t
    
    extracted_features = np.zeros((all_data.shape[0], 2 * R))
    
    for i, epoch in enumerate(all_data):    
        epoch_fourier = np.fft.fft(epoch)
        
        for j, filter_pair in enumerate(c1_filter_pairs + c2_filter_pairs):
            spectral_filter = filter_pair[0]  
            spatial_filter = filter_pair[1]
            extracted_feature = np.log(spatial_filter.T @ 
                                       epoch_fourier @ 
                                       np.diag(spectral_filter) @ 
                                       np.matrix(epoch_fourier).H @ 
                                       spatial_filter)
            
            extracted_features[i, j] = extracted_feature[0, 0]
            
    return extracted_features


class SACSP_LDA_classifier:
    
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
            
        ## Apply SACSP to transform data
        self.c1_filter_pairs, self.c2_filter_pairs = SACSP2(data_c1, data_c2, M=self.n_top)
        extracted_features = SACSP2_extract_features(self.X_train, self.c1_filter_pairs, self.c2_filter_pairs)
        
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

            X_test_ = SACSP2_extract_features(X_test, self.c1_filter_pairs, self.c2_filter_pairs)
            predictions = self.classifier.predict(X_test_)
            accuracy = accuracy_score(y_test, predictions)

            print('Testing accuracy: ', accuracy)
            return predictions, accuracy
            
        else:

            X_train_ = SACSP2_extract_features(self.X_train, self.c1_filter_pairs, self.c2_filter_pairs)
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
        self.SACSP_filter_pairs = []
        cross_val_scores = []
        
        for i in range(self.num_unique_labels):
            for j in range(i + 1, self.num_unique_labels):
                
                ## Extract data and labels for two classes
                data_c1 = self.X_train[self.y_train == unique_labels[i]]
                data_c2 = self.X_train[self.y_train == unique_labels[j]]
                labels_c1 = self.y_train[self.y_train == unique_labels[i]]
                labels_c2 = self.y_train[self.y_train == unique_labels[j]]

                ## Apply SACSP to transform data
                c1_filter_pairs, c2_filter_pairs = SACSP2(data_c1, data_c2, M=self.n_top)
                self.SACSP_filter_pairs.append([c1_filter_pairs, c2_filter_pairs])
                data_c1 = SACSP2_extract_features(data_c1, c1_filter_pairs, c2_filter_pairs)
                data_c2 = SACSP2_extract_features(data_c2, c1_filter_pairs, c2_filter_pairs)

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
            
            for classifier, filter_pairs in zip(self.classifiers, self.SACSP_filter_pairs):   

                X_test_ = SACSP2_extract_features(X_test, filter_pairs[0], filter_pairs[1])
                predictions = classifier.predict(X_test_) 
                
                for i, prediction in enumerate(predictions):
                    votes[i, prediction - self.label_offset] += 1
                      
            final_predictions = np.argmax(votes, axis=1) + self.label_offset
            accuracy = accuracy_score(y_test, final_predictions)
            
            print('Testing accuracy: ', accuracy)
            return final_predictions, accuracy
    
        else:
            votes = np.zeros((len(self.y_train), self.num_unique_labels))
            
            for classifier, filter_pairs in zip(self.classifiers, self.SACSP_filter_pairs):   
                
                X_train_ = SACSP2_extract_features(self.X_train, filter_pairs[0], filter_pairs[1])                
                predictions = classifier.predict(X_train_) 

                for i, prediction in enumerate(predictions):
                    votes[i, prediction - self.label_offset] += 1
                      
            final_predictions = np.argmax(votes, axis=1) + self.label_offset
            accuracy = accuracy_score(self.y_train, final_predictions)
            
            print('Training accuracy: ', accuracy)
            return final_predictions, accuracy
