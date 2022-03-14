import sys
import numpy as np
import scipy as sp
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

sys.path.append('/home/inffzy/Desktop/cogs189/cogs189_final_project/scripts')
import classification_utils as utils



def cost_function(data_fourier, h_or_l, w_or_v, avg_cov_sum):
    ## Equation 4 and 5 in paper
    
    numerator = (w_or_v.T @ data_fourier @ np.diag(h_or_l) @ np.matrix(data_fourier).H @ w_or_v)[0, 0]
    denominator = (w_or_v.T @ avg_cov_sum @ w_or_v)[0, 0]
    
    return numerator / denominator


def update_spatial_filters(data_fourier, h_or_l, avg_cov_sum, top_n):
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
        W = update_spatial_filters(c1_data_avg_fourier, h, avg_cov_sum, top_n=R)
        V = update_spatial_filters(c2_data_avg_fourier, l, avg_cov_sum, top_n=R)
        
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
                w = update_spatial_filters(c1_data_avg_fourier, h, avg_cov_sum, top_n=1)
                W[:, r] = w.flatten()

                ## Calculate cost
                c1_cost_prev = c1_cost
                c1_cost = cost_function(c1_data_avg_fourier, h, w, avg_cov_sum)
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
                v = update_spatial_filters(c2_data_avg_fourier, l, avg_cov_sum, top_n=1)
                V[:, r] = v.flatten()

                ## Calculate cost
                c2_cost_prev = c2_cost
                c2_cost = cost_function(c2_data_avg_fourier, l, v, avg_cov_sum)
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



class SACSP_LDA_classifier:
    
    def __init__(self, X_train, y_train, cross_val=None):
        self.X_train = X_train
        self.y_train = y_train
        self.cross_val = cross_val

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
        self.c1_filter_pairs, self.c2_filter_pairs = SACSP(data_c1, data_c2)
        extracted_features = SACSP_extract_features(self.X_train, self.c1_filter_pairs, self.c2_filter_pairs)
        
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

            X_test_ = SACSP_extract_features(X_test, self.c1_filter_pairs, self.c2_filter_pairs)
            predictions = self.classifier.predict(X_test_)
            accuracy = accuracy_score(y_test, predictions)

            print('Testing accuracy: ', accuracy)
            return predictions, accuracy
            
        else:

            X_train_ = SACSP_extract_features(self.X_train, self.c1_filter_pairs, self.c2_filter_pairs)
            predictions = self.classifier.predict(X_train_)
            accuracy = accuracy_score(self.y_train, predictions)

            print('Training accuracy: ', accuracy)
            return predictions, accuracy

    """
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
                CSP_transform = CSP(data_c1, data_c2)
                self.CSP_transforms.append(CSP_transform)
                data_c1 = apply_CSP(CSP_transform, data_c1)
                data_c2 = apply_CSP(CSP_transform, data_c2)
        
                ## Concatenate data and labels
                data_concat = np.concatenate((data_c1, data_c2), axis=0)
                labels_concat = np.concatenate((labels_c1, labels_c2), axis=0)
        
                ## Downsample with windowed means
                data_concat = utils.windowed_means(data_concat, self.num_samples)
                
                ## Flatten data
                data_concat = utils.flatten_dim12(data_concat)
                
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
                X_test_ = utils.windowed_means(X_test_, self.num_samples)
                X_test_ = utils.flatten_dim12(X_test_)

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
                X_train_ = utils.windowed_means(X_train_, self.num_samples)
                X_train_ = utils.flatten_dim12(X_train_)
                
                predictions = classifier.predict(X_train_) 

                for i, prediction in enumerate(predictions):
                    votes[i, prediction - self.label_offset] += 1
                      
            final_predictions = np.argmax(votes, axis=1) + self.label_offset
            accuracy = accuracy_score(self.y_train, final_predictions)
            
            print('Training accuracy: ', accuracy)
            return final_predictions, accuracy
    """
