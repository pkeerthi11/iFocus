from os import path, listdir
from pathlib import Path
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import math
import pickle
import time

DATA_DIR = path.join(path.dirname(__file__),"original_data")
CLASSIFIER_NAME = "predict_attention.pkl"

def binArray(data, axis, binstep, binsize, func=np.nanmean):
    """
    Alexandre Kempf
    https://stackoverflow.com/questions/21921178/binning-a-numpy-array/42024730#42024730
    """

    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]

    data = data.transpose(argdims)
    data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

#Generate windows for training data
def generate_epochs(file_path, TBuffer=15, time_index = 0):
    full_EEG_data = np.genfromtxt(file_path, delimiter = ',')
    full_EEG_data = full_EEG_data[1:]
    duration = full_EEG_data[-1,time_index]-full_EEG_data[0,time_index]

    #Remove timestamps and AUX channel
    full_EEG_data =  full_EEG_data[:,1:-1]

    #Split into TBuffer sized epochs (seconds)
    num_segs = duration/TBuffer 
    num_segs = math.ceil(num_segs)
    EEG_epochs = np.array_split(full_EEG_data, num_segs,axis=0)

    return EEG_epochs

#Generate features in frequency domain
def feature_generation(EEG_data,Fs=250,deltaT=15):
    f,t,S_tw = signal.spectrogram(EEG_data,fs=Fs,window='blackman',nperseg=Fs,nfft=1000,axis=0) 

    #Use keep frequencies 18 Hz and lower, group into 0.5 Hz bins, average spectral power time window 
    S_tw = np.delete(S_tw, np.where(f >= 18),axis=0)
    f = np.delete(f, np.where(f >= 18))

    f_05_bins = binArray(f,0,2,2,np.min)
    specpower_05bins = binArray(S_tw, 0, 2, 2, np.mean)
    specpower_05bins_smooth = np.mean(specpower_05bins,axis=2) 

    return specpower_05bins_smooth.flatten() #Reshape to 1D vector

def prepare_data(data_directory):
    data_files = [x for x in listdir(DATA_DIR) if x.endswith(".csv")]
    
    features = []
    target  = []
    
    for file in data_files:
        string_labels = file.split("-")

        #Code 1 as concentration state, 0 as non-concentration state
        if(string_labels[1]=="concentrating"):
            label = 1
        else:
            label = 0
            
        current_file = path.join(DATA_DIR,file)

        #Split into default 15s intervals
        epochs = generate_epochs(current_file)

        #Calculate features for each windowed segment
        for epoch in epochs:
            EEG_features = feature_generation(epoch)
            features.append(EEG_features)
            target.append(label)
            
    return features, target

def run_pipeline(data_directory,save_classifier_file):
    features, true_target = prepare_data(data_directory)

    #80-20 split of training data, use SVM classifier
    train_feat, test_feat, train_targ, test_targ = train_test_split(features, true_target, test_size=0.2, random_state=109)
    clf = SVC(kernel='linear')
    clf.fit(train_feat, train_targ)

    pred_targ = clf.predict(test_feat)
    print("Accuracy:",metrics.accuracy_score(test_targ, pred_targ))
    print("Precision:",metrics.precision_score(test_targ, pred_targ))
    print("Recall:",metrics.recall_score(test_targ, pred_targ))

    #Save classifier
    with open(save_classifier_file,'wb') as fid:
        pickle.dump(clf,fid)


    
if __name__ == '__main__':
    run_pipeline(DATA_DIR,CLASSIFIER_NAME)
        
    
