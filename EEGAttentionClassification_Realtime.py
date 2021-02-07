from os import path, listdir
from pathlib import Path
import numpy as np
import scipy as sp
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import math
import pickle
from RingBuffer import RingBuffer
import time
from datetime import datetime
from EEGAttentionClassification import feature_generation

CLASSIFIER_FILE = "predict_attention.pkl"
PATH_TO_MODEL = path.dirname(__file__)
PATH_TO_SAMPLESET = path.dirname(__file__)
SAMPLESET = "DemoSet.csv"
fullfile_path = path.join(PATH_TO_SAMPLESET,SAMPLESET)
full_EEG_data = np.genfromtxt(fullfile_path, delimiter = ',')
full_EEG_data = full_EEG_data[1:]
#Remove First and last columns from file
full_EEG_data =  full_EEG_data[:,1:-1]




#Iterate through EEG data at sampling frequency to simulate real-time nature of device
def main(Fs=250,TBuffer=15):
    period = 1/Fs

    #Load SVM model
    fullmodel_path = path.join(PATH_TO_MODEL,CLASSIFIER_FILE)
    svm_model = pickle.load(open(fullmodel_path,'rb'))

    #Load Sample Set for Demo 
    fullfile_path = path.join(PATH_TO_SAMPLESET,SAMPLESET)
    full_EEG_data = np.genfromtxt(fullfile_path, delimiter = ',')
    full_EEG_data = full_EEG_data[1:]
    full_EEG_data =  full_EEG_data[:,1:-1]


    #Gain data in 15 second window before performing feature calculations  
    eeg_buffer = RingBuffer(Fs*TBuffer)
    for i in range(Fs*TBuffer):
        timepoint_data = full_EEG_data[i,:]
        eeg_buffer.append(timepoint_data)
        time.sleep(period)
        if i%250 == 0:
            print('Time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.now())," - State Unknown")

    #Iterate through remaining EEG values
    while i<full_EEG_data.shape[0]-1:
        timepoint_data = full_EEG_data[i,:]
        eeg_buffer.append(timepoint_data)
        time.sleep(period)

        #Use last 15s of data for feature extraction
        eeg_array = np.array(eeg_buffer.get())
        eeg_features = feature_generation(eeg_array)
        eeg_features = eeg_features.reshape(1,-1)
        state_prediction = svm_model.predict(eeg_features) 
        
        #State prediction of 0, coded as non-attentive state
        if state_prediction == 0:

            #Vibrate vibration motor at this point
            print("Whoops, you aren't paying attention!")
            button = input("Would you like to stop the vibration? Press Y to continue: ")

            if button == "Y":
                print("Welcome back! ")

                #Stop vibration, clear the buffer and start again - previous state should have no affect
                eeg_buffer = RingBuffer(Fs*TBuffer)

                for j in range(Fs*TBuffer):
                    if i>=full_EEG_data.shape[0]:
                        break
                    timepoint_data = full_EEG_data[i,:]
                    eeg_buffer.append(timepoint_data)
                    time.sleep(period)
                    i+=1
                    
                    if i%250 == 0:
                        print('Time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.now())," - State Unknown")

            
        i+=1
        if i%250 == 0:
            print('Time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.now())," - Focused")

if __name__== '__main__':
    main()

