# iFocus
Implementing an SVM classifier to determine attention states. Classifier is built in EEGAttentionClassification.py
Test proof-of-concept real-time implementation using the EEGAttentionClassification_Realtime.py script. 

Key functions:

*generate_epochs(file_path, TBuffer=15, time_index = 0)*\
Given input data file, calculate the duration of the recording and split into segments of length TBuffer

*feature_generation(EEG_data,Fs=250,deltaT=15)*\
Given data over a time interval of deltaT, extract spectral power in 0.5 Hz bins from 0-18 Hz

*prepare_data(data_directory)*\
Extract features for data and label based on file names

*run_pipeline(data_directory,save_classifier_file)*\
Run the linear SVM classifier and save the model object in a .pkl file for future deployment

# Sources:

Dataset from: \
https://www.kaggle.com/birdy654/eeg-mental-state-v2

Signal processing and machine learning techniques:\ 
Acı Çİ, Kaya M, Mishchenko Y. Distinguishing mental attention states of humans via an EEG-based passive BCI using machine learning methods. Expert Systems with Applications. 2019 Nov 15;134:153-66. https://doi.org/10.1016/j.eswa.2019.05.057

