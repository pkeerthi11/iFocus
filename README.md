# iFocus
Implementing an SVM classifier to determine attention states. 

Test proof-of-concept real-time implementation using the EEGAttentionClassification_Realtime.py script. 

Key functions:

generate_epochs(file_path, TBuffer=15, time_index = 0)
Given input data file, calculate the duration of the recording and split into segments of length TBuffer

feature_generation(EEG_data,Fs=250,deltaT=15)
Given data over a time interval of deltaT, extract spectral power in 0.5 Hz bins from 0-18 Hz

prepare_data(data_directory)
Extract features for data and label based on file names

run_pipeline(data_directory,save_classifier_file)
Run the linear SVM classifier and save the model object in a .pkl file for future deployment

