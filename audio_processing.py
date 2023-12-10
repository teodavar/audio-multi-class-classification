"""
Created on Wed Apr 12 15:24:26 2023

@author: davarakis
"""

import librosa
import numpy as np
import os

# Get MFCC coefficients
def get_mfcc(y, sr):
  #y, sr = librosa.load(wav_file_path, offset=0, duration=30)
  mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr))
  return mfcc

# Get Mel spectogram
def get_melspectrogram(y, sr):
  #y, sr = librosa.load(wav_file_path, offset=0, duration=30)
  melspectrogram = np.array(librosa.feature.melspectrogram(y=y, sr=sr))
  return melspectrogram

# Get Chroma vector
def get_chroma_vector(y, sr):
  #y, sr = librosa.load(wav_file_path)
  chroma = np.array(librosa.feature.chroma_stft(y=y, sr=sr))
  return chroma

# Get Tonal centroid features
def get_tonnetz(y, sr):
  #y, sr = librosa.load(wav_file_path)
  tonnetz = np.array(librosa.feature.tonnetz(y=y, sr=sr))
  return tonnetz

# Workflow 1 - For each audio feaure calculate statistic metrics:
# min, max and mean
def get_feature(song, sr):

  # Extracting MFCC feature
  mfcc = get_mfcc(song, sr)
  mfcc_mean = mfcc.mean(axis=1)
  mfcc_min = mfcc.min(axis=1)
  mfcc_max = mfcc.max(axis=1)
  mfcc_feature = np.concatenate( (mfcc_mean, mfcc_min, mfcc_max) )

  # Extracting Mel Spectrogram feature
  melspectrogram = get_melspectrogram(song, sr)
  melspectrogram_mean = melspectrogram.mean(axis=1)
  melspectrogram_min = melspectrogram.min(axis=1)
  melspectrogram_max = melspectrogram.max(axis=1)
  melspectrogram_feature = np.concatenate( (melspectrogram_mean, melspectrogram_min, melspectrogram_max) )

  # Extracting chroma vector feature
  chroma = get_chroma_vector(song, sr)
  chroma_mean = chroma.mean(axis=1)
  chroma_min = chroma.min(axis=1)
  chroma_max = chroma.max(axis=1)
  chroma_feature = np.concatenate( (chroma_mean, chroma_min, chroma_max) )

  # Extracting tonnetz feature
  tntz = get_tonnetz(song, sr)
  tntz_mean = tntz.mean(axis=1)
  tntz_min = tntz.min(axis=1)
  tntz_max = tntz.max(axis=1)
  tntz_feature = np.concatenate( (tntz_mean, tntz_min, tntz_max) ) 
  
  # concatenate the extracted features
  feature = np.concatenate( (chroma_feature, melspectrogram_feature, mfcc_feature, tntz_feature) )
  return feature

# Workflow 1 - Read Original dataset and extract audio features 
# based on statistic measures
def calculateStatisticsFeatures():
  
    directory = 'archive/Data/genres_original'
    genres = ['blues','classical','country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock' ]

    features = []
    labels = []
    for genre in genres:
        print("Calculating features for genre : " + genre)
        for file in os.listdir(directory+"/"+genre):
            print("Calculating features for file : " + file)
            # skip jazz00054.wav - it is corrupted
            if file == 'jazz.00054.wav':
                print("Skipping jazz.00054.wav!!!!")
            else:
                file_path = directory+"/"+genre+"/"+file
                song, sr = librosa.load(file_path, offset=0, duration=30)
                features.append(get_feature(song, sr))
                label = genres.index(genre)
                labels.append(label)
    return features, labels

# Workflow 1 - Split Original dataset to create the increased dataset
# and then extract audio features based on statistic measures
def calculateStatisticsFeatures_3sec():
  
    directory = 'archive/Data/genres_original'
    genres = ['blues','classical','country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock' ]
    features = []
    labels = []
    clip_duration = 3
    song_length = 30
    for genre in genres:
        print("Calculating features for genre : " + genre)
        for file in os.listdir(directory+"/"+genre):
            print("Calculating features for file : " + file)
            # skip jazz00054.wav - it is corrupted
            if file == 'jazz.00054.wav':
                print("Skipping jazz.00054.wav!!!!")
            else:
                file_path = directory+"/"+genre+"/"+file
                #print(file)
                # Split each audio file into clips of 3seconds length
                for k in range(0, song_length, 3):
                    song, sr = librosa.load(file_path, offset=k, duration=clip_duration)
                    feat = get_feature(song, sr)
                    #print(feat.shape)
                    features.append(feat)
                    label = genres.index(genre)
                    labels.append(label)
    return features, labels

# Workflow 1 - Prepare the training / validation / testing datasets
def prepareStatisticsDataset(features, labels):
    
    print("FEATURES len:", len(features))
    print("labels len:", len(labels))
    
    np.random.seed(2023)

    # 60% of data : for training
    training_size = int(len(features)*60/100)
    # 20% of data : for validation
    validation_size = training_size + int(len(features)*20/100)
    # last 20% of data : for testing
    
    permutations = np.random.permutation(len(features))
    features = np.array(features)[permutations]
    labels = np.array(labels)[permutations]
    
    features_train = features[0:training_size]
    print("features_train len:", len(features_train))
    labels_train = labels[0:training_size]
    print("labels_train len:", len(labels_train))
    
    features_val = features[training_size:validation_size]
    print("features_val len:", len(features_val))
    labels_val = labels[training_size:validation_size]
    print("labels_val len:", len(labels_val))
    
    features_test = features[validation_size:len(features)]
    print("features_test len:", len(features_test))
    labels_test = labels[validation_size:len(features)]
    print("labels_test len:", len(labels_test))
    
    
    return features_train, labels_train, features_val, labels_val, features_test, labels_test

# Workflow 2 - Read the original dataset and extact MFCC feature
def calculateAudioFeatures():
    directory = 'archive/Data/genres_original'
    audio_data = {
        "labels": [],
        "mfcc": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(directory)):
    
        for file in filenames:
            if file == 'jazz.00054.wav':
                print("Skipping jazz.00054.wav!!!!")
            else:
                #print(file)
                song, sr = librosa.load(os.path.join(dirpath, file), duration=30)
                #song, sr = librosa.load(os.path.join(dirpath, file), offset=0, duration=30)
    
                mfcc = librosa.feature.mfcc(y=song, sr=sr)
                #print("shape of mfcc: ", mfcc.shape)
                if mfcc.shape[0]!=20:
                    print("NOT 20:", file, mfcc.shape)
                if mfcc.shape[1]!=1292:
                    # Some audio files are smaller than 30 seconds
                    # At the end pad then with their edge values!
                    print("NOT 1292:", file, mfcc.shape)
                    print("padding")
                    x = 1292 - mfcc.shape[1]
                    mfcc = np.pad(mfcc, [(0,0),(0, x)], mode='edge')
                    #new_mfcc = np.pad(mfcc, [(0,0),(0, x)], mode='mean')
                    print("NEW SHAPE:", file, mfcc.shape)
                mfcc = mfcc.T
    
                audio_data["labels"].append(i-1)
                audio_data["mfcc"].append(mfcc.tolist())       
    return audio_data

# Workflow 2 - Read the original dataset, prepare the 
# increased dataset (by spliting each audio file to 3secs clips
# and extact MFCC feature
def calculateAudioFeatures_3sec():
    directory = 'archive/Data/genres_original'
    audio_data = {
        "labels": [],
        "mfcc": []
    }
    clip_duration = 3
    song_length = 30
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(directory)):
        for file in filenames:
            if file == 'jazz.00054.wav':
                print("Skipping jazz.00054.wav!!!!")
            else:
                #print(file)
                # split each audio file to 10 clips of 3 secs length
                for k in range(0, song_length, 3):
                    #print(k)
                    y, sr = librosa.load(os.path.join(dirpath, file), offset=k, duration=clip_duration)
                    #print("mfcc")
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)
                    if mfcc.shape[0]!=20:
                        print("NOT 20:", file, mfcc.shape)
                    if mfcc.shape[1]!=130:
                        # Some audio file are smaller that 30 secs
                        # Their last clip is padded to contain their edge values
                        print("NOT 130:", file, mfcc.shape)
                        print("padding")
                        x = 130 - mfcc.shape[1]
                        mfcc = np.pad(mfcc, [(0,0),(0, x)], mode='edge')
                        #new_mfcc = np.pad(mfcc, [(0,0),(0, x)], mode='mean')
                        print("NEW SHAPE:", file, mfcc.shape)
                    mfcc = mfcc.T
                    audio_data["labels"].append(i-1)
                    audio_data["mfcc"].append(mfcc.tolist())       
    return audio_data