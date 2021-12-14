import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import scipy.signal as sig
import numpy as np
from scipy.io import wavfile
import glob
import os
from scipy import stats
import matplotlib.pyplot as plt
import math

# %%
def  block_audio(x,blockSize,hopSize,fs):   
   # allocate memory    
    numBlocks = math.ceil(x.size / hopSize)    
    xb = np.zeros([numBlocks, blockSize])    
    # compute time stamps    
    t = (np.arange(0, numBlocks) * hopSize) / fs    
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)   
    for n in range(0, numBlocks):        
        i_start = n * hopSize        
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])        
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]    
    return (xb,t)

# %%
# RMS
def extract_rms(xb):
    for i in range(np.shape(xb)[0]):
        rms = np.sqrt(np.abs(np.sum((xb[i]**2))/xb[i].size))
        amp = 20 * np.log10(rms)
        if amp < -100:
            amp = -100
        if i == 0:
            amps = [amp]
        else:
            amps = np.concatenate((amps,[amp]))
    return amps

# %%
# Spectral Centroid
def extract_spectral_centroid(xb,fs):
    for i in range(np.shape(xb)[0]):
        f,z,a = sig.stft(xb[i],fs = fs,window = "hann",nperseg = xb[i].size)
        a = abs(a[:,1])
        if np.sum(a) == 0:
            centroid = 0
        else:
            centroid = np.sum(np.dot(f,a))/np.sum(a)
        if i == 0:
            centroids = [centroid]
        else:
            centroids = np.concatenate((centroids,[centroid]))
    return centroids

# %%
# Zero Crossing Rate
def extract_zerocrossingrate(xb):
    for i in range(np.shape(xb)[0]):
        zcr = [.5 * np.mean(abs(np.diff(np.sign(xb[i]))))]
        if i == 0:
            zcrs = zcr
        else:
            zcrs = np.concatenate((zcrs,zcr))
    return zcrs

# %%
# Spectral Crest
def extract_spectral_crest(xb):
    for i in range(np.shape(xb)[0]):
        f,z,a = sig.stft(xb[i],fs = 10000,window = "hann",nperseg = xb[i].size)
        a = abs(a[:,1])
        if np.sum(a)==0:
            cr = [0]
        else:    
            cr = [np.max(a)/np.sum(a)]
        if i == 0:
            crs = cr
        else:
            crs = np.concatenate((crs,cr))
    return crs

# %%
# Spectral Flux
def extract_spectral_flux(xb):
    fls = np.array([])
    for i in range(np.shape(xb)[0]):
        f,z,a = sig.stft(xb[i],fs = 10000,window = "hann",nperseg = xb[i].size)
        a = abs(a[:,1])
        fl = [np.sqrt(np.sum(np.diff(a)**2))/(np.diff(a).size+1)]
        if i == 0:
            fls = fl
        else:
            fls = np.concatenate((fls,fl))
    return fls
def extract_ste(xb):
    ste = np.array([sum(abs(block**2)) for block in xb])
    
    return ste

# %%
# Extract all features
def extract_features(x,blockSize,hopSize,fs, feature_list):
    xb,time = block_audio(x,blockSize,hopSize,fs)
    features = []
    if "spectral_centroid" in feature_list:
        features.append(extract_spectral_centroid(xb, fs))
    if "rms" in feature_list:
        features.append(extract_rms(xb))
    if "zcr" in feature_list:
        features.append(extract_zerocrossingrate(xb))
    if "spectral_crest" in feature_list:
        features.append(extract_spectral_crest(xb))
    if "spectral_flux" in feature_list:
        features.append(extract_spectral_flux(xb))
    if "ste" in feature_list:
        features.append(extract_ste(xb))
        
    features = np.vstack(features)
    # features = np.vstack((extract_spectral_centroid(xb, fs),
    #                    extract_rms(xb),
    #                    extract_zerocrossingrate(xb),
    #                    extract_spectral_crest(xb),
    #                    extract_spectral_flux(xb)))
    return features

# %%
# Return matrix with mean and standard deviation of all features
def aggregate_feature_per_file(features):
    aggFeatures = np.zeros(len(features)*2)

    for n in range(len(features)):
        index = 2*n
        aggFeatures[index] = np.mean(features[n])
        aggFeatures[index+1] = np.std(features[n])
        

    return aggFeatures

# %%
# Obtain feature data for each file in a folder path
def get_feature_data(path, blockSize, hopSize):

    featureData = np.zeros([10])
    for filename in glob.glob(os.path.join(path, '*.wav')):
        (fs, x) = wavfile.read(filename)
        extractFeat = extract_features(x, blockSize, hopSize, fs)
        featureAgg = aggregate_feature_per_file(extractFeat)
        featureData = np.vstack([featureData, featureAgg])
        
    featureData = np.delete(featureData, 0, axis = 0)
    featureData = np.transpose(featureData)
    return featureData



# %%
# Use SciPy zscore normalization function on feature data
# def normalize_zscore(featureData):
#     normFeatureMatrix = stats.zscore(featureData)
#     return normFeatureMatrix
def normalize_zscore(featureData):
    mu = np.mean(featureData)
    std = np.std(featureData)
    
    centered = featureData - mu
    normFeatureMatrix = centered / std

    return normFeatureMatrix

def visualize_features(path_to_musicspeech):
    music = get_feature_data(path_to_musicspeech + "/music_wav",1024,256)
    speech = get_feature_data(path_to_musicspeech + "/speech_wav",1024,256)
    together = np.concatenate((music,speech),axis = 1)
    norm = normalize_zscore(together)
    split = np.split(norm,[music.shape[1]],axis = 1)
    newmus = split[0]
    newspeech = split[1]
    fig,sub = plt.subplots(2,3)
    sub[0,0].plot(newmus[0,:],newmus[6,:],"r.")
    sub[0,0].plot(newspeech[0,:],newspeech[6,:],"b.")
    sub[0,0].set_title("Centroid vs. Crest")
    sub[0,1].plot(newmus[8,:],newmus[4,:],"r.")
    sub[0,1].plot(newspeech[8,:],newspeech[4,:],"b.")
    sub[0,1].set_title("Flux vs. ZCR")
    sub[0,2].plot(newmus[2,:],newmus[3,:],"r.")
    sub[0,2].plot(newspeech[2,:],newspeech[3,:],".b")
    sub[0,2].set_title("RMS Mean vs. STD")
    sub[1,0].plot(newmus[5,:],newmus[7,:],"r.")
    sub[1,0].plot(newspeech[5,:],newspeech[7,:],"b.")
    sub[1,0].set_title("ZCR vs. Crest")
    sub[1,1].plot(newmus[1,:],newmus[9,:],"r.")
    sub[1,1].plot(newspeech[1,:],newspeech[9,:],"b.")
    sub[1,1].set_title("Centroid vs. Flux")
    fig.set_tight_layout(True)
    fig.tight_layout(pad = 1000)
    fig.delaxes(sub[1,2])
    plt.show()
    


def get_z_score_scaler(features):
    scaler = StandardScaler()
    scaler.fit(features)

    return scaler

def get_min_max_scaler(features):
    scaler = MinMaxScaler()
    scaler.fit(features)

    return scaler

def normalize_features(features, scaler):
    return scaler.transform(features)





