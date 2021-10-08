import librosa
import os
import h5py
import scipy
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from os.path import join
import random
# from tensorflow import keras

from tensorflow.keras import utils

 

FS = 16000
FFT_SIZE = 512
HOP_LENGTH=256
WIN_LENGTH=512
n_mels = 80

# dir
DATA_DIR = '../ESD/en/'
BIN_DIR = '../StrengthNet/training_data_en/'

# Strength score for english subset (0011-0020) of ESD data.
list_file = 'Score_List.csv'

emo_label = ['Angry', 'Happy', 'Surprise', 'Sad']
 

def get_melspectrograms(sound_file, fs=FS, fft_size=FFT_SIZE): 
    # Loading sound file
    y, _ = librosa.load(sound_file, sr=fs) # or set sr to hp.sr.
    linear = librosa.stft(y=y,
                     n_fft=fft_size, 
                     hop_length=HOP_LENGTH, 
                     win_length=WIN_LENGTH,
                     window=scipy.signal.hamming,
                     )
    mag = np.abs(linear) #(1+n_fft/2, T)

    # TODO add mel spectrum
    mel_basis = librosa.filters.mel(fs, fft_size, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)
    # shape in (T, 1+n_fft/2)
    return np.transpose(mel.astype(np.float32))  


def read_list(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path

def read(file_path):
    
    data_file = h5py.File(file_path, 'r')
    mel_sgram = np.array(data_file['mel_sgram'][:])
    
    timestep = mel_sgram.shape[0]
    mel_sgram = np.reshape(mel_sgram,(1, timestep, n_mels))
    
    return {
        'mel_sgram': mel_sgram,
    }   

def pad(array, reference_shape):
    
    result = np.zeros(reference_shape)
    result[:array.shape[0],:array.shape[1],:array.shape[2]] = array

    return result

def data_generator(file_list, bin_root, frame=False, batch_size=1):
    index=0
    while True:
            
        filename = [file_list[index+x].split(',')[0] for x in range(batch_size)]
        
        for i in range(len(filename)):
            all_feat = read(join(bin_root,filename[i]+'.h5'))
            sgram = all_feat['mel_sgram']

            # the very first feat
            if i == 0:
                feat = sgram
                max_timestep = feat.shape[1]
            else:
                if sgram.shape[1] > feat.shape[1]:
                    # extend all feat in feat
                    ref_shape = [feat.shape[0], sgram.shape[1], feat.shape[2]]
                    feat = pad(feat, ref_shape)
                    feat = np.append(feat, sgram, axis=0)
                elif sgram.shape[1] < feat.shape[1]:
                    # extend sgram to feat.shape[1]
                    ref_shape = [sgram.shape[0], feat.shape[1], feat.shape[2]]
                    sgram = pad(sgram, ref_shape)
                    feat = np.append(feat, sgram, axis=0)
                else:
                    # same timestep, append all
                    feat = np.append(feat, sgram, axis=0)
        
        strength = [float(file_list[x+index].split(',')[1]) for x in range(batch_size)]
        strength=np.asarray(strength).reshape([batch_size])
        frame_strength = np.array([strength[i]*np.ones([feat.shape[1],1]) for i in range(batch_size)])
        # add Multi-task
        emo_class = [emo_label.index(str(file_list[x+index].split(',')[0].split('/')[1])) for x in range(batch_size)]   
        emo_target = utils.to_categorical(emo_class, num_classes=4) # one-hot encoding
        index += batch_size  
        if index+batch_size >= len(file_list):
            index = 0
            random.shuffle(file_list)
        
        if frame:
            yield feat, [strength, frame_strength, emo_target]   
        else:
            yield feat, [strength, emo_target]  
            
            
def extract_to_h5():
    audio_dir = DATA_DIR
    output_dir = BIN_DIR
    
    print('audio dir: {}'.format(audio_dir))
    print('output_dir: {}'.format(output_dir))
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
  
    
    files = []
    with open(list_file, 'r') as f:
        for line in f:
            files.append(line.split(',')[0])
            out_dir1 = output_dir + '/' + line.split(',')[0].split('/')[0]
            if not os.path.exists(out_dir1):
                os.makedirs(out_dir1)

            out_dir2 = output_dir + '/' + line.split(',')[0].split('/')[0] + '/' + line.split(',')[0].split('/')[1]
            if not os.path.exists(out_dir2):
                os.makedirs(out_dir2)
    
    print('start extracting .wav to .h5, {} files found...'.format(len(files)))
            
    for i in tqdm(range(len(files))):
        f = files[i]
        
        # set audio file path
        audio_file = join(audio_dir, f)
       
        # Mel-spectrogram
        mel = get_melspectrograms(audio_file)
        

        with h5py.File(join(output_dir, '{}.h5'.format(f)), 'w') as hf:
            hf.create_dataset('mel_sgram', data=mel)

            
if __name__ == '__main__':

    extract_to_h5()