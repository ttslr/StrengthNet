import os
import time 
import numpy as np
from tqdm import tqdm
import scipy.stats
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import argparse
import tensorflow as tf
from tensorflow import keras
import model
import utils
import random
 

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000, help="number epochs")
parser.add_argument("--batch_size", type=int, default=64, help="number batch_size")

args = parser.parse_args()

 

print('training with model architecture: {}'.format(args.model))   
print('epochs: {}\nbatch_size: {}'.format(args.epoch, args.batch_size))

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

tf.debugging.set_log_device_placement(False)
# set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

        
# set dir
OUTPUT_DIR = './output'

DATA_DIR = '../ESD/en/'
# AUDIO_DIR = join(DATA_DIR, 'wav')
BIN_DIR = '../StrengthNet/training_data_en/'
list_file = '../ESD/en/Score_List.csv'


EPOCHS = args.epoch
BATCH_SIZE = args.batch_size

 
NUM_TRAIN = 10000
NUM_TEST=1000
NUM_VALID=3000


emo_label = ['Angry', 'Happy', 'Surprise', 'Sad']



if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
            
strength_list = utils.read_list(list_file)
random.shuffle(strength_list)

train_list= strength_list[0:-(NUM_TEST+NUM_VALID)]
random.shuffle(train_list)
valid_list= strength_list[-(NUM_TEST+NUM_VALID):-NUM_TEST]
test_list= strength_list[-NUM_TEST:]

print('{} for training; {} for valid; {} for testing'.format(NUM_TRAIN, NUM_VALID, NUM_TEST))        

    

# init model
StrengthNet = model.CNN_BLSTM()
model = StrengthNet.build()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={'avg':'mae',
          'frame':'mae',
          'class': 'categorical_crossentropy'}  # TODO add SER loss
          )
    
CALLBACKS = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR,'strengthnet.h5'),
        save_best_only=True,
        monitor='val_loss',
        verbose=1),
    keras.callbacks.TensorBoard(
        log_dir=os.path.join(OUTPUT_DIR,'tensorboard.log'),
        update_freq='epoch'), 
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0,
        patience=30,
        verbose=1)
]

# data generator
train_data = utils.data_generator(train_list, BIN_DIR, frame=True, batch_size=BATCH_SIZE)
valid_data = utils.data_generator(valid_list, BIN_DIR, frame=True, batch_size=BATCH_SIZE)

tr_steps = int(NUM_TRAIN/BATCH_SIZE)
val_steps = int(NUM_VALID/BATCH_SIZE)


# start fitting model
hist = model.fit_generator(train_data,
                          steps_per_epoch=tr_steps,
                          epochs=EPOCHS,
                          callbacks=CALLBACKS,
                          validation_data=valid_data,
                          validation_steps=val_steps,
                          verbose=1,)
    

# plot testing result
model.load_weights(os.path.join(OUTPUT_DIR,'strengthnet.h5'),)   # Load the best model   

print('testing...')
Strength_Predict=np.zeros([len(test_list),])
Strength_true   =np.zeros([len(test_list),])
class_predict=np.zeros([len(test_list),])
class_true=np.zeros([len(test_list),])
df = pd.DataFrame(columns=['audio', 'true_strength', 'predict_strength', 'true_class', 'predict_class'])


for i in tqdm(range(len(test_list))):
    
    filepath=test_list[i].split(',')
    filename=filepath[0]
    
    _feat = utils.read(os.path.join(BIN_DIR,filename+'.h5'))
    _mel = _feat['mel_sgram']    
        
    strength=float(filepath[1])
    class_true[i] = emo_label.index(filename.split('/')[1])
    
    [Average_score, Frame_score, emo_class]=model.predict(_mel, verbose=0, batch_size=1)
    Strength_Predict[i]=Average_score
    Strength_true[i]   =strength
    class_predict[i] = np.argmax(emo_class,1)
    df = df.append({'audio': filepath[0], 
                    'true_strength': Strength_true[i], 
                    'predict_strength': Strength_Predict[i],
                    'true_class': class_true[i],
                    'predict_class': class_predict[i]}, 
                   ignore_index=True)
    
    

plt.style.use('seaborn-deep')
x = df['true_strength']
y = df['predict_strength']
bins = np.linspace(0, 1, 40)
plt.figure(2)
parameters = {'xtick.labelsize': 16,
          'ytick.labelsize': 16}
plt.rcParams.update(parameters)
plt.hist([x, y], bins, label=['true_strength', 'predict_strength'])
plt.legend(loc='upper left', prop = {'size':14})
plt.xlabel('Strength',fontsize=17)
plt.ylabel('Number',fontsize=17) 
plt.show()
plt.savefig('./output/StrengthNet_distribution.png', bbox_inches='tight', dpi=150)



SER_MSE=accuracy_score(class_predict,class_true)
print('[UTTERANCE] SER Test error= %f' % SER_MSE)
 
MAE=np.mean(np.abs(Strength_true-Strength_Predict))
print('[UTTERANCE] MAE= %f' % MAE)
