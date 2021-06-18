import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import wave
import pylab
from pathlib2 import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools

print(tf.__version__)
# Set paths to input and output data
INPUT_DIR = '/home/sharad/Desktop/free-spoken-digit-dataset-master/recordings/'
OUTPUT_DIR = '/home/sharad/Desktop/free-spoken-digit-dataset-master/out'

# Print names of 10 WAV files from the input path
parent_list = os.listdir(INPUT_DIR)
for i in range(10):
    print(parent_list[i])
    

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

print(tf.__version__)    
if not os.path.exists(os.path.join(OUTPUT_DIR, 'audio-images')):
    os.mkdir(os.path.join(OUTPUT_DIR, 'audio-images'))
print(tf.__version__)    
for filename in os.listdir(INPUT_DIR):
    if "wav" in filename:
        print("1")
        file_path = os.path.join(INPUT_DIR, filename)
        print("2")
        file_stem = Path(file_path).stem
        print("3")
        target_dir = 'class_'+ file_stem[0]
        print("4")
        dist_dir = os.path.join(os.path.join(OUTPUT_DIR, 'audio-images'), target_dir)
        print("5")
        file_dist_path = os.path.join(dist_dir, file_stem)
        print("6")
        if not os.path.exists(file_dist_path + '.png'):
            print("7")
            if not os.path.exists(dist_dir):
                print("8")
                os.mkdir(dist_dir)
                print("9")
            file_stem = Path(file_path).stem
            print("10")
            sound_info, frame_rate = get_wav_info(file_path)
            print("11")
            pylab.specgram(sound_info, Fs=frame_rate)
            print("12")
            pylab.savefig(file_dist_path +'.png')
            print("13")
            pylab.close()
            print("14")
print(tf.__version__)
# Print the ten classes in our dataset
path_list = os.listdir(os.path.join(OUTPUT_DIR, 'audio-images'))
print("Classes: \n")
for i in range(10):
    print(path_list[i])
    
# File names for class 1
path_list = os.listdir(os.path.join(OUTPUT_DIR, 'audio-images/class_1'))
print("\nA few example files: \n")
for i in range(10):
    print(path_list[i])    
print(tf.__version__)
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 10
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=os.path.join(OUTPUT_DIR, 'audio-images'),
                                             shuffle=True,
                                             color_mode='rgb',
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="training",
                                             seed=0)

# Make a dataset containing the validation spectrogram
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=os.path.join(OUTPUT_DIR, 'audio-images'),
                                             shuffle=True,
                                             color_mode='rgb',
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="validation",
                                             seed=0)
print(tf.__version__)   
def prepare(ds, augment=False):
    # Define our one transformation
    rescale = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])
    flip_and_rotate = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])
    
    # Apply rescale to both datasets and augmentation only to training
    ds = ds.map(lambda x, y: (rescale(x, training=True), y))
    if augment: ds = ds.map(lambda x, y: (flip_and_rotate(x, training=True), y))
    return ds

train_dataset = prepare(train_dataset, augment=False)
valid_dataset = prepare(valid_dataset, augment=False) 
print(type(train_dataset))

iterator = train_dataset.__iter__()
print(type(iterator))
lines = []
ne = iterator.get_next()
iterations = 0
while True:
    try:
        iterations += 1
        pr = ne[0]
        en = ne[1]
#         print(pr.numpy()[1].shape)
        print(pr.numpy().shape)
        for i in range(pr.numpy().shape[0]):
            arr = pr.numpy()[i]
            arr = arr.flatten().tolist()
            arr = arr + [en.numpy()[i].tolist()]
            lines.append(",".join(map(str, arr)))
        print(en.numpy())
        ne = iterator.get_next()
    except Exception as e:
        print(e)
        break
print(iterations)
print(lines[0])                                        