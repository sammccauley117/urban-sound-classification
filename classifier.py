import librosa
import librosa.display
import numpy as np
import pandas as pd
import glob, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

TRAIN_PATH = './train/train/'
TRAIN_INDEX = './train/train.csv'
TRAIN_IMG = './train_img/'
VALIDATION_IMG = './validation_img/'
CLASSIFICATIONS = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

def load_data(split=.8):
    '''
    Description: loads the validation and training data into a their respective folders.
        For example, the './train_img/' root directory will have subdirectories for each possible classification:
        - ./train_img/air_conditioner/
        - ./train_img/car_horn/
        - ./train_img/children_playing/
        - etc.
    Args:
        split: the ratio at which to split the data for testing and traing
            for example, a split of .8 means that 80% of the data will be for training
            and 20% for validation
    '''
    # Read the index file and split the data into a test set and a train set
    data = pd.read_csv(TRAIN_INDEX)
    train_len = int(len(data)*split) # The ammount of training data is the split ratio
    data = data.reindex(np.random.permutation(data.index)) # Randomly rearrange the data
    train_data = data[:train_len]
    validation_data = data[train_len:]

    # Save the images to their directories
    init_directory(TRAIN_IMG)
    init_directory(VALIDATION_IMG)
    # m = 0
    for i, row in train_data.iterrows():
        save_spectrogram(row['ID'], row['Class'], TRAIN_IMG)
        # m += 1
        # if m == 20: break
    # m = 0
    for i, row in validation_data.iterrows():
        save_spectrogram(row['ID'], row['Class'], VALIDATION_IMG)
        # m += 1
        # if m == 20: break

def load_wave(num, path=TRAIN_PATH):
    '''
    Description: loads a specific .wav file from the path. Example load_wave(5) loads 5.wav
    Args:
        index: number of which .wav file to load
        path: parent path (./train/train/)
    Returns:
        samples: a np.array of the .wav file sample data
        sr: the sample rate of the recording
    '''
    filename = path+str(num)+'.wav'
    return librosa.load(filename, sr=None)

def save_spectrogram(num, classification, dir):
    '''
    Description: saves spectrogram image of the given file index to the correct classification subfolder
    Args:
        num: id of which file to load (ex: if 2 is passed then 2.wav is loaded)
        classification: the label of the sound - this determines which subdirectory the spectrogram image goes to
        dir: which directory to save the image to (ex: './train_img/')
    TODO: right now the pixel settings aren't working right: 384x128 is actually saved as 297x98
    '''
    # Variable initialization
    dpi = 128 # Figure pixel density
    x_pixels = dpi*3 # Image width in pixels
    y_pixels = dpi # Image height in pixels

    # Load the wave file and calculate the Short Time Fourier Transform
    samples, sr = load_wave(num)
    stft = np.absolute(librosa.stft(samples)) # Get the magnitude of the Short Time Fourier Transform
    db = librosa.amplitude_to_db(stft, ref=np.max) # Convert the amplitudes to decibels

    # Configure the matplotlib figure
    fig = plt.figure(figsize=(x_pixels//dpi, y_pixels//dpi))
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    # Plot and save the spectrogram
    librosa.display.specshow(db, y_axis='linear') # Create a spectrogram with linear frequency axis
    path = dir + classification + '/' + str(num) + '.jpg'
    plt.savefig(path, dpi=dpi, bbox_inches='tight',pad_inches=0)
    plt.close(fig) # Need to close to prevent unecessary memory consumtion

def init_directory(dir):
    '''
    Description: initializes the image container directories and its classification subdirectories for training
    Args:
        dir: path to the directory (ex: './train_img/')
    '''
    # Make the root directory if necessary
    if not os.path.isdir(dir): os.mkdir(dir)

    # Check if each classification directory exists. If it does, delete all the images in the directory.
    # If the classification directory does not exist, then create one.
    for classification in CLASSIFICATIONS:
        if os.path.isdir(dir+classification):
            files = glob.glob(dir+classification+'/*')
            for file in files:
                os.remove(file)
        else:
            os.mkdir(dir+classification)

if __name__ == '__main__':
    load_data()
