import librosa
import librosa.display
import numpy as np
import pandas as pd
import glob, os
import matplotlib.pyplot as plt

TRAIN_PATH = './train/train/'
TRAIN_INDEX = './train/train.csv'
TRAIN_IMG = './train_img/'
VALIDATION_IMG = './validation_img/'

def load_data(split=.8):
    '''
    Description: loads the validation and training features and labels
    Args:
        split: the ratio at which to split the data for testing and traing
            for example, a split of .8 means that 80% of the data will be for training
            and 20% for validation
    Returns:
        train_x: train features
        train_y: train labels
        validation_x: validation features
        validation_y: validation labels
    '''
    # Read the index file and split the data into a test set and a train set
    data = pd.read_csv(TRAIN_INDEX)
    train_len = int(len(data)*split) # The ammount of training data is the split ratio
    data = data.reindex(np.random.permutation(data.index)) # Randomly rearrange the data
    train_data = data[:train_len]
    validation_data = data[train_len:]

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

def save_spectrogram(num):
    '''
    Description: saves spectrogram image of the given file index
    Args:
        num: id of which file to load (ex: if 2 is passed then 2.wav is loaded)
    TODO: right now the pixel settings aren't working right: 384x128 is actually saved as 297x98
    '''
    # Variable initialization
    dpi = 128 # Figure pixel density
    x_pixels = 384 # Image width in pixels
    y_pixels = 128 # Image height in pixels

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
    plt.savefig(TRAIN_IMG+str(num)+'.jpg', dpi=dpi, bbox_inches='tight',pad_inches=0)

def init_directory(dir):
    '''
    Description: clears the given CNN image directory if it exists or creates a new directory otherwise
    Args:
        dir: path to the directory (ex: './train_img/')
    '''
    if os.path.isdir(dir):
        files = glob.glob(dir+'*')
        for file in files:
            os.remove(file)
    else:
        os.mkdir(dir)

if __name__ == '__main__':
    init_directory(TRAIN_IMG)
    init_directory(VALIDATION_IMG)
    save_spectrogram(22)
