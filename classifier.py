import librosa
import librosa.display
import numpy as np
import pandas as pd

TRAIN_PATH = './train/train/'
TRAIN_INDEX = './train/train.csv'

def load_data(split=.8):
    '''
    Description: loads the testing and training features and labels
    Args:
        split: the ratio at which to split the data for testing and traing
            for example, a split of .8 means that 80% of the data will be for training
            and 20% for testing
    Returns:
        train_x: train features
        train_y: train labels
        test_x: test features
        test_y: test labels
    '''
    # Read the index file and split the data into a test set and a train set
    data = pd.read_csv(TRAIN_INDEX)
    train_len = int(len(data)*split) # The ammount of training data is the split ratio
    data = data.reindex(np.random.permutation(data.index)) # Randomly rearrange the data
    train_data = data[:train_len]
    test_data = data[train_len:]

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

if __name__ == '__main__':
    # train_x, train_y, test_x, test_y = load_data()
    load_data()
