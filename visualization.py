import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
This script shows various visualizations of the .wav file data. The visualization types are:
  1) Time Domain: a depiction of each of the classifications as time v. amplitude
  2) Frequency Domain: a spectra of the Fourier Transform of the .wav file samples
  3) Spectrogram: a colored graph that shows the Frequency Domain over Short Time Fourier Transforms
'''

TRAIN_PATH = './train/train/'
CLASSIFICATIONS = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

def load_wave(index, path=TRAIN_PATH):
    '''
    Description: loads a specific .wav file from the path. Example load_wave(5) loads 5.wav
    Args:
        index: number of which .wav file to load
        path: parent path (./train/train/)
    Returns:
        samples: a np.array of the .wav file sample data
        sr: the sample rate of the recording
    '''
    filename = path+str(index)+'.wav'
    return librosa.load(filename, sr=None)

def plot_fft(samples, sr, title='Frequency Domain'):
    '''
    Plots the FFT Frequency Domain of a given audio sample
    Args:
        samples: np.array of .wav file samples
        sr: sample rate
        title: the desired title for the graph
    '''
    n = len(samples) # Number of samples
    fft = np.fft.fft(samples) # Calculate the FFT of the samples
    fft = fft[:n//2] # Only take the real values of the frequency spectra
    fft = fft / n # Normalize the data
    fft = np.absolute(fft) # Find each frequency's magnitude
    f = np.linspace(0,sr//2,len(fft)) # Determine the frequency range vector
    plt.title(title)
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency')
    plt.plot(f,fft)

def plot_wave(samples, sr, title='Time Domain'):
    '''
    Plots the Time Domain of a given audio sample
    Args:
        samples: np.array of .wav file samples
        sr: sample rate
        title: the desired title for the graph
    '''
    librosa.display.waveplot(samples, sr=sr)
    plt.title(title)
    plt.ylim(-1,1)
    plt.ylabel('Amplitude')

def plot_spectrogram(samples, sr, title='Spectrogram'):
    '''
    Plots a spectrogram for a given audio sample
    Args:
        samples: np.array of .wav file samples
        sr: sample rate
        title: the desired title for the graph
    '''
    stft = np.absolute(librosa.stft(samples)) # Get the magnitude of the Short Time Fourier Transform
    db = librosa.amplitude_to_db(stft, ref=np.max) # Convert the amplitudes to decibels
    librosa.display.specshow(db, y_axis='linear') # Create a spectrogram with linear frequency axis
    plt.colorbar(format='%+2.0f dB') # Configure dB color bar
    plt.xlabel('Time')
    plt.title(title)

def show_all_fft(data, random=False):
    '''
    Shows an example Frequency Domain plot for all possible classifications
    Args:
        data: pd.dataframe of the data .csv file (ID, Class)
        random: whether or not the charts should be chosen randomly from the data
    '''
    classifications = sample_of_each(data, random) # Get a sample for each classification
    plt.figure(figsize=(15,15))
    plt.subplots_adjust(hspace=.75)
    i = 1
    for classification, audio in sorted(classifications.items()):
        plt.subplot(5,2,i)
        plot_fft(audio[0], audio[1], classification)
        i += 1
    plt.suptitle('Frequency Domain', x=0.5, y=0.95, fontsize=18)
    plt.show()

def show_all_waves(data, random=False):
    '''
    Shows an example Time Domain plot for all possible classifications
    Args:
        data: pd.dataframe of the data .csv file (ID, Class)
        random: whether or not the charts should be chosen randomly from the data
    '''
    classifications = sample_of_each(data, random)
    plt.figure(figsize=(15,15))
    plt.subplots_adjust(hspace=.75)
    i = 1
    for classification, audio in sorted(classifications.items()):
        plt.subplot(5,2,i)
        plot_wave(audio[0], audio[1], classification)
        plt.xlim(0,4) # Set xlim to (0,4) so that all graphs have the same time compression
        i += 1
    plt.suptitle('Time Domain', x=0.5, y=0.95, fontsize=18)
    plt.show()

def show_all_spectrograms(data, random=False):
    '''
    Shows an example spectrogram for all possible classifications
    Args:
        data: pd.dataframe of the data .csv file (ID, Class)
        random: whether or not the charts should be chosen randomly from the data
    '''
    classifications = sample_of_each(data, random)
    plt.figure(figsize=(15,15))
    plt.subplots_adjust(hspace=.75)
    i = 1
    for classification, audio in sorted(classifications.items()):
        plt.subplot(5,2,i)
        plot_spectrogram(audio[0], audio[1], classification)
        i += 1
    plt.suptitle('Spectrogram', x=0.5, y=0.95, fontsize=18)
    plt.show()

def sample_of_each(data, random=False):
    '''
    Loads an example of each audio file classification
    Args:
        data: pd.dataframe of the data .csv file (ID, Class)
        random: whether or not the charts should be chosen randomly from the data
    Returns: dictionary <str: classification, tuple<np.array: sample data, float: sample_rate>>
        which is a dictionary of data for each possible classification
    '''
    classifications = {}
    indexes = np.random.permutation(data.index) if random else range(len(data))
    for i in indexes:
        if data['Class'][i] not in classifications:
            classifications[data['Class'][i]] = (load_wave(data['ID'][i]))
        if len(classifications) == len(CLASSIFICATIONS): break
    return classifications

if __name__ == '__main__':
    data = pd.read_csv('./train/train.csv')
    show_all_waves(data)
    show_all_fft(data)
    show_all_spectrograms(data)
