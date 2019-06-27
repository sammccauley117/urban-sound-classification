import librosa
import librosa.display
import numpy as np
import pandas as pd
import glob, os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import time

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
    m = 0
    for i, row in train_data.iterrows():
        save_spectrogram(row['ID'], row['Class'], TRAIN_IMG)
        m += 1
        if m % 10 == 0: print('Train Progress:', m, '/', train_len)
    m = 0
    for i, row in validation_data.iterrows():
        save_spectrogram(row['ID'], row['Class'], VALIDATION_IMG)
        m += 1
        if m % 10 == 0: print('Validation Progress:', m, '/', int(len(data)*(1-split)))

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
    print(db)

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
    # Load data to their respective image directories
    load_data()

    # Set up data generator for the training data
    train_datagen = keras.preprocessing.image.ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        directory = TRAIN_IMG,
        target_size = (297, 98),
        color_mode = 'rgb',
        batch_size = 128,
        class_mode = 'categorical',
        shuffle = True,
    )

    # Set up data generator for the validation data
    validation_datagen = keras.preprocessing.image.ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(
        directory = VALIDATION_IMG,
        target_size = (297, 98),
        color_mode = 'rgb',
        batch_size = 64,
        class_mode = 'categorical',
        shuffle = True,
    )

    # Set up Convolutional Neural Network:
    model = keras.Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(297,98,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.1))
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.1))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.1))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])


    # Train model
    start = time.time()
    model.fit_generator(
        generator = train_generator,
        steps_per_epoch = train_generator.n // train_generator.batch_size,
        validation_data = validation_generator,
        validation_steps = validation_generator.n // validation_generator.batch_size,
        epochs = 10
    )
    end = time.time()
    print('Training Time:', end - start)
