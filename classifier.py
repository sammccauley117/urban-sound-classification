import librosa
import librosa.display
import numpy as np
import pandas as pd
import glob, os, time, uuid
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD

# CNN configuration
KERNEL_SIZE = (6,6)
POOL_SIZE = (2,2)
DROPOUT = .2
LEARNING_RATE = .001
EPOCHS = 256
NOISE = .01
PITCH = 5

TRAIN_PATH = './train/train/'
TRAIN_INDEX = './train/train.csv'
TRAIN_IMG = './train_img/'
VALIDATION_IMG = './validation_img/'
CLASSIFICATIONS = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

def load_data(split=.8, normalize=4, noise=True, pitch_shift=True):
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
    data = data.reindex(np.random.permutation(data.index)) # Randomly rearrange the data
    validation_len = len(data)*(1-split) # The ammount of validation data
    train_len = int(len(data)*split) # The ammount of training data is the split ratio
    train_data = data[:train_len]
    validation_data = data[train_len:]
    if noise and pitch_shift: train_len *= 3
    elif noise or pitch_shift: train_len *= 2 

    # Save the images to their directories
    init_directory(TRAIN_IMG)
    init_directory(VALIDATION_IMG)
    m = 0 # Used for progress updates
    for i, row in train_data.iterrows():
        samples, sr = load_wave(row['ID'])
        save_spectrogram(samples=samples, sr=sr, num=row['ID'], classification=row['Class'], dir=TRAIN_IMG)
        m += 1
        if m % 10 == 0: print('Train Progress:', m, '/', train_len)
        if noise:
            noisy = np.random.normal(0, NOISE, len(samples)) + samples
            save_spectrogram(samples=noisy, sr=sr, classification=row['Class'], dir=TRAIN_IMG)
            m += 1
            if m % 10 == 0: print('Train Progress:', m, '/', train_len)
        if pitch_shift:
            steps = int(np.random.random_sample() * PITCH) + 1
            if int(np.random.random_sample() * 2): steps = -steps # Randomly determine shift up vs. shift down
            pitch_shift = librosa.effects.pitch_shift(samples, sr, n_steps=steps)
            save_spectrogram(samples=pitch_shift, sr=sr, classification=row['Class'], dir=TRAIN_IMG)
            m += 1
            if m % 10 == 0: print('Train Progress:', m, '/', train_len)
        print(row['Class'])
        while 1:
            x = 1
    m = 0 # Used for progress updates
    for i, row in validation_data.iterrows():
        save_spectrogram(num=row['ID'], classification=row['Class'], dir=VALIDATION_IMG)
        m += 1
        if m % 10 == 0: print('Validation Progress:', m, '/', int(len(data)*(1-split)))

def load_wave(num, normalize=4, path=TRAIN_PATH):
    '''
    Description: loads a specific .wav file from the path. Example load_wave(5) loads 5.wav
    Args:
        index: number of which .wav file to load
        normalize: length in time to normalize the sample to--does not normalize if None
        path: parent path (./train/train/)
    Returns:
        samples: a np.array of the .wav file sample data
        sr: the sample rate of the recording
    '''
    # Load the file's samples and sample rate (sr)
    filename = path+str(num)+'.wav'
    samples, sr = librosa.load(filename, sr=None)

    # Check to see if we need to normalize the duration. If so, keep doubling the
    # sample until it surpasses the proper duration and then cut off the excess samples
    if normalize:
        while(len(samples)/sr < normalize):
            samples = np.append(samples,samples) # Double samples
        samples = samples[:sr*normalize] # Cut off excess samples

    return samples, sr

def save_spectrogram(num=None, samples=None, sr=None, classification=None, dir='./', normalize=4):
    '''
    Description: saves spectrogram image of the given file index to the correct classification subfolder
    Args:
        num: id of which file to load (ex: if 2 is passed then 2.wav is loaded)
        classification: the label of the sound - this determines which subdirectory the spectrogram image goes to
        dir: which directory to save the image to (ex: './train_img/')
        normalize: length in time to normalize the sample to--does not normalize if None
    TODO: right now the pixel settings aren't working right: 384x128 is actually saved as 297x98
    '''
    # Variable initialization
    dpi = 128 # Figure pixel density
    x_pixels = dpi*3 # Image width in pixels
    y_pixels = dpi # Image height in pixels
    filename = str(num) if (num is not None) else str(uuid.uuid4()) # Uses the file number that was passed if possible
    if classification:
        path = dir + classification + '/' + filename + '.jpg'
    else:
        path = dir + filename + '.jpg'

    # Load the wave file from file number if the number was passed
    if num is not None: samples, sr = load_wave(num, normalize)

    # Calculate the Short Time Fourier Transform
    S = librosa.feature.melspectrogram(samples, sr)
    db = librosa.power_to_db(S, ref=np.max)

    # Configure the matplotlib figure
    fig = plt.figure(figsize=(x_pixels//dpi, y_pixels//dpi))
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    # Plot and save the spectrogram
    librosa.display.specshow(db, cmap='gray_r', y_axis='mel') # Create a spectrogram with mel frequency axis
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

def build_generator(dir, batch_size, color_mode='grayscale'):
    '''
    Description: creates data generators for training/validation
    Args:
        dir: which subdirectory to use (ex: ./train_img/ or ./validation_img/)
        batch_size: the batch size that the generator returns
        color_mode: either 'grayscale' or 'rgb'
    Returns: keras data generator
    '''
    data_generator = keras.preprocessing.image.ImageDataGenerator()
    return data_generator.flow_from_directory(
        directory = dir,
        target_size = (297, 98),
        color_mode = color_mode,
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True,
    )

def build_model(kernel_size, pool_size, dropout, learning_rate):
    '''
    Description: creates, configures, and compiles a convolutional neural network
        with tweakable hyperparameters
    Args:
        kernel_size: the kernel size for each convolutional layer in the form (kernel_size, kernel_size)
        pool_size: the max pooling window size in the form (pool_size, pool_size)
        dropout: threshold for the dropout regularization
        learning_rate: the learning rate for the SGD optimizer
    Returns: compiled keras model
    '''

    # Set up Convolutional Neural Network:
    model = keras.Sequential()
    model.add(Conv2D(16, kernel_size=KERNEL_SIZE, activation='relu', input_shape=(297,98,1)))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(DROPOUT))
    model.add(Conv2D(32, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(DROPOUT))
    model.add(Conv2D(64, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(DROPOUT))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_model(history, show=True, save=False, filename='accuracy.jpg'):
    '''
    Description: shows and/or saves a graph of the training and validation accuracy v. epoch
    Args:
        history: history object returned from the keras model fit function
        show: whether or not to show the image in a window
        save: whether or not to save the image
        filename: what to call the saved file
    '''

    # Plot and configure graph
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Display settings
    if save: plt.savefig(filename)
    if show: plt.show()

if __name__ == '__main__':
    # Load data to their respective image directories
    start = time.time()
    load_data()
    end = time.time()
    print('Data Collection Time:', end - start)

    # Use the test and validation image directories to set up data generators for
    # training and validation.
    train_generator = build_generator(TRAIN_IMG, 128)
    validation_generator = build_generator(VALIDATION_IMG, 128)

    # Congigure and compile a model
    model = build_model(KERNEL_SIZE, POOL_SIZE, DROPOUT, LEARNING_RATE)

    # Train model
    start = time.time()
    history = model.fit_generator(
        generator = train_generator,
        steps_per_epoch = train_generator.n // train_generator.batch_size,
        validation_data = validation_generator,
        validation_steps = validation_generator.n // validation_generator.batch_size,
        epochs = EPOCHS
    )
    end = time.time()
    print('Training Time:', end - start)

    # Show graph of model accuracy vs. epoch
    plot_model(history)
