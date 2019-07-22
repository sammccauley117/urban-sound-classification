import glob, os, time, uuid, argparse
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix

# Set up and parse command line arguments
parser = argparse.ArgumentParser(description='Create a video of a .wav file\'s audio spectrum')
parser.add_argument('-L', '--loadmodel', type=str, default='', help='Name of saved .h5 model to load instead of training a new model')
parser.add_argument('-f', '--filename', type=str, default='', help='Filename of the accuracy plot and saved model')
parser.add_argument('-s', '--split', type=float, default=.8, help='Train : Validation split ratio (default: .8)')
parser.add_argument('-b', '--batchsize', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('-k', '--kernel', nargs='+', type=int, default=[6,6], help='Kernel window size (defualt: (6,6))')
parser.add_argument('-p', '--pool', nargs='+', type=int, default=[2,2], help='Max pooling size (defualt: (2,2))')
parser.add_argument('-d', '--dropout', type=float, default=.5, help='Dropout threshold (default: .5)')
parser.add_argument('-l', '--learningrate', type=float, default=.001, help='Learning rate (default: .001)')
parser.add_argument('-e', '--epochs', type=int, default=512, help='Number of epochs (default: 512)')
parser.add_argument('-N', '--noise', type=float, default=0, help='How much noise to add to the training data (default: 0)')
parser.add_argument('-P', '--pitch', type=float, default=0, help='Amplitude of pitch shifting applied to the data (default: 0)')
parser.add_argument('--load', default=True)
parser.add_argument('--normalize', default=True)
parser.add_argument('--color', default=True)
parser.add_argument('--no-load', dest='load', action='store_false', help='Prevents the image directories from being overwritten')
parser.add_argument('--no-normalize', dest='normalize', action='store_false', help='Prevents audio clips from being normalized to four seconds')
parser.add_argument('--no-color', dest='color', action='store_false', help='Forces the images to be saved as grayscale')
args = parser.parse_args()

# Show all arguments
for key, value in vars(args).items():
    print(key, '=', value)
print()

# Global variables
TRAIN_PATH = './train/train/'
TRAIN_INDEX = './train/train.csv'
TRAIN_IMG = './train_img/'
VALIDATION_IMG = './validation_img/'
CLASSIFICATIONS = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
NORMALIZE_LEN = 4

def load_data():
    '''
    Description: loads the validation and training data into a their respective folders.
        For example, the './train_img/' root directory will have subdirectories for each possible classification:
        - ./train_img/air_conditioner/
        - ./train_img/car_horn/
        - ./train_img/children_playing/
        - etc.
    '''
    # Read the index file and split the data into a test set and a train set
    data = pd.read_csv(TRAIN_INDEX)
    data = data.reindex(np.random.permutation(data.index)) # Randomly rearrange the data
    validation_len = len(data)*(1-args.split) # The ammount of validation data
    train_len = int(len(data)*args.split) # The ammount of training data is the split ratio
    train_data = data[:train_len]
    validation_data = data[train_len:]
    if args.noise and args.pitch: train_len *= 3
    elif args.noise or args.pitch: train_len *= 2

    # Save the images to their directories
    init_directory(TRAIN_IMG)
    init_directory(VALIDATION_IMG)
    m = 0 # Used for progress updates
    for i, row in train_data.iterrows():
        samples, sr = load_wave(row['ID'])
        save_spectrogram(samples=samples, sr=sr, num=row['ID'], classification=row['Class'], dir=TRAIN_IMG)
        m += 1
        if m % 10 == 0: print('Train Progress:', m, '/', train_len)
        if args.noise:
            noisy = np.random.normal(0, args.noise, len(samples)) + samples
            save_spectrogram(samples=noisy, sr=sr, classification=row['Class'], dir=TRAIN_IMG)
            m += 1
            if m % 10 == 0: print('Train Progress:', m, '/', train_len)
        if args.pitch:
            steps = int(np.random.random_sample() * args.pitch) + 1
            if int(np.random.random_sample() * 2): steps = -steps # Randomly determine shift up vs. shift down
            shifted = librosa.effects.pitch_shift(samples, sr, n_steps=steps)
            save_spectrogram(samples=shifted, sr=sr, classification=row['Class'], dir=TRAIN_IMG)
            m += 1
            if m % 10 == 0: print('Train Progress:', m, '/', train_len)
    m = 0 # Used for progress updates
    for i, row in validation_data.iterrows():
        samples, sr = load_wave(row['ID'])
        save_spectrogram(samples=samples, sr=sr, num=row['ID'], classification=row['Class'], dir=VALIDATION_IMG)
        m += 1
        if m % 10 == 0: print('Validation Progress:', m, '/', int(len(data)*(1-args.split)))

def load_wave(num, path=TRAIN_PATH):
    '''
    Description: loads a specific .wav file from the path. Example load_wave(5) loads 5.wav
    Args:
        index: number of which .wav file to load
        path: root path (ex: ./train/train/)
    Returns:
        samples: a np.array of the .wav file sample data
        sr: the sample rate of the recording
    '''
    # Load the file's samples and sample rate (sr)
    filename = path+str(num)+'.wav'
    samples, sr = librosa.load(filename, sr=None)

    # Check to see if we need to normalize the duration of each clip to [NORMALIZE_LEN] seconds long
    if args.normalize:
        duration = len(samples) / sr
        if duration > NORMALIZE_LEN:
            return samples[:int(NORMALIZE_LEN*sr)], sr
        elif duration == NORMALIZE_LEN:
            return samples, sr
        else:
            zeros = np.zeros((NORMALIZE_LEN*sr) - len(samples))
            return np.hstack((samples, zeros)), sr
    return samples, sr

def save_spectrogram(num=None, samples=None, sr=None, classification=None, dir='./'):
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
    filename = str(num) if (num is not None) else str(uuid.uuid4()) # Uses the file number that was passed if possible
    if classification:
        path = dir + classification + '/' + filename + '.jpg'
    else:
        path = dir + filename + '.jpg'

    # Calculate the Short Time Fourier Transform
    S = librosa.feature.melspectrogram(samples, sr)
    db = librosa.power_to_db(S, ref=np.max)

    # Configure the matplotlib figure
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    # Plot and save the spectrogram
    if args.color:
        librosa.display.specshow(db, y_axis='mel') # Create a spectrogram with mel frequency axis
    else:
        librosa.display.specshow(db, cmap='gray_r', y_axis='mel') # Create a spectrogram with mel frequency axis
    plt.savefig(path, bbox_inches='tight',pad_inches=0)
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

def build_generator(dir, batch_size):
    '''
    Description: creates data generators for training/validation
    Args:
        dir: which subdirectory to use (ex: ./train_img/ or ./validation_img/)
        batch_size: the batch size that the generator returns
    Returns: keras data generator
    '''
    color_mode = 'rgb' if args.color else 'grayscale'
    data_generator = keras.preprocessing.image.ImageDataGenerator()
    return data_generator.flow_from_directory(
        directory = dir,
        target_size = (77, 77),
        color_mode = color_mode,
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True,
    )

def build_model():
    '''
    Description: creates, configures, and compiles a convolutional neural network
        with tweakable hyperparameters
    Returns: compiled keras model
    '''
    shape_depth = 3 if args.color else 1
    kernel_size = tuple(args.kernel)
    pool_size = tuple(args.pool)
    dropout = args.dropout
    lr = args.learningrate

    # Set up Convolutional Neural Network:
    model = keras.Sequential()
    model.add(Conv2D(16, kernel_size=kernel_size, activation='relu', input_shape=(77,77,shape_depth)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(.2))
    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    optimizer = SGD(lr=lr)
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
    plt.ylim((0,1))
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Display settings
    if save: plt.savefig(filename)
    if show: plt.show()

def plot_matrix(labels, predicted):
    # Calculate confusion matrix
    matrix = confusion_matrix(labels, predicted)

    # Set up plot, add matrix data, and configure colormap
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.viridis)
    ax.figure.colorbar(im, ax=ax)

    # Configure plot axises
    classes = [c.replace('_', ' ') for c in CLASSIFICATIONS]
    ax.set(xticks=np.arange(matrix.shape[1]),
       yticks=np.arange(matrix.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=classes,
       title='Urban Sound Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

    # Rotate the x-axis labels so they don't overlap
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
        rotation_mode='anchor')

    # Add the numbers coresponding number to each square in the matrix
    threshold = matrix.max() / 2 # Threshold for white vs. black text due to colormap
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], 'd'),
                ha='center', va='center',
                color='white' if matrix[i, j] < threshold else 'black')

    # Display
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Either load or train a model
    if args.loadmodel:
        # Load the model
        if args.loadmodel.endswith('.h5'):
            model = load_model(args.loadmodel)
        else:
            model = load_model(args.loadmodel+'.h5')

        # Still need to create a validation generator for the confusion matrix
        validation_generator = build_generator(VALIDATION_IMG, args.batchsize)
    else:
        # Load data to their respective image directories
        if args.load:
            start = time.time()
            load_data()
            end = time.time()
            print('Data Collection Time:', end - start)

        # Use the test and validation image directories to set up data generators for training and validation.
        train_generator = build_generator(TRAIN_IMG, args.batchsize)
        validation_generator = build_generator(VALIDATION_IMG, args.batchsize)

        # Congigure and compile a model
        model = build_model()

        # Train model
        start = time.time()
        history = model.fit_generator(
            generator = train_generator,
            steps_per_epoch = train_generator.n // train_generator.batch_size,
            validation_data = validation_generator,
            validation_steps = validation_generator.n // validation_generator.batch_size,
            epochs = args.epochs
        )
        end = time.time()
        print('Training Time:', end - start)

        # Save model and show graph of model accuracy vs. epoch
        if args.filename:
            model.save(args.filename+'.h5')
            plot_model(history, save=True, filename=args.filename+'.jpg')
        else:
            plot_model(history)

    # Show confusion matrix of model
    validation_generator.shuffle = False
    predicted = np.argmax(model.predict_generator(validation_generator), axis=1)
    plot_matrix(validation_generator.classes, predicted)
