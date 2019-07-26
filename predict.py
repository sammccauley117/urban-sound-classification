import os, sys, argparse, random, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import h5py
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import logging
import numpy as np
import winsound
import librosa
logging.getLogger('tensorflow').disabled = True

# Set up and parse command line arguments
parser = argparse.ArgumentParser(description='Predict a sound using a convolutional model and an image')
parser.add_argument('model', type=str, help='Path of saved .h5 convolutional nerual network model')
parser.add_argument('-i', '--img_dir', type=str, default='test_img/', help='Path to the directory containing spectrograms to predict (default: \'test_img/\')')
parser.add_argument('-w', '--wav_dir', type=str, default='test/test/', help='Path to the directory containing .wav files to play (default: \'test/test/\')')
parser.add_argument('-n', '--num', type=int, default=None, help='Specific number to play, otherwise random')
args = parser.parse_args()

CLASSIFICATIONS = ['Air conditioner', 'Car horn', 'Children playing', 'Dog bark',
    'Drilling', 'Engine idling', 'Gun shot', 'Jackhammer', 'Siren', 'Street music']

if __name__ == '__main__':
    # Determine actual path variables (decide whether or not extensions need to be added)
    model_path = args.model if args.model.endswith('.h5') else args.model+'.h5'
    img_path = args.img_dir if args.img_dir.endswith('/') else args.img_dir+'/'
    wav_path = args.wav_dir if args.wav_dir.endswith('/') else args.wav_dir+'/'

    # Load the image
    if args.num: # If specific number was passed
        path = img_path+str(args.num)+'.jpg'
        if os.path.exists(path):
            print('Image path:', path.replace('\\','/')) # Replace for consistency
            num = args.num
            img = image.load_img(path)
        else:
            print('Error: image \'{}\' does not exist'.format(path))
            sys.exit()
    else: # Load random
        files = glob.glob(img_path+'*.jpg')
        path = random.choice(files)
        print('Image path:', path.replace('\\','/')) # Replace for consistency
        num = int(path.split('\\')[-1].replace('.jpg',''))
        img = image.load_img(path)

    # Calculate the duration (in seconds) of the file
    path = wav_path+str(num)+'.wav'
    print('Audio path:',path)
    samples, sr = librosa.load(path, sr=None) # Get the .wav file samples and sample rate
    duration = len(samples) / sr

    # Play the sound
    winsound.PlaySound(path, winsound.SND_ASYNC)
    time.sleep(duration)

    # Get image and prepare it to be an input to our model
    img_tensor = image.img_to_array(img) #  Tensor of shape (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0) # Shape to (1, height, width, channels) where 1 is the batch size

    # Get model
    model = keras.models.load_model(model_path)

    # Make prediction
    prediction = model.predict(img_tensor)
    prediction = np.argmax(prediction) # Pick most probable classification
    prediction = CLASSIFICATIONS[prediction] # Decode classification from number to name
    print('Prediction:',prediction)
