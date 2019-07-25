import os, sys, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import logging
import numpy as np
logging.getLogger('tensorflow').disabled = True

# Set up and parse command line arguments
parser = argparse.ArgumentParser(description='Predict a sound using a convolutional model and an image')
parser.add_argument('model', type=str, help='Path of saved .h5 convolutional nerual network model')
parser.add_argument('img', type=str, help='Path of image to predict')
args = parser.parse_args()

CLASSIFICATIONS = ['Air conditioner', 'Car horn', 'Children playing', 'Dog bark',
    'Drilling', 'Engine idling', 'Gun shot', 'Jackhammer', 'Siren', 'Street music']

if __name__ == '__main__':
    # Get image and prepare it to be an input to our model
    img_path = args.img if args.img.endswith('.jpg') else args.img+'.jpg'
    img = image.load_img(img_path)
    img_tensor = image.img_to_array(img) #  Tensor of shape (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0) # Shape to (1, height, width, channels) where 1 is the batch size

    # Get model
    model_path = args.model if args.model.endswith('.h5') else args.model+'.h5'
    model = keras.models.load_model(model_path)
    # 'exp/default.h5'

    # Make prediction
    prediction = model.predict(img_tensor)
    prediction = np.argmax(prediction) # Pick most probable classification
    prediction = CLASSIFICATIONS[prediction] # Decode classification from number to name
    print(prediction)
