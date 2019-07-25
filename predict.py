import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import logging
import numpy as np
logging.getLogger('tensorflow').disabled = True

CLASSIFICATIONS = ['Air conditioner', 'Car horn', 'Children playing', 'Dog bark',
    'Drilling', 'Engine idling', 'Gun shot', 'Jackhammer', 'Siren', 'Street music']

if __name__ == '__main__':
    # Get image and prepare it to be an input to our model
    img = image.load_img('validation_img_default/children_playing/44.jpg')
    img_tensor = image.img_to_array(img) #  Tensor of shape (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0) # Shape to (1, height, width, channels) where 1 is the batch size

    # Get model
    model = keras.models.load_model('exp/default.h5')

    # Make prediction
    prediction = model.predict(img_tensor)
    prediction = np.argmax(prediction) # Pick most probable classification
    prediction = CLASSIFICATIONS[prediction] # Decode classification from number to name
    print(prediction)
