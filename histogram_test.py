# All required imports
import cv2
import numpy as np
import pandas as pd

from model import kFolderName, kAddingFactor, add_curvy_images
from keras.models import model_from_json
from keras.utils.visualize_util import plot

def main():
    # Read the whole csv file containing the udacity input data
    input_data = pd.read_csv(kFolderName + '/driving_log.csv')

    print("Original size of input data: " + str(len(input_data)))
    input_data = add_curvy_images(input_data)
    print("Equalized size of input data: " + str(len(input_data)))


def plot_model():
    with open('model.json', 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    plot(model, to_file='model.png', show_shapes=True)


if __name__ == "__main__":
    plot_model()
