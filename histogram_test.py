# All required imports
import cv2
import numpy as np
import pandas as pd

from model import kFolderName, kAddingFactor, add_curvy_images

def main():
    # Read the whole csv file containing the udacity input data
    input_data = pd.read_csv(kFolderName + '/driving_log.csv')
    
    print("Original size of input data: " + str(len(input_data)))
    input_data = add_curvy_images(input_data)
    print("Equalized size of input data: " + str(len(input_data)))
    
if __name__ == "__main__":
    # Call the main routine
    main()
    