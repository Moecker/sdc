# All required imports
import argparse
import cv2
import numpy as np
import pandas as pd

from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

# Image from data
kInputCols = 320
kInputRows = 160
kInputImageSize = (kInputCols, kInputRows)

# Image for training
kCols = 64
kRows = 64
kChannels = 3
kTargetImageSize = (kCols, kRows)

# Training
kEpochs = 12
kSamplePerEpoch = 20000
kBatchSize = 64
kTrainingSplit = 0.8

# Adding of images
kAddingFactor = 2
kSteeringThresholds = [0.20, 0.40, 0.60]

# Directories
kModelName = 'model'
kFolderName = 'C:/_HAF/sdc/behavioral-cloning/udacity_data'

# Preprocessing
kCropTop = 55
kCropBottom = kInputRows - 25
kSteeringAdjustments = 0.20
kMinBrightness = 0.25
kAddedSteering = 0.001
kXRange = 80
kYRange = 40
	
def main(model_name):
    # Read the whole csv file containing the udacity input data
    input_data = pd.read_csv(kFolderName + '/driving_log.csv')
   
    # Increase number of images with high steering angle
    print("Original size of input data: " + str(len(input_data)))
    input_data = add_curvy_images(input_data, kSteeringThresholds)
    print("Total size of input data: " + str(len(input_data)))
    
    # Shuffle the data
    input_data = input_data.sample(frac=1).reset_index(drop=True)
    
    # Split training and validation into 80/20
    num_rows = input_data.shape[0]
    num_rows_training = int(num_rows * kTrainingSplit)

    # Setup training and validation based on split
    training_data = input_data.loc[0:num_rows_training-1]
    validation_data = input_data.loc[num_rows_training:num_rows-1]
    
    print("Target image size: " + str(kTargetImageSize))
    print("Number of training data: " + str(training_data.shape[0]))
    print("Number of validation data: " + str(validation_data.shape[0]))
 
    # Delete the main input_data from memory
    del input_data

    # Init the generators for training and validation
    training_generator = build_generator(training_data, batch_size=kBatchSize)
    validation_generator = build_generator(validation_data, batch_size=kBatchSize)

    # Build the to be trained model
    if model_name == None:
        model = build_model()
    else:
        model = load_model(model_name)

    # Define samples per epoch
    samples_per_epoch = int(kSamplePerEpoch / kBatchSize) * kBatchSize
    print("Number of epochs: "+ str(kEpochs))
    print("Samples per epoch: "+ str(kSamplePerEpoch))

    # Perform the training
    model.fit_generator(generator=training_generator, 
                        validation_data=validation_generator,
                        samples_per_epoch=samples_per_epoch, 
                        nb_epoch=kEpochs, 
                        nb_val_samples=validation_data.shape[0])

    # Finally save the model and its weights
    print("Saving model architecture and trained weights ...")
    model.save_weights(kModelName + '.h5')
    with open(kModelName + '.json', 'w') as outfile:
        outfile.write(model.to_json())
 
 
def add_curvy_images(input_data, thresholds):    
    abs_steering = np.abs(input_data['steering'])

    for threshold in thresholds:
        is_curve = abs_steering > threshold
        curve_rows = input_data[is_curve]
            
        for i in range(kAddingFactor):
            input_data = input_data.append(curve_rows, ignore_index=False)
            
    return input_data.reindex()
    
    
def load_model(model_name):
    print("Opening previously trained model ...")
    
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    return model
    
def build_model():
    # Create a sequential model
    model = Sequential()

    # Layer 1: Convolution + ELU, Output shape is 32x32x32
    input_shape = (kRows, kCols, kChannels)
    model.add(Convolution2D(32, 5, 5, input_shape=input_shape, subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    # Layer 2: Convolution + MaxPooling + ELU + Dropout, Output shape is 15x15x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # Layer 3: Convolution + ELU + Droput, Output shape is 13x13x8
    model.add(Convolution2D(8, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))

    # Flatten the input, Output is o shape 1352
    model.add(Flatten())

    # Layer 4: Dense + Dropout + ELU, Output is o shape 1024
    model.add(Dense(1024))
    model.add(Dropout(.3))
    model.add(ELU())

    # Layer 5: Dense + ELU, Output is o shape 512
    model.add(Dense(512))
    model.add(ELU())

    # Finally add a single dense layer, since this is a regression problem, Output is o shape 1
    model.add(Dense(1))
	
	# Compile using mse as loss function (softmax would be wrong since we do regression)
    model.compile(optimizer="adam", loss="mse")
    return model
       
       
def build_generator(input_data, batch_size=kBatchSize):
    number_of_inputs = input_data.shape[0]
    batches_per_epoch = int(number_of_inputs / batch_size)
    
    batch_index = 0
    while(True):
        start_batch = batch_index * batch_size
        end_batch = start_batch + batch_size - 1

        # Create empty numpy arrays for storing the X and y data
        X_batch = np.zeros((batch_size, kRows, kCols, kChannels), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        # For the current batch get the image (preprocessed and augmented) and the
		# corresponding steering angle
        slice_index = 0
        for index, csv_row in input_data.loc[start_batch:end_batch].iterrows():
            X_batch[slice_index], y_batch[slice_index] = get_image_with_steering(csv_row)
            slice_index += 1

        batch_index += 1
        if batch_index == (batches_per_epoch - 1):
            # Reset the index to enable for further loops
            batch_index = 0
            
		# Yield is part of the generator and semantically close to "return"
        yield X_batch, y_batch
     
    
def get_image_with_steering(csv_row):
    steering = csv_row['steering']

    # Randomly choose the camera to take the image from
    camera = np.random.choice(['center', 'left', 'right'])

    # Adjust the steering angle for left and right cameras
    if camera == 'left':
        steering += kSteeringAdjustments
    elif camera == 'right':
        steering -= kSteeringAdjustments

    # Loads the image
    image = load_img(kFolderName + '/' + csv_row[camera].strip())
    image = img_to_array(image)

    # Decide whether to horizontally flip the image
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # Flip the image and reverse the steering angle
        steering = -steering
        image = cv2.flip(image, 1)

    # Apply brightness augmentation
    image = alter_image_brightness(image)
    
    # Add a slice of random shadow
    image = add_shadow(image)
    
    # Apply traversion into x and y direction
    image, added_steering = traverse_image(image)
    steering += added_steering

    # Crop, resize and normalize the image
    image = preprocess_image(image)
    return image, steering
    
    
def alter_image_brightness(image):
    # Convert to HSV space so that its easy to adjust brightness
    altered_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_brightness = kMinBrightness + np.random.uniform()

    # Apply the brightness reduction to the V channel
    altered_image[:,:,2] = altered_image[:,:,2] * random_brightness

    # Convert to RBG again
    altered_image = cv2.cvtColor(altered_image, cv2.COLOR_HSV2RGB)
    return altered_image       
       
       
def traverse_image(image):
    # Translation in x direction
    translation_x = kXRange * np.random.uniform() - kXRange / 2
    
    # Changed steering angle based on x-direction
    added_steering_angle = kAddedSteering + translation_x / kXRange * 2 * 0.2
    
    # Translation in x direction
    translation_y = kYRange * np.random.uniform() - kYRange / 2
    
    # Apply the translation using warp method from opencv
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (kInputCols, kInputRows))
    
    return translated_image, added_steering_angle

    
def add_shadow(image):
    top_y = kInputCols * np.random.uniform()
    top_x = 0
    botom_x = kInputRows
    botom_y = kInputCols * np.random.uniform()
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    shadow_mask = 0 * image[:,:,1]

    mesh_x = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    mesh_y = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    
    is_mask = ((mesh_x - top_x) * (botom_y - top_y) - (botom_x - top_x) * (mesh_y - top_y)) >= 0
    shadow_mask[is_mask] = 1
   
    if np.random.randint(2) == 1:
        random_brightness = kMinBrightness + np.random.uniform()
        mask_on = shadow_mask == 1
        mask_off = shadow_mask == 0
        
        if np.random.randint(2) == 1:
            image[:,:,2][mask_on] = image[:,:,2][mask_on] * random_brightness
        else:
            image[:,:,2][mask_off] = image[:,:,2][mask_off] * random_brightness    
            
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

       
def preprocess_image(image):
    image = crop_and_resize(image)
    image = image.astype(np.float32)

    # Normalize the image to range [-0.5, +0.5]
    image = (image / 255.0) - 0.5
    return image
       
       
def crop_and_resize(image):
    cropped_image = image[kCropTop:kCropBottom, :, :]
    processed_image = resize_to_target_image_size(cropped_image)
    return processed_image
    
    
def resize_to_target_image_size(image):
    return cv2.resize(image, kTargetImageSize)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('--model', 
                        type=str, 
                        help='Path to model definition json. Model weights should be on the same path.',
                        required=False,
                        default=None)
                        
    args = parser.parse_args()
    
    # Call the main routine
    main(args.model)
    
