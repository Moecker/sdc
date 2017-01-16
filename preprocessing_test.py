import cv2
from model import crop_and_resize, alter_image_brightness, traverse_image, add_shadow, kFolderName
from keras.preprocessing.image import img_to_array, load_img

kTargetFolder = 'samples'

image_files = [kFolderName + '/IMG/left_2016_12_01_13_30_48_287.jpg', 
               kFolderName + '/IMG/center_2016_12_01_13_30_48_287.jpg',
               kFolderName + '/IMG/right_2016_12_01_13_30_48_287.jpg',
               kFolderName + '/IMG/left_2016_12_01_13_43_52_984.jpg', 
               kFolderName + '/IMG/center_2016_12_01_13_43_52_984.jpg',
               kFolderName + '/IMG/right_2016_12_01_13_43_52_984.jpg',
               kFolderName + '/IMG/left_2016_12_01_13_37_33_786.jpg',
               kFolderName + '/IMG/center_2016_12_01_13_37_33_786.jpg',
               kFolderName + '/IMG/right_2016_12_01_13_37_33_786.jpg']
               
              
# Just some test method to check the preprocessing
def test_image_processing():
    for i, image_file in enumerate(image_files):
        image = load_img(image_file)    
        image = img_to_array(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        save_image(image, 'raw', i)
        
        image = cv2.flip(image, 1)
        save_image(image, 'flipped', i)
 
        image = alter_image_brightness(image)
        save_image(image, 'altered', i)
        
        image = add_shadow(image)
        save_image(image, 'shadow', i)
        
        image = traverse_image(image)[0]
        save_image(image, 'traversed', i)
        
        image = crop_and_resize(image)
        save_image(image, 'cropped', i)
            
        print("Image size after preprocessing: " + str(image.shape))

        
def save_image(image, name, i):
    cv2.imwrite(kTargetFolder + '/' +  name + str(i) + '.jpg', image)  
        
        
test_image_processing()