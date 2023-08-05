from keras.preprocessing.image import ImageDataGenerator
import cv2
import os 

image_gen = ImageDataGenerator(rotation_range=180, # rotate the image 180 degrees
                               width_shift_range=0.1, # Shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allow horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


#add or remove items from this list as required, ensure file structures are correct
#change this to function as required 
for item in ['banana','carrot','pepper','pickle']:
    directory = os.path.join(os.getcwd(),'items',item)
    images = os.listdir(directory)
    
    for i in range(len(images)):
        
        for j in range(5):
            image = cv2.imread(os.path.join(directory,images[i]))
            new_image = image_gen.random_transform(image)
            filename = str(i) + '-' + str(j) + '.jpg'
            filepath = os.path.join(directory, filename)
            cv2.imwrite(filepath,new_image)
            
