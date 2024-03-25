import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    datagen = ImageDataGenerator(
            rotation_range=40,
            height_shift_range=0.2,
            width_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    i = 0
    for batch in datagen.flow_from_directory('path/to/dir/augm', 
                                         batch_size=6, 
                                         target_size=(256, 256),
                                         save_to_dir='paht/to/dir/augm/val_augm', 
                                         save_format='png'):
        i += 1
        if i > 30:
            break

if(__name__=='__main__'):
    main()
    