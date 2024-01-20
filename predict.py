import numpy as np
import tensorflow as tf
from PIL import Image

from siamese import Siamese

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":
    model = Siamese()
    image_1 = r"datasets\images_background\2\28_with_mask.jpg"
    try:
        image_1 = Image.open(image_1)
    except:
        print('Image_1 Open Error! Try again!')

    image_2 = r'datasets\images_background\1\19.jpg'
    try:
        image_2 = Image.open(image_2)
    except:
        print('Image_2 Open Error! Try again!')
    probability = model.detect_image(image_1,image_2)
    print(probability)
    # while True:
    #     image_1 = r"datasets\images_background\1\10.jpg"
    #     try:
    #         image_1 = Image.open(image_1)
    #     except:
    #         print('Image_1 Open Error! Try again!')
    #         continue
    #
    #     image_2 = r'datasets\images_background\1\12.jpg'
    #     try:
    #         image_2 = Image.open(image_2)
    #     except:
    #         print('Image_2 Open Error! Try again!')
    #         continue
    #     probability = model.detect_image(image_1,image_2)
    #     print(probability)
