from keras.models import load_model
from PIL import Image
image_1 = input(r"datasets\images_background\1\10.jpg")
image_1 = Image.open(image_1)

model=load_model(r'D:\research\资料\Siamese-tf2-master\model\2')
model.show()