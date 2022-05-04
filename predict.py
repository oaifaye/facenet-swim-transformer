import numpy as np
from PIL import Image

from facenet import Facenet
import  os

def p1():
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue

        probability = model.detect_image(image_1, image_2)
        print(probability)

if __name__ == "__main__":
    model = Facenet()
    check_path = r'C:\Users\Administrator\Desktop\zawu\20220116\imgs_f\y'
    for img_name1 in os.listdir(check_path):
        for img_name2 in os.listdir(check_path):
            image_1 = Image.open(os.path.join(check_path, img_name1))
            image_2 = Image.open(os.path.join(check_path, img_name2))
            probability = model.detect_image(image_1, image_2, show_plt=False)
            print(img_name1+'-'+img_name2+'\t'+ str(probability[0]))

