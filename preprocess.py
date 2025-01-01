import pandas as pd
import numpy as np


import cv2


df = pd.read_csv('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/updated_orca_image_data_expanded.csv')
image_dir = '/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/'
mask_dir = '/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/masks/'

def load_image(image_path, mask_path):
    img = cv2.imread(image_dir + image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    
    mask = cv2.imread(mask_dir + mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (224, 224))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    
    return img, mask

