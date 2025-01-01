import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras 

from keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.model_selection import train_test_split
# import cv2
import preprocess


df = pd.read_csv('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/updated_orca_image_data_expanded.csv')
image_dir = '/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/'
mask_dir = '/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/masks/'

X_images = []
X_masks = []
y = []

for _, row in df.iterrows():
    img, mask = preprocess.load_image(row['directory_name'], row['mask_name'])
    X_images.append(img)
    X_masks.append(mask)
    y.append(row['tags'])

X_images = np.array(X_images)
X_masks = np.array(X_masks)
y = np.array([eval(tag)[0] for tag in y])

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train_img, X_test_img, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
    X_images, X_masks, y_encoded, test_size=0.2, random_state=42
)
# Separate inputs for image and mask
input_image = Input(shape=(224, 224, 3))
input_mask = Input(shape=(224, 224, 1))

# Process image through ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
image_features = base_model(input_image)

# Process mask 
mask_features = Conv2D(64, (3, 3), activation='relu')(input_mask)
mask_features = MaxPooling2D((2, 2))(mask_features)

# Combine image and mask features
combined_features = Concatenate()([GlobalAveragePooling2D()(image_features), Flatten()(mask_features)])

# Add dense layers for classification
x = Dense(256, activation='relu')(combined_features)
output = Dense(len(le.classes_), activation='softmax')(x)

# Create the model
resNet_model = Model(inputs=[input_image, input_mask], outputs=output)