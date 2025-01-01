import os
import cv2
import numpy as np
import resNet

import matplotlib.pyplot as plt

image_files = [os.path.join('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/testModel', f) for f in os.listdir('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/testModel') if f.endswith('.jpg') or f.endswith('.png')]
print(image_files)



import loadDataset

model = loadDataset.version.model
def process_image(image_file):
  results = model.predict(image_file).json() 

  # Create an empty mask 
  image = cv2.imread(image_file)
  height, width, _ = image.shape
  mask = np.zeros((height, width, 3), dtype=np.uint8)  # Create a 3-channel mask to visualize colors

  # Process the segmentation results
  for prediction in results['predictions']:
      class_name = prediction['class']

      # Assign colors for each class
      if class_name == 'orca-body':
          color = (0, 0, 255)  # Red for orca-body
      elif class_name == 'saddle-patch':
          color = (0, 255, 0)  # Green for saddle-patch
      else:
          continue  # Skip other classes (if any)

      # Extract the polygon points for the segmentation mask
      points = prediction['points']
      polygon = np.array([[point['x'], point['y']] for point in points], dtype=np.int32)

      # Fill the polygon into the mask with the assigned color
      cv2.fillPoly(mask, [polygon], color)

  return image, mask


t=0
# Loop over all images, process them, and save the results
for image_file in image_files:
    # Process each image and generate its segmentation mask
    if t == 0:
      imageTest, maskTest = process_image(image_file)
    
    t += 1

cv2.imwrite('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/testModel/testMask.png', maskTest)
cv2.imwrite('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/testModel/tesImage.jpg', imageTest)

def preprocess_input(image, mask, target_size=(224, 224)):
    # Resize and normalize image
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    
    # Resize and normalize mask
    mask = cv2.resize(mask, target_size)
    mask = mask.astype(np.float32) / 255.0
    
    # Ensure mask 3 dimensions 
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
    
    image = np.expand_dims(image, axis=0)
    mask = np.expand_dims(mask, axis=0)
    
    return image, mask

image = cv2.imread('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/testModel/tesImage.jpg')
mask = cv2.imread('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/testModel/testMask.png', cv2.IMREAD_GRAYSCALE)

image_input, mask_input = preprocess_input(image, mask)

prediction = resNet.resNet_model.predict([image_input, mask_input])

print(prediction)

orca_tags = ['L84', 'L88', 'J37', 'J39', 'J27', 'J26']  
predicted_tag = orca_tags[np.argmax(prediction)]
print(image_file)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

print(predicted_tag)