import numpy as np
import os
import cv2


import matplotlib.pyplot as plt

import loadDataset

model = loadDataset.version.model
def process_image(image_file):
  results = model.predict(image_file).json()  # Pass the file path here

  # Create an empty mask (use the same size as the input image)
  image = cv2.imread(image_file)
  height, width, _ = image.shape
  mask = np.zeros((height, width, 3), dtype=np.uint8)  # Create a 3-channel mask to visualize colors

  # Process the segmentation results
  for prediction in results['predictions']:
      class_name = prediction['class']

      # Assign colors for each class (for visualization purposes)
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

  # Display the image and the corresponding segmentation mask
  # plt.subplot(1, 2, 1)
  # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  # plt.title("Image")

  # plt.subplot(1, 2, 2)
  # plt.imshow(mask)  # Show the mask with colored regions
  # plt.title("Segmentation Mask")
  # plt.show()

  # Optionally, save the segmentation mask
  cv2.imwrite("/path/to/save/mask.png", mask)

  return image, mask


# Function to process all images in a given directory
def process_dataset(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    # Get the list of all images in the directory
    image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png')]

    images = []
    masks = []
    # Loop over all images, process them, and save the results
    for image_file in image_files:
        # Process each image and generate its segmentation mask
        image, mask = process_image(image_file)

        # Save the segmentation mask
        mask_filename = os.path.join(output_dir, os.path.basename(image_file).replace('.jpg', '_mask.png').replace('.png', '_mask.png'))
        cv2.imwrite(mask_filename, mask)

        # # Optionally, display the image and the corresponding segmentation mask
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.title("Image")

        # plt.subplot(1, 2, 2)
        # plt.imshow(mask)  # Show the mask with colored regions
        # plt.title("Segmentation Mask")
        # plt.show()

        # Example images and masks
        images.append(image)  # List of your images
        masks.append(mask)     # Corresponding list of masks

       


        # Optionally, save the annotated image with the mask
        annotated_image_filename = os.path.join(output_dir, os.path.basename(image_file).replace('.jpg', '_annotated.png').replace('.png', '_annotated.png'))
        cv2.imwrite(annotated_image_filename, mask)
    
    # Create a figure with a vertical layout
    fig, axes = plt.subplots(len(images), 2, figsize=(10, 5 * len(images)))
    fig.subplots_adjust(hspace=0.5)  # Adjust spacing between rows

    # If there's only one pair, axes won't be iterable; handle single pair case
    if len(images) == 1:
      axes = [axes]

    # Plot each image and its corresponding mask
    for i, (img, mask) in enumerate(zip(images, masks)):
      # Display the image
      axes[i][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      axes[i][0].set_title(f"Image {i+1}")
      axes[i][0].axis("off")
      
      # Display the segmentation mask
      axes[i][1].imshow(mask)
      axes[i][1].set_title(f"Segmentation Mask {i+1}")
      axes[i][1].axis("off")

    # Show all the images in one window
    # plt.show()



# Directories for the dataset
base_image_dir = "/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2"  
train_dir = os.path.join(base_image_dir, "train")
valid_dir = os.path.join(base_image_dir, "valid")
test_dir = os.path.join(base_image_dir, "test")

# Output directories to save the masks
train_output_dir = os.path.join(base_image_dir, "masks/train")  
valid_output_dir = os.path.join(base_image_dir, "masks/valid")  
test_output_dir = os.path.join(base_image_dir, "masks/test") 

# Process the entire dataset (train, valid, test)
process_dataset(train_dir, train_output_dir)
process_dataset(valid_dir, valid_output_dir)
process_dataset(test_dir, test_output_dir)

