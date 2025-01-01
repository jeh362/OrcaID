import pandas as pd
import numpy as np
import os

import requests
import shutil



# Define your Roboflow API Key and Project details
API_KEY = os.environ["ROBOFLOW_API_KEY"]  # Replace with your actual API key
WORKSPACE = 'jacqueline-eb2ts'  # Replace with your Roboflow workspace
PROJECT = 'orca-dnre4'  # Replace with your Roboflow project name
VERSION_ID = 2  # Replace with the dataset version you're using

# Define the base URL for the search API
BASE_URL = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/search"

# List of orca tags 
tags = ["L84", "L88", "J37", "J39", "J27", "J26"]  

records = []
image_mask_tag_data = []


for tag in tags:
    payload = {
        "api_key": API_KEY,
        "tag": tag,  # Search by the tag
        "in_dataset": True,  # searching within the dataset
        "limit": 100,  
        "fields": ["id", "name", "tags"],  # Specify the fields to retrieve
    }

    # Send a POST request to the search API
    response = requests.post(BASE_URL, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Add the images from the response to the records list
        records.extend(data['results'])
    else:
        print(f"Error fetching data for tag {tag}: {response.status_code}")

# Convert the records to a pandas DataFrame
df = pd.DataFrame(records)

# Display the first few rows of the DataFrame
print(df.head())

df.to_csv('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/orca_tags.csv', index=False)

base_dir = "/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
valid_dir = os.path.join(base_dir, "valid")

masks_dir = os.path.join(base_dir, "masks")
masks_train_dir = os.path.join(masks_dir, "train")
masks_test_dir = os.path.join(masks_dir, "test")
masks_valid_dir = os.path.join(masks_dir, "valid")

def move_files(src_dir, dest_dir):
    if os.path.exists(src_dir):
        for file_name in os.listdir(src_dir):
            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)
            if os.path.isfile(src_path):  
                shutil.move(src_path, dest_path)

# Move image files
move_files(train_dir, base_dir)
move_files(test_dir, base_dir)
move_files(valid_dir, base_dir)

# Move mask files
move_files(masks_train_dir, masks_dir)
move_files(masks_test_dir, masks_dir)
move_files(masks_valid_dir, masks_dir)



image_dir = "/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2"  
mask_dir = "/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/masks" 

df = pd.read_csv("/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/orca_tags.csv") 

updated_rows = []

for _, row in df.iterrows():
    image_name = row['name']  
    tag = row['tags'][0] if row['tags'] else "" 

    base_image_name = os.path.splitext(image_name)[0]
    print(base_image_name)

    # Initialize mask_path as None
    mask_path = "/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/masks/"


    for mask_file in os.listdir(mask_dir):
          if base_image_name in mask_file:  # Check if base image name is in the mask filename
              mask_path = os.path.join(mask_dir, mask_file)
  

    # Append the data (image name, mask path, and tag) to the list
    row['mask_path'] = mask_path  # Add the mask path (or None if not found)
    updated_rows.append(row)

updated_df = pd.DataFrame(updated_rows)

updated_df.to_csv('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/updated_orca_image_data.csv', index=False)

print("Data saved to 'updated_orca_image_data.csv'")

################################################################

# Load the CSV file
csv_file_path = '/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/updated_orca_image_data.csv'
data = pd.read_csv(csv_file_path)

# Get the list of image files in the directory
image_directory = '/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/'
mask_directory = '/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/masks'
image_files = os.listdir(image_directory)
mask_files = os.listdir(mask_directory)


# Function to find matching directory names
def find_matching_files(name, dir):
    base_name = os.path.splitext(name)[0]
    print(base_name)
    return [file for file in dir if file.startswith(base_name)]

# Create a new list to store the expanded data
expanded_data = []

# Iterate through each row in the original DataFrame
for _, row in data.iterrows():
    matching_files = find_matching_files(row['name'], image_files)
    
    # Create a new row for each matching file
    for file in matching_files:
        new_row = row.copy()
        new_row['directory_name'] = file
        expanded_data.append(new_row)

# Create a new DataFrame from the expanded data
updated_df = pd.DataFrame(expanded_data)
new_expanded = []

# Iterate through each row in the original DataFrame
for _, row in updated_df.iterrows():
    matching_files = find_matching_files(row['directory_name'], mask_files)
    
    # Create a new row for each matching file
    for file in matching_files:
        new_row = row.copy()
        new_row['mask_name'] = file
        new_expanded.append(new_row)

updated_df = pd.DataFrame(new_expanded)

# Save the updated DataFrame to a new CSV file
updated_df.to_csv('/Users/jacquelinehui/Desktop/cs/Projects/orca-id/orca--2/updated_orca_image_data_expanded.csv', index=False)

print("Updated CSV file created with expanded rows.")

