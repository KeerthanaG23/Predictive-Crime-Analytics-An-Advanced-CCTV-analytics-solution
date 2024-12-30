# import os
# import numpy as np

# # Define the categories
# crime_categories = [
#     'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
#     'Explosion', 'Fighting', 'RoadAccident', 'Robbery',
#     'Shooting', 'Shoplifting', 'Stealing', 'Vandalism', 'Normal'
# ]

# # Path to the main data directory
# data_directory = 'E:/Projects/MGFN/data/ucf_tencrop_1d'  # Change this to your actual path

# # Function to create ground truth labels
# def create_ground_truth_labels(directory):
#     gt_labels = {}

#     # Iterate through the files in the directory
#     for filename in os.listdir(directory):
#         if filename.endswith('.npy'):
#             # Extract the category from the filename
#             category_found = None
#             for category in crime_categories:
#                 if category.lower() in filename.lower():
#                     category_found = category
#                     break
            
#             # Assign ground truth label
#             if category_found:
#                 gt_labels[filename] = category_found  # Use the specific crime category
#             else:
#                 print(f"Warning: No category found for {filename}")

#     # Convert the labels to a numpy array
#     video_names = list(gt_labels.keys())
#     ground_truth_array = np.array(list(gt_labels.values()))

#     # Save the ground truth labels to a .npy file
#     #np.save(os.path.join(directory, 'gt-ucf.npy'), ground_truth_array)
#     np.save('gt-ucf.npy', ground_truth_array)


#     print(f"Ground truth labels created and saved successfully.")
# create_ground_truth_labels(os.path.join(data_directory, 'train'))

import os
import numpy as np

# Define the categories
crime_categories = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
    'Explosion', 'Fighting', 'RoadAccident', 'Robbery',
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism', 'Normal'
]

# Path to the main data directory
data_directory = 'E:/Projects/MGFN/data/ucf_tencrop_1d'  # Change this to your actual path

# Function to create ground truth labels
def create_ground_truth_labels(directory):
    gt_labels = []

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            # Check if the filename contains "Normal"
            if 'normal' in filename.lower():
                gt_labels.append(0)  # Normal
            else:
                gt_labels.append(1)  # Abnormal

    # Convert the labels to a numpy array
    ground_truth_array = np.array(gt_labels)

    # Save the ground truth labels to a .npy file
    np.save('gt-ucf.npy', ground_truth_array)

    print(f"Ground truth labels created and saved successfully.")

# Create ground truth labels for the training directory
create_ground_truth_labels(os.path.join(data_directory, 'train'))