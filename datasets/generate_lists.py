import os

def generate_list_files(train_dir, test_dir, train_list_file, test_list_file):
    normal_videos = []
    abnormal_videos = []

    # Process training videos
    for filename in os.listdir(train_dir):
        if filename.endswith('.npy'):
            # Check if the video is normal or abnormal based on the filename
            if 'Normal' in filename:
                normal_videos.append(os.path.join(train_dir, filename))
            else:
                abnormal_videos.append(os.path.join(train_dir, filename))

    # Create training list
    with open(train_list_file, 'w') as train_file:
        for video in normal_videos:
            train_file.write(f"{video}\n")
        for video in abnormal_videos:
            train_file.write(f"{video}\n")  # Include all abnormal videos in training

    # Process testing videos
    for filename in os.listdir(test_dir):
        if filename.endswith('.npy'):
            # Check if the video is normal or abnormal based on the filename
            if 'Normal' in filename:
                normal_videos.append(os.path.join(test_dir, filename))
            else:
                abnormal_videos.append(os.path.join(test_dir, filename))

    # Create testing list
    with open(test_list_file, 'w') as test_file:
        for video in normal_videos:
            test_file.write(f"{video}\n")
        for video in abnormal_videos:
            test_file.write(f"{video}\n")  # Include all abnormal videos in testing

if __name__ == "__main__":
    
    train_directory = 'E:/Projects/MGFN/data/ucf_tencrop_1d/train'  # Update this path to point to the train directory
    test_directory = 'E:/Projects/MGFN/data/ucf_tencrop_1d/test'    # Update this path to point to the test directory
    train_list_path = 'ucf-i3d.list'  # Path for training list
    test_list_path = 'ucf-i3d-test.list'  # Path for testing list

    generate_list_files(train_directory, test_directory, train_list_path, test_list_path)