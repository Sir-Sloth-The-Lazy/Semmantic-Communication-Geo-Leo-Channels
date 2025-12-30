import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

def save_images(data, labels, base_dir, class_names):
    """
    Saves CIFAR-10 images to directories: base_dir/class_name/image_id.png
    """
    print(f"Saving images to {base_dir}...")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for i in tqdm(range(len(data))):
        img_array = data[i]
        label = labels[i][0]
        class_name = class_names[label]
        
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        img = Image.fromarray(img_array)
        img.save(os.path.join(class_dir, f"{i}.png"))

def main():
    # Define paths
    ROOT_DIR = 'dataset/CIFAR10'
    TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
    TEST_DIR = os.path.join(ROOT_DIR, 'test')

    # CIFAR-10 Class Names
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Download Data
    print("Downloading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Save Train Images
    save_images(x_train, y_train, TRAIN_DIR, CLASS_NAMES)

    # Save Test Images
    save_images(x_test, y_test, TEST_DIR, CLASS_NAMES)

    print("\nDataset successfully saved to:")
    print(f" - {TRAIN_DIR}")
    print(f" - {TEST_DIR}")

if __name__ == "__main__":
    main()
