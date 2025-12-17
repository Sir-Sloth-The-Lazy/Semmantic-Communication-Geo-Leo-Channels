from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import os

from config.train_config import BATCH_SIZE

def dataset_generator(dir, mode=None, shuffle=True):
    
    # Check if directory exists
    if not os.path.exists(dir):
        print(f"Dataset directory '{dir}' not found. Downloading CIFAR-10 fallback...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        if 'test' in dir or 'val' in dir:
            x_data, y_data = x_test, y_test
        else:
            x_data, y_data = x_train, y_train
            
        # Create Dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        
        # Mimic image_dataset_from_directory output behavior
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(x_data), seed=0)
            
        dataset = dataset.batch(BATCH_SIZE)
        
    else:
        if mode:
            dataset = image_dataset_from_directory(
                directory=dir,
                label_mode='int',
                labels='inferred',
                color_mode='rgb',
                batch_size=BATCH_SIZE,
                image_size=(32, 32),
                shuffle=shuffle,
                interpolation='bilinear',
                validation_split=0.1,
                subset=mode,
                seed=0
            )
        else:
            dataset = image_dataset_from_directory(
                directory=dir,
                label_mode='int',
                labels='inferred',
                color_mode='rgb',
                batch_size=BATCH_SIZE,
                image_size=(32, 32), 
                shuffle=shuffle,
                interpolation='bilinear'
            )

    return dataset
