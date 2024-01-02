from PIL import Image
import numpy as np
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import sys

import db

INPUT_SIZE = 256

def load_training_data(limit: int = None, img_size=(INPUT_SIZE, INPUT_SIZE)) -> (np.ndarray, np.ndarray):
    '''Loads and preprocesses images and labels for multi-label classification.'''
    training_images = db.get_training_data(limit)
    
    images = []
    labels = []

    # Replace the path in each image with the actual image data
    for screenshot_path, domain_labels in training_images:
        # Load and resize the image
        with Image.open(screenshot_path, 'r') as image:
            rgb_image = image.convert('RGB')
            resized_image = rgb_image.resize(img_size)

            # Convert the image to a numpy array
            images.append(np.array(resized_image))

        # Add the label to the array (as a binary vector for multi-label classification)
        labels.append(domain_labels)  # 'domain_labels' should already be a binary vector

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def get_model(input_size: int = INPUT_SIZE) -> models.Sequential:
    '''Creates a convolutional neural network model'''
    # Define a simple CNN model
    topic_mapping_count = len(db.get_topics())

    if topic_mapping_count == 0:
        raise Exception("No topic mappings found in the database")

    '''
        If you have a binary classification problem(i.e., your labels are 
        either 0 or 1), you should ensure that the last layer of your model has
        only one neuron and uses a sigmoid activation function. If you have a 
        multi-class classification problem, the last layer should have as many 
        neurons as there are classes, and you should use a softmax activation 
        function.
    '''
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_size, input_size, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(input_size, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(topic_mapping_count, activation='sigmoid')  # Adjust the output layer based on the number of topics
    ])

    # Compile the model
    model.compile(optimizer='adam',
                # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model

def make_model_diagram(model: models.Sequential):
    '''Creates a diagram of the model'''
    plot_model(model, to_file='research_data/model.png', show_shapes=True, show_layer_names=True)

def train():
    '''Loads data and trains the model'''
    # Load and preprocess images and labels
    print("📷 Preparing images and labels...")

    # Load the images and labels for the 100 most popular screenshotted domains
    images, labels = load_training_data(limit=100)

    # Split the data into training and testing sets
    print("📊 Preparing test and training datasets...")
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define a simple CNN model
    print("🦾 Building model...")
    model = get_model()

    # Train the model
    print("🦾 Training model...")
    model.fit(images_train, labels_train, epochs=10, validation_data=(images_test, labels_test))

if __name__ == "__main__":
    # Check args for 'plot' to plot the model
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        model = get_model()
        make_model_diagram(model)
    else:
        train()