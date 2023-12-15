from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import sys

import db

INPUT_SIZE = 256

# Function to load and preprocess images
def load_training_data(limit: int = -1, img_size=(INPUT_SIZE, INPUT_SIZE)) -> (np.array, np.array):
    training_images = db.get_training_data(limit)

    # Create an empty array to hold the images
    images = np.array()
    labels = np.array()

    # Replace the path in each image with the actual image data
    for entry in training_images:
        # Load and resize the image
        img_path = entry[0]
        image = Image.open(img_path, 'r')
        image = image.resize(img_size)

        # Convert the image to a numpy array
        image = np.array(image)

        # Add the image to the array
        images.append(image)

        # Add the label to the array
        labels.append(entry[1])

    return (images, labels)

def get_model(input_size: int = INPUT_SIZE, ) -> models.Sequential:
    # Define a simple CNN model
    topic_mapping_count = len(db.get_topics())

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
    plot_model(model, to_file='research_data/model.png', show_shapes=True, show_layer_names=True)

def main():
    # Load and preprocess images and labels
    print("📷 Preparing images and labels...")

    # Load the images and labels for the 100 most popular screenshotted domains
    images, labels = load_training_data(100)

    # Split the data into training and testing sets
    print("📊 Preparing test and training datasets...")
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    topic_mappings = db.get_topics()

    # Define a simple CNN model
    print("🦾 Building model...")
    model = get_model(len(topic_mappings))

    # Train the model
    print("🦾 Training model...")
    model.fit(images_train, labels_train, epochs=10, validation_data=(images_test, labels_test))

if __name__ == "__main__":
    # Check args for 'plot' to plot the model
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        model = get_model()
        make_model_diagram(model)
    else:
        main()