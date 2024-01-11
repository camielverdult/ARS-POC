from PIL import Image
import numpy as np
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

import sys
import datetime

import db

INPUT_SIZE = db.INPUT_SIZE

def get_model(input_size: int = INPUT_SIZE) -> models.Sequential:
    '''Creates a convolutional neural network model'''
    # Define a simple CNN model
    # topic_mapping_count = len(db.get_topics())
    topic_mapping_count = db.get_count('topics')

    if topic_mapping_count == 0:
        raise Exception("No topic mappings found in the database")

    
    # We have a multi-class classification problem, the last layer should have
    # as many  neurons as there are classes, and you should use a softmax 
    # activation function. The loss function should be categorical_crossentropy.
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_size, input_size, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(topic_mapping_count, activation='sigmoid')  # Adjust the output layer based on the number of topics
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model

def make_model_diagram(model: models.Sequential):
    '''Creates a diagram of the model'''
    plot_model(model, to_file='research_data/model.png', show_shapes=True, show_layer_names=True)

def train():
    '''Loads data and trains the model'''
    db.randomise_use_in_training()

    # Load and preprocess images and labels
    print("📷 Preparing images and labels...")

    # Split the data into training and testing sets
    print("📊 Preparing test and training datasets...")
    training_images, training_labels, validation_images, validation_labels = db.get_training_data()

    # Define a simple CNN model
    print("🦾 Building model...")
    model = get_model()

    # Train the model
    print("🦾 Training model...")
    model.fit(training_images, training_labels, epochs=10, validation_data=(validation_images, validation_labels))

    # Save the model with timestamp
    print("💾 Saving model...")

    timestamp = datetime.datetime.now().strftime("%H%M%S-%d%m%Y")
    model_name = f"model-{timestamp}"
    model.save(f"research_data/models/{model_name}.keras")

    # # Evaluate the model
    # print("🧾 Evaluating model...")
    # test_loss, test_acc = model.evaluate(images_test, labels_test, verbose=2)

if __name__ == "__main__":
    # Check args for 'plot' to plot the model
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        model = get_model()
        make_model_diagram(model)
    else:
        train()
