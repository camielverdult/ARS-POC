from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import evaluate
import numpy as np

import datetime
import os
import pathlib
import sys

import db

INPUT_SIZE = db.INPUT_SIZE
DROPOUT_RATE = 0.25

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
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(input_size, input_size, 3)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(topic_mapping_count, activation='sigmoid') 
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

    augmented_data_gen = ImageDataGenerator(
        rotation_range=20,       # Degree range for random rotations
        width_shift_range=0.2,   # Fraction of total width for horizontal shift
        height_shift_range=0.2,  # Fraction of total height for vertical shift
        shear_range=0.2,         # Shear Intensity
        zoom_range=0.2,          # Range for random zoom
        horizontal_flip=True,    # Randomly flip inputs horizontally
        fill_mode='nearest'      # Strategy to fill newly created pixels
    )

    batch_size = 32

    augmented_training_data = augmented_data_gen.flow(training_images, training_labels, batch_size=batch_size)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(augmented_training_data, steps_per_epoch=len(training_images) // batch_size, validation_data=(validation_images, validation_labels), epochs=100, callbacks=[early_stopping])

    # Save the model with timestamp
    print("💾 Saving model...")

    timestamp = datetime.datetime.now().strftime("%H%M%S-%d%m%Y")
    model_name = f"model-{timestamp}"

    if not os.path.exists("research_data/models"):
        os.mkdir("research_data/models")

    model.save(f"research_data/models/{model_name}.keras")

def evaluate_model(model_path: pathlib.Path):
    '''Evaluates a model'''
    # Load the model
    model = models.load_model(model_path)

    # Load and preprocess images and labels
    print("📷 Preparing images and labels...")

    # Split the data into training and testing sets
    print("📊 Preparing test and training datasets...")
    _training_images, _training_labels, validation_images, validation_labels = db.get_training_data(validation_data_only=True)

    # Get predictions on validation images
    print("🦾 Getting predictions...")
    raw_predictions = model.predict(validation_images)
    predictions = np.argmax(raw_predictions, axis=1).astype('int32')

    validation_labels = np.argmax(validation_labels, axis=1).astype('int32')

    print(f"Predictions type: {predictions.dtype}, Labels type: {validation_labels.dtype}")

    # Use evaluation library, example: accuracy.compute(references=[0,1,0,1], predictions=[1,0,0,1])
    print("📈 Evaluating model...")
    accuracy = evaluate.load("accuracy")
    accuracy_results = accuracy.compute(references=validation_labels, predictions=predictions)
    print(accuracy_results)

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <init|seed|backup|purge>")
        os._exit(1)

    command = sys.argv[1]

    # Check args for 'plot' to plot the model
    if command == 'plot':
        model = get_model()
        make_model_diagram(model)
    elif command == 'train':
        train()
    elif command == 'evaluate':
        model_path = sys.argv[2]

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            os._exit(1)

        evaluate_model(pathlib.Path(model_path))
    else:
        print(f"Unknown command: {command}")
        os._exit(1)
