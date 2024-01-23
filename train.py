from tensorflow.keras import layers, models, callbacks, regularizers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall

from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(topic_mapping_count, activation='sigmoid')
    ])

    # For multi-label classification, binary accuracy is a suitable metric
    binary_accuracy = BinaryAccuracy(name='accuracy')

    # Precision and recall are also useful metrics
    precision = Precision(name='precision')
    recall = Recall(name='recall')

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=[binary_accuracy, precision, recall])

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

    # Make model name
    timestamp = datetime.datetime.now().strftime("%H%M%S-%d%m%Y")
    model_name = f"{timestamp}"
    model_dir = "research_data/models"

    if not os.path.exists("research_data/models"):
        os.mkdir("research_data/models")

    # make new folder for this training
    model_dir = os.path.join(model_dir, model_name)
    os.mkdir(model_dir)

    # Make model paths
    model_path = os.path.join(model_dir, "model.keras")
    best_model_path = os.path.join(model_dir, "best.keras")

    # Callback for saving the best model
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # Include checkpoint callback in the list
    callbacks_list = [early_stopping, checkpoint]

    history = model.fit(augmented_training_data, steps_per_epoch=len(training_images) // batch_size, validation_data=(validation_images, validation_labels), epochs=100, callbacks=callbacks_list)

    # Save the model with timestamp
    print("💾 Saving model...")

    model.save(model_path)

    # Get best model
    model = models.load_model(best_model_path)

    # This code was originally in the evaluate function
    raw_predictions = model.predict(validation_images)
    predictions = np.argmax(raw_predictions, axis=1).astype('int32')

    validation_labels = np.argmax(validation_labels, axis=1).astype('int32')

    print(f"Predictions type: {predictions.dtype}, Labels type: {validation_labels.dtype}")

    # Confusion Matrix
    cm = confusion_matrix(validation_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_filename = os.path.join(model_dir, 'confusion-matrix.png')
    plt.savefig(cm_filename)

    # Save Classification Report
    report = classification_report(validation_labels, predictions)
    print("Classification Report:\n", report)
    report_filename = os.path.join(model_dir, 'classification-report.txt')
    with open(report_filename, 'w') as file:
        file.write(report)

    # Save Performance Graphs
    def plot_history_key(history_key, title, ylabel, filename):
        plt.figure(figsize=(8, 6))
        plt.plot(history.history[history_key], label=f'Training {ylabel}')
        plt.plot(history.history[f'val_{history_key}'], label=f'Validation {ylabel}')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend(loc='bottom right')
        plot_filename = os.path.join(model_dir, f'{filename}.png')
        plt.savefig(plot_filename)

    # Accuracy Plot
    plot_history_key('accuracy', 'Model Accuracy', 'Accuracy', 'accuracy')

    # Loss Plot
    plot_history_key('loss', 'Model Loss', 'Loss', 'loss')

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <plot|train")
        os._exit(1)

    command = sys.argv[1]

    # Check args for 'plot' to plot the model
    if command == 'plot':
        model = get_model()
        make_model_diagram(model)
    elif command == 'train':
        train()
    else:
        print(f"Unknown command: {command}")
        os._exit(1)
