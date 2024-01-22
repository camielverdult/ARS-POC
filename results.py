import db
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

import seaborn as sns
from tensorflow.keras import models
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_screenshot_results():
    '''
    Plots the distribution between succesful screenshots and exceptions, 
    which we can use to support our hypothesis that a lot of domains are not
    appropiate for use in classification.
    '''

    # Load db conn
    conn = db.get_conn()

    # Get data
    cur = conn.cursor()

    """
    The exception column contains the following types of exceptions:

    neterror: 55 occurrences
    ublock: 22 occurrences
    timeout: 6 occurrences
    404 (Not Found): 3 occurrences
    403 (Forbidden): 1 occurrence

    Additionally, there are cases where there is no exception, 
    which can be considered successful processing. 
    Let's create a bar chart that shows the count of each exception type along
    with the count of successful processes (no exception). 
    This visualization will provide a clear overview of the distribution of 
    successes and different types of exceptions in your dataset.
    """

    total_count = cur.execute("""
        SELECT COUNT(*) FROM metrics
    """).fetchone()[0]

    # Count the successes (no exception) and each type of exception
    success_count = cur.execute("""
        SELECT COUNT(*) FROM metrics WHERE metrics.exception IS NULL
    """).fetchone()[0]

    exceptions_with_counts = cur.execute("""
        SELECT exception, COUNT(*) FROM metrics 
        WHERE metrics.exception IS NOT NULL GROUP BY metrics.exception
        ORDER BY COUNT(*) ASC
    """).fetchall()

    # Find the ratio of successes to total exceptions and per exception type
    success_ratio = success_count / total_count
    
    # Find the ratio of each exception type to total exceptions
    exception_ratios = [(exception, count / total_count) for exception, count in exceptions_with_counts]

    # Write the ratios to a csv file
    with open('research_data/screenshot_results.csv', 'w') as f:
        f.write('exception,exception_ratio\n')
        f.write(f'success,{success_ratio}\n')
        for exception, ratio in exception_ratios:
            f.write(f'{exception},{ratio}\n')

    # Close the database connection
    conn.close()

    # Convert the data to a pandas Series for easy plotting
    exception_counts = pd.Series(dict(exceptions_with_counts), name='Count')
    exception_counts['success'] = success_count

    # Plotting
    plt.figure(figsize=(10, 6))
    exception_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Successes and Exceptions')
    plt.xlabel('Outcome')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Remove padding around the figure and margins
    plt.tight_layout()

    # Make the text in the plot a bit bigger
    plt.rcParams.update({'font.size': 10})

    # Save the plot as a svg file
    plt.savefig('research_data/screenshot_results.svg')
    plt.savefig('research_data/screenshot_results.png')

def plot_label_distribution():
    conn = db.get_conn()
    cur = conn.cursor()

    """    # Create domains table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS domains (
            domain_id INTEGER PRIMARY KEY,
            ranking INTEGER,
            domain TEXT UNIQUE,
            screenshot TEXT,
            used_in_training BOOLEAN DEFAULT FALSE
        )
    ''')

    # JOIN optimalization: Create index on domain_id column in domains table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_domains_domain_id ON domains(domain_id)
    ''')

    # Create topics table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS topics
        (topic_id INTEGER PRIMARY KEY, name TEXT UNIQUE)
    ''')

    # Create labels table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS labels
        (domain_id INTEGER,
         topic_id INTEGER,
         FOREIGN KEY(domain_id) REFERENCES domains(domain_id),
         FOREIGN KEY(topic_id) REFERENCES topics(id))
    ''')"""

    # Get the label counts for succesful domains (metric not having an exception)
    label_counts = cur.execute("""
        SELECT topics.name, COUNT(*) FROM labels 
        JOIN metrics ON labels.domain_id = metrics.domain_id 
        JOIN topics ON labels.topic_id = topics.topic_id
        WHERE metrics.exception IS NULL
        GROUP BY topics.name
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """).fetchall()

    # Close the database connection
    conn.close()

    # Convert the data to a pandas Series for easy plotting
    label_counts = pd.Series(dict(label_counts), name='Count')

    # Plotting, making sure all of the topic names on the x-axis are readable in the plot
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of 10 most common labels')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.rcParams.update({'font.size': 10})
    plt.savefig('research_data/label_distribution.png')

def get_latest_model_dir():
    # Get the latest model directory
    model_dir = max(glob.glob('research_data/models/*'), key=os.path.getmtime)
    return model_dir

def make_confusion_matrix(model_dir):
    model_path = os.path.join(model_dir, "model.keras")
    best_model_path = os.path.join(model_dir, "best.keras")

    # Get best model
    model = models.load_model(best_model_path)

    class_names = db.get_topic_id_to_name_mapping()

    _, _, validation_images, validation_labels = db.get_training_data(validation_data_only=True)

    # This code was originally in the evaluate function
    raw_predictions = model.predict(validation_images)
    predictions = np.argmax(raw_predictions, axis=1).astype('int32')
    validation_labels = np.argmax(validation_labels, axis=1).astype('int32')

    validation_labels_named = np.vectorize(class_names.get)(validation_labels)
    predictions_named = np.vectorize(class_names.get)(predictions)

    # Confusion Matrix
    cm = confusion_matrix(validation_labels_named, predictions_named, labels=list(class_names.values()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names.values(), yticklabels=class_names.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Add diagonal line as a reference for a perfect classifier
    num_classes = len(class_names)
    plt.plot([-0.5, num_classes-0.5], [-0.5, num_classes-0.5], color='red', lw=2)

    plt.tight_layout()

    # Save confusion matrix
    cm_filename = os.path.join(model_dir, 'confusion-matrix.png')
    plt.savefig(cm_filename, dpi=300)

if __name__ == "__main__":
    make_confusion_matrix('research_data/models/big one')
    os._exit(0)
    plot_screenshot_results()
    plot_label_distribution()