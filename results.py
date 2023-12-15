import db
import matplotlib.pyplot as plt
import pandas as pd

def plot_screenshot_results():
    '''
    Plots the distribution between succesful screenshots and exceptions, 
    which we can use to support our hypothesis that a lot of domains are not
    appropiate for use in classification.
    '''

    # Load db conn
    conn = db.get_conn("research_data/succesful_first_200.sqlite3")

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
    """).fetchall()

    # Find the ratio of successes to total exceptions and per exception type
    total_exceptions = sum([count for _, count in exceptions_with_counts])
    success_ratio = success_count / total_count
    
    # Find the ratio of each exception type to total exceptions
    exception_ratios = [(exception, count / total_count) for exception, count in exceptions_with_counts]

    # Write the ratios to a csv file
    with open('research_data/succesful_first_200.csv', 'w') as f:
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
    plt.savefig('research_data/succesful_first_200.svg')
    plt.savefig('research_data/succesful_first_200.png')

if __name__ == "__main__":
    plot_screenshot_results()