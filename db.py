import os
import shutil
import sqlite3
import sys
import time
import random

import numpy as np
import pandas as pd
from PIL import Image
import pathlib

DB_FILENAME = 'screenshots.sqlite3'
DOMAIN_AMOUNT = 110000
VALIDATION_AMOUNT = 0.2
INPUT_SIZE = 512

# These domains triggered MalwareBytes and contain Riskware, Trojans, Malvertising, are compromised, etc.
# AKA bad stuff that we don't want to visit
BAD_DOMAINS = ["e7z9t4x6a0v5mk3zo1a0xj2z7c6g8sa6js5z7s2c3h9x0s5fh3a6sjwb8q7m.xyz", "onlyindianx.cc", "tokyomotion.net", "heylink.me", "tamilyogi.plus", "zcswet.com", "bidmachine.io", "doodstream.io", "1024tera.com", "mobile-tracker-free.com", "yadongtube.net", "vlxx.moe", "ai-lawandorder.com", "worldfcdn2.com", "edgesuite.net"]

def get_conn(filename=None):
    # Open a thread-safe connection to the database
    # to prevent: 'database is locked' errors
    if filename:
        return sqlite3.connect(filename, check_same_thread=False)
    
    return sqlite3.connect(DB_FILENAME, check_same_thread=False)

def get_count(table_name: str, cur: sqlite3.Cursor = None, select: str = None, where: str = None, error_callback_fn = None) -> int:
    """Returns the amount of rows in a table"""

    # Allow for row operation in select
    if not select:
        select = "*"

    # Format the WHERE clause
    if where:
        where = f" WHERE {where}"
    else:
        where = ""

    # Track if we started the cursor
    cur_started = False

    # Use the passed cursor or create a new one
    if not cur:
        conn = get_conn()
        cur = conn.cursor()
        cur_started = True

    # Use the passed error callback or use a default
    if not error_callback_fn:
        def _error_callback(error: str):
            print(f"🚨 Error getting {table_name} count: {error}")
            os._exit(1)
        error_callback_fn = _error_callback

    # Run the query
    cur.execute(f"SELECT COUNT({select}) FROM {table_name}{where}")
    count = cur.fetchone()

    if not count:
        # No rows in the table
        count = -1
        error_callback_fn(table_name, "COUNT(*) returned None")
    else:
        # Get the first column of the first row from the result
        count = count[0]

    if cur_started:
        cur.close()
        conn.close()

    return count

def start_session(device_info: str) -> int:
    """Starts a new session and returns the session ID"""
    # Connect to the database
    conn = get_conn()
    cur = conn.cursor()

    # Get the session ID and device info
    cur.execute("INSERT INTO sessions (start_time, device_info) VALUES (datetime('now'), ?)", (device_info,))
    conn.commit()

    # Get the session ID
    session_id = cur.lastrowid

    cur.close()
    conn.close()

    return session_id

def get_latest_session_id(cur: sqlite3.Cursor = None) -> int:
    """Returns the latest session ID"""
    # Connect to the database
    cur_started = False
    if not cur:
        conn = get_conn()
        cur = conn.cursor()
        cur_started = True

    # Get the session ID and device info
    cur.execute("SELECT MAX(session_id) FROM sessions")

    # Get the session ID
    session_id = cur.fetchone()[0]

    if cur_started:
        cur.close()
        conn.close()

    return session_id

def insert_metric(domain_id: int, start_time: float, end_time: float, exception: str|None, session_id: int, cur: sqlite3.Cursor = None, conn: sqlite3.Connection = None) -> int:
    """Inserts a metric and returns the metric ID"""
    # Connect to the database
    cur_started = False
    if not cur:
        conn = get_conn()
        cur = conn.cursor()
        cur_started = True

    # Get the session ID and device info
    cur.execute("INSERT INTO metrics (domain_id, start_time, end_time, exception) VALUES (?, ?, ?, ?)", (domain_id, start_time, end_time, exception))
    cur.execute("INSERT INTO metrics_sessions (session_id, metric_id) VALUES (?, ?)", (session_id, cur.lastrowid))
    conn.commit()

    # Get the session ID
    metric_id = cur.lastrowid

    if cur_started:
        cur.close()
        conn.close()

    return metric_id

def get_session_metrics(session_id: int) -> dict:
    """Returns the session data including various metrics for the specific session."""

    # Connect to the database
    conn = get_conn()
    cur = conn.cursor()

    # Get the session start time
    session_start_time = cur.execute(f"SELECT start_time FROM sessions WHERE session_id = {session_id}").fetchone()
    if not session_start_time:
        print(f"🚨 Error getting session metrics: session start time is None for session ID {session_id}!")
        os._exit(1)

    session_start_time = session_start_time[0]

    # Construct the query to fetch all metrics for the specific session
    query = f"""
        SELECT 
            COUNT(DISTINCT CASE WHEN d.screenshot IS NOT NULL THEN m.metric_id END) AS num_pictures,
            SUM(CASE WHEN m.exception IS NULL THEN m.end_time - m.start_time END) AS virtual_time,
            COUNT(CASE WHEN m.exception IS NOT NULL THEN m.metric_id END) AS skipped_domains,
            AVG(CASE WHEN m.exception IS NULL THEN m.end_time - m.start_time END) AS avg_duration,
            SUM(CASE WHEN m.exception IS NOT NULL THEN m.end_time - m.start_time END) AS time_lost
        FROM metrics m
        INNER JOIN metrics_sessions ms ON m.metric_id = ms.metric_id
        INNER JOIN domains d ON m.domain_id = d.domain_id
        WHERE ms.session_id = {session_id}
    """

    cur.execute(query)
    num_pictures, virtual_time, skipped_domains, avg_duration, time_lost = cur.fetchone()

    # Check for none and assign default value of 0
    num_pictures = num_pictures or 0
    virtual_time = virtual_time or 0
    skipped_domains = skipped_domains or 0
    avg_duration = avg_duration or 0
    time_lost = time_lost or 0

    cur.close()
    conn.close()

    return {
        "session_id": session_id,
        "num_pictures": num_pictures,
        "virtual_time": virtual_time,
        "skipped_domains": skipped_domains,
        "avg_duration": avg_duration,
        "time_lost": time_lost
    }

def end_session(session_id: int, bandwidth_used: int):
    """Ends a session"""
    # Connect to the database
    conn = get_conn()
    cur = conn.cursor()

    # Get the session ID and device info
    cur.execute("UPDATE sessions SET end_time = datetime('now'), bandwidth_used = ? WHERE session_id = ?", (bandwidth_used, session_id))

    conn.commit()
    cur.close()
    conn.close()

def init_db(filename: str = None):
    conn = get_conn(filename)
    cur = conn.cursor()

    # Create domains table
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
    ''')

    # Create index on domain_id column in labels table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_labels_domain_id ON labels(domain_id)
    ''')

    # Create index on topic_id column in labels table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_topic_id ON labels(topic_id)
    ''')

    # Create table for screenshot (multi)processing metrics
    cur.execute('''
        CREATE TABLE IF NOT EXISTS metrics
        (metric_id INTEGER PRIMARY KEY,
         domain_id INTEGER, 
         start_time TEXT,
         end_time TEXT,
         exception TEXT,
         FOREIGN KEY(domain_id) REFERENCES domains(domain_id))
    ''')

    # JOIN optimalization: Create index on domain_id column in metrics table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_domain_id ON metrics(domain_id)
    ''')

    # Create index on start_time column being NULL in metrics table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_start_time ON metrics(start_time);
    ''')

    # Create a tabel that logs each program session
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sessions
        (session_id INTEGER PRIMARY KEY, 
         start_time TEXT,
         end_time TEXT,
         device_info TEXT,
         bandwidth_used INTEGER
        )
    ''')

    conn.commit()

    # Get new cursor
    cur = conn.cursor()

    # JOIN optimalization: Create index on session_id column in metrics table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON sessions(session_id)
    ''')

    # JOIN optimalization: Create index on session_id column in metrics table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON metrics(session_id)
    ''')

    # Tabel for joining metrics and sessions
    cur.execute('''
        CREATE TABLE IF NOT EXISTS metrics_sessions
        (session_id INTEGER, 
         metric_id INTEGER,
         FOREIGN KEY(session_id) REFERENCES sessions(session_id),
         FOREIGN KEY(metric_id) REFERENCES metrics(metric_id))
    ''')

    # JOIN optimalization: Create index on session_id column in metrics table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON metrics(session_id)
    ''')

    # JOIN optimalization: Create index on domain_id column in metrics table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_domain_id ON metrics(domain_id)
    ''')

    # Model data table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS model_data
        (model_data_id INTEGER PRIMARY KEY,
            model_name TEXT,
            metric_session_id INTEGER,
            timestamp TEXT,
            FOREIGN KEY(metric_session_id) REFERENCES metrics_sessions(metric_id))
    ''')

    # Model training results
    cur.execute('''
        CREATE TABLE IF NOT EXISTS model_results
        (model_result_id INTEGER PRIMARY KEY,
            model_data_id INTEGER,
            test_loss REAL,
            test_accuracy REAL,
            FOREIGN KEY(model_data_id) REFERENCES model_data(model_data_id))
    ''')

    conn.commit()
    conn.close()

def seed_domains_and_labels():
    domain_ranking_directory = pathlib.Path("top_websites_by_country")
    files = [x for x in domain_ranking_directory.glob("*.csv")]

    df_list = []
    for file in files:
        print(f"📝 Inserting domains from {file.name} into the database...")
        df = pd.read_csv(file, delimiter=',', on_bad_lines='warn')
        df_list.append(df)

    merged_df = pd.concat(df_list)

    # Remove entries that have an empty categories column
    merged_df = merged_df.dropna(subset=['categories'])

    # Removing duplicates on the domain entry in the dataframe
    distinct_df = merged_df.drop_duplicates(subset=['domain'])

    # Remove CDN domains
    distinct_df = distinct_df[~distinct_df['domain'].str.contains("cdn", na=False)]

    # Remove entries that have an inappropriate category
    distinct_df = distinct_df[~distinct_df['categories'].str.contains("Porn", na=False)]

    # Convert string of categories seperated by ; into list of category strings
    distinct_df['categories'] = distinct_df['categories'].str.split(';')

    # Converting the distinct DataFrame to a list of dictionaries
    distinct_domains = distinct_df.to_dict(orient='records')

    # Remove urls that are in BAD_DOMAINS
    distinct_domains = [domain for domain in distinct_domains if domain['domain'] not in BAD_DOMAINS]

    # Get all categories from the distinct domains
    categories = []
    for domain in distinct_domains:
        categories.extend(domain['categories'])

    # Remove duplicates from the categories list
    distinct_categories = list(set(categories))

    conn = get_conn()
    cur = conn.cursor()

    # Insert all domains into the domains table
    for domain in distinct_domains:
        cur.execute("INSERT OR IGNORE INTO domains (domain, ranking) VALUES (?, ?)", (domain['domain'], domain['rank']))

    conn.commit()

    # Insert the categories into the topics table
    for category_id, category in enumerate(distinct_categories):
        cur.execute("INSERT OR IGNORE INTO topics (topic_id, name) VALUES (?, ?)", (category_id, category))

    # Prepare distinct_domains for insertion into the labels table by replacing the categories with their index
    for domain in distinct_domains:
        domain['categories'] = [distinct_categories.index(category) for category in domain['categories']]

    # Insert the labels into the labels table
    for domain in distinct_domains:
        domain_url = domain['domain']
        category_ids = domain['categories']
        domain_id = get_domain_id(domain_url, cur)
        for category_id in category_ids:
            cur.execute("INSERT INTO labels (domain_id, topic_id) VALUES (?, ?)", (domain_id, category_id))

    conn.commit()
    cur.close()
    conn.close()

def seed():
    init_db()
    seed_domains_and_labels()

def backup():
    # Define backup directory
    backup_dir = 'db-backups'

    # Create the backup directory if it does not exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # Define the backup filename
    backup_filename = os.path.join(backup_dir, f'{time.time()}.{DB_FILENAME}_backup')

    print(f"💾 Backing up {DB_FILENAME} to {backup_filename}...")

    # Copy the file to the backup location
    shutil.copyfile(DB_FILENAME, backup_filename)

def purge():
    if os.path.exists(DB_FILENAME):
        backup()

        conn = get_conn()
        cur = conn.cursor()

        print("🗑️ Purging database... (keeping succesful domains!)")

        # Drop all tables but keep metrics without an exception
        cur.execute("DROP TABLE IF EXISTS topics")
        cur.execute("DROP TABLE IF EXISTS labels")

        # Delete all metrics without an exception
        cur.execute("DELETE FROM metrics WHERE exception IS NULL")

        # Delete domains that don't have an exception
        cur.execute("DELETE FROM domains WHERE domain_id NOT IN (SELECT domain_id FROM metrics)")

        # Drop indexes
        cur.execute("DROP INDEX IF EXISTS idx_domain_id")
        cur.execute("DROP INDEX IF EXISTS idx_topic_id")
        cur.execute("DROP INDEX IF EXISTS idx_start_time_null")

        conn.commit()
        cur.close()
        conn.close()

    # Backup the screenshots directory
    if os.path.exists('screenshots'):
        backup_dir = 'screenshots-backups'

        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)

        print(f"💾 Backing up screenshots to {backup_dir}...")
        shutil.copytree('screenshots', backup_dir)
        
        print("🗑️ Purging screenshots directory...")
        shutil.rmtree('screenshots')

def get_domain_id(domain, open_cur: sqlite3.Cursor = None) -> int:
    if not open_cur:
        conn = get_conn()
        cur = conn.cursor()
    else:    
        cur = open_cur

    # Get the domain ID
    cur.execute(f"SELECT domain_id FROM domains WHERE domain = '{domain}'")
    domain_id = cur.fetchone()

    if not domain_id:
        print(f"🚨 Error getting domain ID for {domain}!")
        os._exit(1)

    if not open_cur:
        cur.close()
        conn.close()

    return domain_id[0]

def get_labels(open_cur: sqlite3.Cursor = None) -> list:
    """Returns the domain_id with its corresponding labels from the database"""
    if not open_cur:
        conn = get_conn()
        cur = conn.cursor()
    else:
        cur = open_cur

    # There might be multiple topics per domain, so we need to group them into a list
    sql_query = f'''
        SELECT domain_id, GROUP_CONCAT(topic_id)
        FROM labels
        GROUP BY domain_id
    '''

    cur.execute(sql_query)

    rows = cur.fetchall()
    
    # rows now looks like this:
    # domain_id, topic_ids
    # [
    #     (0, '1,2,3'),
    #     (1, '4,5,6'),
    #     (2, '7,8,9')
    # ]

    if not open_cur:
        cur.close()
        conn.close()

    # Convert the concatenated topic_id string to a list of integers
    rows = [(row[0], [int(topic_id) for topic_id in row[1].split(',')]) for row in rows]

    return rows

def get_screenshot_path_for_domain(domain: str) -> str:
    domain_name = domain.replace('.', '-')
    return f"screenshots/{domain_name}.png"

def get_unprocessed_domains() -> list[int]:
    """Returns a list of domain IDs that have not been screenshotted yet."""
    # Connect to the database
    conn = get_conn()
    cur = conn.cursor()

    # Check if we have an empty metrics table, because then we need to process all domains
    metrics_count = get_count('metrics', cur)

    # If the metrics table is empty, we need to process all domains
    if metrics_count == 0:
        # Get all domains
        cur.execute(f"""
            SELECT domain_id, domain FROM domains
        """)

    else:    
        """This part is ran when we have already processed domains before"""

        # Get all domains that have no exception
        cur.execute("""
            SELECT DISTINCT d.domain_id, d.domain
            FROM domains d
            INNER JOIN metrics m ON d.domain_id = m.domain_id
            WHERE (m.exception IS NULL)
        """)

    unprocessed_domain_ids = cur.fetchall()

    domains_to_process = []

    latest_session_id = get_latest_session_id(cur)

    # Remove domains that have a screenshot file
    for domain_id, domain_url in unprocessed_domain_ids:
        filename = get_screenshot_path_for_domain(domain_url)
        if os.path.exists(filename):
            metric = cur.execute(f"SELECT metric_id FROM metrics WHERE domain_id = {domain_id}").fetchone()
            screenshot_set = cur.execute(f"SELECT screenshot FROM domains WHERE domain_id = {domain_id}").fetchone()
            # Domain already has a screenshot, so we only need to add a metric if it does not exist
            if not metric:
                insert_metric(domain_id, None, None, None, latest_session_id, cur, conn)

            if not screenshot_set:
                cur.execute(
                    "UPDATE domains SET screenshot = ? WHERE domain_id = ?",
                    (filename, domain_id)
                )
        else:
            # Domain will be screenshotted and a metric will be added
            domains_to_process.append(domain_id)

    # We now have a list of new domain IDs that have not been captured yet
    conn.commit()

    cur.close()
    conn.close()

    return domains_to_process

def get_succesful_domain_ids() -> list[int]:
    """Returns the succesful domains from the latest screenshot session"""
    conn = get_conn()
    cur = conn.cursor()

    # Verify tables that we join in the query are not empty
    domains_count = get_count('domains', cur)

    if domains_count == 0:
        print("🚨 Missing domains or labels in the database!")
        os._exit(1)

    # Get metrics for each domain without exception from last session
    sql_query = '''
        SELECT DISTINCT d.domain_id
        FROM domains d
        INNER JOIN metrics AS m ON d.domain_id = m.domain_id
        WHERE m.exception IS NULL
        ORDER BY d.ranking ASC
    '''

    cur.execute(sql_query)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [row[0] for row in rows]

def randomise_use_in_training(training_data_amount: float = float(1 - VALIDATION_AMOUNT)):
    """Checks if the succesful domains->used_in_training values have been divided into train and validate data and if not, sets them proportionally randomly."""
    conn = get_conn()
    cur = conn.cursor()

    # Check if the domains table is empty
    domains_count = get_count('domains', cur)

    if domains_count == 0:
        print("🚨 Missing domains in the database!")
        os._exit(1)

    # Check if the current amount of domains used in training is correct
    domains_used_in_training_count = get_count('domains', cur, select='*', where='used_in_training IS TRUE')

    # Calculate amount of domains without exceptions to update
    human_domains = get_succesful_domain_ids()
    domains_to_update_amount = int(len(human_domains) * training_data_amount)

    # The current amount of domains used in training is already set, we're done here
    if domains_used_in_training_count == domains_to_update_amount:
        print(f"✅ Correct amount of domains used in training already set to {training_data_amount*100:.0f}%!")
        return

    # Get a random sample of domains to update to use in training
    ids_to_update = random.sample(human_domains, domains_to_update_amount)

    # Update the domains to be used in training
    print(f"📊 Randomly updating {training_data_amount*100:.0f}% of domains to be used in training...")
    sql = "UPDATE domains SET used_in_training = TRUE WHERE domain_id IN ("
    for id_to_exclude in ids_to_update:
        sql += f"{id_to_exclude},"
    sql = sql[:-1] + ")"

    cur.execute(sql)

    # Commit the changes to the database
    conn.commit()

def get_succesful_domain_urls() -> list[str]:
    domain_ids = get_succesful_domain_ids()

    conn = get_conn()
    cur = conn.cursor()

    # Get URLs from domain IDs
    sql_query = f'''
        SELECT domain
        FROM domains
        WHERE domain_id IN ({','.join(map(str, domain_ids))})
    '''

    cur.execute(sql_query)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [row[0] for row in rows]

def get_topic_id_to_name_mapping() -> dict:
    """Returns a dictionary that maps topic IDs to their names"""
    conn = get_conn()
    cur = conn.cursor()

    # Get the topic IDs and names
    cur.execute("SELECT topic_id, name FROM topics")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {row[0]: row[1] for row in rows}

def get_training_data(limit: int = None, validation_data_only = False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
        Returns the screenshot image data and labels/topics for all labeled domains. Only succesful domains are used in training/validation. The labels are returned as a binary vector.
        @param limit: The amount of domains to return
        @return: (training_images, training_labels, validation_images, validation_labels)
    """
    conn = get_conn()
    cur = conn.cursor()

    labels_count = get_count('labels', cur)
    if labels_count == 0:
        print("🚨 Missing labels in the database!")
        os._exit(1)

    extra_premise = ""

    if validation_data_only:
        extra_premise = "AND d.used_in_training IS FALSE"

    sql_query = f'''
        SELECT d.screenshot, GROUP_CONCAT(l.topic_id), d.used_in_training
        FROM domains d
        INNER JOIN labels AS l ON d.domain_id = l.domain_id
        INNER JOIN metrics AS m ON d.domain_id = m.domain_id
        WHERE m.exception IS NULL {extra_premise}
        GROUP BY d.domain_id
        ORDER BY d.ranking DESC
    '''

    # Limit the amount of domains to the specified limit
    if limit:
        sql_query += f" LIMIT {limit}"

    cur.execute(sql_query)

    rows = cur.fetchall()
    
    # rows now looks like this:
    #      path, topic_ids, used_in_training
    # [
    #     ('screenshots/1.png', '1,2,3', 1),
    #     ('screenshots/2.png', '4,5,6', 1),
    #     ('screenshots/3.png', '7,8,9', 0)
    # ]

    cur.close()
    conn.close()

    # Convert the concatenated topic_id string to a list of integers
    rows = [(row[0], [int(topic_id) for topic_id in row[1].split(',')], bool(row[2])) for row in rows]

    # Get the total number of topics
    topic_count = get_count('topics')

    # These are the training and validation data
    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []

    # Initialize a binary label vector for each row, that would look like this for 12 labels:
    # [ 
    #    ('screenshots/1.png', [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),
    #    ('screenshots/2.png', [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
    # ]
    # Also sort the data into training and validation data
    for screenshot_path, topic_ids, used_in_training in rows:
        # Load the image
        with Image.open(screenshot_path, 'r') as image:
            rgb_image = image.convert('RGB')
            resized_image = rgb_image.resize((INPUT_SIZE, INPUT_SIZE))
            image_data = np.array(resized_image)

        binary_label = [0] * topic_count  # Initialize a binary label vector with all zeros
        for topic_id in topic_ids:
            if topic_id <= topic_count:  # Ensure the topic_id is within the range of topics
                binary_label[topic_id - 1] = 1  # Set to 1 at the index corresponding to the topic_id

        match used_in_training:
            case True:
                training_images.append(image_data)
                training_labels.append(binary_label)
            case False:
                validation_images.append(image_data)
                validation_labels.append(binary_label)
            case _:
                # This should never happen
                raise Exception("Used in training is not a boolean!")
            
    # Convert lists to numpy arrays
    return (np.array(training_images), np.array(training_labels), np.array(validation_images), np.array(validation_labels))

if __name__ == '__main__':
    seed()
    os._exit(0)
    # Check arguments
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <init|seed|backup|purge>")
        os._exit(1)

    # Parse arguments
    command = sys.argv[1]

    # Run command
    if command == 'init':
        init_db()
    elif command == 'seed':
        seed()
    elif command == 'backup':
        backup()
    elif command == 'purge':
        purge()
    else:
        print(f"Unknown command: {command}")
        os._exit(1)