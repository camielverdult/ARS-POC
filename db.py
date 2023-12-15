import os
import shutil
import sqlite3
import sys
import time
import yaml

import pandas as pd

DB_FILENAME = 'screenshots.sqlite3'
DOMAIN_AMOUNT = 200

# These domains triggered MalwareBytes and contain Riskware, Trojans, Malvertising, are compromised, etc.
# AKA bad stuff that we don't want to visit
BAD_DOMAINS = ["e7z9t4x6a0v5mk3zo1a0xj2z7c6g8sa6js5z7s2c3h9x0s5fh3a6sjwb8q7m.xyz", "onlyindianx.cc", "tokyomotion.net", "heylink.me", "tamilyogi.plus", "zcswet.com", "bidmachine.io", "doodstream.io", "1024tera.com", "mobile-tracker-free.com", "yadongtube.net", "vlxx.moe", "ai-lawandorder.com", "worldfcdn2.com", "edgesuite.net"]

def get_conn(filename=None):
    if filename:
        return sqlite3.connect(filename)
    
    return sqlite3.connect(DB_FILENAME)

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

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Create domains table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS domains (
            domain_id INTEGER PRIMARY KEY,
            ranking INTEGER,
            domain TEXT,
            screenshot TEXT,
            ignored BOOLEAN DEFAULT FALSE
        )
    ''')

    # JOIN optimalization: Create index on domain_id column in domains table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_domains_domain_id ON domains(domain_id)
    ''')

    # Index on ignored column in domains table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_domains_ignored ON domains(ignored);
    ''')

    # Create topics table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS topics
        (topic_id INTEGER PRIMARY KEY, name TEXT)
    ''')

    # Create labels table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS labels
        (domain_id INTEGER, topic_id INTEGER,
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
        (domain_id INTEGER, 
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
         domain_id INTEGER,
         FOREIGN KEY(session_id) REFERENCES sessions(session_id),
         FOREIGN KEY(domain_id) REFERENCES domains(domain_id))
    ''')

    # JOIN optimalization: Create index on session_id column in metrics table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON metrics(session_id)
    ''')

    # JOIN optimalization: Create index on domain_id column in metrics table
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_domain_id ON metrics(domain_id)
    ''')
    
    conn.commit()
    conn.close()

def seed_domains():
    # Read domains from CSV file
    domains_df = pd.read_csv('tranco_N7VPW.csv', header=None)

    conn = get_conn()
    cur = conn.cursor()

    # Build SQL statement to bulk-insert domains with popularity ranking and domain URL keys into the database
    print(f"📝 Inserting first {DOMAIN_AMOUNT} domains into the database...")
    sql = "INSERT INTO domains (ranking, domain) VALUES "

    # Grab domains from the CSV file
    domains = domains_df[:DOMAIN_AMOUNT].values.tolist()

    # Add the domains to the SQL statement that are not in the BAD_DOMAINS list
    for domain in domains:
        if domain[1] not in BAD_DOMAINS:
            sql += f"({domain[0]}, '{domain[1]}'),"

    # Remove the trailing comma
    sql = sql[:-1]
    
    # Execute the SQL statement and check for errors
    try:
        cur.execute(sql)
    except sqlite3.Error as e:
        print(f"Error inserting domains: {e}")
        exit(1)

    # Commit the changes to the database
    conn.commit()

    cur.close()
    conn.close()

def seed_topics():
    # Connect to the database
    conn = get_conn()
    cur = conn.cursor()

    # Build SQL statement to bulk-insert topics into the database
    print("📝 Inserting topics into the database...")

    # Read the topics from the topics.yml file and convert to SQL statement
    with open('topics.yml', 'r') as file:
        lines = file.readlines()
        for line in lines:
            topic, topic_id = line.strip().split(':')
            topic = topic.strip()
            topic_id = topic_id.strip()
            sql = f"INSERT INTO topics (topic_id, name) VALUES ({topic_id}, '{topic}')"
            try:
                cur.execute(sql)
            except sqlite3.Error as e:
                print(f"Error inserting topics: {e}")
                exit(1)

    # Commit the changes to the database
    conn.commit()

    cur.close()
    conn.close()

def seed_labels():
    # Connect to the database
    conn = get_conn()
    cur = conn.cursor()

    # Check if topics and domains tables are empty
    topics_count = get_count('topics', cur)
    domains_count = get_count('domains', cur)

    if topics_count == 0 or domains_count == 0:
        print("🚨 Missing topics or domains in the database!")
        exit(1)

    # Build SQL statement to bulk-insert labels into the database
    print("📝 Inserting labels into the database...")

    # Read the labels from the labels.yml file and convert to SQL statement
    # The labels.yml file entry is structured like this:
    # mozilla.org: [2, 6]
    yaml_file = open('labels.yml')
    yaml_data = yaml.safe_load(yaml_file)
    for domain, topic_ids in yaml_data.items():
        for topic_id in topic_ids:
            # Insert each entry in the topics list as a label for the domain
            sql = f"INSERT INTO labels (domain_id, topic_id) VALUES ((SELECT domain_id FROM domains WHERE domain = '{domain}'), {topic_id})"
            try:
                cur.execute(sql)
            except sqlite3.Error as e:
                print(f"Error inserting labels: {e}")
                exit(1)

    # Commit the changes to the database
    conn.commit()

    cur.close()
    conn.close()

def seed():
    init_db()
    seed_domains()
    seed_topics()
    seed_labels()

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

        print("🗑️ Purging database... (keeping ignored domains!)")

        # Drop all tables but keep ignored metrics
        cur.execute("DROP TABLE IF EXISTS topics")
        cur.execute("DROP TABLE IF EXISTS labels")

        # Delete all metrics without an exception
        cur.execute("DELETE FROM metrics WHERE exception IS NULL")

        # Delete domains that don't have an exception
        cur.execute("DELETE FROM domains WHERE domain_id NOT IN (SELECT domain_id FROM metrics)")

        # Drop indexes
        cur.execute("DROP INDEX IF EXISTS idx_ignored")
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

def get_topics() -> list:
    """Returns a list of topics from the database"""
    conn = get_conn()
    cur = conn.cursor()

    # Get all topics
    cur.execute('SELECT * FROM topics')

    # Convert the topics to a list
    topics = cur.fetchall()

    cur.close()
    conn.close()

    return topics

def get_unprocssed_domains() -> (list[int], str):
    """Returns a list of domain IDs that have not been processed yet and are not ignored or have had an exception."""

    # Connect to the database
    conn = get_conn()
    cur = conn.cursor()

    # Check if we have an empty metrics table, because then we need to process all domains
    metrics_count = get_count('metrics', cur)

    # Check if the amount of metrics we have is less than the amount of domains we have
    # If so, we need to process all domains
    domains_count = get_count('domains', cur)

    description = ""

    # If the metrics table is empty, we need to process all domains
    if metrics_count == 0 or metrics_count < domains_count:
        # Get all domain IDs that are not ignored and not in the BAD_DOMAINS list
        sql_query = f"""
            SELECT domain_id FROM domains
        """
        description = "all"
    else:
        # Get all domain IDs that have not been processed yet and are not ignored or have had a timeout exception
        sql_query = f"""
            SELECT DISTINCT d.domain_id 
            FROM domains d
            LEFT JOIN metrics m ON d.domain_id = m.domain_id
            WHERE (m.start_time IS NULL OR m.start_time < datetime('now', '-1 day'))
            AND (d.ignored IS FALSE)
            AND (m.exception IS NULL)
        """
        description = "exceptionless"

    # Execute the query
    cur.execute(sql_query)
    unprocessed_domain_ids = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()

    return (unprocessed_domain_ids, description)

def get_training_data(limit: int = None) -> list: # list[(str, list[int])]
    """Returns the id, screenshot path and labels/topics for all labeled domains"""
    conn = get_conn()
    cur = conn.cursor()

    # There might be multiple topics per domain, so we need to group them into a list
    # We also need to ignore domains that:
    # - have been marked as ignored
    # - have not been processed yet
    # - have had an exception
    # There may be multiple metrics per domain, so we need to find the latest metric per domain
    # We want to collect multiple topics per domain, so we need to group them into a list

    # Verify tables that we join in the query are not empty
    domains_count = get_count('domains', cur)
    labels_count = get_count('labels', cur)

    if domains_count == 0 or labels_count == 0:
        print("🚨 Missing domains or labels in the database!")
        exit(1)

    # Get metrics for each domain without exception
    # Use the latest metrics table as a subquery to get the domain IDs
    sql_query = '''
        SELECT d.screenshot, GROUP_CONCAT(l.topic_id)
        FROM domains d
        INNER JOIN labels AS l ON d.domain_id = l.domain_id
        INNER JOIN metrics AS m ON d.domain_id = m.domain_id
        INNER JOIN (
            SELECT domain_id, MAX(start_time) AS latest_start_time
            FROM metrics
            GROUP BY domain_id
        ) subq ON d.domain_id = subq.domain_id AND m.start_time = subq.latest_start_time
        WHERE d.ignored IS FALSE
            AND m.exception IS NULL
        GROUP BY d.domain_id
        ORDER BY d.ranking DESC
    '''

    # Limit the amount of domains to the specified limit
    if limit:
        sql_query += f" LIMIT {limit}"

    cur.execute(sql_query)

    rows = cur.fetchall()
    
    # rows now looks like this:
    # [
    #     ('screenshots/1.png', '1,2,3'),
    #     ('screenshots/2.png', '4,5,6'),
    #     ('screenshots/3.png', '7,8,9')
    # ]


    rows = [(row[0], [int(topic_id) for topic_id in row[1].split(',')]) for row in rows]

    cur.close()
    conn.close()

    return rows

if __name__ == '__main__':
    # Check arguments
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <init|seed|backup|purge>")
        exit(1)

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
        exit(1)