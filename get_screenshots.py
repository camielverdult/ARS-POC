import platform
import os
import multiprocessing
import pathlib
import time

import psutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common import exceptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions

# from train import INPUT_SIZE
INPUT_SIZE = 256
import db

# This scales the screenshots to k times the size of the CNN model input size
# Pictures are in square format, so the width and height are the same
# This will increase file size, but might also provide better quality downsampled images for the CNN
WINDOW_SIZE = INPUT_SIZE * 4

# Number of browsers to run in parallel per CPU thread
BROWSERS_PER_CORE = 1.5

def get_full_path(path: pathlib.Path) -> str:
    return str(path.absolute())

def setup_driver():
    # Configure webdriver options for Firefox
    firefox_options = FirefoxOptions()

    firefox_options.add_argument("--headless")  # Run in headless mode
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Firefox(options=firefox_options)

    # print("Adding extensions to browser...")
    driver.install_addon("browser_extensions/i_dont_care_about_cookies-3.5.0.xpi")
    driver.install_addon("browser_extensions/uBlock0_1.54.1b6.firefox.signed.xpi")
    driver.set_window_size(WINDOW_SIZE, WINDOW_SIZE)
    
    return driver

def calc_process_count():
    # Get the number of CPU cores
    cpu_cores = psutil.cpu_count(logical=True)

    # Get the total memory and calculate the available memory (considering 70% usage)
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
    available_memory = total_memory * 0.8  # Assuming 85% of memory is available for processes

    # Estimate memory usage per process (in GB) - adjust based on your observation
    memory_per_process = 1.5

    # Calculate the number of processes based on CPU cores and memory
    processes_based_on_cpu = cpu_cores
    processes_based_on_memory = int(available_memory / memory_per_process)

    # Return the minimum of the two calculations as the optimal process count
    return min(processes_based_on_cpu, processes_based_on_memory)

def get_screenshot_path_for_domain(domain: str) -> str:
    domain_name = domain.replace('.', '-')
    return f"screenshots/{domain_name}.png"

def take_screenshots(domain_ids: list):
    driver = setup_driver()
    conn = db.get_conn()
    cur = conn.cursor()

    session_id = db.get_latest_session_id(cur)

    # Get both domain_id and domain
    query = "SELECT domain_id, domain FROM domains WHERE domain_id IN ({})".format(','.join('?' * len(domain_ids)))
    cur.execute(query, domain_ids)

    # Map domain_id to domain
    domain_mapping = {row[0]: row[1] for row in cur.fetchall()}

    # Exception handling for the process running this function 
    try:
        for domain_id, domain_url in domain_mapping.items():
            filename = get_screenshot_path_for_domain(domain_url)
            ignored = False

            start_time = time.time()
            exception_type = None

            # Exception handling for the domain being processed
            try:
                driver.get(f"https://{domain_url}")
                time.sleep(4)

                # Check for 404 on website title or content
                if "404" in driver.title or "404 Not found" in driver.page_source:
                    print(f"🚫 {domain_url} (404)")
                    ignored = True
                    exception_type = "404"
                elif "403" in driver.title or "403 Forbidden" in driver.page_source:
                    print(f"🚫 {domain_url} (403)")
                    ignored = True
                    exception_type = "403"
                # Check if the domain is blocked by the uBlock extension
                elif "Page blocked" in driver.title:
                    print(f"🚫 {domain_url} (uBlocked)")
                    ignored = True
                    exception_type = "ublock"
                else:
                    try:
                        # Destroy abnoxious popups
                        driver.find_element(By.ID, "dialog-close").click()
                        time.sleep(1)
                    except:
                        pass

                    # Scroll to the top of the page
                    driver.execute_script("window.scrollTo(0, 0);")
                    time.sleep(1)

                    print(f"📷 {domain_url}")
                    driver.save_screenshot(filename)
            except exceptions.TimeoutException:
                filename = None  # Set filename to None if screenshot failed
                exception_type = "timeout" # Set exception type to timeout
            except exceptions.WebDriverException as e:
                error = str(e).lower()
                # print(f"{type(e)}.{e}")
                if "timeout" in error:
                    filename = None 
                    exception_type = "timeout"
                    print(f"🌐 {domain_url} (timeout error)")
                elif "dnsNotFound" in error:
                    filename = None
                    ignored = True # Ignore this domain in the future
                    exception_type = "dnsNotFound"
                    print(f"🤖 {domain_url} (ignored in future)")
                elif "neterror" in error:
                    filename = None
                    exception_type = "neterror"
                    print(f"🌐 {domain_url} (network error)")
                elif "InsecureCertificateError" in error:
                    filename = None
                    ignored = True
                    exception_type = "insecure"
                    print(f"🤔 {domain_url} (insecure, ignored in future)")
                else:
                    # Error taking screenshot for yandex.net: <class 'selenium.common.exceptions.WebDriverException'>
                    print(f"Error taking screenshot for {domain_url}: {type(e)}\n{e}")
                    exception_type = "unknown"
                    filename = None
            except Exception as e:
                # Some other exception
                exception_type = "unknown"
                print(f"Unexpected error when taking screenshot for {domain_url}: {e}")

            if ignored:
                filename = None

            cur.execute(
                "UPDATE domains SET screenshot = ?, ignored = ? WHERE domain_id = ?",
                (filename, ignored, domain_id)
            )

            end_time = time.time()

            # Insert screenshot process metrics
            cur.execute(
                "INSERT INTO metrics (domain_id, start_time, end_time, exception) VALUES (?, ?, ?, ?)",
                (domain_id, start_time, end_time, exception_type)
            )

            cur.execute(
                "INSERT INTO metrics_sessions (session_id, domain_id) VALUES (?, ?)",
                (session_id, domain_id)
            )

            conn.commit()
    except Exception as e:
        print(f"Error in Process: {e}")
    finally:
        driver.quit()
        cur.close()
        conn.close()

def chunker(seq, size):
    # Yield successive chunks of size from seq
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def screenshot_domains():
    if not os.path.exists(db.DB_FILENAME):
        db.seed()
    else:
        db.init_db()

    # Create session with given hardware info
    session_id = db.start_session(device_info=platform.platform())

    # Get start time for benchmarking
    start_time = time.time()

    # Fetch domain IDs that are not ignored
    domain_ids, description = db.get_unprocessed_domains()

    if not domain_ids: # <=> len(domain_ids) == 0
        print("🎉 All domains have already been processed!")
        return
    
    # Create a directory for screenshots
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    # Define the number of processes and chunk size for multiprocessing, based on the number of cores in the system
    # The number of processes will not exceed over 16 due to memory constraints
    num_processes = calc_process_count() # TODO: find upper limit of num_processes
    chunk_size = len(domain_ids) // num_processes + (len(domain_ids) % num_processes > 0)

    # Use multiprocessing to take screenshots
    print(f"🚀 Visiting {description} domains ({len(domain_ids)}) with {num_processes} processes...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        for _ in pool.imap_unordered(take_screenshots, chunker(domain_ids, chunk_size)):
            pass

    db.end_session(session_id, 1)

    # Get end time for benchmarking
    end_time = time.time()
    total_time = end_time - start_time

    # Get metrics for the session
    session_metrics = db.get_session_metrics(session_id)
    session_id = session_metrics["session_id"]
    num_pictures = session_metrics["num_pictures"]
    virtual_time = session_metrics["virtual_time"]
    skipped_domains = session_metrics["skipped_domains"]
    avg_duration = session_metrics["avg_duration"]
    time_lost = session_metrics["time_lost"]

    # Print capturing results
    print(f"🕑 Took {total_time:.1f} real & {virtual_time:.1f} virtual seconds to take {num_pictures} screenshots ({skipped_domains} skipped, {avg_duration:.1f} avg. threaded sec/pic, {time_lost:.1f} threaded seconds lost on exceptions)")

def main():
    screenshot_domains()

if __name__ == "__main__":
    main()
