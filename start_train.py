import platform

def run_script():
    os_name = platform.system()

    if os_name == 'Windows':
        # Command to run on Windows using Docker
        import subprocess
        command = 'docker-compose up tensorflow'
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")

    elif os_name == 'Darwin':
        # Call the function directly on macOS
        import train
        train.main()

    else:
        raise ValueError(f"Unsupported operating system: {os_name}")

if __name__ == '__main__':
    run_script()
