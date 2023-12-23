# Use the TensorFlow image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Add a volume pointing our working directory and set the working directory
VOLUME /app
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY pip-requirements.txt /app/
RUN pip install --no-cache-dir -r pip-requirements.txt

# Copy the rest of your application's code
# COPY . /app

# Install Firefox, wget, and Geckodriver for Selenium
RUN apt-get clean
RUN apt-get update --fix-missing
RUN apt-get install -y --no-install-recommends firefox wget
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.33.0/geckodriver-v0.33.0-linux64.tar.gz
RUN tar -xvzf geckodriver-v0.33.0-linux64.tar.gz
RUN chmod +x geckodriver
RUN mv geckodriver /usr/local/bin/
RUN rm geckodriver-v0.33.0-linux64.tar.gz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Command to run when starting the container
CMD ["python", "/app/main.py"]
