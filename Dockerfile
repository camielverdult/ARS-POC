# Use the TensorFlow image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY pip-requirements.txt /app/
RUN pip install --no-cache-dir -r pip-requirements.txt

# Copy the rest of your application's code
COPY . /app

# Command to run when starting the container
CMD ["python", "/app/train.py"]
