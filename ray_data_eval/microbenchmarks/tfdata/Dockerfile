# Use an official lightweight Python image.
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR .

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "hello", "/bin/bash", "-c"]

# Make RUN commands use the new environment:
RUN echo "Make sure conda is in the path" 

# Make port 80 available to the world outside this container
EXPOSE 80   

# Run app.py when the container launches
# CMD ["conda", "run", "-n", "hello", "python", "producer_consumer_microbenchmark.py"]
