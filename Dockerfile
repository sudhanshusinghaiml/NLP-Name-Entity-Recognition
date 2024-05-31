# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.10.14
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Setting up Working directory
WORKDIR /app

# Copy the source code into the container.
COPY . /app

# To signify root directory of application
RUN touch /app/.project-root

# Run requirements.txt to install all dependencies
RUN pip install -r requirements.txt

# Expose the port that the application listens on.
EXPOSE 8080

# Run the application.
CMD uvicorn 'app:NameEntityRecognitionApp' --host=0.0.0.0 --port=8080
