# Use the exact Python version you specified
FROM python:3.11.9-slim

# Prevent Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install SUMO and necessary system dependencies
RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    sumo-doc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set SUMO_HOME environment variable (Critical for TraCI to work)
ENV SUMO_HOME=/usr/share/sumo

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code into the container
COPY . .

# Expose the port your FastAPI dashboard runs on
EXPOSE 8000

# Command to run the FastAPI server (assuming you use uvicorn)
CMD ["uvicorn", "dashboard_server:app", "--host", "0.0.0.0", "--port", "8000"]