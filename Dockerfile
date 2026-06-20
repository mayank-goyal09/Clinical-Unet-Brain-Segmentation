# Use a lightweight official Python runtime
FROM python:3.10-slim

# Install system packages needed by OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000 for Hugging Face compatibility
RUN useradd -m -u 1000 user

# Set the working directory
WORKDIR /code

# Copy and install dependencies first to leverage Docker layer caching
COPY --chown=user:user requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files (model, clean data, and dashboard code)
COPY --chown=user:user . /code

# Switch to the non-root user
USER user

# Set home directory environment variable
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port 7860 (Hugging Face expects this port)
EXPOSE 7860

# Command to run FastAPI server binding to 0.0.0.0 on port 7860
CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "7860"]
