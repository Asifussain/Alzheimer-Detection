# Stage 1: Define the build environment
FROM python:3.11.9-slim as builder

# Set the working directory
WORKDIR /app

# Install build tools
RUN pip install --upgrade pip setuptools wheel cython

# Copy requirements and install dependencies
COPY backend/requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# Stage 2: Define the final, lean production environment
FROM python:3.11.9-slim

WORKDIR /app

# Copy the pre-built wheels from the builder stage
COPY --from=builder /app/wheels /wheels/

# Install the dependencies from the wheels
RUN pip install --no-cache /wheels/*

# Copy the backend application code into the container
COPY backend/ .

# Expose the port the app runs on
EXPOSE 5000

# Set the command to run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]