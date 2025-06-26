# Start with an official, lightweight version of Python.
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the file that lists your dependencies
COPY requirements.txt .

# Install all the dependencies, including gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copy all of your application code into the container
COPY . .

# Tell Docker the container will listen on port 10000
EXPOSE 10000

# The command to run when the container starts.
# This will now work because gunicorn is installed.
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]