# Step 1: Use a specific Python 3.9 image as our base
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file into the container
COPY requirements.txt .

# Step 4: Install the CPU-only version of PyTorch first
RUN pip install torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Step 5: Install the rest of the dependencies
RUN pip install -r requirements.txt

# Step 6: Copy the rest of your backend code into the container
COPY . .

# Step 7: Expose the port the app will run on
EXPOSE 8000

# Step 8: Define the command to run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]