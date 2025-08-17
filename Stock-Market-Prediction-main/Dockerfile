# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Install the package
RUN pip install -e .

# Make port 8000 available
EXPOSE 8000

# Define environment variable
ENV NAME StockPrediction

# Run the application
CMD ["uvicorn", "stock_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 