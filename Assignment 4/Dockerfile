# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .
COPY score.py .
COPY model.pkl .
COPY vectorizer.pkl .

# Expose the port
EXPOSE 5001

# Run the app
CMD ["python", "app.py"]