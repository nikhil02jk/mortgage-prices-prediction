# Mortgage Prices Prediction API

This repository contains the code for a Flask-based API to predict mortgage prices using machine learning models. The API is deployed using **Cloud Run** and the Docker image is stored in **Artifact Registry**.

## Project Structure

- `flask-app/`: Contains the main Flask application and Dockerfile.
- `cloudbuild.yaml`: Cloud Build configuration for CI/CD pipeline, including deployment to Cloud Run.
- `requirements.txt`: Python dependencies for the Flask app.
- `README.md`: Project documentation.

## Installation

To get started with the project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nikhil02jk/mortgage-prices-prediction.git
   cd mortgage-prices-prediction

2. **Create a virtual environment:**
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. **Install dependencies:**
    pip install -r requirements.txt

4. **Run the Flask app locally:**
    python flask-app/app.py


**Cloud Deployment**
This project is set up for CI/CD using Cloud Build and Cloud Run. Upon pushing to the main branch, a build is automatically triggered and deployed to Cloud Run.

Steps for deployment:
The cloudbuild.yaml file contains the configuration for building the Docker image and deploying it to Cloud Run.

The Docker image is stored in Artifact Registry and used for the deployment on Cloud Run.

Logs are stored in Google Cloud Storage.


