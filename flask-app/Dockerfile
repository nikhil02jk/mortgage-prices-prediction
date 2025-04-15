# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first
COPY requirements.txt /app/

# Install dependencies (including the downgraded Prophet version)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir holidays==0.25
RUN pip install --no-cache-dir -r requirements.txt

# Install CmdStan (Fix CmdStanPy issues)
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# Copy the rest of the files
COPY . .

# Expose the correct port for Flask
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=app.py
ENV PORT=8080
ENV PROPHET_BACKEND=cmdstanpy

# Command to run the Flask app
CMD ["python", "app.py"]
