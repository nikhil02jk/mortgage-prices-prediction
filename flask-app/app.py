from flask import Flask, request, jsonify
import pandas as pd
import os
import sys
import logging
import cmdstanpy
cmdstanpy.cmdstan_path()
from cmdstanpy import cmdstan_path, set_cmdstan_path
from google.cloud import storage
import uuid
import io
from flask_cors import CORS




import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot




# Set environment variables and configure logging
os.environ["PROPHET_BACKEND"] = "cmdstanpy"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing Prophet with error handling
try:
    from prophet import Prophet

    logger.info("Prophet imported successfully")
except Exception as e:
    logger.error(f"Failed to import Prophet: {str(e)}")
    sys.exit(1)

import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from matplotlib.ticker import FuncFormatter

# Set up the GCS client and bucket name
client = storage.Client()  # Uses default credentials from environment variable
bucket_name = "bigdataproject_model"  # Replace with your GCS bucket name
bucket = client.bucket(bucket_name)

# Initialize Flask app
app = Flask(__name__)
CORS(app)


def money_formatter(x, pos):
    """Format y-axis labels as currency"""
    return f'${x:,.0f}'


# Load dataset (ensure the CSV file is included in your container's build context)
try:
    data_cleaned = pd.read_csv('data/Merged file 2.csv').dropna(subset=['StateName', 'RegionName', 'HouseType'])
    logger.info(f"Data loaded successfully with {len(data_cleaned)} rows")
except Exception as e:
    logger.error(f"Failed to load data: {str(e)}")
    data_cleaned = None

# Define the color palette and aesthetic style
COLORS = {
    'historical': '#1f77b4',  # Blue
    'forecast': '#ff7f0e',  # Orange
    'background': '#f5f5f5',  # Light gray
    'grid': '#dcdcdc',  # Gray
    'highlight': '#2ca02c',  # Green
    'text': '#333333'  # Dark gray
}


# Home page endpoint
@app.route("/", methods=["GET"])
def home():
    return """
    <html>
      <head>
        <title>Housing Price Forecast API</title>
      </head>
      <body style="font-family: Arial, sans-serif; margin: 2em;">
        <h1>Welcome to the Housing Price Forecast API</h1>
        <p>This API provides housing price forecasts based on historical data.</p>
        <h2>Available Endpoints:</h2>
        <ul>
          <li>
            <strong>Home</strong>: <code>GET /</code> — This page.
          </li>
          <li>
            <strong>Health</strong>: <code>GET /health</code> — Check service status.
          </li>
          <li>
            <strong>Forecast</strong>: <code>POST /forecast</code> — Submit parameters and get forecasts.
          </li>
        </ul>
        <h3>Example Request for Forecast</h3>
        <pre>
{
  "state": "CO",
  "region": "Boulder, CO",
  "house_type": "Single Family",
  "start_date": "2024-07-31",
  "prediction_months": 12
}
        </pre>
        <p>Use a tool like <a href="https://www.postman.com/">Postman</a> or <code>curl</code> to send requests.</p>
      </body>
    </html>
    """


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if data_cleaned is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": data_cleaned is not None,
        "prophet_backend": os.environ.get("PROPHET_BACKEND", "unknown")
    }), 200 if data_cleaned is not None else 500


# Forecasting endpoint
@app.route("/forecast", methods=["POST"])
def forecast_endpoint():
    if data_cleaned is None:
        return jsonify({"error": "Data not loaded. Service is not ready."}), 503

    try:
        logger.info("Received forecast request")
        req_data = request.get_json()
        logger.info(f"Request data: {req_data}")

        state = req_data.get("state")
        region = req_data.get("region")
        house_type = req_data.get("house_type")
        start_date = req_data.get("start_date")
        prediction_months = req_data.get("prediction_months")

        if not all([state, region, house_type, prediction_months]):
            return jsonify({"error": "Missing parameters"}), 400

        filtered_data = data_cleaned[
            (data_cleaned['StateName'] == state) &
            (data_cleaned['RegionName'] == region) &
            (data_cleaned['HouseType'] == house_type)
            ]
        if filtered_data.empty:
            return jsonify({"error": "No data found for the selected filters."}), 404

        logger.info(f"Filtered data: {len(filtered_data)} rows")

        price_data = filtered_data.iloc[:, 6:].T  # Transpose price data
        price_data = price_data.reset_index()
        price_data.columns = ["ds", "y"]
        price_data["ds"] = pd.to_datetime(price_data["ds"], errors="coerce")
        price_data["y"] = pd.to_numeric(price_data["y"], errors="coerce")
        price_data = price_data.dropna(subset=["ds", "y"])
        if price_data.empty:
            return jsonify({"error": "No valid data after cleaning."}), 400

        training_start_date = pd.to_datetime("2014-01-31")
        training_end_date = pd.to_datetime("2025-01-31")
        price_data_train = price_data[(price_data['ds'] >= training_start_date) &
                                      (price_data['ds'] <= training_end_date)]
        if price_data_train.empty:
            return jsonify({"error": "No data available in the fixed training period."}), 400

        logger.info(f"Training data: {len(price_data_train)} rows")

        # Configure Prophet with more conservative settings to avoid optimization issues
        model = Prophet(
            changepoint_prior_scale=0.05,  # Default is 0.05
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',  # More stable than multiplicative
            mcmc_samples=0,  # Disable MCMC sampling which can be problematic
        )

        logger.info("Fitting Prophet model")
        # Reduce number of iterations for Cloud Run compatibility
        # model.fit(price_data_train, iter=10000)
        model.fit(price_data_train,iter=10000)
        logger.info("Model fitting complete")

        future_start = pd.to_datetime("2025-02-28")
        future_dates = pd.DataFrame({
            'ds': pd.date_range(start=future_start, periods=prediction_months, freq='M')
        })

        logger.info("Making predictions")
        forecast = model.predict(future_dates)
        logger.info("Predictions complete")

        # Compute trend metrics
        historical_first_price = price_data_train['y'].iloc[0]
        historical_last_price = price_data_train['y'].iloc[-1]
        historical_growth_rate = ((historical_last_price / historical_first_price) - 1) * 100
        historical_trend = "Rising" if historical_growth_rate > 0 else "Falling"
        forecast_first_price = forecast['yhat'].iloc[0]
        forecast_last_price = forecast['yhat'].iloc[-1]
        forecast_change = ((forecast_last_price / forecast_first_price) - 1) * 100
        forecast_trend = "Rising" if forecast_change > 0 else "Falling"

        # Plot generation
        logger.info("Generating plot")
        plt.clf()
        plt.figure(figsize=(12, 7), dpi=120, facecolor=COLORS['background'])
        dynamic_start_date = pd.to_datetime(start_date)
        jan_2025 = pd.to_datetime("2025-01-31")
        price_data_dynamic = price_data[(price_data['ds'] >= dynamic_start_date) &
                                        (price_data['ds'] <= jan_2025)]
        ax = plt.gca()
        ax.set_facecolor(COLORS['background'])
        plt.plot(
            price_data_dynamic['ds'],
            price_data_dynamic['y'],
            marker='o',
            markersize=8,
            color=COLORS['historical'],
            label='Historical Prices',
            linewidth=2.5,
            alpha=0.9,
            zorder=5
        )
        plt.fill_between(
            price_data_dynamic['ds'],
            price_data_dynamic['y'].min() * 0.95,
            price_data_dynamic['y'],
            color=COLORS['historical'],
            alpha=0.15
        )
        plt.plot(
            forecast['ds'],
            forecast['yhat'],
            marker='D',
            markersize=7,
            color=COLORS['forecast'],
            label='Forecasted Prices',
            linewidth=2.5,
            linestyle='-',
            alpha=0.9,
            zorder=5
        )
        plt.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            color=COLORS['forecast'],
            alpha=0.15,
            label='95% Confidence Interval'
        )
        plt.axvline(
            x=jan_2025,
            color=COLORS['highlight'],
            linestyle='--',
            linewidth=2,
            label='Historical/Forecast Boundary',
            alpha=0.8,
            zorder=4
        )
        plt.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
        plt.xlabel('Month & Year', fontsize=12, fontweight='bold', color=COLORS['text'])
        plt.ylabel('Average Home Value', fontsize=12, fontweight='bold', color=COLORS['text'])
        title_text = f'Housing Price Trends & Forecast\n{house_type} in {region}, {state}'
        plt.title(title_text, fontsize=16, fontweight='bold', pad=20, color=COLORS['text'])
        ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
        all_dates = pd.concat([price_data_dynamic['ds'], forecast['ds']])
        date_min = all_dates.min()
        date_max = all_dates.max()
        xticks = pd.date_range(start=date_min, end=date_max, freq='M')[::2]
        xticklabels = xticks.strftime('%b %Y')
        plt.xticks(xticks, labels=xticklabels, rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='upper left', frameon=True, framealpha=0.95, facecolor='white', edgecolor=COLORS['grid'],
                   fontsize=10)

        if len(price_data_dynamic) > 1:
            first_price = price_data_dynamic['y'].iloc[0]
            last_price = price_data_dynamic['y'].iloc[-1]
            first_date = price_data_dynamic['ds'].iloc[0].strftime('%b-%Y')
            last_date = price_data_dynamic['ds'].iloc[-1].strftime('%b-%Y')
            growth_rate = ((last_price / first_price) - 1) * 100
            plt.annotate(
                f"Historical :\\${first_price:,.0f} ({first_date}) → \\${last_price:,.0f} ({last_date})\n{abs(growth_rate):.1f}% {'Rising' if growth_rate > 0 else 'Falling'}",
                xy=(price_data_dynamic['ds'].iloc[-1], last_price),
                xytext=(10, -30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=COLORS['text'], alpha=0.6),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS['grid'], alpha=0.8),
                fontsize=9
            )

        forecast_first_price = forecast['yhat'].iloc[0]
        forecast_last_price = forecast['yhat'].iloc[-1]
        forecast_first_date = forecast['ds'].iloc[0].strftime('%b-%Y')
        forecast_last_date = forecast['ds'].iloc[-1].strftime('%b-%Y')
        forecast_change = ((forecast_last_price / forecast_first_price) - 1) * 100
        plt.annotate(
            f"Forecast:\\${forecast_first_price:,.0f} ({forecast_first_date}) → \\${forecast_last_price:,.0f} ({forecast_last_date})\n{abs(forecast_change):.1f}% {'Rising' if forecast_change > 0 else 'Falling'}",
            xy=(forecast['ds'].iloc[-1], forecast['yhat'].iloc[-1]),
            xytext=(-120, 30),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=COLORS['forecast'], alpha=0.6),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS['grid'], alpha=0.8),
            fontsize=9
        )

        plt.figtext(
            0.99, 0.01,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
            ha='right',
            fontsize=8,
            color='gray'
        )

        date_range = date_max - date_min
        padding = date_range * 0.03
        plt.xlim(date_min - padding, date_max + padding)
        y_min = min(price_data_dynamic['y'].min(), forecast['yhat_lower'].min())
        y_max = max(price_data_dynamic['y'].max(), forecast['yhat_upper'].max())
        y_range = y_max - y_min
        plt.ylim(y_min - y_range * 0.05, y_max + y_range * 0.1)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Generate a unique filename for the plot
        plot_filename = f"housing_forecast_{str(uuid.uuid4())}.png"

        # Upload the plot to GCS
        blob = bucket.blob(plot_filename)
        blob.upload_from_file(buf, content_type='image/png')

        # Get the public URL of the uploaded plot
        plot_url = f"https://storage.googleapis.com/{bucket_name}/{plot_filename}"

        logger.info("Creating response")
        response_data = {
            "forecast": {row['ds'].strftime('%b %Y'): row['yhat'] for _, row in forecast.iterrows()},
            "historical_trend": historical_trend,
            "forecast_trend": forecast_trend,
            "plot": plot_url
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
