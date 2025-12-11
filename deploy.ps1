# GCP Deployment Script for LSTM Sentiment API (PowerShell)
# Minimal cost deployment to Cloud Run

# Set your GCP project ID
$PROJECT_ID = "stately-arbor-479417-b8"
$REGION = "us-central1"
$SERVICE_NAME = "lstm-sentiment-api"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Get the service URL
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format "value(status.url)"

Write-Host ""
Write-Host "Your API is live at: $SERVICE_URL" -ForegroundColor Cyan
Write-Host "API Documentation: $SERVICE_URL/docs" -ForegroundColor Cyan
Write-Host "Health Check: $SERVICE_URL/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Cost optimization tips:" -ForegroundColor Yellow
Write-Host "  - Free tier: 2M requests/month"
Write-Host "  - Pay only when requests are served"
Write-Host "  - Min instances = 0 (no idle costs)"
Write-Host ""
