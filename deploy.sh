#!/bin/bash

# GCP Deployment Script for LSTM Sentiment API
# Minimal cost deployment to Cloud Run

# Set your GCP project ID
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
SERVICE_NAME="lstm-sentiment-api"

echo "🚀 Deploying LSTM Sentiment API to GCP Cloud Run..."

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📦 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
echo "🔨 Building and deploying..."
gcloud builds submit --config cloudbuild.yaml

# Get the service URL
echo "✅ Deployment complete!"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo ""
echo "🌐 Your API is live at: $SERVICE_URL"
echo "📚 API Documentation: $SERVICE_URL/docs"
echo "❤️  Health Check: $SERVICE_URL/health"
echo ""
echo "💰 Cost optimization tips:"
echo "   - Free tier: 2M requests/month"
echo "   - Pay only when serving requests"
echo "   - Min instances = 0 (no idle costs)"
echo ""
