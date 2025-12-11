# GCP Deployment Checklist - Minimal Cost Setup

Follow this step-by-step checklist to deploy your LSTM Sentiment Analysis app on Google Cloud Platform with minimal costs.

## ✅ Pre-Deployment Checklist

### 1. Prerequisites Setup
- [ ] **Google Cloud SDK Installed**
  - Download: https://cloud.google.com/sdk/docs/install
  - Verify: `gcloud --version`

- [ ] **GCP Account Created**
  - Free trial: $300 credits for 90 days
  - URL: https://console.cloud.google.com

- [ ] **Docker Installed** (optional, for local testing)
  - Download: https://www.docker.com/products/docker-desktop
  - Verify: `docker --version`

- [ ] **Python Dependencies Installed**
  ```powershell
  pip install -r requirements.txt
  pip install -r requirements_api.txt
  ```

### 2. Local Testing
- [ ] **Test the API locally**
  ```powershell
  python run_api.py
  ```
  - Open: http://localhost:8000/docs
  - Test: `/health` endpoint
  - Test: `/predict` endpoint with sample text

- [ ] **Test the Web UI locally** (optional)
  ```powershell
  streamlit run app.py
  ```

- [ ] **Verify model files exist**
  - [ ] `models/improved_lstm_model_20251106_003134.pth`
  - [ ] `models/improved_lstm_model_20251106_003134_vocabulary.pth`

## 🚀 GCP Setup & Deployment

### 3. Initial GCP Configuration

- [ ] **Authenticate with GCP**
  ```powershell
  gcloud auth login
  gcloud auth configure-docker
  ```

- [ ] **Create GCP Project**
  ```powershell
  # Replace 'your-unique-project-id' with your actual project ID
  gcloud projects create your-unique-project-id --name="LSTM Sentiment API"
  ```

- [ ] **Set Default Project**
  ```powershell
  gcloud config set project your-unique-project-id
  ```

- [ ] **Enable Billing**
  - Go to: https://console.cloud.google.com/billing
  - Link billing account to your project
  - ⚠️ **Set up budget alerts** (recommended: $5-10/month)

- [ ] **Enable Required APIs**
  ```powershell
  gcloud services enable cloudbuild.googleapis.com
  gcloud services enable run.googleapis.com
  gcloud services enable containerregistry.googleapis.com
  ```

### 4. Deployment Configuration

- [ ] **Update deploy.ps1 with your Project ID**
  ```powershell
  notepad deploy.ps1
  # Change: $PROJECT_ID = "your-gcp-project-id"
  # To: $PROJECT_ID = "your-actual-project-id"
  ```

- [ ] **Review cloudbuild.yaml settings**
  - Memory: 2Gi (reduce to 1Gi for lower cost)
  - CPU: 1 (reduce to 0.5 for lower cost)
  - Max instances: 3 (reduce to 2 for lower cost)
  - Min instances: 0 (keep at 0 for no idle costs)

### 5. Deploy to Cloud Run

- [ ] **Run deployment script**
  ```powershell
  .\deploy.ps1
  ```
  
  **OR manually:**
  ```powershell
  gcloud builds submit --config cloudbuild.yaml
  ```

- [ ] **Wait for deployment** (5-10 minutes)
  - Watch build progress in terminal
  - Or monitor at: https://console.cloud.google.com/cloud-build

- [ ] **Note your Service URL**
  ```powershell
  gcloud run services describe lstm-sentiment-api --platform managed --region us-central1 --format 'value(status.url)'
  ```
  - Save this URL for testing

## 🧪 Post-Deployment Testing

### 6. Verify Deployment

- [ ] **Test Health Endpoint**
  ```powershell
  $SERVICE_URL = "YOUR_SERVICE_URL_HERE"
  curl "$SERVICE_URL/health"
  ```
  - Expected: `{"status": "healthy", ...}`

- [ ] **Test Prediction Endpoint**
  ```powershell
  curl -X POST "$SERVICE_URL/predict" -H "Content-Type: application/json" -d '{"text": "This movie was absolutely amazing!"}'
  ```
  - Expected: Sentiment prediction with confidence score

- [ ] **Check API Documentation**
  ```powershell
  start "$SERVICE_URL/docs"
  ```
  - Should open interactive Swagger UI

- [ ] **Test Batch Prediction** (optional)
  ```powershell
  curl -X POST "$SERVICE_URL/predict/batch" -H "Content-Type: application/json" -d '{"texts": ["Great movie!", "Terrible film."]}'
  ```

## 💰 Cost Optimization Setup

### 7. Configure Budget Alerts

- [ ] **Create Budget Alert**
  ```powershell
  # Go to: https://console.cloud.google.com/billing/budgets
  # Or use CLI:
  gcloud billing budgets create --billing-account=BILLING_ACCOUNT_ID --display-name="LSTM API Budget" --budget-amount=5
  ```

- [ ] **Set Alert Thresholds**
  - 50% of budget ($2.50)
  - 75% of budget ($3.75)
  - 100% of budget ($5.00)

### 8. Monitor & Optimize

- [ ] **Review Resource Allocation**
  ```powershell
  gcloud run services describe lstm-sentiment-api --region us-central1
  ```
  
  **Optimization options:**
  - If performance is good: Reduce memory to 1Gi
  - If rarely used: Set timeout to 60s (default: 300s)
  - If high traffic: Increase max-instances limit

- [ ] **Update Service Configuration** (if needed)
  ```powershell
  gcloud run services update lstm-sentiment-api --region us-central1 --memory 1Gi --cpu 1 --timeout 60 --max-instances 2
  ```

- [ ] **Enable Request Caching**
  - Already configured in `api_config.yaml`
  - Verify cache is working in logs

## 📊 Monitoring & Maintenance

### 9. Set Up Monitoring

- [ ] **View Service Logs**
  ```powershell
  gcloud run logs tail lstm-sentiment-api --region us-central1
  ```

- [ ] **Access Cloud Console Dashboard**
  - URL: https://console.cloud.google.com/run
  - Bookmark for regular monitoring

- [ ] **Check Metrics**
  - Request count
  - Request latency
  - Error rate
  - Container CPU/Memory usage

### 10. Cost Tracking

- [ ] **View Current Usage**
  ```powershell
  # Go to: https://console.cloud.google.com/billing
  ```

- [ ] **Weekly Cost Review** (recommended)
  - Monitor billing dashboard
  - Check for unexpected charges
  - Adjust resources if needed

## 🔒 Security & Best Practices

### 11. Security Hardening (Production)

- [ ] **Enable Authentication** (for production use)
  ```powershell
  gcloud run services update lstm-sentiment-api --no-allow-unauthenticated --region us-central1
  ```

- [ ] **Add Rate Limiting**
  - Already configured in API code
  - Verify it's working with load tests

- [ ] **Set Up Custom Domain** (optional)
  ```powershell
  gcloud beta run domain-mappings create --service lstm-sentiment-api --domain your-domain.com --region us-central1
  ```

- [ ] **Enable HTTPS** (automatic with Cloud Run)
  - Verify SSL certificate is active
  - Test with: `https://your-service-url`

### 12. Backup & Recovery

- [ ] **Tag Production Image**
  ```powershell
  docker tag gcr.io/PROJECT_ID/lstm-sentiment-api:latest gcr.io/PROJECT_ID/lstm-sentiment-api:production
  docker push gcr.io/PROJECT_ID/lstm-sentiment-api:production
  ```

- [ ] **Document Rollback Procedure**
  ```powershell
  # To rollback to previous version:
  gcloud run services update lstm-sentiment-api --image gcr.io/PROJECT_ID/lstm-sentiment-api:PREVIOUS_SHA --region us-central1
  ```

- [ ] **Backup Model Files**
  - Store in Google Cloud Storage (optional)
  - Keep local copies

## 📈 Advanced Features (Optional)

### 13. CI/CD Setup (Advanced)

- [ ] **Connect GitHub Repository**
  - Go to: https://console.cloud.google.com/cloud-build/triggers
  - Create trigger for automatic deployment on push

- [ ] **Configure Build Trigger**
  ```yaml
  # Automatically deploy on git push to main branch
  trigger: push to main
  build-config: cloudbuild.yaml
  ```

### 14. Load Testing

- [ ] **Test with Multiple Requests**
  ```powershell
  # Simple load test (requires Apache Bench or similar)
  ab -n 100 -c 10 -p test_request.json -T application/json "$SERVICE_URL/predict"
  ```

- [ ] **Monitor Performance Under Load**
  - Check response times
  - Verify auto-scaling works
  - Ensure no errors

## 🎯 Cost Optimization Checklist Summary

### Essential Cost-Saving Measures

✅ **Configured:**
- [x] Min instances: 0 (no idle costs)
- [x] Scale to zero when inactive
- [x] Request caching enabled
- [x] Optimized container size with .dockerignore

🔄 **Monitor & Adjust:**
- [ ] Review resource usage weekly
- [ ] Reduce memory if possible (2Gi → 1Gi)
- [ ] Reduce CPU if possible (1 → 0.5)
- [ ] Set realistic max-instances limit

💡 **Expected Monthly Cost:**
- **Low traffic** (<10K requests): **$0** (free tier)
- **Medium traffic** (10K-100K requests): **$0-2**
- **High traffic** (100K-500K requests): **$2-10**

## 📞 Troubleshooting

### Common Issues

**Build Fails:**
```powershell
# Check build logs
gcloud builds log $(gcloud builds list --limit=1 --format='value(id)')

# Common fixes:
# - Verify Dockerfile syntax
# - Check requirements.txt dependencies
# - Ensure model files are committed to repo
```

**Service Won't Start:**
```powershell
# Check service logs
gcloud run logs tail lstm-sentiment-api --region us-central1

# Common fixes:
# - Verify PORT environment variable
# - Check model file paths
# - Ensure all dependencies installed
```

**High Costs:**
```powershell
# Check current configuration
gcloud run services describe lstm-sentiment-api --region us-central1

# Reduce resources:
gcloud run services update lstm-sentiment-api --memory 1Gi --cpu 0.5 --max-instances 2 --region us-central1
```

**Cold Start Issues:**
```powershell
# If slow to respond after inactivity, consider:
# 1. Optimize model loading (lazy loading)
# 2. Use lighter model
# 3. Set min-instances to 1 (adds cost)
```

## ✨ Deployment Complete!

Once all checkboxes are complete, your LSTM Sentiment Analysis API is:
- ✅ Deployed on GCP Cloud Run
- ✅ Auto-scaling (0 to 3 instances)
- ✅ Cost-optimized (<$5/month expected)
- ✅ Production-ready with monitoring
- ✅ Secure with HTTPS
- ✅ Documented and maintainable

### Your Deployed Service:
- **URL:** `https://lstm-sentiment-api-xxxxx-uc.a.run.app`
- **Docs:** `https://lstm-sentiment-api-xxxxx-uc.a.run.app/docs`
- **Health:** `https://lstm-sentiment-api-xxxxx-uc.a.run.app/health`

### Quick Access Commands:
```powershell
# Service URL
gcloud run services describe lstm-sentiment-api --region us-central1 --format 'value(status.url)'

# View logs
gcloud run logs tail lstm-sentiment-api --region us-central1

# Update service
gcloud run services update lstm-sentiment-api --region us-central1 [OPTIONS]

# Delete service (if needed)
gcloud run services delete lstm-sentiment-api --region us-central1
```

---

**🎉 Congratulations!** Your AI model is now live on Google Cloud Platform!

**Need help?** 
- GCP Documentation: https://cloud.google.com/run/docs
- Cloud Run Pricing: https://cloud.google.com/run/pricing
- Support: https://cloud.google.com/support
