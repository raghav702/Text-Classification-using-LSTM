# GCP Deployment Guide - Minimal Cost Setup

Deploy your LSTM Sentiment Analysis API on Google Cloud Platform with minimal costs.

## 💰 Cost Breakdown

### Cloud Run (Recommended)
- **Free Tier**: 2M requests/month, 180K vCPU-seconds, 360K GiB-seconds
- **After Free Tier**: ~$0.0000025 per request
- **Expected Cost**: $0-5/month for low to moderate traffic
- **Scaling**: Auto-scales to zero (no idle costs)

### Alternative: Compute Engine f1-micro
- **Free Tier**: 1 f1-micro instance/month (US regions only)
- **Specs**: 0.6 GB RAM, 1 vCPU (shared)
- **Expected Cost**: FREE in free tier, ~$5/month after
- **Note**: Always running, even with no traffic

## 🚀 Quick Deployment Steps

### Prerequisites

1. **Install Google Cloud SDK**
   ```powershell
   # Download from: https://cloud.google.com/sdk/docs/install
   # Or use PowerShell:
   (New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
   & $env:Temp\GoogleCloudSDKInstaller.exe
   ```

2. **Authenticate**
   ```powershell
   gcloud auth login
   gcloud auth configure-docker
   ```

3. **Create GCP Project** (if needed)
   ```powershell
   gcloud projects create your-project-id --name="LSTM Sentiment API"
   gcloud config set project your-project-id
   ```

### Deploy to Cloud Run

1. **Update deployment script**
   ```powershell
   # Edit deploy.ps1 and set your PROJECT_ID
   notepad deploy.ps1
   ```

2. **Run deployment**
   ```powershell
   .\deploy.ps1
   ```

   Or manually:
   ```powershell
   # Set project
   gcloud config set project your-project-id

   # Enable APIs
   gcloud services enable cloudbuild.googleapis.com run.googleapis.com

   # Deploy
   gcloud builds submit --config cloudbuild.yaml
   ```

3. **Access your API**
   - Your API URL will be displayed after deployment
   - Format: `https://lstm-sentiment-api-xxxxx-uc.a.run.app`
   - Documentation: Add `/docs` to the URL
   - Health check: Add `/health` to the URL

## 📊 Cost Optimization Tips

### 1. Use Cloud Run (Not Always-On VMs)
- Scales to zero when not in use
- Pay only for actual request processing time
- Free tier covers most personal/demo projects

### 2. Optimize Container Size
- Use slim Python image (already configured)
- Remove unnecessary files with `.dockerignore`
- Keep only the best model, delete checkpoints

### 3. Configure Resource Limits
```powershell
gcloud run deploy lstm-sentiment-api \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 3 \
  --min-instances 0 \
  --timeout 60
```

### 4. Enable Request Caching
The API already has caching configured in `api_config.yaml`

### 5. Set Up Budget Alerts
```powershell
# Create budget alert at $5
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="LSTM API Budget" \
  --budget-amount=5.00
```

## 🔧 Configuration

### Adjust Memory/CPU (in cloudbuild.yaml)
```yaml
args:
  - '--memory'
  - '2Gi'        # Reduce to 1Gi for lower cost
  - '--cpu'
  - '1'          # Reduce to 0.5 for lower cost
```

### Reduce Timeout (faster = cheaper)
```yaml
args:
  - '--timeout'
  - '60'         # Reduce if predictions are fast
```

### Limit Concurrent Requests
```powershell
gcloud run services update lstm-sentiment-api \
  --concurrency 80 \
  --max-instances 3
```

## 🧪 Testing After Deployment

```powershell
# Get your service URL
$SERVICE_URL = gcloud run services describe lstm-sentiment-api \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'

# Test health endpoint
curl "$SERVICE_URL/health"

# Test prediction
curl -X POST "$SERVICE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'

# View API docs in browser
start "$SERVICE_URL/docs"
```

## 📈 Monitoring Costs

### View Current Usage
```powershell
# Cloud Run metrics
gcloud run services describe lstm-sentiment-api \
  --platform managed \
  --region us-central1

# Billing report
gcloud billing accounts list
```

### Check Logs
```powershell
gcloud logs read \
  --project=your-project-id \
  --limit=50 \
  --format=json
```

## 🎯 Alternative: Compute Engine Free Tier

If you prefer always-on VM (still minimal cost):

### Create f1-micro Instance
```powershell
gcloud compute instances create lstm-api-vm \
  --zone=us-central1-a \
  --machine-type=f1-micro \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=10GB \
  --tags=http-server,https-server
```

### Deploy Application
```powershell
# SSH into instance
gcloud compute ssh lstm-api-vm --zone=us-central1-a

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip git
git clone https://github.com/your-repo/lstm-sentiment.git
cd lstm-sentiment
pip3 install -r requirements.txt -r requirements_api.txt

# Run with screen/tmux
screen -S api
python3 run_api.py
# Press Ctrl+A, then D to detach
```

### Setup Firewall
```powershell
gcloud compute firewall-rules create allow-http \
  --allow tcp:8000 \
  --target-tags http-server
```

## 🛡️ Security Best Practices

1. **Enable Authentication** (for production)
   ```powershell
   gcloud run services update lstm-sentiment-api \
     --no-allow-unauthenticated
   ```

2. **Add Rate Limiting** (already configured in API)

3. **Use Secret Manager** for sensitive configs
   ```powershell
   echo -n "api-key" | gcloud secrets create api-key --data-file=-
   ```

4. **Enable Cloud Armor** (if needed)

## 📞 Support & Troubleshooting

### View Logs
```powershell
gcloud run logs tail lstm-sentiment-api --region us-central1
```

### Common Issues

**Build Fails**
- Check Dockerfile syntax
- Ensure all dependencies in requirements.txt
- Verify model files exist

**High Costs**
- Check `--min-instances` is set to 0
- Reduce memory/CPU allocation
- Add request caching
- Set max-instances limit

**Slow Cold Starts**
- Optimize container size
- Use lighter model
- Consider min-instances=1 (adds cost)

## 🎉 Success Checklist

- [ ] Google Cloud SDK installed
- [ ] GCP project created and billing enabled
- [ ] Updated `deploy.ps1` with your PROJECT_ID
- [ ] Model files included in repository
- [ ] Ran `.\deploy.ps1` successfully
- [ ] Tested `/health` endpoint
- [ ] Tested `/predict` endpoint
- [ ] Checked API documentation at `/docs`
- [ ] Set up budget alerts

## 📚 Additional Resources

- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Cloud Run Quickstart](https://cloud.google.com/run/docs/quickstarts)
- [Free Tier Details](https://cloud.google.com/free)
- [Cost Calculator](https://cloud.google.com/products/calculator)

---

**Estimated Total Cost**: $0-5/month with proper configuration! 🎉
