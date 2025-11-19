# 🚀 Railway Deployment Guide

This guide will help you deploy your Credit Risk Application on Railway.

## 📋 Prerequisites

- [Railway Account](https://railway.app/) (free tier available)
- GitHub repository with your code
- Railway CLI (optional but recommended)

## 🏗️ Application Architecture

Your application is deployed as a **single service** that includes:
- **Web Service**: FastAPI application (main interface)
- **ML Model**: Integrated prediction service (direct inference)
- **Redis**: Optional message queue (for local development with separate services)

## 🔧 Pre-deployment Setup

### 1. Environment Variables
The application uses the following environment variables (automatically configured by Railway):

```env
# Redis Configuration (Auto-configured by Railway)
REDIS_URL=redis://...
REDIS_QUEUE=service_queue

# Application Settings
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
API_SLEEP=0.5
SERVER_SLEEP=0.1
```

### 2. File Structure
```
project/
├── railway.json          # Railway configuration
├── src/
│   ├── app/              # Web application
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app.py
│   ├── models/           # ML model service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── ml_model.py
│   └── docker-compose.yml
└── env.example           # Environment variables example
```

## 🚀 Deployment Steps

### Method 1: GitHub Integration (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Railway deployment ready"
   git push origin main
   ```

2. **Create Railway Project**
   - Go to [railway.app](https://railway.app/)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure Services**
   Railway will automatically detect the `railway.json` file and deploy:
   - Single combined service (FastAPI app + ML model)

4. **Set Environment Variables**
   Railway will automatically configure most variables. Required variables:
   - `SECRET_KEY`: Generate a secure secret key for JWT tokens
   - `ALGORITHM`: JWT algorithm (default: HS256)
   - `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration (default: 60)
   - `API_SLEEP`: Sleep time between requests (default: 0.5)

### Method 2: Railway CLI

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**
   ```bash
   railway login
   ```

3. **Initialize Project**
   ```bash
   railway init
   ```

4. **Deploy**
   ```bash
   railway up
   ```

## 🔍 Monitoring and Troubleshooting

### Health Checks
Both services include health checks:
- **Web Service**: HTTP endpoint check
- **Model Service**: Redis connection check

### Logs
Monitor logs in Railway dashboard:
- Web service logs: Application requests and responses
- Model service logs: ML prediction processing
- Redis logs: Queue operations

### Common Issues

1. **Redis Connection Errors**
   - Ensure Redis service is running
   - Check environment variables

2. **Model Loading Issues**
   - Verify `predict_model.pkl` exists in models directory
   - Check model dependencies in requirements.txt

3. **Port Configuration**
   - Railway automatically assigns ports
   - Don't hardcode port numbers

## 📊 Application URLs

After deployment, you'll get:
- **Main Application**: `https://your-app-name.railway.app`
- **Login**: `https://your-app-name.railway.app/login`
- **API Docs**: `https://your-app-name.railway.app/docs`

## 🔐 Default Credentials

- **Username**: `anyoneai`
- **Password**: `secret`

## 🎯 Testing Your Deployment

1. **Access the Application**
   ```bash
   curl https://your-app-name.railway.app/
   ```

2. **Test Health Check**
   ```bash
   curl https://your-app-name.railway.app/health
   ```

3. **Test Login**
   - Navigate to `/login`
   - Use default credentials
   - Try making a prediction

## 📈 Scaling and Performance

Railway automatically handles:
- **Auto-scaling**: Based on traffic
- **Load balancing**: For high availability
- **Monitoring**: Built-in metrics

## 💰 Cost Optimization

- **Free Tier**: 512MB RAM, 1GB storage
- **Pro Plan**: $5/month for higher limits
- **Resource Monitoring**: Track usage in dashboard

## 🛠️ Maintenance

### Updates
```bash
git push origin main  # Automatic deployment
```

### Environment Variables
Update in Railway dashboard under "Variables"

### Logs
Monitor in Railway dashboard under "Logs"

## 🆘 Support

- **Railway Docs**: [docs.railway.app](https://docs.railway.app/)
- **Railway Discord**: [discord.gg/railway](https://discord.gg/railway)
- **GitHub Issues**: For application-specific issues

## 🎉 Success!

Your Credit Risk Application is now running on Railway! 🚀 