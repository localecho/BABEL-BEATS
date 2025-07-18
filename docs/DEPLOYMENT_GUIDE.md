# BABEL-BEATS Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Environment Configuration](#environment-configuration)
8. [Database Setup](#database-setup)
9. [Monitoring & Logging](#monitoring--logging)
10. [Troubleshooting](#troubleshooting)

## Overview

BABEL-BEATS is a full-stack application consisting of:
- **Backend**: FastAPI Python application with audio processing and music generation
- **Frontend**: HTML5/JavaScript web interface
- **Database**: PostgreSQL (production) or SQLite (development)
- **Cache**: Redis for session and file caching
- **Storage**: Local filesystem or cloud storage for generated audio files

## Prerequisites

### System Requirements
- Python 3.9+
- Node.js 16+ (for frontend build tools)
- Redis 6+
- PostgreSQL 13+ (for production)
- FFmpeg (for audio processing)
- 4GB RAM minimum
- 10GB disk space

### Required Dependencies
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv ffmpeg libsndfile1 redis-server postgresql

# For macOS
brew install python@3.9 ffmpeg libsndfile redis postgresql
```

## Local Development

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/babel-beats.git
cd babel-beats
```

### 2. Backend Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your configuration
nano .env
```

Required environment variables:
```env
# Application
ENVIRONMENT=development
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key

# Database
DATABASE_URL=sqlite:///./babel_beats.db  # For development
# DATABASE_URL=postgresql://user:password@localhost/babel_beats  # For production

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Audio Processing
MODEL_PATH=models/
SAMPLE_RATE=44100

# Storage
STORAGE_TYPE=local  # or 's3', 'gcs'
STORAGE_PATH=generated/

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

### 4. Database Setup
```bash
# Run migrations
alembic upgrade head

# Create initial data (optional)
python scripts/seed_database.py
```

### 5. Start Services
```bash
# Start Redis
redis-server

# Start backend (in another terminal)
source venv/bin/activate
python main.py

# Or use the enhanced version
python backend/enhanced_main.py
```

### 6. Frontend Setup
```bash
# Serve frontend (in another terminal)
cd frontend
python -m http.server 3000

# Or use a proper web server
npx serve .
```

Access the application at `http://localhost:3000`

## Docker Deployment

### 1. Create Dockerfile
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p generated logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/babel_beats
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - db
      - redis
    volumes:
      - ./generated:/app/generated
      - ./logs:/app/logs

  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./frontend:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend

  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=babel_beats
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 3. Nginx Configuration
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    upstream backend {
        server backend:8000;
    }

    server {
        listen 80;
        server_name localhost;

        # Frontend
        location / {
            root /usr/share/nginx/html;
            index index.html;
            try_files $uri $uri/ /index.html;
        }

        # API proxy
        location /api {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket support
        location /ws {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

### 4. Build and Run
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Kubernetes Deployment

### 1. Create Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: babel-beats
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: babel-beats-config
  namespace: babel-beats
data:
  ENVIRONMENT: "production"
  CORS_ORIGINS: "*"
  LOG_LEVEL: "INFO"
```

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: babel-beats-secret
  namespace: babel-beats
type: Opaque
stringData:
  DATABASE_URL: "postgresql://user:password@postgres:5432/babel_beats"
  REDIS_PASSWORD: "your-redis-password"
  SECRET_KEY: "your-secret-key"
```

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: babel-beats-backend
  namespace: babel-beats
spec:
  replicas: 3
  selector:
    matchLabels:
      app: babel-beats-backend
  template:
    metadata:
      labels:
        app: babel-beats-backend
    spec:
      containers:
      - name: backend
        image: babel-beats:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: babel-beats-config
        - secretRef:
            name: babel-beats-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: audio-storage
          mountPath: /app/generated
      volumes:
      - name: audio-storage
        persistentVolumeClaim:
          claimName: babel-beats-pvc
```

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: babel-beats-backend
  namespace: babel-beats
spec:
  selector:
    app: babel-beats-backend
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: babel-beats-ingress
  namespace: babel-beats
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - babel-beats.example.com
    secretName: babel-beats-tls
  rules:
  - host: babel-beats.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: babel-beats-backend
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: babel-beats-frontend
            port:
              number: 80
```

### 2. Deploy to Kubernetes
```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n babel-beats

# View logs
kubectl logs -f deployment/babel-beats-backend -n babel-beats

# Scale deployment
kubectl scale deployment babel-beats-backend --replicas=5 -n babel-beats
```

## Cloud Deployment

### AWS Deployment

#### 1. Using Elastic Beanstalk
```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init -p python-3.9 babel-beats

# Create environment
eb create babel-beats-prod

# Deploy
eb deploy

# Open in browser
eb open
```

#### 2. Using ECS with Fargate
```json
// task-definition.json
{
  "family": "babel-beats",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "babel-beats",
      "image": "your-ecr-repo/babel-beats:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:babel-beats-db"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/babel-beats",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Using App Engine
```yaml
# app.yaml
runtime: python39
entrypoint: gunicorn -b :$PORT main:app

env_variables:
  ENVIRONMENT: "production"

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.6

resources:
  cpu: 2
  memory_gb: 2.0
  disk_size_gb: 10

handlers:
- url: /static
  static_dir: static
- url: /.*
  script: auto
```

```bash
# Deploy to App Engine
gcloud app deploy

# View logs
gcloud app logs tail
```

### Azure Deployment

#### Using App Service
```bash
# Create resource group
az group create --name babel-beats-rg --location eastus

# Create App Service plan
az appservice plan create --name babel-beats-plan --resource-group babel-beats-rg --sku B2 --is-linux

# Create web app
az webapp create --resource-group babel-beats-rg --plan babel-beats-plan --name babel-beats-app --runtime "PYTHON|3.9"

# Configure deployment
az webapp deployment source config-local-git --resource-group babel-beats-rg --name babel-beats-app

# Deploy
git remote add azure <deployment-url>
git push azure main
```

## Database Setup

### PostgreSQL Setup
```sql
-- Create database
CREATE DATABASE babel_beats;

-- Create user
CREATE USER babel_beats_user WITH PASSWORD 'secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE babel_beats TO babel_beats_user;

-- Connect to database
\c babel_beats;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

### Run Migrations
```bash
# Initialize Alembic
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Initial migration"

# Apply migrations
alembic upgrade head
```

## Monitoring & Logging

### 1. Prometheus Setup
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'babel-beats'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### 2. Grafana Dashboard
Import the provided dashboard JSON:
```json
{
  "dashboard": {
    "title": "BABEL-BEATS Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(babel_beats_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Active Users",
        "targets": [
          {
            "expr": "babel_beats_active_users"
          }
        ]
      }
    ]
  }
}
```

### 3. Logging Configuration
```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/babel_beats.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}
```

## Performance Optimization

### 1. Enable Caching
```python
# Use Redis for caching
from functools import lru_cache
import redis

redis_client = redis.Redis(decode_responses=True)

@lru_cache(maxsize=100)
def get_cached_analysis(analysis_id):
    return redis_client.get(f"analysis:{analysis_id}")
```

### 2. Database Optimization
```sql
-- Create indexes
CREATE INDEX idx_user_id ON analysis_results(user_id);
CREATE INDEX idx_created_at ON analysis_results(created_at);
CREATE INDEX idx_language ON analysis_results(language);

-- Analyze tables
ANALYZE analysis_results;
ANALYZE generated_music;
```

### 3. Load Balancing
Use Nginx for load balancing:
```nginx
upstream babel_beats {
    least_conn;
    server backend1:8000 weight=3;
    server backend2:8000 weight=2;
    server backend3:8000 weight=1;
}
```

## Troubleshooting

### Common Issues

#### 1. Audio Processing Errors
```bash
# Check FFmpeg installation
ffmpeg -version

# Install missing codecs
sudo apt-get install libavcodec-extra
```

#### 2. Database Connection Issues
```bash
# Test database connection
psql -h localhost -U postgres -d babel_beats

# Check database logs
tail -f /var/log/postgresql/postgresql-*.log
```

#### 3. Memory Issues
```bash
# Monitor memory usage
free -h
top -p $(pgrep -f "uvicorn")

# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Performance Issues
```bash
# Profile application
python -m cProfile -o profile.stats main.py

# Analyze profile
python -m pstats profile.stats
```

### Debug Mode
Enable debug mode for detailed logging:
```python
# In .env file
DEBUG=true
LOG_LEVEL=DEBUG

# In code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks
```bash
# Check application health
curl http://localhost:8000/health

# Check all services
docker-compose ps
kubectl get pods -n babel-beats
```

## Security Best Practices

1. **Use HTTPS in production**
   ```nginx
   server {
       listen 443 ssl http2;
       ssl_certificate /etc/ssl/certs/babel-beats.crt;
       ssl_certificate_key /etc/ssl/private/babel-beats.key;
   }
   ```

2. **Secure environment variables**
   - Use secrets management (AWS Secrets Manager, Azure Key Vault)
   - Never commit `.env` files

3. **Rate limiting**
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   
   @app.get("/api/v1/analyze")
   @limiter.limit("10/minute")
   async def analyze():
       pass
   ```

4. **Input validation**
   - Validate audio file size and format
   - Sanitize all user inputs

5. **Regular updates**
   ```bash
   # Update dependencies
   pip list --outdated
   pip install --upgrade -r requirements.txt
   ```

## Backup and Recovery

### Database Backup
```bash
# Backup PostgreSQL
pg_dump -U postgres babel_beats > backup_$(date +%Y%m%d).sql

# Restore from backup
psql -U postgres babel_beats < backup_20240115.sql
```

### Audio Files Backup
```bash
# Backup to S3
aws s3 sync ./generated s3://babel-beats-backup/generated/

# Restore from S3
aws s3 sync s3://babel-beats-backup/generated/ ./generated
```

## Maintenance

### Regular Tasks
1. **Clean up old files**
   ```bash
   # Remove files older than 7 days
   find ./generated -type f -mtime +7 -delete
   ```

2. **Database maintenance**
   ```sql
   -- Vacuum and analyze
   VACUUM ANALYZE;
   
   -- Reindex
   REINDEX DATABASE babel_beats;
   ```

3. **Log rotation**
   ```bash
   # Configure logrotate
   /var/log/babel-beats/*.log {
       daily
       rotate 7
       compress
       delaycompress
       missingok
       notifempty
   }
   ```

## Support

For additional support:
- Documentation: https://docs.babel-beats.ai
- Issues: https://github.com/yourusername/babel-beats/issues
- Email: support@babel-beats.ai