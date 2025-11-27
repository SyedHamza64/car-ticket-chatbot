# Hosting Guide for AI Support Assistant

## Why Shared Hosting Won't Work

This application has specific requirements that shared hosting cannot meet:

1. **Ollama Local LLM**: Requires running a background service on localhost:11434
2. **Streamlit Server**: Needs a long-running process on port 8501
3. **Large Files**: Model files (2GB+), vector database, and data files
4. **Resource Intensive**: LLM inference requires significant CPU/RAM
5. **Custom Services**: Shared hosting doesn't allow custom background processes

## Recommended Hosting Options

### Option 1: VPS (Virtual Private Server) - **RECOMMENDED**

**Shock Hosting VPS** or similar providers:
- Full root access
- Can install Ollama, Python, and all dependencies
- Sufficient resources for LLM inference
- Control over ports and services

**Minimum Requirements:**
- **RAM**: 8GB+ (16GB recommended for larger models)
- **CPU**: 4+ cores
- **Storage**: 50GB+ SSD
- **OS**: Ubuntu 22.04 LTS or similar

**Setup Steps:**
1. Purchase VPS plan
2. Install Ubuntu 22.04
3. Install Python 3.10+, Ollama, and dependencies
4. Upload application files
5. Set up systemd service for auto-start
6. Configure firewall (ports 8501, 11434)
7. Set up reverse proxy (Nginx) for domain access

### Option 2: Cloud Platforms (Easiest)

**Streamlit Cloud** (Free tier available):
- ✅ Handles Streamlit hosting automatically
- ❌ Still need Ollama running elsewhere (separate VPS)
- Best for: Quick deployment, managed hosting

**Railway.app** or **Render.com**:
- ✅ Can run both Streamlit and Ollama
- ✅ Auto-deployment from Git
- ✅ Free tier available (with limitations)
- Best for: Easy deployment, managed services

**Setup on Railway/Render:**
1. Connect GitHub repository
2. Set environment variables
3. Add Ollama service
4. Deploy automatically

### Option 3: Hybrid Approach

**Host App on Shared Hosting + Ollama on VPS/Cloud:**

1. Convert Streamlit app to Flask/FastAPI
2. Host Flask/FastAPI on shared hosting (via cPanel)
3. Run Ollama on separate VPS/cloud service
4. Connect app to remote Ollama via API

**Pros:**
- Can use existing shared hosting
- Lower cost for app hosting

**Cons:**
- More complex setup
- Network latency between app and LLM
- Need to maintain two services

### Option 4: Docker Deployment

**Any VPS or Cloud with Docker support:**

1. Create Dockerfile for the application
2. Use Docker Compose for multi-container setup
3. Deploy to any Docker-compatible host

**Benefits:**
- Consistent environment
- Easy scaling
- Portable deployment

## Detailed VPS Setup Guide

### Step 1: Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.10 python3-pip python3-venv git

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required model
ollama pull gemma2:2b
```

### Step 2: Application Setup

```bash
# Clone repository
git clone <your-repo-url> /var/www/chatbot
cd /var/www/chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up data files (upload via SFTP or git)
# Ensure data/processed/, data/guides/, data/chroma/ exist
```

### Step 3: Create Systemd Service

Create `/etc/systemd/system/streamlit-app.service`:

```ini
[Unit]
Description=Streamlit AI Assistant
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/chatbot
Environment="PATH=/var/www/chatbot/venv/bin"
ExecStart=/var/www/chatbot/venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable streamlit-app
sudo systemctl start streamlit-app
sudo systemctl status streamlit-app
```

### Step 4: Configure Nginx Reverse Proxy

Create `/etc/nginx/sites-available/chatbot`:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

Enable and reload:
```bash
sudo ln -s /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 5: SSL Certificate (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

### Step 6: Firewall Configuration

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

## Environment Variables

Create `.env` file on server:

```env
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b

# Embedding model
EMBEDDING_MODEL=all-mpnet-base-v2

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

## Monitoring & Maintenance

### Check Service Status
```bash
sudo systemctl status streamlit-app
sudo systemctl status ollama
```

### View Logs
```bash
# Application logs
tail -f /var/www/chatbot/logs/app.log

# Streamlit service logs
sudo journalctl -u streamlit-app -f

# Ollama logs
sudo journalctl -u ollama -f
```

### Restart Services
```bash
sudo systemctl restart streamlit-app
sudo systemctl restart ollama
```

## Cost Comparison

| Option | Monthly Cost | Complexity | Recommended For |
|--------|--------------|------------|-----------------|
| Shock Hosting VPS | $10-30 | Medium | Production use |
| Railway.app | $5-20 | Low | Quick deployment |
| Render.com | $7-25 | Low | Managed hosting |
| Streamlit Cloud | Free-$20 | Low | Streamlit-only (needs separate Ollama) |
| Hybrid (Shared + VPS) | $5-15 | High | Existing shared hosting |

## Security Considerations

1. **Firewall**: Only expose ports 80, 443, 22
2. **SSL**: Always use HTTPS
3. **Authentication**: Add login to Streamlit app
4. **Backups**: Regular backups of data/ and vector DB
5. **Updates**: Keep system and dependencies updated

## Troubleshooting

### Ollama not responding
```bash
# Check if Ollama is running
sudo systemctl status ollama

# Restart Ollama
sudo systemctl restart ollama

# Test connection
curl http://localhost:11434/api/tags
```

### Streamlit not accessible
```bash
# Check if service is running
sudo systemctl status streamlit-app

# Check port
sudo netstat -tlnp | grep 8501

# Check Nginx
sudo nginx -t
sudo systemctl status nginx
```

### High memory usage
- Use smaller model (gemma2:2b instead of larger models)
- Reduce concurrent requests
- Add swap space if needed

## Next Steps

1. **Choose hosting option** based on budget and requirements
2. **Set up server** following the guide above
3. **Test deployment** with sample queries
4. **Configure domain** and SSL
5. **Set up monitoring** and backups
6. **Go live!**

For questions or issues, refer to the main README.md or contact support.





