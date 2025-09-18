# AI4K8s Docker Deployment Guide

## 🐳 **Docker Image: AI-Powered Kubernetes Management Platform**

This guide shows you how to build, deploy, and use the AI4K8s Docker image.

## 📦 **Quick Start**

### **Option 1: Using Docker Hub (Recommended)**

```bash
# Pull the latest image
docker pull pedramnj/ai4k8s:latest

# Run with environment variables
docker run -d \
  --name ai4k8s-web \
  -p 5003:5003 \
  -e ANTHROPIC_API_KEY="your-api-key-here" \
  -v ai4k8s_data:/app/instance \
  --restart unless-stopped \
  pedramnj/ai4k8s:latest
```

### **Option 2: Using Docker Compose**

```bash
# Clone the repository
git clone https://github.com/pedramnj/A14K8s.git
cd A14K8s

# Create .env file with your API key
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env

# Start the application
docker-compose up -d
```

### **Option 3: Build from Source**

```bash
# Clone the repository
git clone https://github.com/pedramnj/A14K8s.git
cd A14K8s

# Build the Docker image
docker build -t ai4k8s:latest .

# Run the container
docker run -d \
  --name ai4k8s-web \
  -p 5003:5003 \
  -e ANTHROPIC_API_KEY="your-api-key-here" \
  -v ai4k8s_data:/app/instance \
  --restart unless-stopped \
  ai4k8s:latest
```

## 🔧 **Configuration**

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | - | **Required** - Your Anthropic API key for AI processing |
| `FLASK_ENV` | `production` | Flask environment |
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `FLASK_HOST` | `0.0.0.0` | Host to bind to |
| `FLASK_PORT` | `5003` | Port to run on |

### **Volume Mounts**

| Mount Point | Description |
|-------------|-------------|
| `/app/instance` | SQLite database storage |

## 🚀 **Usage**

### **Access the Application**

Once running, access the web interface at:
- **Local**: http://localhost:5003
- **Remote**: http://your-server-ip:5003

### **Features**

- ✅ **AI-Powered Natural Language Processing**
- ✅ **Kubernetes Cluster Management**
- ✅ **User Authentication & Authorization**
- ✅ **MCP Tool Integration**
- ✅ **Real-time Chat Interface**
- ✅ **Health Monitoring**

## 🔐 **Security Notes**

1. **API Key**: Always set `ANTHROPIC_API_KEY` environment variable
2. **Network**: Consider using a reverse proxy (nginx) for production
3. **SSL**: Use HTTPS in production environments
4. **Firewall**: Restrict access to port 5003 if needed

## 📊 **Health Check**

The container includes a health check that verifies the application is responding:

```bash
# Check container health
docker ps
# Look for "healthy" status

# View logs
docker logs ai4k8s-web
```

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **Container won't start**
   ```bash
   docker logs ai4k8s-web
   ```

2. **AI not working**
   - Verify `ANTHROPIC_API_KEY` is set correctly
   - Check logs for API errors

3. **Database issues**
   - Ensure volume is mounted correctly
   - Check permissions on `/app/instance`

### **Debug Mode**

To enable debug mode:

```bash
docker run -d \
  --name ai4k8s-web \
  -p 5003:5003 \
  -e ANTHROPIC_API_KEY="your-api-key-here" \
  -e FLASK_DEBUG=true \
  -v ai4k8s_data:/app/instance \
  pedramnj/ai4k8s:latest
```

## 🔄 **Updates**

### **Update the Image**

```bash
# Pull latest version
docker pull pedramnj/ai4k8s:latest

# Stop and remove old container
docker stop ai4k8s-web
docker rm ai4k8s-web

# Run new container (your data persists in the volume)
docker run -d \
  --name ai4k8s-web \
  -p 5003:5003 \
  -e ANTHROPIC_API_KEY="your-api-key-here" \
  -v ai4k8s_data:/app/instance \
  --restart unless-stopped \
  pedramnj/ai4k8s:latest
```

### **Using Docker Compose**

```bash
docker-compose pull
docker-compose up -d
```

## 📈 **Production Deployment**

### **Using Docker Swarm**

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml ai4k8s
```

### **Using Kubernetes**

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s-deployment.yaml
```

## 🏷️ **Image Tags**

- `pedramnj/ai4k8s:latest` - Latest stable version
- `pedramnj/ai4k8s:v1.0.0` - Specific version
- `pedramnj/ai4k8s:dev` - Development version

## 📞 **Support**

- **GitHub Issues**: [Report bugs or request features](https://github.com/pedramnj/A14K8s/issues)
- **Documentation**: [Full documentation](https://github.com/pedramnj/A14K8s/blob/main/README.md)

---

**🎉 Enjoy your AI-powered Kubernetes management platform!**
