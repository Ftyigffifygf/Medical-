# 🚀 MedExpert Deployment Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying MedExpert in production environments. MedExpert is designed for healthcare professionals and requires careful consideration of security, compliance, and performance requirements.

## 🔧 Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Memory**: Minimum 4GB RAM, 8GB+ recommended
- **Storage**: 20GB+ available disk space
- **Network**: Stable internet connection with HTTPS support
- **Domain**: SSL certificate for production deployment

### Software Dependencies
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Git**: For source code management
- **Nginx**: For reverse proxy (optional)

## 🏗️ Deployment Options

### Option 1: Docker Compose (Recommended)

#### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd medexpert

# Copy environment configuration
cp .env.example .env

# Edit environment variables
nano .env

# Deploy with Docker Compose
./deploy.sh
```

#### Manual Docker Compose Deployment
```bash
# Build and start services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f medexpert
```

### Option 2: Standalone Docker

#### Build and Run
```bash
# Build the Docker image
docker build -t medexpert:latest .

# Run the container
docker run -d \
  --name medexpert-app \
  -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  medexpert:latest
```

### Option 3: Direct Python Deployment

#### Setup Virtual Environment
```bash
# Create virtual environment
python3 -m venv medexpert-env
source medexpert-env/bin/activate

# Install dependencies
pip install -r requirements_simple.txt

# Run the application
streamlit run medexpert_production.py --server.port 8501
```

## ⚙️ Configuration

### Environment Variables

#### Core Application Settings
```bash
# Application Configuration
APP_NAME=MedExpert
APP_VERSION=2.0.0
APP_ENV=production
DEBUG=false

# Server Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

#### Security Settings
```bash
# Security Configuration
SECRET_KEY=your-secure-secret-key-here
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

#### Medical Compliance
```bash
# HIPAA and Medical Compliance
HIPAA_COMPLIANCE=true
AUDIT_LOGGING=true
DATA_RETENTION_DAYS=2555  # 7 years as per medical standards
MAX_FILE_SIZE_MB=50
ALLOWED_FILE_TYPES=jpg,jpeg,png,dcm,pdf
```

### SSL/TLS Configuration

#### Using Let's Encrypt with Nginx
```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 🔒 Security Considerations

### Network Security
- **Firewall**: Configure firewall to allow only necessary ports
- **VPN**: Consider VPN access for sensitive environments
- **Network Segmentation**: Isolate medical systems from general network

### Application Security
- **HTTPS Only**: Enforce SSL/TLS encryption
- **Strong Authentication**: Implement robust user authentication
- **Rate Limiting**: Prevent abuse and DoS attacks
- **Input Validation**: Sanitize all user inputs

### Data Protection
- **No Data Persistence**: System doesn't store patient data
- **Secure Transmission**: All data encrypted in transit
- **Audit Logging**: Track all system access and usage

## 🏥 Medical Compliance

### HIPAA Compliance
- **Access Controls**: Implement proper user authentication
- **Audit Trails**: Log all system interactions
- **Data Encryption**: Encrypt data in transit and at rest
- **Risk Assessment**: Regular security assessments

### Professional Requirements
- **Licensed Users Only**: Restrict access to healthcare professionals
- **Medical Disclaimers**: Clear warnings about AI limitations
- **Clinical Oversight**: Require professional supervision
- **Documentation**: Maintain proper medical records

## 📊 Monitoring and Maintenance

### Health Checks
```bash
# Application Health Check
curl -f http://localhost:8501/_stcore/health

# Docker Container Health
docker ps --filter "name=medexpert"

# System Resources
docker stats medexpert-app
```

### Log Management
```bash
# View Application Logs
docker-compose logs -f medexpert

# System Logs
tail -f /var/log/syslog

# Nginx Logs (if using reverse proxy)
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### Performance Monitoring
- **Response Times**: Monitor API response times
- **Resource Usage**: Track CPU, memory, and disk usage
- **User Activity**: Monitor consultation and analysis volumes
- **Error Rates**: Track application errors and failures

## 🔄 Backup and Recovery

### Data Backup
```bash
# Backup Configuration
tar -czf medexpert-config-$(date +%Y%m%d).tar.gz .env docker-compose.yml

# Backup Logs
tar -czf medexpert-logs-$(date +%Y%m%d).tar.gz logs/
```

### Disaster Recovery
1. **System Backup**: Regular system snapshots
2. **Configuration Backup**: Version control for configurations
3. **Recovery Testing**: Regular disaster recovery drills
4. **Documentation**: Maintain recovery procedures

## 🚀 Production Deployment Checklist

### Pre-Deployment
- [ ] Review and update environment variables
- [ ] Configure SSL/TLS certificates
- [ ] Set up monitoring and alerting
- [ ] Perform security assessment
- [ ] Test backup and recovery procedures

### Deployment
- [ ] Deploy application using chosen method
- [ ] Verify all services are running
- [ ] Test application functionality
- [ ] Confirm SSL/TLS configuration
- [ ] Validate monitoring systems

### Post-Deployment
- [ ] Monitor system performance
- [ ] Review application logs
- [ ] Test user access and authentication
- [ ] Verify compliance requirements
- [ ] Document deployment details

## 🛠️ Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check Docker logs
docker-compose logs medexpert

# Verify environment variables
docker-compose config

# Check port availability
netstat -tlnp | grep 8501
```

#### SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in /path/to/certificate.crt -text -noout

# Renew Let's Encrypt certificate
certbot renew --dry-run
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check system resources
htop
df -h
```

### Support Resources
- **Documentation**: Comprehensive system documentation
- **Logs**: Detailed application and system logs
- **Monitoring**: Real-time performance metrics
- **Community**: Healthcare IT professional networks

## 📞 Support and Maintenance

### Regular Maintenance Tasks
- **Security Updates**: Apply system and application updates
- **Certificate Renewal**: Renew SSL certificates before expiration
- **Log Rotation**: Manage log file sizes and retention
- **Performance Review**: Regular performance assessments

### Emergency Procedures
- **Incident Response**: Documented response procedures
- **Escalation Paths**: Clear escalation procedures
- **Communication Plan**: Stakeholder notification procedures
- **Recovery Procedures**: Step-by-step recovery instructions

## 📚 Additional Resources

### Documentation
- [User Guide](USER_GUIDE.md)
- [API Documentation](API_REFERENCE.md)
- [Security Guidelines](SECURITY.md)

### External Resources
- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
- [HIPAA Compliance Guide](https://www.hhs.gov/hipaa/index.html)

---

## 📝 Version Information

**Version**: 2.0.0  
**Last Updated**: 2025-01-02  
**Compatibility**: Docker 20.10+, Python 3.11+  
**License**: Medical Use Only

---

*For licensed healthcare professionals only. Ensure compliance with all applicable medical and data protection regulations.*