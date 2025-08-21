#!/bin/bash
set -e

echo "🚀 Starting Contact Scraper Build Process..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update

# Install Chrome/Chromium
echo "🌐 Installing Chrome/Chromium..."
apt-get install -y chromium-browser chromium-driver

# Set Chrome environment variables
export CHROME_BIN=/usr/bin/chromium-browser
export CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "
import fastapi
import DrissionPage
print('✅ FastAPI version:', fastapi.__version__)
print('✅ DrissionPage imported successfully')
"

# Test Chrome installation
echo "🧪 Testing Chrome installation..."
chromium-browser --version || echo "⚠️ Chrome test failed (normal in build environment)"

echo "🎉 Build completed successfully!"
