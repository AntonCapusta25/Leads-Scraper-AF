#!/usr/bin/env python3
"""
Simple monitoring script
"""
import time
import requests
import json
from datetime import datetime

def monitor_api(base_url="https://your-app.onrender.com"):
    """Monitor API endpoints"""
    endpoints = [
        "/health",
        "/legal/disclaimer",
        "/deduplicate/config"
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "base_url": base_url,
        "status": "unknown",
        "endpoints": {}
    }
    
    all_healthy = True
    
    for endpoint in endpoints:
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            response_time = round((time.time() - start_time) * 1000, 2)
            
            results["endpoints"][endpoint] = {
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "healthy": response.status_code == 200
            }
            
            if response.status_code != 200:
                all_healthy = False
                
        except Exception as e:
            results["endpoints"][endpoint] = {
                "error": str(e),
                "healthy": False
            }
            all_healthy = False
    
    results["status"] = "healthy" if all_healthy else "unhealthy"
    
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    monitor_api()
