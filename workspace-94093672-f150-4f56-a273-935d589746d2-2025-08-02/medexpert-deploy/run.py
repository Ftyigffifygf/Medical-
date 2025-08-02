#!/usr/bin/env python3
"""
MedExpert Deployment Runner
Simple script to launch the MedExpert application
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Main function to run the MedExpert application"""
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not found. Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run the streamlit app
    print("🚀 Starting MedExpert...")
    
    # Use the simplified app for deployment
    app_file = "app_simplified.py"
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()