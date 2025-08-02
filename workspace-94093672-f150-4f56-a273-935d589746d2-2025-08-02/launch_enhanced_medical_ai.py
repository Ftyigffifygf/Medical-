#!/usr/bin/env python3
"""
Enhanced Medical AI System Launcher
Comprehensive launcher for the medical AI system with setup validation and configuration

Features:
- Environment validation
- Dependency checking
- API key validation
- System initialization
- Error handling and troubleshooting
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalAILauncher:
    """Enhanced Medical AI System Launcher"""
    
    def __init__(self):
        """Initialize the launcher"""
        self.required_packages = [
            'streamlit', 'openai', 'pandas', 'numpy', 'matplotlib', 
            'plotly', 'requests', 'python-dotenv'
        ]
        self.optional_packages = [
            'datasets', 'transformers', 'biopython', 'opencv-python'
        ]
        self.system_ready = False
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        logger.info("Checking Python version...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
            return True
        else:
            logger.error(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Requires Python 3.8+")
            return False
    
    def check_package_installation(self, package_name: str) -> bool:
        """Check if a package is installed"""
        try:
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except ImportError:
            return False
    
    def install_package(self, package_name: str) -> bool:
        """Install a package using pip"""
        try:
            logger.info(f"Installing {package_name}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name, "--no-cache-dir"
            ])
            logger.info(f"✅ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package_name}: {e}")
            return False
    
    def check_and_install_dependencies(self) -> bool:
        """Check and install required dependencies"""
        logger.info("Checking dependencies...")
        
        missing_packages = []
        
        # Check required packages
        for package in self.required_packages:
            if not self.check_package_installation(package):
                missing_packages.append(package)
        
        if missing_packages:
            logger.info(f"Missing packages: {missing_packages}")
            
            # Try to install missing packages
            for package in missing_packages:
                if not self.install_package(package):
                    logger.error(f"Failed to install required package: {package}")
                    return False
        
        logger.info("✅ All required dependencies are available")
        return True
    
    def check_environment_file(self) -> bool:
        """Check if .env file exists and has required variables"""
        logger.info("Checking environment configuration...")
        
        env_file = Path(".env")
        if not env_file.exists():
            logger.warning("⚠️ .env file not found. Creating template...")
            self.create_env_template()
            return False
        
        # Check for required environment variables
        required_vars = ['OPENAI_API_KEY']
        
        try:
            with open(env_file, 'r') as f:
                content = f.read()
            
            missing_vars = []
            for var in required_vars:
                if var not in content or f"{var}=" not in content:
                    missing_vars.append(var)
            
            if missing_vars:
                logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
                return False
            
            logger.info("✅ Environment file is properly configured")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reading .env file: {e}")
            return False
    
    def create_env_template(self):
        """Create a template .env file"""
        template = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Medical AI Configuration
MEDICAL_AI_MODE=enhanced
ENABLE_FINE_TUNING=true
ENABLE_MEDICAL_IMAGING=true
ENABLE_EVIDENCE_SYNTHESIS=true

# Dataset Configuration
MIMIC_DATA_PATH=./data/mimic
PUBMED_DATA_PATH=./data/pubmed
MEDICAL_IMAGING_PATH=./data/imaging
BIOBERT_MODEL_PATH=./models/biobert

# Security Configuration
HEALTHCARE_PROFESSIONAL_AUTH=true
AUDIT_LOGGING=true
HIPAA_COMPLIANCE=true
"""
        
        with open(".env", "w") as f:
            f.write(template)
        
        logger.info("📝 Created .env template file. Please update with your API keys.")
    
    def validate_openai_api(self) -> bool:
        """Validate OpenAI API key"""
        logger.info("Validating OpenAI API key...")
        
        try:
            from dotenv import load_dotenv
            import openai
            
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key or api_key == 'your_openai_api_key_here':
                logger.error("❌ OpenAI API key not configured")
                return False
            
            # Test API connection
            client = openai.OpenAI(api_key=api_key)
            
            # Simple test call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            logger.info("✅ OpenAI API key is valid and working")
            return True
            
        except Exception as e:
            logger.error(f"❌ OpenAI API validation failed: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        logger.info("Creating necessary directories...")
        
        directories = [
            "data",
            "data/mimic",
            "data/pubmed", 
            "data/medquad",
            "data/dialogues",
            "data/imaging",
            "data/biobert",
            "data/processed",
            "models",
            "logs"
        ]
        
        try:
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ All directories created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating directories: {e}")
            return False
    
    def run_system_checks(self) -> bool:
        """Run comprehensive system checks"""
        logger.info("🔍 Running system checks...")
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_and_install_dependencies),
            ("Environment File", self.check_environment_file),
            ("Directories", self.create_directories),
            ("OpenAI API", self.validate_openai_api)
        ]
        
        all_passed = True
        
        for check_name, check_function in checks:
            logger.info(f"Running {check_name} check...")
            
            try:
                if not check_function():
                    logger.error(f"❌ {check_name} check failed")
                    all_passed = False
                else:
                    logger.info(f"✅ {check_name} check passed")
            except Exception as e:
                logger.error(f"❌ {check_name} check failed with error: {e}")
                all_passed = False
        
        return all_passed
    
    def launch_application(self) -> bool:
        """Launch the Streamlit application"""
        logger.info("🚀 Launching Enhanced Medical AI System...")
        
        try:
            # Launch Streamlit app
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "medical_ai_app.py",
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--server.headless=true"
            ])
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 Application stopped by user")
            return True
        except Exception as e:
            logger.error(f"❌ Error launching application: {e}")
            return False
    
    def display_startup_info(self):
        """Display startup information"""
        print("""
🏥 Enhanced Medical AI System
=====================================

🎯 Features:
• Advanced medical case analysis with GPT-4
• Fine-tuning pipeline for medical datasets
• SOAP note generation
• Drug interaction checking
• Performance analytics
• Medical knowledge base

📚 Datasets Supported:
• MIMIC-III/IV Clinical Records
• PubMed Biomedical Literature
• MedQuAD Q&A Dataset
• Medical Dialogue Corpus

⚠️  Important Notice:
This system is for educational and research purposes only.
Always consult healthcare professionals for medical decisions.

=====================================
        """)
    
    def display_troubleshooting(self):
        """Display troubleshooting information"""
        print("""
🔧 Troubleshooting Guide
=====================================

Common Issues:

1. OpenAI API Key Issues:
   • Ensure your API key is valid and has sufficient credits
   • Update the OPENAI_API_KEY in your .env file
   • Check for any typos or extra spaces

2. Package Installation Issues:
   • Try: pip install --upgrade pip
   • Use: pip install -r requirements_enhanced.txt
   • For disk space issues, use: pip install --no-cache-dir

3. Memory Issues:
   • Close other applications
   • Use lighter models (gpt-3.5-turbo instead of gpt-4)
   • Reduce batch sizes in fine-tuning

4. Network Issues:
   • Check internet connection
   • Verify firewall settings
   • Try using a VPN if needed

5. Permission Issues:
   • Run with appropriate permissions
   • Check file/directory access rights

For more help, check the documentation or contact support.
=====================================
        """)
    
    def run(self):
        """Main launcher function"""
        self.display_startup_info()
        
        # Run system checks
        if self.run_system_checks():
            logger.info("🎉 All system checks passed!")
            self.system_ready = True
            
            # Launch application
            if self.launch_application():
                logger.info("✅ Application launched successfully")
            else:
                logger.error("❌ Failed to launch application")
                self.display_troubleshooting()
        else:
            logger.error("❌ System checks failed")
            self.display_troubleshooting()
            
            # Ask user if they want to continue anyway
            try:
                response = input("\nWould you like to try launching anyway? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    logger.info("🚀 Attempting to launch despite failed checks...")
                    self.launch_application()
            except KeyboardInterrupt:
                logger.info("🛑 Launch cancelled by user")

def main():
    """Main function"""
    launcher = MedicalAILauncher()
    launcher.run()

if __name__ == "__main__":
    main()