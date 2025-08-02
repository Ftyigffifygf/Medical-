#!/usr/bin/env python3
"""
MedExpert Launch Script
Comprehensive launcher for the MedExpert Medical AI System

This script provides multiple launch options for the MedExpert system:
- Basic MedExpert (original version)
- Enhanced MedExpert (full-featured version)
- System diagnostics and health checks
"""

import subprocess
import sys
import os
import importlib.util
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 'plotly', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def check_module_files():
    """Check if all required module files exist"""
    required_files = [
        'medexpert.py',
        'medexpert_enhanced.py',
        'medical_knowledge.py',
        'medical_imaging.py',
        'evidence_synthesis.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    return missing_files

def run_system_diagnostics():
    """Run comprehensive system diagnostics"""
    print("🔍 Running MedExpert System Diagnostics...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    else:
        print("✅ Python version compatible")
    
    print()
    
    # Check dependencies
    print("📦 Checking Dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Install missing dependencies with:")
        print(f"   pip install {' '.join(missing_deps)}")
    else:
        print("✅ All dependencies installed")
    
    print()
    
    # Check module files
    print("📁 Checking Module Files...")
    missing_files = check_module_files()
    
    if missing_files:
        print("❌ Missing module files:")
        for file in missing_files:
            print(f"   - {file}")
    else:
        print("✅ All module files present")
    
    print()
    
    # Check system resources
    print("💻 System Resources...")
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"   CPU Usage: {cpu_percent}%")
        print(f"   Memory Usage: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
        print(f"   Disk Usage: {disk.percent}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)")
        
        if cpu_percent > 80:
            print("⚠️  High CPU usage detected")
        if memory.percent > 80:
            print("⚠️  High memory usage detected")
        if disk.percent > 90:
            print("⚠️  Low disk space")
        
        if cpu_percent <= 80 and memory.percent <= 80 and disk.percent <= 90:
            print("✅ System resources healthy")
            
    except ImportError:
        print("   psutil not available - install with: pip install psutil")
    
    print()
    print("🏥 MedExpert System Diagnostics Complete")
    print("=" * 50)

def launch_basic_medexpert():
    """Launch the basic MedExpert application"""
    print("🚀 Launching MedExpert Basic...")
    print("📍 Starting Streamlit application...")
    print("🌐 Application will open in your default browser")
    print("⏹️  Press Ctrl+C to stop the application")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "medexpert.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 MedExpert Basic stopped by user")
    except Exception as e:
        print(f"❌ Error launching MedExpert Basic: {e}")

def launch_enhanced_medexpert():
    """Launch the enhanced MedExpert application"""
    print("🚀 Launching MedExpert Enhanced...")
    print("📍 Starting advanced medical AI system...")
    print("🌐 Application will open in your default browser")
    print("⏹️  Press Ctrl+C to stop the application")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "medexpert_enhanced.py",
            "--server.port", "8502",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 MedExpert Enhanced stopped by user")
    except Exception as e:
        print(f"❌ Error launching MedExpert Enhanced: {e}")

def show_system_info():
    """Display comprehensive system information"""
    print("🏥 MedExpert Medical AI System")
    print("=" * 50)
    print("Version: 2.0.0")
    print("Build Date: 2025-01-02")
    print("Author: Kortix AI Team")
    print()
    print("📚 Training Datasets:")
    print("   • MIMIC-III Clinical Database")
    print("   • MIMIC-IV Clinical Database")
    print("   • PubMed Biomedical Literature")
    print("   • PMC Open Access Articles")
    print("   • MedQuAD Medical Q&A Dataset")
    print("   • NIH Chest X-ray Dataset")
    print("   • TCIA Cancer Imaging Archive")
    print("   • Medical Dialogue Datasets")
    print("   • BioBERT Medical NLP")
    print("   • MONAI Medical Imaging Framework")
    print()
    print("🚀 Capabilities:")
    print("   • Advanced Clinical Reasoning")
    print("   • Differential Diagnosis Generation")
    print("   • Medical Imaging Analysis (MONAI)")
    print("   • Evidence-Based Literature Synthesis")
    print("   • Comprehensive Drug Information")
    print("   • Automated SOAP Note Generation")
    print("   • Drug Interaction Checking")
    print("   • Clinical Risk Calculators")
    print("   • Treatment Recommendations")
    print("   • Medical Data Analytics")
    print()
    print("⚠️  FOR LICENSED HEALTHCARE PROFESSIONALS ONLY")
    print("   This AI system assists in clinical decision-making")
    print("   but does not replace professional medical judgment.")
    print("=" * 50)

def main():
    """Main launcher interface"""
    print("🏥 MedExpert Medical AI System Launcher")
    print("=" * 50)
    
    while True:
        print("\nSelect an option:")
        print("1. 🚀 Launch MedExpert Basic")
        print("2. 🚀 Launch MedExpert Enhanced (Recommended)")
        print("3. 🔍 Run System Diagnostics")
        print("4. ℹ️  Show System Information")
        print("5. 📋 View Module Files")
        print("6. 🛠️  Install Dependencies")
        print("7. ❌ Exit")
        
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                print()
                missing_files = check_module_files()
                if 'medexpert.py' in missing_files:
                    print("❌ medexpert.py not found!")
                    continue
                
                missing_deps = check_dependencies()
                if missing_deps:
                    print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
                    print("Please install dependencies first (option 6)")
                    continue
                
                launch_basic_medexpert()
                
            elif choice == "2":
                print()
                missing_files = check_module_files()
                required_for_enhanced = [
                    'medexpert_enhanced.py', 'medical_knowledge.py', 
                    'medical_imaging.py', 'evidence_synthesis.py'
                ]
                
                missing_required = [f for f in required_for_enhanced if f in missing_files]
                if missing_required:
                    print(f"❌ Missing required files: {', '.join(missing_required)}")
                    continue
                
                missing_deps = check_dependencies()
                if missing_deps:
                    print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
                    print("Please install dependencies first (option 6)")
                    continue
                
                launch_enhanced_medexpert()
                
            elif choice == "3":
                print()
                run_system_diagnostics()
                
            elif choice == "4":
                print()
                show_system_info()
                
            elif choice == "5":
                print()
                print("📁 MedExpert Module Files:")
                print("-" * 30)
                
                module_files = [
                    ('medexpert.py', 'Basic MedExpert application'),
                    ('medexpert_enhanced.py', 'Enhanced MedExpert with all features'),
                    ('medical_knowledge.py', 'Medical knowledge base and conditions'),
                    ('medical_imaging.py', 'MONAI-based medical imaging analysis'),
                    ('evidence_synthesis.py', 'Biomedical literature synthesis'),
                    ('requirements.txt', 'Python dependencies'),
                    ('todo.md', 'Development progress tracking'),
                    ('launch_medexpert.py', 'This launcher script')
                ]
                
                for filename, description in module_files:
                    status = "✅" if Path(filename).exists() else "❌"
                    print(f"{status} {filename:<25} - {description}")
                
            elif choice == "6":
                print()
                print("🛠️  Installing Dependencies...")
                
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "--no-cache-dir",
                        "streamlit", "pandas", "numpy", "matplotlib", "plotly", "requests"
                    ], check=True)
                    print("✅ Dependencies installed successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error installing dependencies: {e}")
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                
            elif choice == "7":
                print("\n👋 Thank you for using MedExpert!")
                print("🏥 Advancing medical AI for healthcare professionals")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()