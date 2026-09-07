# MOmics Installation Guide

This guide provides step-by-step instructions for installing and running MOmics on different operating systems.

## Quick Start (All Platforms)

### Step 1: Download the Repository
```bash
git clone <repository-url>
cd MOmics
```

### Step 2: Verify Installation
Run the verification script to check if everything is set up correctly:
```bash
python verify_setup.py
```

### Step 3: Launch the Application
Choose your preferred method based on your operating system.

---

## Windows Installation

### Method 1: One-Click Launch (Recommended)
1. **Double-click** `run_app.bat`
2. **Wait** for dependencies to install
3. **Browser opens** automatically to `http://localhost:8501`

### Method 2: PowerShell
1. **Right-click** on `run_app.ps1`
2. **Select** "Run with PowerShell"
3. **Wait** for installation to complete

### Method 3: Command Prompt
1. **Open Command Prompt** in the project folder
2. **Run**: `pip install -r requirements.txt`
3. **Run**: `streamlit run momics_app.py`
4. **Open browser** to `http://localhost:8501`

### Troubleshooting Windows
- **Permission errors**: Run Command Prompt as Administrator
- **Python not found**: Install Python from [python.org](https://python.org)
- **Script execution policy**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell

---

## macOS Installation

1. **Install Python 3.9+** (via [Homebrew](https://brew.sh) or python.org):
   ```bash
   brew install python@3.11
   ```
2. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```
3. **Launch application**:
   ```bash
   streamlit run momics_app.py
   ```

### Troubleshooting macOS
- **Permission errors**: Use `pip3` instead of `pip`
- **Browser issues**: Try Chrome or Safari

---

## Linux Installation

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
pip3 install -r requirements.txt
streamlit run momics_app.py
```

### CentOS/RHEL/Fedora
```bash
sudo dnf install python3 python3-pip
pip3 install -r requirements.txt
streamlit run momics_app.py
```

---

## Advanced Installation

### Using a Virtual Environment (Recommended)
```bash
python -m venv momics_env
# Windows:
momics_env\Scripts\activate
# macOS/Linux:
source momics_env/bin/activate

pip install -r requirements.txt
streamlit run momics_app.py
```

### Using Conda
```bash
conda create -n momics_env python=3.11
conda activate momics_env
pip install -r requirements.txt
streamlit run momics_app.py
```

---

## Common Issues & Solutions

### Python Issues
- **"Python not found"**: Install Python 3.9+ from [python.org](https://python.org)
- **Version conflicts**: Use a virtual environment

### Dependency Issues
- **Missing packages**: Run `pip install -r requirements.txt`
- **Permission errors**: Use `pip install --user -r requirements.txt`

### Application Issues
- **"File not found"**: Ensure `data/`, `models/`, `docs.py`, and `logo.png` are present alongside `momics_app.py`
- **"Port already in use"**: Close other Streamlit instances or change the port with `streamlit run momics_app.py --server.port 8502`

---

## Getting Help

1. **Run the verification script**: `python verify_setup.py`
2. **Check your Python version**: `python --version`
3. **Review `docs/gui_integration_guide.md`** for details on the underlying model pipeline

For questions or bug reports, contact Dr. Sivanesan Dakshanamurthy: sd233@georgetown.edu
