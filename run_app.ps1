Write-Host "Installing dependencies..."
pip install -r requirements.txt
Write-Host "Launching MOmics..."
streamlit run momics_app.py
