# Setup and run script
.\venv\Scripts\activate
pip install -r requirements.txt

Write-Host "Starting Streamlit Dashboard..."
# Run local host streamlit inside powershell background jobs
Start-Job -ScriptBlock { 
    param($dir)
    cd $dir
    .\venv\Scripts\activate
    streamlit run ui/app.py
} -ArgumentList (Get-Location)

Write-Host "Starting FastAPI Server..."
uvicorn api.main:app --host 127.0.0.1 --port 8000
