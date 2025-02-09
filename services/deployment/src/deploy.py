import subprocess
from pathlib import Path
import os


def deploy_frontend():
    """Deploy Streamlit frontend"""
    frontend_dir = Path(__file__).parent
    app_path = frontend_dir / "main.py"

    # Start Streamlit
    process = subprocess.Popen(
        ["streamlit", "run", str(app_path)], env=os.environ.copy()
    )
    return {"status": "success", "pid": process.pid}
