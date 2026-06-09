# pytorch_deep_learning/vision/utils/setup.py

import subprocess
import sys

def setup_project():
    """
    Installs missing dependencies and prepares project utilities.
    """

    try:
        import torchinfo
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "torchinfo"
        ])

    print("[INFO] Setup complete ✔️")