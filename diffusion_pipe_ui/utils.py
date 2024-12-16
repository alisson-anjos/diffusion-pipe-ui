# project/utils.py

from datetime import datetime

def generate_unique_filename(base_name):
    """Generate a unique filename based on the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.toml"
