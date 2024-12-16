# project/utils.py

from datetime import datetime

def generate_unique_filename(base_name, extension=".toml"):
    """Generate a unique filename based on the current timestamp.
    
    Args:
        base_name (str): Base name for the file
        extension (str, optional): File extension. Defaults to ".toml"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"
