import os
import hashlib
import joblib

def calculate_file_hash(filepath: str) -> str:
    """Calculate the SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def save_secure_model(model, filepath: str) -> str:
    """
    Serialize model using joblib and generate a companion SHA-256 hash manifest.
    Returns the computed checksum.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(model, filepath)
    
    file_hash = calculate_file_hash(filepath)
    hash_file_path = f"{filepath}.sha256"
    
    with open(hash_file_path, "w", encoding="utf-8") as f:
        f.write(file_hash)
        
    print(f"[SECURITY] Model saved securely to: {filepath}")
    print(f"[SECURITY] SHA-256 Hash recorded: {file_hash} -> {hash_file_path}")
    return file_hash

def load_secure_model(filepath: str, enforce_hash_check: bool = True):
    """
    Load a serialized model with integrity verification against its SHA-256 manifest.
    Raises SecurityError if integrity check fails.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
        
    hash_file_path = f"{filepath}.sha256"
    
    if enforce_hash_check:
        if not os.path.exists(hash_file_path):
            raise SecurityError(
                f"[SECURITY ERROR] Integrity signature missing: '{hash_file_path}'. "
                f"Refusing to deserialize unverified joblib model."
            )
        with open(hash_file_path, "r", encoding="utf-8") as f:
            expected_hash = f.read().strip()
            
        current_hash = calculate_file_hash(filepath)
        if current_hash != expected_hash:
            raise SecurityError(
                f"[SECURITY ERROR] Model integrity check FAILED for '{filepath}'! "
                f"Expected: {expected_hash}, Got: {current_hash}. File may be tampered."
            )
        print(f"[SECURITY] Integrity verification PASSED for: {filepath} (SHA-256: {current_hash[:12]}...)")
        
    return joblib.load(filepath)

class SecurityError(Exception):
    """Exception raised when an integrity or security constraint fails."""
    pass
