import os
import urllib.request
import sys
import hashlib

def calculate_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def download_dataset(expected_sha256=None):
    url = "https://raw.githubusercontent.com/mayukh18/DEAP/master/city_pollution_data.csv"
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    dest_path = os.path.join(raw_dir, "city_pollution_data.csv")
    
    print(f"[FETCH] Starting secure download from: {url}")
    print(f"[FETCH] Target destination: {dest_path}")
    
    try:
        # Enforce User-Agent and a 30-second socket timeout to prevent hang / DoS
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status != 200:
                raise ValueError(f"HTTP response status failed: {response.status}")
            data = response.read()
            with open(dest_path, "wb") as f:
                f.write(data)
                
        file_size = os.path.getsize(dest_path)
        sha256_hash = calculate_sha256(dest_path)
        print(f"[FETCH] Download completed successfully! Size: {file_size / (1024 * 1024):.2f} MB")
        print(f"[FETCH] SHA-256 Checksum: {sha256_hash}")
        
        if expected_sha256 and sha256_hash != expected_sha256:
            raise ValueError(f"Checksum mismatch! Expected {expected_sha256}, got {sha256_hash}")
            
    except Exception as e:
        print(f"[FETCH ERROR] Error occurred during dataset download: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()

