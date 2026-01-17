import os
import csv
import requests
import concurrent.futures
from PIL import Image
from io import BytesIO
import time
import warnings
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
Image.MAX_IMAGE_PIXELS = None 
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", category=UserWarning) 

CSV_FILE = "data.csv"
OUTPUT_DIR = "gallery_images"
TARGET_COUNT = 10000           
MAX_WORKERS = 64               
TIMEOUT = 5                    
LOG_INTERVAL = 50
PROXIES = None 
PROXIES = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"} 

session = requests.Session()
retries = Retry(total=2, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

def download_single_image(args):
    idx, url = args
    save_path = os.path.join(OUTPUT_DIR, f"{idx}.jpg")
    
    if os.path.exists(save_path):
        if os.path.getsize(save_path) > 0:
            return True
        else:
            try:
                os.remove(save_path)
            except:
                pass

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = session.get(url, headers=headers, timeout=TIMEOUT, proxies=PROXIES)
        
        if response.status_code != 200:
            return False

        img_data = response.content
        with BytesIO(img_data) as f:
            img = Image.open(f)
            
            if img.mode in ('P', 'RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
                
            w, h = img.size
            if w < 10 or h < 10: 
                return False   
            img.save(save_path, "JPEG", quality=85)
            
        return True

    except Exception:
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Reading {CSV_FILE}...", flush=True)
    all_rows = []
    try:
        with open(CSV_FILE, "r", encoding="utf-8", errors='ignore') as f:
            reader = csv.reader(f)
            next(reader) 
            for i, row in enumerate(reader):
                if row and len(row) > 0:
                    all_rows.append((i, row[0]))
    except FileNotFoundError:
        print("Error: data.csv not found.")
        return

    total_csv_lines = len(all_rows)
    print(f"Loaded {total_csv_lines} candidates.", flush=True)
    
    existing_files = len([name for name in os.listdir(OUTPUT_DIR) if name.endswith('.jpg')])
    print(f"Found {existing_files} images already downloaded. Resuming...", flush=True)
    
    if existing_files >= TARGET_COUNT:
        print("Target already reached! You can start retrieval now.")
        return

    print(f"Goal: Reach {TARGET_COUNT} images.", flush=True)

    success_count = 0 
    processed_count = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        iterator = executor.map(download_single_image, all_rows)
        
        for is_success in iterator:
            processed_count += 1
            if is_success:
                success_count += 1
            
            if processed_count % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                speed = (success_count - existing_files) / (elapsed + 1e-5) 
                if speed < 0: speed = 0 
                
                print(f"Scanned: {processed_count}/{total_csv_lines} | Total Saved: {success_count}/{TARGET_COUNT} | Speed: {speed:.2f} img/s", flush=True)
            
            if success_count >= TARGET_COUNT:
                print(f"\nTarget reached! Stopping download.")
                break

    print(f"\nDone. Final valid images: {success_count}", flush=True)

if __name__ == "__main__":
    main()