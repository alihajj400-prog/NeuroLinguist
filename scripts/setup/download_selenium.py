import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

BASE_TARGET = Path(r"C:\FYP\Pitt\raw")

FOLDERS = [
    ("https://media.talkbank.org/dementia/English/Pitt/Control/cookie/", "Control", "cookie"),
    ("https://media.talkbank.org/dementia/English/Pitt/Control/fluency/", "Control", "fluency"),
    ("https://media.talkbank.org/dementia/English/Pitt/Control/recall/", "Control", "recall"),
    ("https://media.talkbank.org/dementia/English/Pitt/Control/sentence/", "Control", "sentence"),
    ("https://media.talkbank.org/dementia/English/Pitt/Dementia/cookie/", "Dementia", "cookie"),
    ("https://media.talkbank.org/dementia/English/Pitt/Dementia/fluency/", "Dementia", "fluency"),
    ("https://media.talkbank.org/dementia/English/Pitt/Dementia/recall/", "Dementia", "recall"),
    ("https://media.talkbank.org/dementia/English/Pitt/Dementia/sentence/", "Dementia", "sentence"),
]


def create_driver_for_folder(download_path):
    """Create a new Chrome driver with specific download folder"""
    options = Options()
    
    prefs = {
        "download.default_directory": str(download_path),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option("detach", True)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    return driver


def main():
    print("=" * 50)
    print("DementiaBank Downloader - Direct to Subfolders")
    print("=" * 50)
    
    # Create all folders
    for _, group, task in FOLDERS:
        (BASE_TARGET / group / task).mkdir(parents=True, exist_ok=True)
    
    # Start with first folder to login
    first_url, first_group, first_task = FOLDERS[0]
    first_download_path = BASE_TARGET / first_group / first_task
    
    driver = create_driver_for_folder(first_download_path)
    
    print("\n[STEP 1] Opening TalkBank...")
    print("         Please log in manually.")
    print("         After logging in, press ENTER here.\n")
    
    driver.get(first_url)
    input(">>> Press ENTER after you have logged in... ")
    
    # Get cookies from logged-in session
    cookies = driver.get_cookies()
    
    total_downloaded = 0
    
    for url, group, task in FOLDERS:
        download_path = BASE_TARGET / group / task
        
        print(f"\n[FOLDER] {group}/{task}")
        print("-" * 40)
        
        # Change download directory using Chrome DevTools Protocol
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": str(download_path)
        })
        
        driver.get(url)
        time.sleep(2)
        
        try:
            links = driver.find_elements(By.XPATH, "//a[contains(@href, '.mp3')]")
            mp3_urls = [link.get_attribute("href") for link in links]
            
            print(f"    Found {len(mp3_urls)} MP3 files")
            
            for mp3_url in mp3_urls:
                filename = mp3_url.split("/")[-1]
                filepath = download_path / filename
                
                # Skip if already exists
                if filepath.exists() and filepath.stat().st_size > 10000:
                    print(f"    [SKIP] {filename}")
                    continue
                
                print(f"    [DOWNLOADING] {filename}...", end=" ")
                
                driver.execute_script(f"""
                    var a = document.createElement('a');
                    a.href = '{mp3_url}';
                    a.download = '{filename}';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                """)
                
                time.sleep(0.5)
                total_downloaded += 1
                print("OK")
                
        except Exception as e:
            print(f"    [ERROR] {e}")
    
    print("\n" + "=" * 50)
    print(f"DONE! Downloaded {total_downloaded} files")
    print("=" * 50)
    print("\n>>> Browser will stay open. Close it manually when done.")


if __name__ == "__main__":
    main()