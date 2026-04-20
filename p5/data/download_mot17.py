import os
import requests
import zipfile
from tqdm import tqdm

def download_mot17(dest_path='data/raw'):
    """
    Télécharge et extrait le dataset MOT17 depuis Kaggle ou un miroir.
    """
    dataset_slug = "wenhoujinjust/mot-17"
    extract_path = os.path.join(dest_path, "MOT17")
    zip_path = os.path.join(dest_path, "mot-17.zip")

    if os.path.exists(extract_path):
        print(f"Dataset déjà présent dans {extract_path}")
        return

    os.makedirs(dest_path, exist_ok=True)

    # Tentative via Kaggle API
    try:
        import kaggle
        print(f"Tentative de téléchargement depuis Kaggle ({dataset_slug})...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_slug, path=dest_path, unzip=False)
        
        # Le fichier téléchargé par Kaggle s'appelle souvent [slug].zip
        # On doit le renommer ou vérifier son nom
        expected_zip = os.path.join(dest_path, "mot-17.zip")
        if os.path.exists(expected_zip):
            print("Extraction du fichier Kaggle...")
            with zipfile.ZipFile(expected_zip, 'r') as zip_ref:
                zip_ref.extractall(dest_path)
            os.remove(expected_zip)
            print("Installation via Kaggle terminée.")
            return
    except Exception as e:
        print(f"Kaggle API non configurée ou erreur : {e}")
        print("Passage au miroir alternatif...")

    # Fallback sur le miroir HyperAI
    url = "https://data.hyper.ai/download/datasets/mot17/MOT17.zip"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    zip_path_alt = os.path.join(dest_path, "MOT17.zip")
    
    print(f"Téléchargement depuis le miroir {url}...")
    response = requests.get(url, stream=True, headers=headers)
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path_alt, "wb") as f, tqdm(
        desc="Progression",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    print("Extraction du fichier ZIP...")
    with zipfile.ZipFile(zip_path_alt, 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    os.remove(zip_path_alt)
    print("Installation terminée.")

if __name__ == "__main__":
    download_mot17()
