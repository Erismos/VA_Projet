import os
import glob
import requests
import zipfile
from tqdm import tqdm

def download_mot17(dest_path='data/raw'):
    """
    Télécharge et extrait le dataset MOT17 depuis Kaggle en priorité,
    avec un miroir HTTP en secours.
    """
    dataset_slug = "wenhoujinjust/mot-17"
    extract_path = os.path.join(dest_path, "MOT17")

    if os.path.exists(extract_path):
        print(f"Dataset déjà présent dans {extract_path}")
        return

    os.makedirs(dest_path, exist_ok=True)

    def extract_zip(zip_path: str) -> None:
        print(f"Extraction du fichier ZIP: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        os.remove(zip_path)

    # Tentative via Kaggle API
    try:
        import kaggle

        print(f"Tentative de téléchargement depuis Kaggle ({dataset_slug})...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_slug, path=dest_path, unzip=False)

        zip_candidates = sorted(
            glob.glob(os.path.join(dest_path, "*.zip")),
            key=os.path.getmtime,
            reverse=True,
        )
        if zip_candidates:
            extract_zip(zip_candidates[0])
            print("Installation via Kaggle terminée.")
            return

        print("Aucune archive Kaggle trouvée, passage au miroir HTTP...")
    except Exception as e:
        print(f"Kaggle API non disponible ou erreur: {e}")
        print("Passage au miroir HTTP...")

    url = "https://data.hyper.ai/download/datasets/mot17/MOT17.zip"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    zip_path_alt = os.path.join(dest_path, "MOT17.zip")
    
    print(f"Téléchargement depuis le miroir {url}...")
    response = requests.get(url, stream=True, headers=headers, timeout=120)
    response.raise_for_status()
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

    extract_zip(zip_path_alt)
    print("Installation terminée.")

if __name__ == "__main__":
    download_mot17()
