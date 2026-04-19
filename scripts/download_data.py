import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DEST = Path("data")


def download():
    DEST.mkdir(exist_ok=True)
    zip_path = DEST / "online_retail_ii.zip"
    print("Downloading...")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DEST)
    print("Done. Files:", list(DEST.iterdir()))


if __name__ == "__main__":
    download()
