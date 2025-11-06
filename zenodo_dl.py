#!/usr/bin/env python3
import os, sys, time, math, hashlib, requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from tqdm import tqdm

# -------- Requests session with sensible retries --------
def make_session():
    s = requests.Session()
    retry = Retry(
        total=6, connect=6, read=6,
        backoff_factor=1.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    token = os.environ.get("ZENODO_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s

# -------- Records API helpers --------
def fetch_record_files(session, record_id):
    """
    Returns list of dicts: {'key', 'download', 'checksum', 'size'}
    Works with both old (files=[]) and new (assets.files=[]) layouts.
    """
    url = f"https://zenodo.org/api/records/{record_id}"
    r = session.get(url)
    r.raise_for_status()
    data = r.json()

    files = []
    raw_files = data.get("files", None)
    if raw_files is None:
        raw_files = data.get("assets", {}).get("files", [])
    for f in raw_files:
        key = f.get("key") or f.get("filename")
        links = f.get("links", {})
        download = links.get("download") or (links.get("self") + "?download=1" if links.get("self") else None)
        checksum = f.get("checksum")
        size = f.get("size")
        if key and download:
            files.append({"key": key, "download": download, "checksum": checksum, "size": size})
    if not files:
        raise RuntimeError("No downloadable files found in record JSON.")
    return files

# -------- Checksums --------
def compute_hash(path, algo_name):
    h = hashlib.new(algo_name)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_checksum(path, expected_checksum):
    """
    expected_checksum looks like 'md5:abc123...' or 'sha256:deadbeef...'
    """
    if not expected_checksum:
        return True  # nothing to verify against
    if ":" in expected_checksum:
        algo, hexdigest = expected_checksum.split(":", 1)
    else:
        # fallback (rare)
        algo, hexdigest = "md5", expected_checksum
    algo = algo.lower()
    if algo not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported checksum algorithm from Zenodo: {algo}")
    return compute_hash(path, algo) == hexdigest

# -------- Download with resume & progress --------
def get_total_from_headers(resp, existing_bytes):
    # If server provides Content-Range, parse total; else add content-length+existing
    cr = resp.headers.get("Content-Range")
    if cr and "/" in cr:
        try:
            total = int(cr.split("/")[-1])
            return total
        except Exception:
            pass
    cl = resp.headers.get("Content-Length")
    if cl is not None:
        try:
            return int(cl) + existing_bytes
        except Exception:
            pass
    return None

def download_with_resume(session, url, dest_path, expected_checksum):
    tmp_path = dest_path + ".part"
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    existing = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    # HEAD for a nicer total size (optional)
    total_size = None
    try:
        hr = session.head(url, allow_redirects=True)
        if hr.ok and "Content-Length" in hr.headers:
            total_size = int(hr.headers["Content-Length"])
            if existing and "Accept-Ranges" not in hr.headers and existing < total_size:
                # Server might not support range; start fresh
                existing, headers = 0, {}
    except requests.RequestException:
        pass

    # GET with streaming; if server ignores Range we’ll overwrite
    with session.get(url, headers=headers, stream=True) as r:
        if existing > 0 and r.status_code == 200:
            # Range ignored; start over
            existing = 0
        r.raise_for_status()

        # Update total for progress bar
        total = get_total_from_headers(r, existing) or total_size
        mode = "ab" if existing > 0 and r.status_code == 206 else "wb"

        with open(tmp_path, mode) as f, tqdm(
            total=total, initial=existing if total else 0,
            unit="B", unit_scale=True, desc=os.path.basename(dest_path)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    # Verify and finalize
    if verify_checksum(tmp_path, expected_checksum):
        os.replace(tmp_path, dest_path)
        return True
    else:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        return False

# -------- Main --------
def main(record_id):
    out_dir = str(record_id)
    os.makedirs(out_dir, exist_ok=True)

    s = make_session()
    files = fetch_record_files(s, record_id)

    for f in files:
        filename = f["key"]
        url = f["download"]
        checksum = f.get("checksum")
        dest = os.path.join(out_dir, filename)

        # If already present and checksum matches, skip
        if os.path.exists(dest):
            print(f"Checking existing file: {filename}")
            try:
                if verify_checksum(dest, checksum):
                    print(f"  OK — checksum matches. Skipping.")
                    continue
                else:
                    print(f"  Mismatch — will re-download (with resume support).")
            except Exception as e:
                print(f"  Checksum check skipped ({e}). Will ensure file is complete.")

        # Try a few attempts around the whole transfer (in addition to per-request retries)
        attempts = 3
        for i in range(1, attempts + 1):
            try:
                ok = download_with_resume(s, url, dest, checksum)
                if ok:
                    print(f"{filename}: Downloaded and verified.")
                    break
                else:
                    print(f"{filename}: Checksum failed (attempt {i}/{attempts}).")
            except requests.RequestException as e:
                print(f"{filename}: Network error (attempt {i}/{attempts}): {e}")
            # brief backoff
            time.sleep(int(1.5 ** i))
        else:
            print(f"{filename}: Failed after {attempts} attempts.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python zenodo_dl.py <record_id>")
        sys.exit(1)
    main(sys.argv[1])

