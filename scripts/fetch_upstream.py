"""Fetch only the pinned upstream runtime files; verify normalized SHA256."""
from pathlib import Path
import hashlib,json,urllib.request

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'configs/upstream.json').read_text())
dest=root/'third_party/crossmpt';dest.mkdir(parents=True,exist_ok=True)
for item in manifest['files']:
    path=dest/item['path']
    if path.exists():body=path.read_bytes().replace(b'\r\n',b'\n')
    else:body=urllib.request.urlopen(item['url'],timeout=60).read().replace(b'\r\n',b'\n')
    if hashlib.sha256(body).hexdigest()!=item['sha256_lf']:raise ValueError(f"Upstream hash mismatch: {item['path']}")
    path.write_bytes(body);print('Verified',item['path'])
