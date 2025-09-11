# services/yolo-local/tests/yolo_http_test.py
# services/yolo-local/tests/yolo_http_test.py
import argparse, base64, json, os, sys, pathlib, requests

def to_b64(p: pathlib.Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")

def hit(server: str, img_path: pathlib.Path, timeout: int = 30):
    b64 = to_b64(img_path)
    url = server.rstrip("/") + "/classify"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {"image_b64": b64}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        # 🔎 서버가 왜 400을 보냈는지 바디를 그대로 보여줌
        print(f"[HTTP {r.status_code}] {url}")
        print("---- response text ----")
        print(r.text[:2000])
        print("-----------------------")
        raise
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:7001", help="YOLO server base URL")
    ap.add_argument("--path", required=True, help="image file or directory")
    args = ap.parse_args()

    p = pathlib.Path(args.path)
    if p.is_dir():
        imgs = sorted([x for x in p.iterdir() if x.suffix.lower() in {".png",".jpg",".jpeg",".bmp"}])
        if not imgs:
            print("[ERR] no images in dir", file=sys.stderr); sys.exit(2)
        for im in imgs:
            try:
                out = hit(args.server, im)
                print(json.dumps({"file": str(im), "out": out}, ensure_ascii=False))
            except Exception as e:
                print(json.dumps({"file": str(im), "error": str(e)}), file=sys.stderr)
    else:
        out = hit(args.server, p)
        print(json.dumps({"file": str(p), "out": out}, ensure_ascii=False))

if __name__ == "__main__":
    main()

# python services/yolo-local/tests/yolo_http_test.py --server http://127.0.0.1:7001 --path C:/Users/t-yooyeunkim/Pictures/test.jpg