import os, base64, json, requests
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

# ---------- Blob ----------
class BlobClientLite:
    def __init__(self):
        conn = os.getenv("BLOB_CONNECTION_STRING")
        cont = os.getenv("BLOB_CONTAINER", "chartqa")
        if not conn:
            raise RuntimeError("Missing BLOB_CONNECTION_STRING")
        self.client = BlobServiceClient.from_connection_string(conn)
        self.container = self.client.get_container_client(cont)
        try:
            self.container.create_container()
        except Exception:
            pass

    def upload_file(self, local_path: str, remote_path: str) -> str:
        with open(local_path, "rb") as f:
            self.container.upload_blob(name=remote_path, data=f, overwrite=True)
        return remote_path

# ---------- YOLO cls (AML endpoint) ----------
YOLO_URL = os.getenv("YOLO_URL"); YOLO_KEY = os.getenv("YOLO_KEY")

def call_yolo_cls(image_b64: str) -> dict:
    headers = {"Authorization": f"Bearer {YOLO_KEY}", "Content-Type": "application/json"}
    payload = {"image_b64": image_b64}
    r = requests.post(YOLO_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

# ---------- ChartGemma (TGI style) ----------
CG_URL = os.getenv("CHARTGEMMA_URL"); CG_KEY = os.getenv("CHARTGEMMA_KEY")

def call_chartgemma(chart_b64: str, prompt: str) -> str:
    data_uri = f"data:image/png;base64,{chart_b64}"
    inputs = f"![]({data_uri})\n{prompt}\n\n"
    payload = {"inputs": inputs}
    headers = {"Authorization": f"Bearer {CG_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
    r = requests.post(CG_URL, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()
    # TGI returns a list or an object; normalize
    if isinstance(j, list) and j and isinstance(j[0], dict) and "generated_text" in j[0]:
        return j[0]["generated_text"]
    if isinstance(j, dict) and "generated_text" in j:
        return j["generated_text"]
    # fallback
    return json.dumps(j)[:4000]

# ---------- Azure OpenAI (text route) ----------
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AOAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
AOAI_API_VER = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

def call_aoai(context: str, question: str) -> str:
    client = AzureOpenAI(
        api_key=AOAI_KEY, azure_endpoint=AOAI_ENDPOINT, api_version=AOAI_API_VER
    )
    sys_prompt = (
        "You answer questions about a PDF. If the question requires reading a chart image, say you need a chart image. "
        "Otherwise, use the provided OCR text only. Answer concisely."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Context from OCR (may be partial):\n{context[:8000]}"},
        {"role": "user", "content": question}
    ]
    resp = client.chat.completions.create(model=AOAI_DEPLOYMENT, messages=messages, temperature=0.2)
    return resp.choices[0].message.content