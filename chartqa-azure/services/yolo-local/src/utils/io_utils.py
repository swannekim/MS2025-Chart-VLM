import base64, re
from io import BytesIO
from PIL import Image

_DATAURI_RE = re.compile(r"^data:image/[^;]+;base64,")

def pil_from_base64(b64: str) -> Image.Image:
    if _DATAURI_RE.match(b64):
        b64 = _DATAURI_RE.sub("", b64)
    data = base64.b64decode(b64)
    img = Image.open(BytesIO(data)).convert("RGB")
    return img
