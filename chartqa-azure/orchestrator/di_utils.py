import os, io, json, pathlib
from typing import Tuple, Dict, List
from PIL import Image, ImageFile
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption

ImageFile.LOAD_TRUNCATED_IMAGES = True

DI_ENDPOINT = os.getenv("DI_ENDPOINT")
DI_KEY = os.getenv("DI_KEY")

client = DocumentIntelligenceClient(DI_ENDPOINT, AzureKeyCredential(DI_KEY))


def _save_png(image_bytes: bytes, out_path: pathlib.Path):
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    im.save(out_path, format="PNG")


def run_read_and_layout(pdf_path: pathlib.Path, workdir: pathlib.Path) -> Tuple[Dict, Dict, List[str]]:
    pdf_bytes = pdf_path.read_bytes()

    poller_read = client.begin_analyze_document(
        "prebuilt-read", pdf_bytes, content_type="application/pdf"
    )
    read_res = poller_read.result(); read_dict = read_res.as_dict()

    poller_layout = client.begin_analyze_document(
        "prebuilt-layout", pdf_bytes, content_type="application/pdf",
        output=[AnalyzeOutputOption.FIGURES]
    )
    layout_res = poller_layout.result(); layout_dict = layout_res.as_dict()

    # collect figure ids
    def collect_ids(obj):
        ids = []
        def walk(o):
            if isinstance(o, dict):
                if isinstance(o.get("figures"), list):
                    for f in o["figures"]:
                        fid = f.get("id");
                        if isinstance(fid, str): ids.append(fid)
                for v in o.values(): walk(v)
            elif isinstance(o, list):
                for v in o: walk(v)
        walk(obj); return ids

    figs = []
    ids = collect_ids(layout_dict)
    meta = getattr(poller_layout, "details", {}) or {}
    op_id = meta.get("result_id") or meta.get("operation_id")
    figs_dir = workdir/"figures"; figs_dir.mkdir(parents=True, exist_ok=True)
    for i, fid in enumerate(ids):
        try:
            it = client.get_analyze_result_figure("prebuilt-layout", op_id, fid)
            b = b"".join(it)
            out = figs_dir/f"figure_{i:03d}.png"; _save_png(b, out)
            figs.append(str(out))
        except Exception as e:
            print("[DI] figure fetch failed:", fid, e)
    return read_dict, layout_dict, figs