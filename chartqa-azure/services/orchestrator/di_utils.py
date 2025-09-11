# orchestrator/di_utils.py

import os
import io
import pathlib
from typing import Tuple, Dict, List, Optional
from PIL import Image, ImageFile

# (선택) .env 자동 로드
try:
    from pathlib import Path
    from dotenv import load_dotenv  # optional
    HERE = Path(__file__).resolve()
    ROOT = HERE.parents[2]
    for cand in [ROOT / ".env.orchestrator", ROOT / ".env", HERE.parent / ".env"]:
        if cand.exists():
            load_dotenv(dotenv_path=cand, override=False)
except Exception:
    pass

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---- Context crop params (env tunable) ----
CTX_PAD_W_FRAC = float(os.getenv("CTX_PAD_W_FRAC", "0.12"))   # bbox 폭 대비 가로 여백 비율
CTX_PAD_H_FRAC = float(os.getenv("CTX_PAD_H_FRAC", "0.18"))   # bbox 높이 대비 세로 여백 비율
CTX_MIN_PAD_PX = int(os.getenv("CTX_MIN_PAD_PX", "16"))       # 최소 여백 픽셀
CTX_MAX_W      = int(os.getenv("CTX_MAX_W", "1600"))          # 최종 크롭 최대 폭
CTX_MAX_H      = int(os.getenv("CTX_MAX_H", "1600"))          # 최종 크롭 최대 높이

# ---- Lazy init ----
_client: DocumentIntelligenceClient | None = None

def _get_env(name: str) -> str:
    v = os.getenv(name)
    if not isinstance(v, str) or not v.strip():
        raise RuntimeError(f"Missing required env var: {name}")
    return v.strip()

def _get_client() -> DocumentIntelligenceClient:
    global _client
    if _client is None:
        endpoint = _get_env("DI_ENDPOINT")
        key = _get_env("DI_KEY")
        _client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
    return _client

def _save_png(image_bytes: bytes, out_path: pathlib.Path):
    with Image.open(io.BytesIO(image_bytes)) as im:
        im.convert("RGB").save(out_path, format="PNG")

# ---------- layout 파싱 유틸 ----------

def _collect_figure_regions(obj) -> List[Dict]:
    """
    layout_dict 어디에 있든 figures를 찾아
    [{'id': '2.1', 'page': 2, 'polygon': [x1,y1,...]}] 리스트로 반환.
    순서는 탐색 순서(= DI가 준 순서)에 맞춰 유지.
    """
    results: List[Dict] = []

    def first_region(fig: Dict) -> Optional[Dict]:
        brs = fig.get("boundingRegions") or []
        if brs and isinstance(brs, list):
            br = brs[0]
            page = int(br.get("pageNumber", 0) or 0)
            poly = br.get("polygon") or []
            if isinstance(poly, list) and len(poly) >= 8:
                return {"page": page, "polygon": poly}
        return None

    def walk(o):
        if isinstance(o, dict):
            figs = o.get("figures")
            if isinstance(figs, list):
                for f in figs:
                    fid = f.get("id")
                    if isinstance(fid, str):
                        reg = first_region(f)
                        if reg:
                            results.append({"id": fid, "page": reg["page"], "polygon": reg["polygon"]})
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return results

def _page_dims_map(layout_dict: Dict) -> Dict[int, Dict]:
    """
    {pageNumber: {'width':..., 'height':..., 'unit': str}} 맵 생성
    """
    mp: Dict[int, Dict] = {}
    for p in (layout_dict.get("pages") or []):
        n = int(p.get("pageNumber", 0) or 0)
        if n:
            mp[n] = {
                "width": float(p.get("width", 0) or 0),
                "height": float(p.get("height", 0) or 0),
                "unit": p.get("unit") or "",
            }
    return mp

def extract_page_texts(read_dict: Dict) -> Dict[int, str]:
    texts: Dict[int, str] = {}
    pages = read_dict.get("pages") or []
    for i, pg in enumerate(pages, start=1):
        lines = pg.get("lines") or []
        s = "\n".join((ln.get("content") or "") for ln in lines).strip()
        texts[i] = s
    return texts

# ---------- 메인: READ+LAYOUT ----------

def run_read_and_layout(pdf_path: pathlib.Path, workdir: pathlib.Path) -> Tuple[Dict, Dict, List[str], List[int]]:
    """
    Returns:
      - read_dict
      - layout_dict (meta 필드 포함: _op_id, _fig_ids_order, _fig_polygons, _page_dims)
      - figure_pngs : DI가 잘라준 원본 figure 크롭
      - fig_pages   : 각 figure의 페이지 번호(1-base)
    """
    client = _get_client()
    pdf_bytes = pdf_path.read_bytes()

    # READ
    poller_read = client.begin_analyze_document("prebuilt-read", pdf_bytes, content_type="application/pdf")
    read_res = poller_read.result()
    read_dict = read_res.as_dict()

    # LAYOUT (FIGURES)
    poller_layout = client.begin_analyze_document(
        "prebuilt-layout", pdf_bytes, content_type="application/pdf", output=[AnalyzeOutputOption.FIGURES]
    )
    layout_res = poller_layout.result()
    layout_dict = layout_res.as_dict()

    # 메타
    meta = getattr(poller_layout, "details", {}) or {}
    op_id = meta.get("result_id") or meta.get("operation_id")
    regions = _collect_figure_regions(layout_dict)
    page_dims = _page_dims_map(layout_dict)

    # 저장 폴더
    figs_dir = workdir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    figure_pngs: List[str] = []
    fig_pages: List[int] = []
    fig_ids_order: List[str] = []
    fig_polygons: Dict[str, List[float]] = {}

    for i, item in enumerate(regions):
        fid, page, poly = item["id"], item["page"], item["polygon"]
        try:
            it = client.get_analyze_result_figure("prebuilt-layout", op_id, fid)
            b = b"".join(it)
            out = figs_dir / f"figure_{i:03d}.png"
            _save_png(b, out)
            figure_pngs.append(str(out))
            fig_pages.append(page)
            fig_ids_order.append(fid)
            fig_polygons[fid] = poly
        except Exception as e:
            print("[DI] figure fetch failed:", fid, e)

    # layout_dict에 메타 심기 (후속 함수에서 사용)
    layout_dict["_op_id"] = op_id
    layout_dict["_fig_ids_order"] = fig_ids_order
    layout_dict["_fig_polygons"] = fig_polygons
    layout_dict["_page_dims"] = page_dims

    return read_dict, layout_dict, figure_pngs, fig_pages

# ---------- 페이지 이미지 기반 '맥락 크롭' 생성 ----------

def _fetch_page_png_bytes(pdf_path: pathlib.Path, layout_dict: Dict, page: int) -> Optional[bytes]:
    """
    1) DI page image API 시도
    2) PyMuPDF(fitz) 폴백
    3) pdf2image 폴백
    """
    client = _get_client()
    op_id = layout_dict.get("_op_id")
    # 1) DI page image
    try:
        it = client.get_analyze_result_page_image("prebuilt-layout", op_id, page)  # SDK 1.0+
        return b"".join(it)
    except Exception:
        pass

    # 2) PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pg = doc.load_page(page - 1)
        pix = pg.get_pixmap()
        return pix.tobytes("png")
    except Exception:
        pass

    # 3) pdf2image
    try:
        from pdf2image import convert_from_path
        imgs = convert_from_path(str(pdf_path), fmt="png", first_page=page, last_page=page)
        if imgs:
            buf = io.BytesIO()
            imgs[0].save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        pass

    return None

def _poly_to_bbox(poly: List[float]) -> Tuple[float, float, float, float]:
    xs = poly[0::2]; ys = poly[1::2]
    return min(xs), min(ys), max(xs), max(ys)

def _clip(v, lo, hi):  # noqa
    return max(lo, min(hi, v))

def make_contextual_crops(
    pdf_path: pathlib.Path,
    workdir: pathlib.Path,
    read_dict: Dict,
    layout_dict: Dict,
) -> List[Optional[str]]:
    """
    layout 메타(도형 polygon, page dims)를 사용해
    '페이지 이미지에서 여백을 준 차트 크롭'을 생성.
    Returns: ctx_paths (len == number of figures), 실패 시 None.
    """
    fig_ids = layout_dict.get("_fig_ids_order", []) or []
    fig_polys: Dict[str, List[float]] = layout_dict.get("_fig_polygons", {}) or {}
    page_dims: Dict[int, Dict] = layout_dict.get("_page_dims", {}) or {}

    # 페이지 이미지 캐시
    pages_dir = workdir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_cache: Dict[int, pathlib.Path] = {}

    ctx_dir = workdir / "figures_ctx"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    ctx_paths: List[Optional[str]] = [None] * len(fig_ids)

    for i, fid in enumerate(fig_ids):
        poly = fig_polys.get(fid)
        # fid "P.I" → page
        try:
            page = int(str(fid).split(".")[0])
        except Exception:
            page = 0
        if not poly or page <= 0:
            continue

        # 페이지 PNG 준비
        if page not in page_cache:
            b = _fetch_page_png_bytes(pdf_path, layout_dict, page)
            if not b:
                continue
            ppath = pages_dir / f"page_{page:03d}.png"
            _save_png(b, ppath)
            page_cache[page] = ppath
        pimg_path = page_cache[page]

        # 배율 계산: layout 좌표 → 페이지 픽셀
        dims = page_dims.get(page) or {}
        pw = float(dims.get("width", 0) or 0)
        ph = float(dims.get("height", 0) or 0)
        if pw <= 0 or ph <= 0:
            # 페이지 실제 픽셀 크기로 대체 (비율 추정은 문제 없게 1:1)
            with Image.open(pimg_path) as im:
                pw, ph = im.size

        x0, y0, x1, y1 = _poly_to_bbox(poly)
        # 정규화 후 픽셀 bbox
        with Image.open(pimg_path) as pim:
            W, H = pim.size
            rx0 = x0 / pw; ry0 = y0 / ph; rx1 = x1 / pw; ry1 = y1 / ph
            bx0 = int(_clip(rx0 * W, 0, W))
            by0 = int(_clip(ry0 * H, 0, H))
            bx1 = int(_clip(rx1 * W, 0, W))
            by1 = int(_clip(ry1 * H, 0, H))

            bw = max(1, bx1 - bx0); bh = max(1, by1 - by0)
            pad_w = max(CTX_MIN_PAD_PX, int(bw * CTX_PAD_W_FRAC))
            pad_h = max(CTX_MIN_PAD_PX, int(bh * CTX_PAD_H_FRAC))

            cx0 = _clip(bx0 - pad_w, 0, W)
            cy0 = _clip(by0 - pad_h, 0, H)
            cx1 = _clip(bx1 + pad_w, 0, W)
            cy1 = _clip(by1 + pad_h, 0, H)

            crop = pim.crop((cx0, cy0, cx1, cy1))

            # 과도히 큰 경우 리사이즈
            cw, ch = crop.size
            scale = min(1.0, CTX_MAX_W / max(1, cw), CTX_MAX_H / max(1, ch))
            if scale < 1.0:
                crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)

            out = ctx_dir / f"figure_{i:03d}_ctx.png"
            crop.save(out, format="PNG")
            ctx_paths[i] = str(out)

    return ctx_paths

# import os
# import io
# import pathlib
# from typing import Tuple, Dict, List
# from PIL import Image, ImageFile

# # (선택) .env 자동 로드
# try:
#     from pathlib import Path
#     from dotenv import load_dotenv  # optional
#     HERE = Path(__file__).resolve()
#     ROOT = HERE.parents[2]
#     for cand in [ROOT / ".env.orchestrator", ROOT / ".env", HERE.parent / ".env"]:
#         if cand.exists():
#             load_dotenv(dotenv_path=cand, override=False)
# except Exception:
#     pass

# from azure.core.credentials import AzureKeyCredential
# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from azure.ai.documentintelligence.models import AnalyzeOutputOption

# # figure 바이트 조각 허용
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # ---- Lazy init ----
# _client: DocumentIntelligenceClient | None = None

# def _get_env(name: str) -> str:
#     v = os.getenv(name)
#     if not isinstance(v, str) or not v.strip():
#         raise RuntimeError(f"Missing required env var: {name}")
#     return v.strip()

# def _get_client() -> DocumentIntelligenceClient:
#     global _client
#     if _client is None:
#         endpoint = _get_env("DI_ENDPOINT")
#         key = _get_env("DI_KEY")
#         _client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
#     return _client

# def _save_png(image_bytes: bytes, out_path: pathlib.Path):
#     im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     im.save(out_path, format="PNG")

# def _collect_figure_ids(obj) -> List[str]:
#     """layout_dict 깊은 곳 어디에 있든 figures.id 리스트를 모아 반환"""
#     ids: List[str] = []
#     def walk(o):
#         if isinstance(o, dict):
#             figs = o.get("figures")
#             if isinstance(figs, list):
#                 for f in figs:
#                     fid = f.get("id")
#                     if isinstance(fid, str):
#                         ids.append(fid)
#             for v in o.values():
#                 walk(v)
#         elif isinstance(o, list):
#             for v in o:
#                 walk(v)
#     walk(obj)
#     return ids

# def extract_page_texts(read_dict: Dict) -> Dict[int, str]:
#     """
#     READ 결과에서 페이지별 텍스트 합치기 (페이지 번호는 1부터).
#     """
#     texts: Dict[int, str] = {}
#     pages = read_dict.get("pages") or []
#     for i, pg in enumerate(pages, start=1):
#         lines = pg.get("lines") or []
#         s = "\n".join((ln.get("content") or "") for ln in lines).strip()
#         texts[i] = s
#     return texts

# def run_read_and_layout(pdf_path: pathlib.Path, workdir: pathlib.Path) -> Tuple[Dict, Dict, List[str], List[int]]:
#     """
#     Returns:
#       - read_dict: READ OCR dict
#       - layout_dict: LAYOUT dict
#       - figure_pngs: 저장된 figure PNG 경로 리스트
#       - fig_pages: 각 figure_png가 속한 페이지 번호(1-base). 알 수 없으면 0
#     """
#     client = _get_client()
#     pdf_bytes = pdf_path.read_bytes()

#     # 1) READ
#     poller_read = client.begin_analyze_document(
#         "prebuilt-read", pdf_bytes, content_type="application/pdf"
#     )
#     read_res = poller_read.result()
#     read_dict = read_res.as_dict()

#     # 2) LAYOUT (FIGURES)
#     poller_layout = client.begin_analyze_document(
#         "prebuilt-layout", pdf_bytes, content_type="application/pdf",
#         output=[AnalyzeOutputOption.FIGURES]
#     )
#     layout_res = poller_layout.result()
#     layout_dict = layout_res.as_dict()

#     # 3) figure id 수집 및 페이지 번호 추출
#     ids = _collect_figure_ids(layout_dict)
#     # id 예시: "2.1" → page=2
#     def _parse_page(fid: str) -> int:
#         try:
#             return int(str(fid).split(".")[0])
#         except Exception:
#             return 0

#     meta = getattr(poller_layout, "details", {}) or {}
#     op_id = meta.get("result_id") or meta.get("operation_id")

#     figs_dir = workdir / "figures"
#     figs_dir.mkdir(parents=True, exist_ok=True)

#     figure_pngs: List[str] = []
#     fig_pages: List[int] = []
#     for i, fid in enumerate(ids):
#         try:
#             it = client.get_analyze_result_figure("prebuilt-layout", op_id, fid)
#             b = b"".join(it)
#             out = figs_dir / f"figure_{i:03d}.png"
#             _save_png(b, out)
#             figure_pngs.append(str(out))
#             fig_pages.append(_parse_page(fid))
#         except Exception as e:
#             print("[DI] figure fetch failed:", fid, e)

#     return read_dict, layout_dict, figure_pngs, fig_pages

# import os
# import io
# import json
# import pathlib
# from typing import Tuple, Dict, List
# from PIL import Image, ImageFile

# # (선택) .env 자동 로드: 루트/.env.orchestrator → 루트/.env → 현재/.env 순
# try:
#     from pathlib import Path
#     from dotenv import load_dotenv  # python-dotenv가 없으면 그냥 패스
#     HERE = Path(__file__).resolve()
#     ROOT = HERE.parents[2]  # chartqa-azure/ 루트
#     for cand in [ROOT / ".env.orchestrator", ROOT / ".env", HERE.parent / ".env"]:
#         if cand.exists():
#             load_dotenv(dotenv_path=cand, override=False)
# except Exception:
#     pass

# from azure.core.credentials import AzureKeyCredential
# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from azure.ai.documentintelligence.models import AnalyzeOutputOption

# # DI figure 바이트가 조각나도 저장하도록 설정
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # ---- Lazy init to avoid import-time crash ----
# _client: DocumentIntelligenceClient | None = None

# def _get_env(name: str) -> str:
#     v = os.getenv(name)
#     if not isinstance(v, str) or not v.strip():
#         raise RuntimeError(f"Missing required env var: {name}")
#     return v.strip()

# def _get_client() -> DocumentIntelligenceClient:
#     global _client
#     if _client is None:
#         endpoint = _get_env("DI_ENDPOINT")
#         key = _get_env("DI_KEY")
#         _client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
#     return _client

# def _save_png(image_bytes: bytes, out_path: pathlib.Path):
#     im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     im.save(out_path, format="PNG")

# def run_read_and_layout(pdf_path: pathlib.Path, workdir: pathlib.Path) -> Tuple[Dict, Dict, List[str]]:
#     client = _get_client()  # ← 실제 호출 시점에 생성
#     pdf_bytes = pdf_path.read_bytes()

#     # 1) READ (OCR)
#     poller_read = client.begin_analyze_document(
#         "prebuilt-read", pdf_bytes, content_type="application/pdf"
#     )
#     read_res = poller_read.result()
#     read_dict = read_res.as_dict()

#     # 2) LAYOUT (FIGURES)
#     poller_layout = client.begin_analyze_document(
#         "prebuilt-layout",
#         pdf_bytes,
#         content_type="application/pdf",
#         output=[AnalyzeOutputOption.FIGURES]
#     )
#     layout_res = poller_layout.result()
#     layout_dict = layout_res.as_dict()

#     # 3) figure id 수집 (dict 어디에 있든 재귀 탐색)
#     def collect_ids(obj):
#         ids = []
#         def walk(o):
#             if isinstance(o, dict):
#                 if isinstance(o.get("figures"), list):
#                     for f in o["figures"]:
#                         fid = f.get("id")
#                         if isinstance(fid, str):
#                             ids.append(fid)
#                 for v in o.values():
#                     walk(v)
#             elif isinstance(o, list):
#                 for v in o:
#                     walk(v)
#         walk(obj)
#         return ids

#     ids = collect_ids(layout_dict)

#     # 4) figure PNG 저장
#     meta = getattr(poller_layout, "details", {}) or {}
#     op_id = meta.get("result_id") or meta.get("operation_id")
#     figs_dir = workdir / "figures"
#     figs_dir.mkdir(parents=True, exist_ok=True)

#     figs: List[str] = []
#     for i, fid in enumerate(ids):
#         try:
#             it = client.get_analyze_result_figure("prebuilt-layout", op_id, fid)
#             b = b"".join(it)
#             out = figs_dir / f"figure_{i:03d}.png"
#             _save_png(b, out)
#             figs.append(str(out))
#         except Exception as e:
#             print("[DI] figure fetch failed:", fid, e)

#     return read_dict, layout_dict, figs


# import os, io, json, pathlib
# from typing import Tuple, Dict, List
# from PIL import Image, ImageFile
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from azure.ai.documentintelligence.models import AnalyzeOutputOption

# ImageFile.LOAD_TRUNCATED_IMAGES = True  # figure 바이트 저장시 안전

# DI_ENDPOINT = os.getenv("DI_ENDPOINT")
# DI_KEY = os.getenv("DI_KEY")

# client = DocumentIntelligenceClient(DI_ENDPOINT, AzureKeyCredential(DI_KEY))

# def _save_png(image_bytes: bytes, out_path: pathlib.Path):
#     im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     im.save(out_path, format="PNG")

# def run_read_and_layout(pdf_path: pathlib.Path, workdir: pathlib.Path) -> Tuple[Dict, Dict, List[str]]:
#     pdf_bytes = pdf_path.read_bytes()

#     poller_read = client.begin_analyze_document(
#         "prebuilt-read", pdf_bytes, content_type="application/pdf"
#     )
#     read_res = poller_read.result()
#     read_dict = read_res.as_dict()

#     poller_layout = client.begin_analyze_document(
#         "prebuilt-layout", pdf_bytes, content_type="application/pdf",
#         output=[AnalyzeOutputOption.FIGURES]
#     )
#     layout_res = poller_layout.result()
#     layout_dict = layout_res.as_dict()

#     # layout_dict 전역에서 figure id 수집
#     def collect_ids(obj):
#         ids = []
#         def walk(o):
#             if isinstance(o, dict):
#                 if isinstance(o.get("figures"), list):
#                     for f in o["figures"]:
#                         fid = f.get("id")
#                         if isinstance(fid, str):
#                             ids.append(fid)
#                 for v in o.values():
#                     walk(v)
#             elif isinstance(o, list):
#                 for v in o:
#                     walk(v)
#         walk(obj)
#         return ids

#     figs = []
#     ids = collect_ids(layout_dict)
#     meta = getattr(poller_layout, "details", {}) or {}
#     op_id = meta.get("result_id") or meta.get("operation_id")

#     figs_dir = workdir / "figures"
#     figs_dir.mkdir(parents=True, exist_ok=True)

#     for i, fid in enumerate(ids):
#         try:
#             it = client.get_analyze_result_figure("prebuilt-layout", op_id, fid)
#             b = b"".join(it)
#             out = figs_dir / f"figure_{i:03d}.png"
#             _save_png(b, out)
#             figs.append(str(out))
#         except Exception as e:
#             print("[DI] figure fetch failed:", fid, e)

#     return read_dict, layout_dict, figs