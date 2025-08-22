import os, io, json, argparse, pathlib, sys
from dotenv import load_dotenv
from typing import Dict, List, Set
from PIL import Image, ImageFile

# Pillow가 바이트 스트림/부분 손상에 관대하도록 설정 (DI figure 바이트 저장시 유용)
# 참고: "Image file is truncated" 관련 Q&A 및 PIL 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True  # noqa

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption

load_dotenv()

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def read_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"[ERR] Missing env var: {name}", file=sys.stderr)
        sys.exit(2)
    return val

def summarize_read_result(rdict: Dict) -> Dict:
    pages = rdict.get("pages", [])
    n_pages = len(pages)
    n_lines = 0
    sample_lines = []
    for p in pages:
        for line in p.get("lines", []):
            text = line.get("content", "")
            n_lines += 1
            if len(sample_lines) < 5 and text:
                sample_lines.append(text[:120])
    # 합쳐진 텍스트 길이
    all_text = []
    for p in pages:
        for line in p.get("lines", []):
            all_text.append(line.get("content", ""))
    return {
        "pages": n_pages,
        "lines": n_lines,
        "text_chars": sum(len(x) for x in all_text),
        "sample_lines": sample_lines
    }

def collect_figure_ids(result_dict: Dict) -> List[str]:
    """결과 dict 안에서 'figures' 항목을 찾아 id 리스트를 뽑는다(위치가 상/하위 어디든 재귀 탐색)."""
    ids: List[str] = []
    seen: Set[str] = set()

    def walk(obj):
        if isinstance(obj, dict):
            if "figures" in obj and isinstance(obj["figures"], list):
                for fig in obj["figures"]:
                    fid = fig.get("id")
                    if isinstance(fid, str) and fid not in seen:
                        ids.append(fid); seen.add(fid)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(result_dict)
    return ids

def save_png(image_bytes: bytes, out_path: pathlib.Path):
    with Image.open(io.BytesIO(image_bytes)) as im:
        im.convert("RGB").save(out_path, format="PNG")

def write_text_outputs(read_dict: Dict, out_dir: pathlib.Path, use_paragraphs: bool = False):
    """
    read_dict로부터 텍스트만 추출해서 파일로 저장.
    - out/text.txt : 전체 페이지 텍스트(페이지 구분 포함)
    - out/text_by_page/page_XXX.txt : 페이지별 텍스트
    - (옵션) use_paragraphs=True면 paragraphs 기반 단일 파일도 생성
    """
    # 1) 페이지/라인 기반 (가장 견고)
    pages = read_dict.get("pages", []) or []
    bypage_dir = out_dir / "text_by_page"
    ensure_dir(bypage_dir)

    all_parts = []
    for i, p in enumerate(pages, start=1):
        lines = p.get("lines", []) or []
        page_text = "\n".join(line.get("content", "") for line in lines).strip()
        (bypage_dir / f"page_{i:03d}.txt").write_text(page_text, encoding="utf-8")
        all_parts.append(f"=== [Page {i}] ===\n{page_text}")

    all_text_path = out_dir / "text.txt"
    all_text_path.write_text("\n\n".join(all_parts).strip(), encoding="utf-8")

    # 2) (옵션) paragraph 기반 단일 파일도 원하면 켜기
    if use_paragraphs and read_dict.get("paragraphs"):
        para_text = "\n\n".join(
            (p.get("content", "") or "").strip()
            for p in read_dict["paragraphs"]
            if p.get("content")
        ).strip()
        (out_dir / "text_paragraphs.txt").write_text(para_text, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="테스트할 PDF 경로")
    ap.add_argument("--out", default="./out", help="결과 저장 폴더")
    ap.add_argument("--save-searchable-pdf", action="store_true",
                    help="prebuilt-read 호출 시 PDF 출력 옵션 활성화 후 get_analyze_result_pdf로 저장")
    args = ap.parse_args()

    endpoint = read_env("DI_ENDPOINT")
    key = read_env("DI_KEY")

    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
    print(f"[INFO] Document Intelligence Client initialized!")

    pdf_path = pathlib.Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERR] PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = pathlib.Path(args.out)
    figs_dir = out_dir / "figures"
    ensure_dir(out_dir); ensure_dir(figs_dir)

    pdf_bytes = pdf_path.read_bytes()

    # 1) READ (OCR) - 텍스트 추출 (옵션으로 PDF 출력)
    read_outputs = [AnalyzeOutputOption.PDF] if args.save_searchable_pdf else None
    poller_read = client.begin_analyze_document(
        "prebuilt-read",
        pdf_bytes,                             # <-- body (positional)
        content_type="application/pdf",        # <-- 중요
        output=read_outputs                    # e.g., [AnalyzeOutputOption.PDF] or None
    )
    read_result = poller_read.result()
    # read_dict = read_result.to_dict() if hasattr(read_result, "to_dict") else json.loads(read_result.as_json())  # 안전처리
    read_dict = read_result.as_dict()
    # read_operation_id = poller_read.details.get("operation_id", "")
    read_meta = getattr(poller_read, "details", {}) or {}
    read_operation_id = read_meta.get("result_id") or read_meta.get("operation_id") or ""

    read_summary = summarize_read_result(read_dict)
    (out_dir / "read_result.json").write_text(json.dumps(read_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "read_summary.json").write_text(json.dumps(read_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_text_outputs(read_dict, out_dir, use_paragraphs=False)  # True로 주면 paragraphs 버전도 생성

    # (옵션) 검색 가능한 PDF 저장
    saved_pdf = None
    if args.save_searchable_pdf and read_operation_id:
        try:
            chunks = client.get_analyze_result_pdf("prebuilt-read", read_operation_id)
            pdf_data = b"".join(chunks)
            saved_pdf = out_dir / "searchable.pdf"
            saved_pdf.write_bytes(pdf_data)
        except Exception as e:
            print(f"[WARN] get_analyze_result_pdf failed: {e}")

    # 2) LAYOUT (FIGURES) - 그림 메타데이터 + 크롭 이미지 bytes 저장
    poller_layout = client.begin_analyze_document(
        "prebuilt-layout",
        pdf_bytes,                              # <-- body
        content_type="application/pdf",         # <-- 중요
        output=[AnalyzeOutputOption.FIGURES]    # <-- figure 크롭 생성
    )
    layout_result = poller_layout.result()
    # layout_dict = layout_result.to_dict() if hasattr(layout_result, "to_dict") else json.loads(layout_result.as_json())
    layout_dict = layout_result.as_dict()
    # layout_operation_id = poller_layout.details.get("operation_id", "")
    layout_meta = getattr(poller_layout, "details", {}) or {}
    layout_operation_id = layout_meta.get("result_id") or layout_meta.get("operation_id") or ""

    (out_dir / "layout_result.json").write_text(json.dumps(layout_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_ids = collect_figure_ids(layout_dict)

    saved_figs = []
    for i, fid in enumerate(figure_ids):
        try:
            fig_iter = client.get_analyze_result_figure("prebuilt-layout", layout_operation_id, fid)
            fig_bytes = b"".join(fig_iter)
            out_png = figs_dir / f"figure_{i:03d}.png"
            save_png(fig_bytes, out_png)
            saved_figs.append(str(out_png))
        except Exception as e:
            print(f"[WARN] figure {fid} download/save failed: {e}")

    summary = {
        "file": str(pdf_path),
        "ocr_pages": read_summary["pages"],
        "ocr_lines": read_summary["lines"],
        "ocr_text_chars": read_summary["text_chars"],
        "sample_lines": read_summary["sample_lines"],
        "figures_detected": len(figure_ids),
        "figures_saved_png": len(saved_figs),
        "searchable_pdf_saved": bool(saved_pdf)
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


# # 기본 검증 (OCR + figure PNG 저장)
# python main.py --pdf "./samples/테스트.pdf"

# # 검색 가능한 PDF까지 저장하고 싶으면:
# python main.py --pdf "./samples/테스트.pdf" --save-searchable-pdf

# # 출력 폴더 바꾸고 싶으면:
# python main.py --pdf "./samples/테스트.pdf" --out "./out_check"
