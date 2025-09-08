import os, io, json, base64, tempfile, pathlib, time
from typing import List, Dict, Tuple
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from clients import BlobClientLite, call_yolo_cls, call_chartgemma, call_aoai
from di_utils import run_read_and_layout
from routing import decide_route

API_KEY = os.getenv("ORCH_API_KEY", "")

app = FastAPI(title="ChartQA Orchestrator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    x_api_key: str | None = Header(default=None)
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    t0 = time.time()
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="chartqa_"))
    pdf_path = tmpdir / file.filename
    pdf_path.write_bytes(await file.read())

    # 1) OCR + Figures via Document Intelligence
    read_dict, layout_dict, figure_pngs = run_read_and_layout(pdf_path, tmpdir)

    # 2) Upload artifacts to Blob (optional but recommended)
    blob = BlobClientLite()
    session_id = f"{int(time.time())}_{pdf_path.stem}"
    remote_base = f"sessions/{session_id}/"
    blob_paths = {}
    try:
        blob_paths["pdf"] = blob.upload_file(str(pdf_path), remote_base + file.filename)
        for p in figure_pngs:
            blob.upload_file(p, remote_base + "figures/" + pathlib.Path(p).name)
        # text dump for AOAI context
        text_txt = (tmpdir/"read_text.txt");
        all_lines = []
        for pg in (read_dict.get("pages") or []):
            for ln in pg.get("lines") or []:
                all_lines.append(ln.get("content", ""))
        text_txt.write_text("\n".join(all_lines), encoding="utf-8")
        blob_paths["text"] = blob.upload_file(str(text_txt), remote_base + "read_text.txt")
    except Exception as e:
        # Blob is optional for the demo. Log and continue.
        print("[blob] WARN:", e)

    # 3) YOLO cls over figure crops
    yolo_results = []
    for fig_path in figure_pngs:
        with open(fig_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        try:
            yolo_results.append(call_yolo_cls(b64))
        except Exception as e:
            yolo_results.append({"error": str(e)})

    # pick top chart by prob
    chart_idx, chart_prob = -1, 0.0
    for i, r in enumerate(yolo_results):
        if r and r.get("label") == "chart" and r.get("prob", 0) > chart_prob:
            chart_idx, chart_prob = i, r.get("prob", 0)
    chosen_fig_b64 = None
    if 0 <= chart_idx < len(figure_pngs):
        chosen_fig_b64 = base64.b64encode(pathlib.Path(figure_pngs[chart_idx]).read_bytes()).decode("utf-8")

    # 4) Routing
    route = decide_route(prompt, has_chart=chosen_fig_b64 is not None, chart_prob=chart_prob)

    # 5) Invoke model according to route
    if route == "chart":
        if not chosen_fig_b64:
            # fallback to text if no crop
            route = "text"
        else:
            answer = call_chartgemma(chosen_fig_b64, prompt)
            meta = {
                "figures_total": len(figure_pngs),
                "chart_idx": chart_idx,
                "chart_prob": chart_prob,
                "blob": blob_paths,
                "ms": int((time.time()-t0)*1000)
            }
            return JSONResponse({
                "route": route,
                "answer": answer,
                "chart_image_b64": chosen_fig_b64,
                "meta": meta
            })

    # → text route (Azure OpenAI)
    # crude context (cap to ~12k chars)
    raw_text = "\n".join(ln.get("content", "") for pg in (read_dict.get("pages") or []) for ln in (pg.get("lines") or []))
    context = raw_text[:12000]
    answer = call_aoai(context, prompt)
    meta = {
        "figures_total": len(figure_pngs),
        "chart_idx": chart_idx,
        "chart_prob": chart_prob,
        "blob": blob_paths,
        "ms": int((time.time()-t0)*1000)
    }
    return JSONResponse({"route": "text", "answer": answer, "meta": meta})