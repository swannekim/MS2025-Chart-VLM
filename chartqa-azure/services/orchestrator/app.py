# orchestrator/app.py
import os, json, base64, tempfile, pathlib, time, logging, re
from typing import Dict, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from PIL import Image

from clients import call_yolo_cls, call_chartgemma, call_aoai
from di_utils import run_read_and_layout, extract_page_texts, make_contextual_crops
from routing import decide_route

API_KEY = os.getenv("ORCH_API_KEY", "")

# 휴리스틱 파라미터 (환경변수로 튜닝 가능)
FALLBACK_MIN_SIDE = int(os.getenv("FALLBACK_MIN_SIDE", "140"))       # px
FALLBACK_MIN_BYTES = int(os.getenv("FALLBACK_MIN_BYTES", "10000"))   # ~10KB
RELEVANCE_MIN_LEN = int(os.getenv("RELEVANCE_MIN_LEN", "3"))         # 단어 최소 길이
YOLO_URL = os.getenv("YOLO_URL", "")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("chartqa-orch")

app = FastAPI(title="ChartQA Orchestrator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def _img_b64_from_path(p: str) -> str:
    return base64.b64encode(pathlib.Path(p).read_bytes()).decode("utf-8")

def _pick_fallback_chart_idx(figure_pngs: List[str]) -> int:
    """YOLO 실패 시 '그럴싸한' 큰 그림 선택"""
    candidates: List[Tuple[int, int, int]] = []  # (idx, bytes, area)
    for i, p in enumerate(figure_pngs):
        try:
            size = os.path.getsize(p)
            with Image.open(p) as im:
                w, h = im.size
            if w >= FALLBACK_MIN_SIDE and h >= FALLBACK_MIN_SIDE and size >= FALLBACK_MIN_BYTES:
                candidates.append((i, size, w * h))
        except Exception:
            continue
    if not candidates:
        return 0 if figure_pngs else -1
    candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return candidates[0][0]

_word_re = re.compile(r"[A-Za-z0-9%]+")  # 간단 토크나이저

def _tokenize(s: str) -> List[str]:
    return [w.lower() for w in _word_re.findall(s or "") if len(w) >= RELEVANCE_MIN_LEN]

def _page_relevance_scores(prompt: str, page_texts: Dict[int, str]) -> Dict[int, float]:
    """
    매우 가벼운 관련도 점수:
    - prompt 토큰 set과 페이지 토큰 set의 교집합 크기 + 빈도 합산
    - % 같은 수치 토큰도 포함(질문이 '몇 %?'라면 유용)
    """
    p_toks = _tokenize(prompt)
    p_set = set(p_toks)
    scores: Dict[int, float] = {}
    for page, text in page_texts.items():
        t_toks = _tokenize(text)
        if not t_toks:
            scores[page] = 0.0
            continue
        inter = [t for t in t_toks if t in p_set]
        scores[page] = len(inter) + (len(inter) / max(len(t_toks), 1))
    return scores

@app.get("/health")
def health():
    return {"status": "ok", "yolo_configured": bool(YOLO_URL)}

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

    # 1) OCR + Figures (+ figure별 페이지 번호)
    try:
        read_dict, layout_dict, figure_pngs, fig_pages = run_read_and_layout(pdf_path, tmpdir)
    except Exception as e:
        log.exception("DI failed")
        raise HTTPException(status_code=502, detail=f"Document Intelligence failed: {e}")

    # 페이지 전체 이미지 기반 '맥락 크롭' (원본 figure 순서와 동일한 인덱스)
    try:
        ctx_pngs = make_contextual_crops(pdf_path, tmpdir, read_dict, layout_dict)
    except Exception as e:
        log.warning("Contextual crop generation failed: %s", e)
        ctx_pngs = [None] * len(figure_pngs)

    # 2) 페이지 관련도 계산
    page_texts = extract_page_texts(read_dict)
    page_scores = _page_relevance_scores(prompt, page_texts)

    # 3) YOLO (있으면 사용, 없으면 빈 dict)
    yolo_results: List[Dict] = []
    for i, fp in enumerate(figure_pngs):
        img_path = ctx_pngs[i] or fp  # YOLO에도 맥락 크롭 우선
        try:
            b64 = _img_b64_from_path(img_path)
            res = call_yolo_cls(b64)  # 미구성 시 None
            yolo_results.append(res if res is not None else {})
        except Exception as e:
            yolo_results.append({"error": str(e)})

    # 후보 테이블 구축
    candidates = []
    for i, fp in enumerate(figure_pngs):
        use_path = ctx_pngs[i] or fp  # 실제로 모델에 전달할 이미지 경로
        try:
            size = os.path.getsize(use_path)
        except Exception:
            size = 0
        pg = fig_pages[i] if i < len(fig_pages) else 0
        y = yolo_results[i] if i < len(yolo_results) else {}
        label = y.get("label")
        prob = float(y.get("prob", 0.0)) if isinstance(y, dict) else 0.0
        candidates.append({
            "idx": i,
            "path_raw": fp,
            "path_ctx": ctx_pngs[i],
            "used_ctx": bool(ctx_pngs[i]),
            "page": pg,
            "page_score": float(page_scores.get(pg, 0.0)),
            "yolo_label": label,
            "yolo_prob": prob,
            "size_bytes": size,
        })

    # 4) 선택 전략
    pnorm = (prompt or "").strip().lower()
    force_chart = pnorm.startswith("#chart")
    wants_chart_kw = any(kw in pnorm for kw in ["chart", "plot", "bar", "line", "axis", "legend", "series", "figure", "graph"])

    def _rank_key_chart(c):
        # 페이지 관련도 > YOLO 확신도 > 파일크기
        return (c["page_score"], c["yolo_prob"], c["size_bytes"])

    def _rank_key_text(c):
        # 페이지 관련도 > 파일크기
        return (c["page_score"], c["size_bytes"])

    chosen_idx = -1
    chosen_reason = None

    # 4-1) YOLO가 chart라고 한 후보들 중 선택
    chart_cands = [c for c in candidates if c.get("yolo_label") == "chart"]
    if chart_cands:
        chart_cands.sort(key=_rank_key_chart, reverse=True)
        chosen_idx = chart_cands[0]["idx"]
        chosen_reason = "yolo+page_score"

    # 4-2) #chart 또는 차트 키워드인데 위가 비어있으면, 페이지 관련도만으로 선택
    if chosen_idx < 0 and (force_chart or wants_chart_kw):
        all_sorted = sorted(candidates, key=_rank_key_text, reverse=True)
        if all_sorted:
            chosen_idx = all_sorted[0]["idx"]
            chosen_reason = "page_score_only"

    # 4-3) 그래도 없으면, 기존 휴리스틱(큰 그림) 폴백
    if chosen_idx < 0:
        fb = _pick_fallback_chart_idx([c["path_ctx"] or c["path_raw"] for c in candidates])
        chosen_idx = fb
        chosen_reason = "size_fallback" if fb >= 0 else "no_figure"

    # 5) 최종 이미지 선택 (컨텍스트 크롭 우선)
    chosen_fig_b64 = None
    used_ctx = False
    if 0 <= chosen_idx < len(candidates):
        used_img_path = candidates[chosen_idx]["path_ctx"] or candidates[chosen_idx]["path_raw"]
        used_ctx = bool(candidates[chosen_idx]["path_ctx"])
        if used_img_path:
            chosen_fig_b64 = _img_b64_from_path(used_img_path)

    # 6) 라우팅
    try:
        chart_prob = candidates[chosen_idx]["yolo_prob"] if 0 <= chosen_idx < len(candidates) else 0.0
        route = "chart" if force_chart else decide_route(
            prompt, has_chart=chosen_fig_b64 is not None, chart_prob=chart_prob
        )
    except Exception:
        route = "text"

    # 7) 호출
    if route == "chart" and chosen_fig_b64:
        try:
            cg_res = call_chartgemma(chosen_fig_b64, prompt)
            # ---- 반환 형태 호환 처리 ----
            # - str
            # - (answer:str, info:dict)
            # - (answer:str, info:dict, *extras)
            if isinstance(cg_res, tuple):
                if len(cg_res) == 2:
                    answer, cg_info = cg_res
                elif len(cg_res) > 2:
                    answer = cg_res[0]
                    second = cg_res[1]
                    cg_info = second if isinstance(second, dict) else {"extra": second}
                    cg_info["extra_return"] = list(cg_res[2:])
                else:
                    # 빈 튜플 같은 비정상
                    answer, cg_info = "", {"mode_used": "invalid_tuple"}
            else:
                answer, cg_info = str(cg_res), {"mode_used": "unknown_api_shape"}
            # --------------------------------
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Chart model failed: {e}")

        meta = {
            "figures_total": len(candidates),
            "chosen_idx": chosen_idx,
            "chosen_page": candidates[chosen_idx]["page"] if 0 <= chosen_idx < len(candidates) else None,
            "chosen_reason": chosen_reason,
            "used_contextual_crop": used_ctx,
            "yolo_configured": bool(YOLO_URL),
            "candidate_ranking": sorted(candidates, key=_rank_key_chart, reverse=True)[:5],
            "chartgemma": cg_info,
            "ms": int((time.time() - t0) * 1000),
        }
        return JSONResponse({
            "route": "chart",
            "answer": answer,
            "chart_image_b64": chosen_fig_b64,  # 컨텍스트 크롭이면 그 이미지
            "meta": meta
        })

    # → text route
    raw_text = "\n".join(
        ln.get("content", "")
        for pg in (read_dict.get("pages") or [])
        for ln in (pg.get("lines") or [])
    )
    context = raw_text[:12000]
    try:
        answer = call_aoai(context, prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Text model failed: {e}")

    meta = {
        "figures_total": len(candidates),
        "chosen_idx": chosen_idx,
        "chosen_page": candidates[chosen_idx]["page"] if 0 <= chosen_idx < len(candidates) else None,
        "chosen_reason": chosen_reason,
        "used_contextual_crop": used_ctx,
        "yolo_configured": bool(YOLO_URL),
        "candidate_ranking": sorted(candidates, key=_rank_key_text, reverse=True)[:5],
        "ms": int((time.time() - t0) * 1000),
    }
    return JSONResponse({"route": "text", "answer": answer, "meta": meta})


# import os, json, base64, tempfile, pathlib, time, logging, re
# from typing import Dict, List, Optional, Tuple
# from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
# from fastapi.responses import JSONResponse
# from starlette.middleware.cors import CORSMiddleware
# from PIL import Image

# from clients import call_yolo_cls, call_chartgemma, call_aoai
# from di_utils import run_read_and_layout, extract_page_texts
# from routing import decide_route

# API_KEY = os.getenv("ORCH_API_KEY", "")

# # 휴리스틱 파라미터 (환경변수로 튜닝 가능)
# FALLBACK_MIN_SIDE = int(os.getenv("FALLBACK_MIN_SIDE", "140"))       # px
# FALLBACK_MIN_BYTES = int(os.getenv("FALLBACK_MIN_BYTES", "10000"))   # ~10KB
# RELEVANCE_MIN_LEN = int(os.getenv("RELEVANCE_MIN_LEN", "3"))         # 단어 최소 길이
# YOLO_URL = os.getenv("YOLO_URL", "")

# logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
# log = logging.getLogger("chartqa-orch")

# app = FastAPI(title="ChartQA Orchestrator")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# def _img_b64_from_path(p: str) -> str:
#     return base64.b64encode(pathlib.Path(p).read_bytes()).decode("utf-8")

# def _pick_fallback_chart_idx(figure_pngs: List[str]) -> int:
#     """YOLO 실패 시 '그럴싸한' 큰 그림 선택"""
#     candidates: List[Tuple[int,int,int]] = []  # (idx, bytes, area)
#     for i, p in enumerate(figure_pngs):
#         try:
#             size = os.path.getsize(p)
#             with Image.open(p) as im:
#                 w, h = im.size
#             if w >= FALLBACK_MIN_SIDE and h >= FALLBACK_MIN_SIDE and size >= FALLBACK_MIN_BYTES:
#                 candidates.append((i, size, w*h))
#         except Exception:
#             continue
#     if not candidates:
#         return 0 if figure_pngs else -1
#     candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)
#     return candidates[0][0]

# _word_re = re.compile(r"[A-Za-z0-9%]+")  # 간단 토크나이저

# def _tokenize(s: str) -> List[str]:
#     return [w.lower() for w in _word_re.findall(s or "") if len(w) >= RELEVANCE_MIN_LEN]

# def _page_relevance_scores(prompt: str, page_texts: Dict[int, str]) -> Dict[int, float]:
#     """
#     매우 가벼운 관련도 점수:
#     - prompt 토큰 set과 페이지 토큰 set의 교집합 크기 + 빈도 합산
#     - % 같은 수치 토큰도 포함(질문이 '몇 %?'라면 유용)
#     """
#     p_toks = _tokenize(prompt)
#     p_set = set(p_toks)
#     scores: Dict[int, float] = {}
#     for page, text in page_texts.items():
#         t_toks = _tokenize(text)
#         if not t_toks:
#             scores[page] = 0.0
#             continue
#         inter = [t for t in t_toks if t in p_set]
#         # 단순: 교집합 개수 + 빈도/길이 보정
#         scores[page] = len(inter) + (len(inter) / max(len(t_toks), 1))
#     return scores

# @app.get("/health")
# def health():
#     return {"status": "ok", "yolo_configured": bool(YOLO_URL)}

# @app.post("/analyze")
# async def analyze(
#     file: UploadFile = File(...),
#     prompt: str = Form(...),
#     x_api_key: str | None = Header(default=None)
# ):
#     if API_KEY and x_api_key != API_KEY:
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     t0 = time.time()
#     tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="chartqa_"))
#     pdf_path = tmpdir / file.filename
#     pdf_path.write_bytes(await file.read())

#     # 1) OCR + Figures (+ figure별 페이지 번호)
#     try:
#         read_dict, layout_dict, figure_pngs, fig_pages = run_read_and_layout(pdf_path, tmpdir)
#     except Exception as e:
#         log.exception("DI failed")
#         raise HTTPException(status_code=502, detail=f"Document Intelligence failed: {e}")

#     blob_paths: Dict[str, str] = {}

#     # 2) 페이지 관련도 계산
#     page_texts = extract_page_texts(read_dict)
#     page_scores = _page_relevance_scores(prompt, page_texts)

#     # 3) YOLO
#     yolo_results: List[Dict] = []
#     for fp in figure_pngs:
#         try:
#             b64 = _img_b64_from_path(fp)
#             res = call_yolo_cls(b64)  # 미구성 시 None
#             yolo_results.append(res if res is not None else {})
#         except Exception as e:
#             yolo_results.append({"error": str(e)})

#     # 후보 테이블 구축
#     candidates = []
#     for i, fp in enumerate(figure_pngs):
#         try:
#             size = os.path.getsize(fp)
#         except Exception:
#             size = 0
#         pg = fig_pages[i] if i < len(fig_pages) else 0
#         y = yolo_results[i] if i < len(yolo_results) else {}
#         label = y.get("label")
#         prob = float(y.get("prob", 0.0)) if isinstance(y, dict) else 0.0
#         candidates.append({
#             "idx": i,
#             "path": fp,
#             "page": pg,
#             "page_score": float(page_scores.get(pg, 0.0)),
#             "yolo_label": label,
#             "yolo_prob": prob,
#             "size_bytes": size,
#         })

#     # 4) 선택 전략
#     pnorm = (prompt or "").strip().lower()
#     force_chart = pnorm.startswith("#chart")
#     wants_chart_kw = any(kw in pnorm for kw in ["chart", "plot", "bar", "line", "axis", "legend", "series", "figure", "graph"])

#     def _rank_key_chart(c):
#         # 페이지 관련도 > YOLO 확신도 > 파일크기
#         return (c["page_score"], c["yolo_prob"], c["size_bytes"])

#     def _rank_key_text(c):
#         # 페이지 관련도 > 파일크기
#         return (c["page_score"], c["size_bytes"])

#     chosen_idx = -1
#     chosen_reason = None

#     # 4-1) YOLO가 chart라고 한 후보들 중 선택
#     chart_cands = [c for c in candidates if c.get("yolo_label") == "chart"]
#     if chart_cands:
#         chart_cands.sort(key=_rank_key_chart, reverse=True)
#         chosen_idx = chart_cands[0]["idx"]
#         chosen_reason = "yolo+page_score"

#     # 4-2) #chart 또는 차트 키워드인데 위가 비어있으면, 페이지 관련도만으로 선택
#     if chosen_idx < 0 and (force_chart or wants_chart_kw):
#         all_sorted = sorted(candidates, key=_rank_key_text, reverse=True)
#         if all_sorted:
#             chosen_idx = all_sorted[0]["idx"]
#             chosen_reason = "page_score_only"

#     # 4-3) 그래도 없으면, 기존 휴리스틱(큰 그림) 폴백
#     if chosen_idx < 0:
#         fb = _pick_fallback_chart_idx(figure_pngs)
#         chosen_idx = fb
#         chosen_reason = "size_fallback" if fb >= 0 else "no_figure"

#     chosen_fig_b64 = None
#     if 0 <= chosen_idx < len(figure_pngs):
#         chosen_fig_b64 = _img_b64_from_path(figure_pngs[chosen_idx])

#     # 5) 라우팅
#     try:
#         route = "chart" if force_chart else decide_route(prompt, has_chart=chosen_fig_b64 is not None, chart_prob=candidates[chosen_idx]["yolo_prob"] if 0 <= chosen_idx < len(candidates) else 0.0)
#     except Exception:
#         route = "text"

#     # 6) 호출
#     if route == "chart" and chosen_fig_b64:
#         try:
#             answer = call_chartgemma(chosen_fig_b64, prompt)
#         except Exception as e:
#             raise HTTPException(status_code=502, detail=f"Chart model failed: {e}")

#         meta = {
#             "figures_total": len(figure_pngs),
#             "chosen_idx": chosen_idx,
#             "chosen_page": candidates[chosen_idx]["page"] if 0 <= chosen_idx < len(candidates) else None,
#             "chosen_reason": chosen_reason,
#             "yolo_configured": bool(YOLO_URL),
#             "candidate_ranking": sorted(candidates, key=_rank_key_chart, reverse=True)[:5],
#             "ms": int((time.time()-t0)*1000),
#         }
#         return JSONResponse({"route": "chart", "answer": answer, "chart_image_b64": chosen_fig_b64, "meta": meta})

#     # → text route
#     raw_text = "\n".join(
#         ln.get("content", "")
#         for pg in (read_dict.get("pages") or [])
#         for ln in (pg.get("lines") or [])
#     )
#     context = raw_text[:12000]
#     try:
#         answer = call_aoai(context, prompt)
#     except Exception as e:
#         raise HTTPException(status_code=502, detail=f"Text model failed: {e}")

#     meta = {
#         "figures_total": len(figure_pngs),
#         "chosen_idx": chosen_idx,
#         "chosen_page": candidates[chosen_idx]["page"] if 0 <= chosen_idx < len(candidates) else None,
#         "chosen_reason": chosen_reason,
#         "yolo_configured": bool(YOLO_URL),
#         "candidate_ranking": sorted(candidates, key=_rank_key_text, reverse=True)[:5],
#         "ms": int((time.time()-t0)*1000),
#     }
#     return JSONResponse({"route": "text", "answer": answer, "meta": meta})

# # services/orchestrator/app.py
# import os, io, json, base64, tempfile, pathlib, time, logging
# from typing import Dict, List, Optional
# from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
# from fastapi.responses import JSONResponse
# from starlette.middleware.cors import CORSMiddleware
# from PIL import Image

# from clients import call_yolo_cls, call_chartgemma, call_aoai
# from di_utils import run_read_and_layout
# from routing import decide_route

# # --------------------
# # Config
# # --------------------
# API_KEY = os.getenv("ORCH_API_KEY", "")
# YOLO_URL = os.getenv("YOLO_URL", "")  # 진단용 메타에 표기
# # 휴리스틱 폴백 최소 조건 (DI figure 중 '그럴싸한' 후보를 고르기 위한 임계값)
# FALLBACK_MIN_SIDE = int(os.getenv("FALLBACK_MIN_SIDE", "140"))     # px
# FALLBACK_MIN_BYTES = int(os.getenv("FALLBACK_MIN_BYTES", "10000")) # ~10KB

# logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
# log = logging.getLogger("chartqa-orchestrator")

# app = FastAPI(title="ChartQA Orchestrator")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# # --------------------
# # Utils
# # --------------------
# def _img_b64_from_path(p: str) -> str:
#     return base64.b64encode(pathlib.Path(p).read_bytes()).decode("utf-8")

# def _pick_fallback_chart_idx(figure_pngs: List[str]) -> int:
#     """
#     YOLO가 없거나 실패한 경우, '그럴싸한' 차트 후보 하나를 고른다.
#     휴리스틱:
#       - 너무 작은 이미지(가로/세로 < FALLBACK_MIN_SIDE) 제외
#       - 너무 작은 파일(바이트 < FALLBACK_MIN_BYTES) 제외
#       - 남은 것 중 파일 크기와 면적이 큰 순으로 정렬하여 1개 선택
#     """
#     candidates = []
#     for i, p in enumerate(figure_pngs):
#         try:
#             size = os.path.getsize(p)
#             with Image.open(p) as im:
#                 w, h = im.size
#             if w >= FALLBACK_MIN_SIDE and h >= FALLBACK_MIN_SIDE and size >= FALLBACK_MIN_BYTES:
#                 candidates.append((i, size, w * h))
#         except Exception:
#             continue
#     if not candidates:
#         return 0 if figure_pngs else -1
#     candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)
#     return candidates[0][0]

# # --------------------
# # Health
# # --------------------
# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "yolo_configured": bool(YOLO_URL),
#     }

# # --------------------
# # Diagnose: 각 figure에 YOLO가 뭐라고 말하는지 한눈에 보기
# # --------------------
# @app.post("/inspect")
# async def inspect(file: UploadFile = File(...)):
#     t0 = time.time()
#     tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="chartqa_"))
#     pdf_path = tmpdir / file.filename
#     pdf_path.write_bytes(await file.read())

#     read_dict, layout_dict, figure_pngs = run_read_and_layout(pdf_path, tmpdir)

#     per_fig: List[Dict] = []
#     yolo_called = 0
#     for i, p in enumerate(figure_pngs):
#         item: Dict[str, Optional[Dict]] = {"idx": i, "path": str(p)}
#         try:
#             b64 = _img_b64_from_path(p)
#             y = call_yolo_cls(b64)  # YOLO_URL 미설정이면 None
#             if y is not None:
#                 yolo_called += 1
#             item["yolo"] = y
#         except Exception as e:
#             item["error"] = str(e)
#         per_fig.append(item)

#     return JSONResponse({
#         "figures_total": len(figure_pngs),
#         "yolo_configured": bool(YOLO_URL),
#         "yolo_called_count": yolo_called,
#         "per_figure": per_fig,
#         "ms": int((time.time() - t0) * 1000)
#     })

# # --------------------
# # Main: analyze
# # --------------------
# @app.post("/analyze")
# async def analyze(
#     file: UploadFile = File(...),
#     prompt: str = Form(...),
#     x_api_key: str | None = Header(default=None)
# ):
#     if API_KEY and x_api_key != API_KEY:
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     t0 = time.time()
#     tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="chartqa_"))
#     pdf_path = tmpdir / file.filename
#     pdf_path.write_bytes(await file.read())

#     # 1) OCR + Figures via Document Intelligence (로컬 파일 기준)
#     try:
#         read_dict, layout_dict, figure_pngs = run_read_and_layout(pdf_path, tmpdir)
#     except Exception as e:
#         log.exception("Document Intelligence error")
#         raise HTTPException(status_code=502, detail=f"Document Intelligence failed: {e}")

#     # Blob 비사용: 메타만 빈 dict로
#     blob_paths: Dict[str, str] = {}

#     # 2) YOLO cls over figure crops (없으면 폴백)
#     yolo_results: List[Dict] = []
#     yolo_called = 0
#     for fig_path in figure_pngs:
#         try:
#             b64 = _img_b64_from_path(fig_path)
#             r = call_yolo_cls(b64)  # YOLO_URL 미구성 시 None
#             if r is not None:
#                 yolo_called += 1
#             yolo_results.append(r if r is not None else {})
#         except Exception as e:
#             yolo_results.append({"error": str(e)})

#     # 2-1) YOLO 결과로 best chart 선택
#     chart_idx, chart_prob = -1, 0.0
#     for i, r in enumerate(yolo_results):
#         if isinstance(r, dict) and r.get("label") == "chart" and r.get("prob", 0) > chart_prob:
#             chart_idx, chart_prob = i, r.get("prob", 0.0)

#     chosen_fig_b64 = None
#     if 0 <= chart_idx < len(figure_pngs):
#         chosen_fig_b64 = _img_b64_from_path(figure_pngs[chart_idx])

#     # 2-2) #chart 명시 또는 차트 관련 키워드인데 YOLO 실패 → 휴리스틱 폴백 선택
#     pnorm = (prompt or "").strip().lower()
#     force_chart = pnorm.startswith("#chart")
#     wants_chart = any(kw in pnorm for kw in ["chart", "plot", "bar", "line", "axis", "legend", "series", "figure", "graph"])
#     fallback_used = None
#     if not chosen_fig_b64 and figure_pngs and (force_chart or wants_chart):
#         fb_idx = _pick_fallback_chart_idx(figure_pngs)
#         if 0 <= fb_idx < len(figure_pngs):
#             chart_idx = fb_idx
#             chart_prob = -1.0  # unknown
#             chosen_fig_b64 = _img_b64_from_path(figure_pngs[fb_idx])
#             fallback_used = "heuristic_max_size"

#     # 3) Routing (has_chart는 폴백 반영 후 평가)
#     try:
#         route = "chart" if force_chart else decide_route(prompt, has_chart=chosen_fig_b64 is not None, chart_prob=chart_prob)
#     except Exception:
#         # routing.py가 뭔가 실패해도 텍스트로 안전 폴백
#         route = "text"

#     # 4) Invoke model according to route
#     if route == "chart" and chosen_fig_b64:
#         try:
#             answer = call_chartgemma(chosen_fig_b64, prompt)
#         except Exception as e:
#             log.exception("ChartGemma call failed")
#             raise HTTPException(status_code=502, detail=f"Chart model failed: {e}")

#         meta = {
#             "figures_total": len(figure_pngs),
#             "chart_idx": chart_idx,
#             "chart_prob": chart_prob,
#             "fallback": fallback_used,
#             "yolo_configured": bool(YOLO_URL),
#             "yolo_called_count": yolo_called,
#             "blob": blob_paths,
#             "ms": int((time.time()-t0)*1000),
#         }
#         return JSONResponse({
#             "route": "chart",
#             "answer": answer,
#             "chart_image_b64": chosen_fig_b64,
#             "meta": meta
#         })

#     # → text route (Azure OpenAI)
#     try:
#         raw_text = "\n".join(
#             ln.get("content", "")
#             for pg in (read_dict.get("pages") or [])
#             for ln in (pg.get("lines") or [])
#         )
#         context = raw_text[:12000]
#         answer = call_aoai(context, prompt)
#     except Exception as e:
#         log.exception("AOAI call failed")
#         raise HTTPException(status_code=502, detail=f"Text model failed: {e}")

#     meta = {
#         "figures_total": len(figure_pngs),
#         "chart_idx": chart_idx,
#         "chart_prob": chart_prob,
#         "fallback": fallback_used,
#         "yolo_configured": bool(YOLO_URL),
#         "yolo_called_count": yolo_called,
#         "blob": blob_paths,
#         "ms": int((time.time()-t0)*1000),
#     }
#     return JSONResponse({"route": "text", "answer": answer, "meta": meta})

# import os, io, json, base64, tempfile, pathlib, time
# from typing import Dict
# from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
# from fastapi.responses import JSONResponse
# from starlette.middleware.cors import CORSMiddleware

# from clients import call_yolo_cls, call_chartgemma, call_aoai
# from di_utils import run_read_and_layout
# from routing import decide_route

# API_KEY = os.getenv("ORCH_API_KEY", "")

# app = FastAPI(title="ChartQA Orchestrator")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/analyze")
# async def analyze(
#     file: UploadFile = File(...),
#     prompt: str = Form(...),
#     x_api_key: str | None = Header(default=None)
# ):
#     if API_KEY and x_api_key != API_KEY:
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     t0 = time.time()
#     tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="chartqa_"))
#     pdf_path = tmpdir / file.filename
#     pdf_path.write_bytes(await file.read())

#     # 1) OCR + Figures via Document Intelligence (로컬 파일 기준)
#     read_dict, layout_dict, figure_pngs = run_read_and_layout(pdf_path, tmpdir)

#     # Blob 비사용: 메타만 빈 dict로
#     blob_paths: Dict[str, str] = {}

#     # 2) YOLO cls over figure crops (없으면 폴백)
#     yolo_results = []
#     for fig_path in figure_pngs:
#         try:
#             with open(fig_path, "rb") as f:
#                 b64 = base64.b64encode(f.read()).decode("utf-8")
#             r = call_yolo_cls(b64)  # 구성 안되면 None 반환
#             yolo_results.append(r if r is not None else {})
#         except Exception as e:
#             yolo_results.append({"error": str(e)})

#     chart_idx, chart_prob = -1, 0.0
#     for i, r in enumerate(yolo_results):
#         if isinstance(r, dict) and r.get("label") == "chart" and r.get("prob", 0) > chart_prob:
#             chart_idx, chart_prob = i, r.get("prob", 0.0)

#     chosen_fig_b64 = None
#     if 0 <= chart_idx < len(figure_pngs):
#         chosen_fig_b64 = base64.b64encode(pathlib.Path(figure_pngs[chart_idx]).read_bytes()).decode("utf-8")

#     # 3) Routing
#     route = decide_route(prompt, has_chart=chosen_fig_b64 is not None, chart_prob=chart_prob)

#     # 4) Invoke model according to route
#     if route == "chart" and chosen_fig_b64:
#         answer = call_chartgemma(chosen_fig_b64, prompt)
#         meta = {
#             "figures_total": len(figure_pngs),
#             "chart_idx": chart_idx,
#             "chart_prob": chart_prob,
#             "blob": blob_paths,
#             "ms": int((time.time()-t0)*1000),
#         }
#         return JSONResponse({
#             "route": "chart",
#             "answer": answer,
#             "chart_image_b64": chosen_fig_b64,
#             "meta": meta
#         })

#     # → text route (Azure OpenAI)
#     raw_text = "\n".join(
#         ln.get("content", "")
#         for pg in (read_dict.get("pages") or [])
#         for ln in (pg.get("lines") or [])
#     )
#     context = raw_text[:12000]
#     answer = call_aoai(context, prompt)
#     meta = {
#         "figures_total": len(figure_pngs),
#         "chart_idx": chart_idx,
#         "chart_prob": chart_prob,
#         "blob": blob_paths,
#         "ms": int((time.time()-t0)*1000),
#     }
#     return JSONResponse({"route": "text", "answer": answer, "meta": meta})