# streamlit run streamlit_app/app.py

import os, io, base64, requests, datetime as dt
import streamlit as st

# ---- env & page config ----
ORCH_URL_DEFAULT = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8000")
ORCH_KEY_DEFAULT = os.getenv("ORCHESTRATOR_KEY", "")

st.set_page_config(page_title="ChartQA on Azure", layout="wide")
st.title("📈 ChartQA on Azure — Demo")

# ---- sidebar (runtime override) ----
with st.sidebar:
    st.header("Orchestrator")
    orch_url = st.text_input("ORCHESTRATOR_URL", ORCH_URL_DEFAULT)
    orch_key = st.text_input("ORCHESTRATOR_KEY (x-api-key)", ORCH_KEY_DEFAULT, type="password")
    st.caption("⬆️ 여기에 값을 바꾸면 바로 반영됩니다.")
    st.markdown("---")
    st.caption("Tip: 차트 우선 ⇒ 프롬프트를 `#chart`로, 텍스트 강제 ⇒ `#text`로 시작")

# ---- session state for history ----
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---- form ----
with st.form("upload"):
    pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    prompt = st.text_area("질문 (차트 우선 처리하려면 '#chart '로 시작)", height=120)
    submitted = st.form_submit_button("분석하기")

def _pretty_error(resp: requests.Response) -> str:
    try:
        j = resp.json()
        return j.get("detail") or j.get("error") or resp.text
    except Exception:
        return resp.text

if submitted and pdf_file and prompt:
    files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
    data = {"prompt": prompt}
    headers = {"x-api-key": orch_key} if orch_key else {}
    with st.spinner("분석 중..."):
        try:
            r = requests.post(f"{orch_url}/analyze", files=files, data=data, headers=headers, timeout=600)
        except Exception as e:
            st.error(f"요청 실패: {e}")
            st.stop()

    if r.ok:
        res = r.json()
        # history push
        st.session_state["history"].insert(0, {
            "ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "route": res.get("route"),
            "answer": res.get("answer"),
            "meta": res.get("meta"),
            "chart_b64": res.get("chart_image_b64"),
            "file": pdf_file.name
        })

        # show latest
        st.subheader("결과")
        st.write(f"**Route:** {res.get('route')}")
        st.write(res.get("answer", "(no answer)"))

        # 미리보기
        if res.get("chart_image_b64"):
            st.image(base64.b64decode(res["chart_image_b64"]), caption="Used chart crop")

        with st.expander("디버그/메타데이터"):
            # answer와 이미지 바이너리는 제외하고 출력
            dbg = {k: v for k, v in res.items() if k not in ["answer", "chart_image_b64"]}
            st.json(dbg)

    else:
        st.error(f"HTTP {r.status_code}: {_pretty_error(r)}")

# ---- history ----
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("세션 히스토리")
    for i, h in enumerate(st.session_state["history"], start=1):
        with st.expander(f"[{h['ts']}] {h['file']} · route={h['route']}"):
            st.code(h["prompt"])
            st.write(h["answer"])
            if h.get("chart_b64"):
                st.image(base64.b64decode(h["chart_b64"]), caption="Used chart crop (history)")
            st.json(h.get("meta", {}))


# import os, io, base64, requests
# import streamlit as st

# ORCH_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
# ORCH_KEY = os.getenv("ORCHESTRATOR_KEY", "")

# st.set_page_config(page_title="ChartQA on Azure", layout="wide")
# st.title("📈 ChartQA on Azure — Demo")

# with st.form("upload"):
#     pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
#     prompt = st.text_area("질문 (차트 우선 처리하려면 '#chart '로 시작)", height=120)
#     submitted = st.form_submit_button("분석하기")

# if submitted and pdf_file and prompt:
#     files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
#     data = {"prompt": prompt}
#     headers = {"x-api-key": ORCH_KEY} if ORCH_KEY else {}
#     with st.spinner("분석 중..."):
#         r = requests.post(f"{ORCH_URL}/analyze", files=files, data=data, headers=headers, timeout=600)
#     if r.ok:
#         res = r.json()
#         st.subheader("결과")
#         st.write(f"**Route:** {res.get('route')}")
#         st.write(res.get("answer", "(no answer)"))

#         # 미리보기
#         if res.get("chart_image_b64"):
#             st.image(base64.b64decode(res["chart_image_b64"]), caption="Used chart crop")
#         with st.expander("디버그/메타데이터"):
#             st.json({k: v for k, v in res.items() if k not in ["answer", "chart_image_b64"]})
#     else:
#         st.error(f"HTTP {r.status_code}: {r.text}")