import os, io, base64, requests
import streamlit as st

ORCH_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
ORCH_KEY = os.getenv("ORCHESTRATOR_KEY", "")

st.set_page_config(page_title="ChartQA on Azure", layout="wide")
st.title("📈 ChartQA on Azure — Demo")

with st.form("upload"):
    pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    prompt = st.text_area("질문 (차트 우선 처리하려면 '#chart '로 시작)", height=120)
    submitted = st.form_submit_button("분석하기")

if submitted and pdf_file and prompt:
    files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
    data = {"prompt": prompt}
    headers = {"x-api-key": ORCH_KEY} if ORCH_KEY else {}
    with st.spinner("분석 중..."):
        r = requests.post(f"{ORCH_URL}/analyze", files=files, data=data, headers=headers, timeout=600)
    if r.ok:
        res = r.json()
        st.subheader("결과")
        st.write(f"**Route:** {res.get('route')}")
        st.write(res.get("answer", "(no answer)"))

        # 미리보기
        if res.get("chart_image_b64"):
            st.image(base64.b64decode(res["chart_image_b64"]), caption="Used chart crop")
        with st.expander("디버그/메타데이터"):
            st.json({k: v for k, v in res.items() if k not in ["answer", "chart_image_b64"]})
    else:
        st.error(f"HTTP {r.status_code}: {r.text}")