# ChartQA on Azure

This repo wires up **Document Intelligence** (OCR + figures), a **YOLO11 chart classifier** (AML), **ChartGemma** (AML), and **Azure OpenAI** (text QA) behind a tiny **FastAPI orchestrator** (deployable to AML Managed Online Endpoint via custom container) and a local **Streamlit** UI.

## Quick start

1. Copy `.env.example` → `.env` and fill your keys/URLs (DI, YOLO, ChartGemma, Azure OpenAI).
   *Minimal vars: `DI_ENDPOINT`, `DI_KEY`, `YOLO_URL`, `CHARTGEMMA_URL`, `CHARTGEMMA_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`.*

2. Start local YOLO classifier:

```bash
python -m services.yolo-local.src.server
```

3. Start the orchestrator:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --env-file ../../.env --log-level debug
```

4. Run the UI:

```bash
streamlit run streamlit_app/app.py
```

That’s it—upload a PDF in the UI, ask a question, and the orchestrator will route to chart or text mode automatically.
