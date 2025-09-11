This repo skeleton wires up existing pieces (Document Intelligence OCR + figures, YOLO11 chart classifier on AML, ChartGemma on AML, and Azure OpenAI for text QA) with a small FastAPI orchestrator (deployable to AML Managed Online Endpoint via custom container) and a local Streamlit UI.

```bash
python -m services.yolo-local.src.server
```
> /services/orchestrator
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --env-file ../../.env --log-level debug
```
```bash
streamlit run streamlit_app/app.py
```