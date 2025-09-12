# MS2025-Chart-VLLM
Chart VLLM with MS Azure: CAIP SE intern

Here’s a tightened-up replacement for just the **Folder details** section—drop it in as-is and keep your Reference section unchanged.

---

## Folder details

1. **`chartqa-azure/` — Core application**

   * **`orchestrator/`**: FastAPI API that runs DI (read/layout/figures), builds contextual crops, scores page relevance, and routes to YOLO → ChartGemma (chart route) or Azure OpenAI (text route).
   * **`clients.py` / `routing.py`** live here with lightweight service clients and the routing heuristic.
   * **`streamlit_app/`**: Minimal UI to upload a PDF, ask a question, and view the chosen route, answer, crop preview, and metadata.
   * **`services/yolo-local/`**: Optional local YOLO11 classifier for dev (`python -m services.yolo-local.src.server`).

2. **`data-asset/` — ChartQA as Azure ML data assets**

   * Scripts/notebooks to register the ChartQA dataset (train/val/test) as AML data assets.
   * Raw dataset directory is **gitignored**; expected layout below for reference:

   ```markdown
   ├── ChartQA Dataset                   
   │   ├── train   
   │   │   ├── train_augmented.json # ChartQA-M (machine-generated) questions/answers. 
   │   │   ├── train_human.json     # ChartQA-H (human-authored) questions/answers. 
   │   │   ├── annotations           # Chart Images Annotations Folder
   │   │   │   ├── chart1_name.json
   │   │   │   ├── chart2_name.json
   │   │   │   ├── ...
   │   │   ├── png                   # Chart Images Folder
   │   │   │   ├── chart1_name.png
   │   │   │   ├── chart2_name.png
   │   │   │   ├── ...
   │   │   ├── tables                # Underlying Data Tables Folder
   │   │   │   ├── chart1_name.csv
   │   │   │   ├── chart2_name.csv
   │   │   │   ├── ...
   │   └── val  
   │   │   │   ...
   │   │   │   ...
   │   └── test  
   │   │   │   ...
   │   │   │   ...
   │   │   |   ...
   ```

3. **`doc-intelligence/` — DI sandboxes**

   * Quick experiments for Azure Document Intelligence: OCR, figure extraction, page-image retrieval, and cropping helpers used by the orchestrator.

4. **`model-deployment/` — ChartGemma on AML (real-time)**

   * Deployment artifacts/configs for serving ChartGemma via Azure ML Managed Online Endpoints (TGI-style), plus example request payloads.

5. **`yolov11-cls-finetuning/` — Chart/non-chart classifier**

   * Azure ML notebooks for fine-tuning YOLOv11 classification; exported endpoint URL/keys are consumed by the orchestrator’s YOLO client.

## Reference
```markdown
@inproceedings{masry-etal-2022-chartqa,
    title = "{C}hart{QA}: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning",
    author = "Masry, Ahmed  and
      Long, Do  and
      Tan, Jia Qing  and
      Joty, Shafiq  and
      Hoque, Enamul",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.177",
    doi = "10.18653/v1/2022.findings-acl.177",
    pages = "2263--2279",
}
@misc{masry2024chartgemmavisualinstructiontuningchart,
      title={ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild}, 
      author={Ahmed Masry and Megh Thakkar and Aayush Bajaj and Aaryaman Kartha and Enamul Hoque and Shafiq Joty},
      year={2024},
      eprint={2407.04172},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.04172}, 
}
```