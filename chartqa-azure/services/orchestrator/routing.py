# orchestrator/routing.py

def decide_route(prompt: str, has_chart: bool, chart_prob: float, threshold: float = 0.6) -> str:
    p = (prompt or "").strip().lower()
    # 명시 오버라이드
    if p.startswith("#chart"):
        return "chart"
    if p.startswith("#text"):
        return "text"

    # YOLO가 chart라고 말했고, 프롬프트도 차트 지시어가 있으면 chart
    if has_chart and chart_prob >= threshold:
        for kw in ["chart", "plot", "bar", "line", "axis", "legend", "series", "figure", "graph"]:
            if kw in p:
                return "chart"

    # 기본은 텍스트
    return "text"


# def decide_route(prompt: str, has_chart: bool, chart_prob: float, threshold: float = 0.6) -> str:
#     p = (prompt or "").strip().lower()
#     if p.startswith("#chart"):  # explicit override
#         return "chart"
#     if has_chart and chart_prob >= threshold:
#         # basic heuristic: if question mentions plot/axis/series or needs reading from chart
#         for kw in ["chart", "plot", "bar", "line", "axis", "legend", "series", "figure", "graph"]:
#             if kw in p:
#                 return "chart"
#     return "text"