def decide_route(prompt: str, has_chart: bool, chart_prob: float, threshold: float = 0.6) -> str:
    p = (prompt or "").strip().lower()
    if p.startswith("#chart"):  # explicit override
        return "chart"
    if has_chart and chart_prob >= threshold:
        # basic heuristic: if question mentions plot/axis/series or needs reading from chart
        for kw in ["chart", "plot", "bar", "line", "axis", "legend", "series", "figure", "graph"]:
            if kw in p:
                return "chart"
    return "text"