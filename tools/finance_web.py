import json
import os

import httpx
import yfinance as yf
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def _tavily() -> TavilyClient:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set TAVILY_API_KEY in .env")
    return TavilyClient(api_key=api_key)

def get_ticker_metrics(ticker: str) -> str:
    """Fetch accurate live price, valuation metrics, and recent news for a stock/ETF ticker."""
    symbol = ticker.upper().strip()
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:
        return f"Could not find market data for ticker '{ticker}': {exc}"
    if "regularMarketPrice" not in info and "currentPrice" not in info:
        return f"Could not find market data for ticker '{ticker}'."

    summary = {
        "ticker": symbol,
        "name": info.get("shortName"),
        "sector": info.get("sector"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "revenue_growth": info.get("revenueGrowth"),
        "profit_margins": info.get("profitMargins"),
        "analyst_target_mean": info.get("targetMeanPrice"),
        "recommendation": info.get("recommendationKey"),
    }
    return json.dumps(summary, indent=2)

def search_web(query: str, is_finance: bool = False, max_results: int = 5) -> str:
    """Search the web. Set is_finance=True for market news, earnings, macro, or company analysis."""
    topic = "finance" if is_finance else "general"
    depth = "advanced" if is_finance else "basic"
    try:
        res = _tavily().search(
            query=query,
            topic=topic,
            search_depth=depth,
            max_results=max_results,
            include_answer=True,
        )
    except Exception as exc:
        return f"Web search failed: {exc}"
    output = []
    if res.get("answer"):
        output.append(f"Quick Synthesis: {res['answer']}\n")
    for r in res.get("results", []):
        output.append(f"- {r['title']} ({r['url']})\n  {r['content'][:450]}")
    return "\n".join(output) or "No results found."

_JOB_BOARDS = (
    "(site:ashbyhq.com OR site:boards.greenhouse.io OR site:jobs.lever.co)"
)
_JOB_HOSTS = ("ashbyhq.com", "boards.greenhouse.io", "jobs.lever.co")


def search_jobs(role: str, location: str = "Remote") -> str:
    """Find roles on Ashby, Greenhouse, and Lever. Returns numbered titles, https URLs, and snippets."""
    query = f"{role.strip()} {location.strip()} {_JOB_BOARDS}"
    try:
        res = _tavily().search(
            query=query,
            search_depth="advanced",
            topic="general",
            max_results=8,
            include_answer=False,
        )
    except Exception as exc:
        return f"Job search failed: {exc}"

    lines = []
    for item in res.get("results", []):
        url = (item.get("url") or "").strip()
        if not url.startswith("https://") or not any(host in url for host in _JOB_HOSTS):
            continue
        title = (item.get("title") or "Untitled role").strip()
        snippet = " ".join((item.get("content") or "").split())[:280]
        lines.append(f"{len(lines) + 1}. {title}\n{url}\n{snippet}")
    return "\n\n".join(lines) or "No jobs found on Ashby, Greenhouse, or Lever."

async def read_webpage_markdown(url: str) -> str:
    """Fetch the full clean Markdown text of an article, earnings summary, or webpage."""
    target = url if url.startswith(("http://", "https://")) else f"https://{url}"
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(f"https://r.jina.ai/{target}")
            resp.raise_for_status()
            return resp.text[:7000]
    except Exception as exc:
        return f"Could not read {url}: {exc}"
