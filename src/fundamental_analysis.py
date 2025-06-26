# This file will be used to read company fundamentals from the SEC website.
# The goal is to extract information that can be translated into some numbers via LLMs.
# The number should be able to be used to calculate/modify weights of the alphas.

# This file is a start but it is far from complete and NOT A PRIORITY.

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------



from bs4 import BeautifulSoup, Comment
import openai, tiktoken
import pandas as pd
import json
from pathlib import Path
import polars as pl

def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup.find_all(["script", "style", "table"]):
        tag.decompose()
    for c in soup(text=lambda t: isinstance(t, Comment)):
        c.extract()
    return soup.get_text(" ", strip=True)



# This scoring system is still not checked. First thought is that it is too simple and hardly deterministic
# Possible improvements: 1. call different LLMS + call the same several times and take averages; 2. Train and use a personal LLM
SYSTEM_PROMPT = """You are a sell-side equity analyst.
Return a JSON object with:
summary: <80 word bullet list>,
sentiment_score: float -1..1
tone_score: float -1..1      # management optimism/pessimism
risk_alert: float 0..1       # higher ⇒ more forward-looking risk
earn_quality: float 0..1     # accrual quality proxy
"""

def analyze_chunk(chunk, model="gpt-4o-mini"):
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)




def upsert_scores(df: pd.DataFrame, path="scores.parquet"):
    if Path(path).exists():
        base = pl.read_parquet(path)
        combined = pl.concat([base, pl.from_pandas(df)]).unique(
            subset=["ticker", "filing_date"], keep="last"
        )
    else:
        combined = pl.from_pandas(df)
    combined.write_parquet(path)

# resulting wide table
# index: MultiIndex(ticker, filing_date)
# columns: sentiment_score, tone_score, risk_alert, earn_quality



def adjust_signal(alpha: pd.Series, sentiment: pd.Series, risk: pd.Series):
    penalty = (1 - 0.5*sentiment.clip(lower=-1)) * (1 + risk)     # >1 ⇒ stronger penalty
    return alpha / penalty