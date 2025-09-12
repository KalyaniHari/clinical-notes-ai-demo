import os
import re
from typing import List

# ---------- Free fallback (regex dictionary) ----------
MED_LIST = [
    "metformin", "insulin", "lisinopril", "atorvastatin", "simvastatin",
    "amlodipine", "losartan", "omeprazole", "levothyroxine", "gabapentin",
    "hydrochlorothiazide", "metoprolol"
]
PATTERN = re.compile(r"\b(" + "|".join([re.escape(m) for m in MED_LIST]) + r")\b", re.IGNORECASE)

def extract_medications_regex(note: str) -> List[str]:
    if not note:
        return []
    return sorted({m.group(0).lower() for m in PATTERN.finditer(note)})

# ---------- Optional LLM path (LangChain + OpenAI) ----------
# If you set OPENAI_API_KEY and install the libs:
#   pip install openai langchain langchain-openai
# this will be used automatically. Otherwise we fall back to regex.
_USE_LLM = False
try:
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        _USE_LLM = True
except Exception:
    _USE_LLM = False  # libs not installed → use regex

def llm_enabled() -> bool:
    return _USE_LLM

def extract_medications_llm(note: str) -> List[str]:
    # Minimal, deterministic prompt
    prompt = ChatPromptTemplate.from_template(
        "Extract medication names from the clinical note. "
        "Return only a comma-separated list of medication names, lowercase. "
        "Note: {note}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    msg = prompt.format_messages(note=note)
    out = model.invoke(msg).content.strip()
    meds = [x.strip().lower() for x in out.split(",") if x.strip()]
    # optional cleanup: keep only alphabetic tokens and known meds union
    return sorted(set(meds + extract_medications_regex(note)))

def extract_medications(note: str) -> List[str]:
    if llm_enabled():
        try:
            return extract_medications_llm(note)
        except Exception:
            # if API call fails, silently fall back to regex
            return extract_medications_regex(note)
    return extract_medications_regex(note)
