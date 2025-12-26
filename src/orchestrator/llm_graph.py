from __future__ import annotations

from typing import TypedDict, Optional, Dict, Any

from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .settings import settings
from .schemas import TriageOut


class State(TypedDict, total=False):
    # input
    text: str
    subject: Optional[str]
    from_addr: Optional[str]

    spam_prob: Optional[float]
    is_spam: Optional[bool]
    spam_model: Optional[str]

    # processed
    llm_input: str

    # output
    triage: Dict[str, Any]


def build_llm():
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in .env or environment variables."
        )
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )


def node_prepare(state: State) -> State:
    subject = state.get("subject") or ""
    from_addr = state.get("from_addr") or ""
    body = state.get("text") or ""

    # LLM input (compact, consistent)
    llm_input = (
        f"From: {from_addr}\n"
        f"Subject: {subject}\n"
        f"---\n"
        f"{body}\n"
    )
    state["llm_input"] = llm_input
    return state


def node_llm_triage(state: State) -> State:
    llm = build_llm()

    parser = JsonOutputParser(pydantic_object=TriageOut)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an email security triage assistant.\n"
         "Return ONLY valid JSON matching the schema.\n"
         "Be conservative: if unsure, category='unknown' and risk_score <= 40.\n"
         "Categories: scam, promo, phishing, adult, malware, social, receipt, job, unknown.\n"
         "risk_score: 0-100.\n"
         "short_summary: 1-2 sentences.\n"
         "recommended_action: clear action (e.g., 'delete', 'ignore', 'review manually', 'report phishing').\n"
         "notes: optional, brief.\n"
        ),
        ("human",
         "Email:\n{email}\n\n"
         "Spam model output (may be None): spam_prob={spam_prob}, is_spam={is_spam}, model={spam_model}\n\n"
         "Now produce JSON.")
    ])

    chain = prompt | llm | parser

    triage = chain.invoke({
        "email": state["llm_input"],
        "spam_prob": state.get("spam_prob"),
        "is_spam": state.get("is_spam"),
        "spam_model": state.get("spam_model"),
    })

    # parser returns dict-like
    state["triage"] = dict(triage)
    return state


def build_graph():
    g = StateGraph(State)
    g.add_node("prepare", node_prepare)
    g.add_node("llm_triage", node_llm_triage)

    g.set_entry_point("prepare")
    g.add_edge("prepare", "llm_triage")
    g.add_edge("llm_triage", END)

    return g.compile()
