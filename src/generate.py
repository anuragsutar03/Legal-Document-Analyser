"""
generate.py
Takes a user query + retrieved chunks and produces a grounded, cited answer.

Supports three LLM providers, checked in this order, plus a fallback:
  1. GEMINI_API_KEY   - Google Gemini (has a genuinely free tier, no card needed)
  2. ANTHROPIC_API_KEY - Claude (paid)
  3. OPENAI_API_KEY    - GPT (paid)
  4. Extractive fallback - if no key is set, returns the most relevant
     retrieved passages directly with citations, so the app is fully
     runnable and demoable without any paid API access.
"""

import os

SYSTEM_PROMPT = (
    "You are a legal research assistant. Answer the user's question using ONLY "
    "the information in the provided excerpts. For every claim you make, cite the "
    "excerpt it came from using the format [Source: <doc title>, <section>]. "
    "If the excerpts do not contain enough information to answer, say so explicitly "
    "rather than guessing. Do not provide information beyond what is in the excerpts. "
    "This is for educational purposes only and is not legal advice."
)


def _build_context(chunks):
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[Source: {c['doc_title']}, {c['section_label']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(context_blocks)


def _extractive_fallback(query, chunks):
    lines = [
        "(No LLM API key detected - showing the most relevant excerpts directly. "
        "Set GEMINI_API_KEY (free), ANTHROPIC_API_KEY, or OPENAI_API_KEY as an "
        "environment variable to enable full generated answers.)\n",
        f"Question: {query}\n",
        "Most relevant excerpts found:\n",
    ]
    for c in chunks:
        lines.append(f"- [{c['doc_title']}, {c['section_label']}]: {c['text'][:400]}...")
    return "\n".join(lines)


def _call_gemini(query, context):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(f"Excerpts:\n\n{context}\n\nQuestion: {query}")
    return response.text


def _call_anthropic(query, context):
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Excerpts:\n\n{context}\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text


def _call_openai(query, context):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {query}"},
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content


def generate_answer(query: str, chunks: list):
    if not chunks:
        return "No relevant excerpts were found in the corpus for this question."

    context = _build_context(chunks)

    if os.environ.get("GEMINI_API_KEY"):
        try:
            return _call_gemini(query, context)
        except Exception as e:
            return f"(Gemini call failed: {e})\n\n" + _extractive_fallback(query, chunks)

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _call_anthropic(query, context)
        except Exception as e:
            return f"(Anthropic call failed: {e})\n\n" + _extractive_fallback(query, chunks)

    if os.environ.get("OPENAI_API_KEY"):
        try:
            return _call_openai(query, context)
        except Exception as e:
            return f"(OpenAI call failed: {e})\n\n" + _extractive_fallback(query, chunks)

    return _extractive_fallback(query, chunks)
