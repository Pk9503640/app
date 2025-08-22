from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
from typing import Optional
import os
import json
import httpx
import asyncio

app = FastAPI()

# Environment variables (Railway ke dashboard me set karo)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Helper to stream chunks ---
async def stream_response(text: str):
    for i in range(0, len(text), 50):
        yield text[i:i+50]
        await asyncio.sleep(0.05)

# --- Model Call Functions ---
async def call_groq(prompt: str):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.groq.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    data = resp.json()
    return data["choices"][0]["message"]["content"]

async def call_gemini(prompt: str):
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            headers=headers,
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

# --- Auto Mode Decision ---
def choose_model(prompt: str) -> str:
    prompt_lower = prompt.lower()
    # simple heuristic (can be improved later with regex / classifier)
    if any(word in prompt_lower for word in ["solve", "equation", "math", "physics", "theorem", "integral", "derivative"]):
        return "gemini"
    return "groq"

@app.post("/solve")
async def solve(
    prompt: str = Form(...),
    model: str = Form("auto"),
    history: str = Form("[]"),
    image: Optional[UploadFile] = None
):
    try:
        history_data = json.loads(history)
    except:
        history_data = []

    text = ""
    try:
        # Explicit Model Selection
        if model == "1.0":
            text = await call_groq(prompt)
        elif model == "mathica":
            text = await call_gemini(prompt)
        else:  # Auto
            selected = choose_model(prompt)
            try:
                if selected == "gemini":
                    text = await call_gemini(prompt)
                else:
                    text = await call_groq(prompt)
            except Exception:
                # fallback if primary fails
                if selected == "gemini":
                    text = await call_groq(prompt)
                else:
                    text = await call_gemini(prompt)

    except Exception as e:
        text = f"Error while processing: {str(e)}"

    return StreamingResponse(stream_response(text), media_type="text/plain")
