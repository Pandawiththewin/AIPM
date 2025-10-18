# automation/main.py
from typing import Optional, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
import asyncio

app = FastAPI(title="Browser Use Automation", version="2.1.0")

# Try to import browser-use. If it's missing, we run in simulated mode.
BROWSER_OK = True
IMPORT_ERROR = ""
try:
    from browser_use import Agent, Browser  # type: ignore
except Exception as e:
    BROWSER_OK = False
    IMPORT_ERROR = str(e)

class Plan(BaseModel):
    intent: Optional[str] = None
    service: Optional[str] = None
    instructions: str

class RunResult(BaseModel):
    status: str
    output: str
    url: Optional[str] = None

API_KEY = os.getenv("BROWSER_USE_API_KEY", "")

@app.get("/health")
async def health():
    return {"ok": True, "browser_use": BROWSER_OK}

async def _run_with_async_browser(instructions: str) -> Any:
    """Preferred path for browser-use builds that support async context managers."""
    async with Browser() as browser:  # type: ignore
        agent = Agent(browser=browser, model="gpt-4o-mini")
        return await agent.run(instructions)

def _worker_sync_browser(instructions: str) -> Any:
    """
    Runs in a background thread for browser-use builds that only support sync 'with Browser()'.
    We spin a private event loop inside the thread to await agent.run().
    """
    import asyncio as _aio
    with Browser() as browser:  # type: ignore
        agent = Agent(browser=browser, model="gpt-4o-mini")
        return _aio.run(agent.run(instructions))

async def _run_browser_use(instructions: str) -> Any:
    """
    Try async-with first; if the implementation raises a TypeError about
    unsupported async context manager, fall back to sync-with in a thread.
    """
    try:
        return await _run_with_async_browser(instructions)
    except TypeError as te:
        msg = str(te)
        if "does not support the asynchronous context manager protocol" in msg:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _worker_sync_browser, instructions)
        raise
    except AttributeError:
        # Some builds expose only sync manager; fall back
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _worker_sync_browser, instructions)

@app.post("/run", response_model=RunResult)
async def run_action(plan: Plan, x_api_key: Optional[str] = Header(default="")):
    # Optional header-based protection
    expected = API_KEY or None
    if expected and (x_api_key or "") != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    instructions = plan.instructions or ""

    if not BROWSER_OK:
        # Simulated mode so the loop still works without installing browser-use.
        return RunResult(
            status="success",
            output=f"[SIMULATED] Would run instructions: {instructions}\n"
                   f"(Install `browser-use` for real execution) Import error: {IMPORT_ERROR}",
            url=None
        )

    try:
        result: Any = await _run_browser_use(instructions)
        # Try to extract a URL if present
        url = None
        if isinstance(result, dict):
            url = result.get("url")
        else:
            url = getattr(result, "url", None)
        return RunResult(status="success", output=str(result), url=url)
    except Exception as e:
        return RunResult(status="error", output=str(e), url=None)
