#!/usr/bin/env python
"""Launch the FastAPI backend: `python scripts/run_app.py`.

The Next.js frontend lives in `web/` — run it separately with `npm run dev`.
"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("deepsearch.api.main:app", host="127.0.0.1", port=8000, reload=True)
