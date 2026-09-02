"""Vercel entry point.

Vercel looks for `app` in this file. Everything real lives in `app/`.

Reading a room and talking about it are vision calls — no GPU, so they run
here happily. Picture *generation* needs a GPU, so keep BACKEND=hosted and let
Replicate do that part; BACKEND=local would try to load Stable Diffusion into a
serverless function, which will not fit and would not have a GPU if it did.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("BACKEND", "hosted")

from app.main import app  # noqa: E402

__all__ = ["app"]
