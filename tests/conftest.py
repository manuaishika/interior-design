"""No test may reach a real model.

The planner and the self-check ask a vision model a question. With a fake key
that is a network call that fails slowly (retries and all), and in an
environment with network and a real key it would be a billed one. Both degrade
to "no plan" / "no faults" on any error, so by default they get an immediate
error; a test that wants an answer patches `planning.ask_json` itself.
"""

import pytest


@pytest.fixture(autouse=True)
def no_real_vision_calls(monkeypatch):
    from app import planning

    async def refuse(*args, **kwargs):
        raise RuntimeError("no model calls in tests")

    monkeypatch.setattr(planning, "ask_json", refuse)
