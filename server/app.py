"""FastAPI application for the Carrom environment."""

import os

from openenv.core.env_server.http_server import create_app

from carrom_env.models import Action, Observation
from server.carrom_environment import CarromEnvironment

# Use custom visual Gradio UI when web interface is enabled
_enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")

if _enable_web:
    from server.gradio_ui import build_carrom_ui

    _gradio_app = build_carrom_ui()
    # Streaming generators (animation frames) require an explicit queue.
    _gradio_app.queue(default_concurrency_limit=4)

    from openenv.core.env_server.http_server import create_fastapi_app

    # Create the base API app (reset/step/state/health/schema endpoints)
    app = create_fastapi_app(CarromEnvironment, Action, Observation)

    # Mount the Gradio UI at "/" so HF Spaces' root-path iframe shows it.
    # Mounting the same Blocks at multiple paths breaks the SSE/queue
    # routing, so keep this to a single mount.
    import gradio as gr
    app = gr.mount_gradio_app(app, _gradio_app, path="/")
else:
    app = create_app(
        CarromEnvironment,
        Action,
        Observation,
        env_name="carrom_env",
    )


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
