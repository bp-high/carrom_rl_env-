"""Custom Gradio web UI for the Carrom environment.

Uses generators to stream physics animation frames so the user sees
coins and striker moving in real-time — like a real carrom board.
"""

from __future__ import annotations

import json
import math
import os
import re
import time

import gradio as gr
import requests

from carrom_env.models import Action, Observation
from carrom_env.renderer import (
    compute_valid_placement,
    render_board,
    render_snapshot_frame,
    _render_static_board,
)
from carrom_env.constants import BoardConfig
from server.carrom_environment import CarromEnvironment


# ──────────────────────────────────────────────────────────────────────────
# LLM call used by the "Auto-play with LLM" panel
# ──────────────────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """You are an expert Carrom player following ICF rules.
You play WHITE coins.  Pocket WHITE coins (+1 pt) and the queen (+3 pt).
Pocketing a BLACK coin is a DUE — the coin returns to the board, your turn ends.

Respond with ONLY a JSON object and nothing else:
{"placement_x": <-0.4 to 0.4>, "angle": <radians, 0 = straight ahead>, "force": <0.0 to 1.0>}"""


def _parse_llm_action(text: str):
    """Extract a {placement_x, angle, force} dict from LLM output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    m = re.search(r"\{[^}]+\}", text)
    if not m:
        return None
    try:
        d = json.loads(m.group())
        return {
            "placement_x": float(d.get("placement_x", 0.0)),
            "angle":       float(d.get("angle",       0.0)),
            "force":       max(0.0, min(1.0, float(d.get("force", 0.5)))),
        }
    except Exception:
        return None


def _llm_call(base_url: str, model: str, api_key: str,
              obs_text: str, max_tokens: int = 2048) -> dict | None:
    """Call an OpenAI-compatible chat endpoint for one shot decision."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user",   "content": obs_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # Reasoning models: answer may be in reasoning_content when content is null
        raw = msg.get("content") or msg.get("reasoning_content") or ""
        return _parse_llm_action(raw)
    except Exception as exc:
        print(f"[Gradio LLM] error: {exc}")
        return None

# Pause durations (seconds)
_AIM_PAUSE = 1.0        # show aiming overlay before physics starts
_FRAME_DELAY = 0.06      # delay between animation frames (~16 FPS)
_POST_ANIM_PAUSE = 0.5   # pause after animation before next phase


def build_carrom_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""

    env = CarromEnvironment()
    last_obs: dict = {"obs": None}
    board = BoardConfig()
    # Pre-render static background for fast animation frames
    _static_bg = _render_static_board(board)

    # ── helpers ──────────────────────────────────────────────────────
    def _pack(obs: Observation, status: str, img=None):
        if img is None:
            img = render_board(obs)
        return (
            img,
            obs.text_summary,
            f"{obs.reward:+.3f}" if obs.reward else "",
            f"You {obs.agent_score}  —  {obs.opponent_score} Opponent",
            f"{obs.remaining_coins}",
            status,
            f"Turn {obs.turn_number} / {obs.max_turns}",
        )

    def _pack_img_only(obs: Observation, status: str, img):
        """Like _pack but only update image + status (faster for animation)."""
        return (
            img,
            obs.text_summary,
            "",
            f"You {obs.agent_score}  —  {obs.opponent_score} Opponent",
            f"{obs.remaining_coins}",
            status,
            f"Turn {obs.turn_number} / {obs.max_turns}",
        )

    def _game_over(obs: Observation) -> str:
        if obs.agent_score > obs.opponent_score:
            return "🏆 YOU WIN!"
        if obs.agent_score < obs.opponent_score:
            return "😞 You lost."
        return "🤝 Draw!"

    def _empty():
        return (None, "No game. Click Reset.", "", "", "", "", "")

    def _stream_snapshots(obs, snapshots, status_prefix, is_opponent=False):
        """Yield animation frames from snapshots."""
        total = len(snapshots)
        for i, snap in enumerate(snapshots):
            frame_img = render_snapshot_frame(
                snap, board=board, static_bg=_static_bg, is_opponent=is_opponent
            )
            progress = f"({i + 1}/{total})"
            yield _pack_img_only(obs, f"{status_prefix} {progress}", frame_img)
            time.sleep(_FRAME_DELAY)

    # ── reset ────────────────────────────────────────────────────────
    def do_reset(seed_text: str = ""):
        seed = int(seed_text) if seed_text and seed_text.strip() else None
        obs = env.reset(seed=seed)
        last_obs["obs"] = obs
        return _pack(obs, "Your turn — aim and shoot!")

    # ──────────────────────────────────────────────────────────────────
    # SHOOT (generator — streams animation frames)
    # ──────────────────────────────────────────────────────────────────
    def do_shoot(placement_x: float, angle_deg: float, force: float):
        obs = last_obs.get("obs")
        if obs is None or obs.done:
            yield _empty()
            return

        angle_rad = math.radians(angle_deg)
        valid_x = compute_valid_placement(obs, placement_x)

        # ── Aiming frame ─────────────────────────────────────────────
        aim_img = render_board(
            obs,
            striker_pos=(valid_x, -0.42),
            last_action_angle=angle_rad,
            last_action_force=force,
        )
        blocked_note = ""
        if abs(valid_x - placement_x) > 0.005:
            blocked_note = f" (nudged {placement_x:.2f}→{valid_x:.2f})"
        yield _pack(obs, f"🎯 Aiming…{blocked_note}", aim_img)
        time.sleep(_AIM_PAUSE)

        # ── Agent shot animation ─────────────────────────────────────
        action = Action(placement_x=valid_x, angle=angle_rad, force=force)
        obs, snapshots = env.step_agent_animated(action)
        last_obs["obs"] = obs

        # Stream animation frames
        yield from _stream_snapshots(obs, snapshots, "🎱 Your shot…")

        # Final result (high-quality render)
        result_img = render_board(obs)
        if obs.done:
            yield _pack(obs, _game_over(obs), result_img)
            return
        yield _pack(obs, f"Your shot landed! (Turn {obs.turn_number})", result_img)

        if not env.needs_opponent_turn:
            return
        time.sleep(_POST_ANIM_PAUSE)

        # ── Opponent aiming ──────────────────────────────────────────
        opp_action = env.get_opponent_action()
        aim_img2 = render_board(obs, opponent_action=opp_action, hide_striker=True)
        yield _pack(
            obs,
            f"🤖 Opponent aiming: pos={opp_action.placement_x:.2f}, "
            f"angle={math.degrees(opp_action.angle):.0f}°, "
            f"force={opp_action.force:.0%}",
            aim_img2,
        )
        time.sleep(_AIM_PAUSE)

        # ── Opponent shot animation ──────────────────────────────────
        obs, _, opp_snapshots = env.step_opponent_animated()
        last_obs["obs"] = obs

        yield from _stream_snapshots(obs, opp_snapshots, "🤖 Opponent shot…", is_opponent=True)

        # Final result
        if obs.done:
            final_img = render_board(obs, hide_striker=True)
            yield _pack(obs, f"Opponent replied. {_game_over(obs)}", final_img)
            return
        time.sleep(_POST_ANIM_PAUSE)
        final_img = render_board(obs)
        yield _pack(obs, f"Opponent replied. (Turn {obs.turn_number}) Your turn!", final_img)

    # ── text shoot (same animation flow) ─────────────────────────────
    def do_text_shoot(text: str):
        obs = last_obs.get("obs")
        if obs is None or obs.done:
            yield _empty()
            return

        yield _pack(obs, f'🔤 Parsing: "{text}"')
        time.sleep(0.5)

        # Agent animated shot
        action = Action(action_type="text", text=text)
        obs, snapshots = env.step_agent_animated(action)
        last_obs["obs"] = obs

        yield from _stream_snapshots(obs, snapshots, "🎱 Your shot…")

        result_img = render_board(obs)
        if obs.done:
            yield _pack(obs, _game_over(obs), result_img)
            return
        yield _pack(obs, f"Your shot landed! (Turn {obs.turn_number})", result_img)

        if not env.needs_opponent_turn:
            return
        time.sleep(_POST_ANIM_PAUSE)

        # Opponent aiming
        opp_action = env.get_opponent_action()
        aim_img = render_board(obs, opponent_action=opp_action, hide_striker=True)
        yield _pack(
            obs,
            f"🤖 Opponent aiming: pos={opp_action.placement_x:.2f}, "
            f"angle={math.degrees(opp_action.angle):.0f}°, "
            f"force={opp_action.force:.0%}",
            aim_img,
        )
        time.sleep(_AIM_PAUSE)

        # Opponent animated shot
        obs, _, opp_snapshots = env.step_opponent_animated()
        last_obs["obs"] = obs

        yield from _stream_snapshots(obs, opp_snapshots, "🤖 Opponent shot…", is_opponent=True)

        if obs.done:
            final_img = render_board(obs, hide_striker=True)
            yield _pack(obs, f"Opponent replied. {_game_over(obs)}", final_img)
            return
        time.sleep(_POST_ANIM_PAUSE)
        final_img = render_board(obs)
        yield _pack(obs, f"Opponent replied. (Turn {obs.turn_number}) Your turn!", final_img)

    # ──────────────────────────────────────────────────────────────────
    # AUTO-PLAY WITH LLM (generator — loops N turns, streams animations)
    # ──────────────────────────────────────────────────────────────────
    def do_auto_play(
        endpoint: str, model: str, api_key: str, num_turns: int,
    ):
        """Play `num_turns` agent shots against the heuristic opponent,
        choosing each shot via an OpenAI-compatible LLM endpoint.
        Streams physics animation frames so the user (and any screen
        recorder) sees each shot play out.
        """
        obs = last_obs.get("obs")
        if obs is None or obs.done:
            # Reset to a fresh board
            obs = env.reset(seed=None)
            last_obs["obs"] = obs
            yield _pack(obs, "🤖 Auto-play starting — fresh board…")
            time.sleep(0.5)

        # Resolve API key: explicit field first, then env vars
        resolved_key = (api_key or "").strip() or (
            os.environ.get("NEBIUS_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("HF_TOKEN")
            or ""
        )

        n = int(num_turns or 10)
        coins_start = obs.remaining_coins
        shots_done  = 0

        for shot in range(1, n + 1):
            if obs.done:
                break

            # ── Query the LLM ────────────────────────────────────────
            yield _pack(obs,
                f"🤖 Auto-play shot {shot}/{n} — asking {model}…")
            parsed = _llm_call(endpoint, model, resolved_key,
                               obs.text_summary)
            if parsed is None:
                # Fallback: benign centre shot so the run continues
                parsed = {"placement_x": 0.0, "angle": 0.0, "force": 0.5}
                yield _pack(obs,
                    f"⚠️ Shot {shot}/{n}: parse failed, using fallback")
                time.sleep(0.5)

            placement_x = parsed["placement_x"]
            angle_rad   = parsed["angle"]
            force       = parsed["force"]
            valid_x     = compute_valid_placement(obs, placement_x)

            # ── Aiming frame ─────────────────────────────────────────
            aim_img = render_board(
                obs,
                striker_pos=(valid_x, -0.42),
                last_action_angle=angle_rad,
                last_action_force=force,
            )
            yield _pack(obs,
                f"🎯 Shot {shot}/{n}: px={valid_x:+.2f} "
                f"angle={math.degrees(angle_rad):+.0f}° force={force:.0%}",
                aim_img)
            time.sleep(_AIM_PAUSE)

            # ── Agent shot animation ─────────────────────────────────
            action = Action(placement_x=valid_x, angle=angle_rad, force=force)
            obs, snapshots = env.step_agent_animated(action)
            last_obs["obs"] = obs
            shots_done += 1
            yield from _stream_snapshots(obs, snapshots,
                f"🎱 LLM shot {shot}/{n}…")

            result_img = render_board(obs)
            if obs.done:
                yield _pack(obs, f"{_game_over(obs)} ({shots_done} LLM shots)",
                            result_img)
                return
            yield _pack(obs,
                f"Shot {shot}/{n} done — score "
                f"{obs.agent_score}-{obs.opponent_score}, "
                f"{obs.remaining_coins} coins left",
                result_img)

            # ── Opponent auto-reply (if their turn) ──────────────────
            if env.needs_opponent_turn:
                time.sleep(_POST_ANIM_PAUSE)
                opp_action = env.get_opponent_action()
                aim_img2 = render_board(obs, opponent_action=opp_action,
                                        hide_striker=True)
                yield _pack(obs, f"🤖 Opponent replying…", aim_img2)
                time.sleep(_AIM_PAUSE)

                obs, _, opp_snapshots = env.step_opponent_animated()
                last_obs["obs"] = obs
                yield from _stream_snapshots(obs, opp_snapshots,
                    "🤖 Opponent shot…", is_opponent=True)
                if obs.done:
                    yield _pack(obs,
                        f"{_game_over(obs)} ({shots_done} LLM shots)",
                        render_board(obs, hide_striker=True))
                    return

            time.sleep(_POST_ANIM_PAUSE)

        # Finished the requested number of shots
        potted = coins_start - obs.remaining_coins
        yield _pack(obs,
            f"✅ Auto-play complete — {shots_done} shots, "
            f"~{potted} coins removed, final {obs.agent_score}-{obs.opponent_score}",
            render_board(obs))

    # ── live preview ─────────────────────────────────────────────────
    def preview_shot(placement_x: float, angle_deg: float, force: float):
        obs = last_obs.get("obs")
        if obs is None or obs.done:
            return None
        angle_rad = math.radians(angle_deg)
        valid_x = compute_valid_placement(obs, placement_x)
        return render_board(
            obs,
            striker_pos=(valid_x, -0.42),
            last_action_angle=angle_rad,
            last_action_force=force,
        )

    # ═════════════════════════════════════════════════════════════════
    # Layout
    # ═════════════════════════════════════════════════════════════════
    with gr.Blocks(title="Carrom RL Environment") as app:
        gr.Markdown(
            "# 🎯 Carrom RL Environment\n"
            "Physics-based Carrom board powered by Pymunk. "
            "Aim with the sliders, click **Shoot!**, and watch "
            "each turn unfold — your shot, then the opponent's.\n\n"
            "**Opponent:** heuristic bot (blue striker) — aims at the "
            "nearest coin from centre, force 0.65."
        )

        with gr.Row():
            with gr.Column(scale=3):
                board_img = gr.Image(
                    label="Carrom Board", type="pil",
                    height=560, interactive=False,
                )
                with gr.Row():
                    score_display = gr.Textbox(
                        label="Score", interactive=False, scale=2)
                    coins_display = gr.Textbox(
                        label="Coins Left", interactive=False, scale=1)
                    reward_display = gr.Textbox(
                        label="Reward", interactive=False, scale=1)
                    turn_display = gr.Textbox(
                        label="Turn", interactive=False, scale=1)

            with gr.Column(scale=2):
                status_display = gr.Textbox(
                    label="Status", interactive=False, lines=2)

                gr.Markdown("### 🎮 Controls")
                with gr.Group():
                    seed_input = gr.Textbox(
                        label="Seed (optional)", placeholder="42", value="")
                    reset_btn = gr.Button(
                        "🔄 Reset Game", variant="secondary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 🏹 Aim Your Shot")
                placement_slider = gr.Slider(
                    minimum=-0.4, maximum=0.4, value=0.0, step=0.01,
                    label="Striker Position (left ← → right)")
                angle_slider = gr.Slider(
                    minimum=-90, maximum=90, value=0, step=1,
                    label="Angle (degrees, 0 = straight ahead)")
                force_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                    label="Force (0 = soft, 1 = maximum)")
                shoot_btn = gr.Button(
                    "🎯 Shoot!", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 💬 Or Use Text Action")
                text_input = gr.Textbox(
                    label="Describe your shot",
                    placeholder="aim at queen_0 with strong force from center")
                text_btn = gr.Button(
                    "📝 Execute Text Action", variant="secondary")

                gr.Markdown("---")
                gr.Markdown(
                    "### 🤖 Auto-play with LLM\n"
                    "Point at any OpenAI-compatible endpoint (Nebius / HF / OpenAI / local vLLM). "
                    "Key defaults to `NEBIUS_API_KEY` / `OPENAI_API_KEY` / `HF_TOKEN` env vars."
                )
                llm_endpoint = gr.Textbox(
                    label="API base URL",
                    value=os.environ.get(
                        "API_BASE_URL",
                        "https://api.tokenfactory.us-central1.nebius.com/v1",
                    ),
                )
                llm_model = gr.Textbox(
                    label="Model name",
                    value=os.environ.get(
                        "MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b",
                    ),
                )
                llm_api_key = gr.Textbox(
                    label="API key (leave blank to use env var)",
                    type="password",
                    value="",
                )
                llm_turns = gr.Slider(
                    minimum=1, maximum=100, value=15, step=1,
                    label="Number of agent shots",
                )
                auto_btn = gr.Button(
                    "🤖 Auto-play with LLM", variant="primary", size="lg",
                )

        with gr.Accordion("📋 Board State Details", open=False):
            obs_text = gr.Textbox(
                label="Full Observation", lines=15, interactive=False)

        # ── Wire events ──────────────────────────────────────────────
        outputs = [board_img, obs_text, reward_display, score_display,
                   coins_display, status_display, turn_display]

        # Auto-start game on page load
        app.load(fn=do_reset, inputs=[seed_input], outputs=outputs)
        reset_btn.click(fn=do_reset, inputs=[seed_input], outputs=outputs)

        # Generators that yield 4 frames each
        shoot_btn.click(
            fn=do_shoot,
            inputs=[placement_slider, angle_slider, force_slider],
            outputs=outputs,
        )
        text_btn.click(
            fn=do_text_shoot,
            inputs=[text_input],
            outputs=outputs,
        )
        auto_btn.click(
            fn=do_auto_play,
            inputs=[llm_endpoint, llm_model, llm_api_key, llm_turns],
            outputs=outputs,
        )

        # Live preview
        for slider in [placement_slider, angle_slider, force_slider]:
            slider.change(
                fn=preview_shot,
                inputs=[placement_slider, angle_slider, force_slider],
                outputs=[board_img],
            )

    return app
