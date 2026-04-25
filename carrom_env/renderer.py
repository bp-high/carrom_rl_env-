"""Visual board renderer for the Carrom environment.

Renders the board state as a PIL Image using matplotlib — works headless
without pygame.  Used by the Gradio web interface and for saving replays.
"""

from __future__ import annotations

import io
import math
from typing import Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as path_effects
from PIL import Image
import numpy as np

from carrom_env.constants import BoardConfig
from carrom_env.models import CoinInfo, Observation

# ── Premium colour palette ──────────────────────────────────────────────
BACKGROUND = "#1a1a2e"           # dark navy background

# Board frame (outer wood)
FRAME_OUTER = "#5C3317"          # dark walnut
FRAME_INNER = "#8B5E3C"         # lighter wood ring
FRAME_EDGE = "#3B1F0E"          # darkest edge

# Playing surface
SURFACE_COLOR = "#F0D9A0"       # warm birch ply
SURFACE_EDGE = "#D4B87A"        # slightly darker border ring

# Board markings
LINE_COLOR = "#BFA06A"          # subtle gold-brown lines
CENTER_DOT = "#C0392B"          # red centre dot (queen placement)

# Pockets
POCKET_OUTER = "#1a1a1a"        # outer pocket rim
POCKET_INNER = "#0a0a0a"        # deep pocket hole
POCKET_RIM = "#333333"          # glint on rim

# Coins
BLACK_COIN_FACE = "#1C1C1C"
BLACK_COIN_EDGE = "#444444"
BLACK_COIN_HIGHLIGHT = "#555555"

WHITE_COIN_FACE = "#EDEDED"
WHITE_COIN_EDGE = "#AAAAAA"
WHITE_COIN_HIGHLIGHT = "#FFFFFF"

QUEEN_FACE = "#C0392B"          # rich red
QUEEN_EDGE = "#922B21"
QUEEN_HIGHLIGHT = "#E74C3C"

STRIKER_FACE = "#F1C40F"        # bright gold
STRIKER_EDGE = "#D4AC0D"
STRIKER_HIGHLIGHT = "#F9E154"

# Shot arrow
ARROW_COLOR = "#E74C3C"
ARROW_GLOW = "#FF6B6B"

# Typography
SCORE_COLOR = "#E0E0E0"
STATUS_GREEN = "#2ECC71"
STATUS_RED = "#E74C3C"
STATUS_ORANGE = "#F39C12"

BLOCKED_ZONE_COLOR = "#E74C3C"   # red blocked zone on baseline


# ── Public utility: valid placement from observation data ────────────

def compute_valid_placement(
    obs: Observation,
    desired_x: float,
    board: Optional[BoardConfig] = None,
    player: str = "agent",
) -> float:
    """Return the nearest valid striker x position given observation data.

    Mirrors the logic of ``CarromEnv._find_valid_placement`` but works
    purely from observation data so the UI can preview the snap.
    """
    board = board or BoardConfig()
    half = board.size / 2.0
    sr = board.striker_radius
    cr = board.coin_radius
    min_gap = sr + cr + 0.002
    placement_limit = half - board.pocket_radius - sr

    baseline_y = -half + board.striker_offset if player == "agent" else half - board.striker_offset

    def _blocked(x: float) -> bool:
        for coin in obs.coins:
            if coin.pocketed:
                continue
            if abs(coin.y - baseline_y) < min_gap:
                if abs(coin.x - x) < min_gap:
                    return True
        return False

    desired_x = float(max(-placement_limit, min(desired_x, placement_limit)))
    if not _blocked(desired_x):
        return desired_x

    step = 0.02
    for offset in range(1, 40):
        for sign in (1, -1):
            candidate = desired_x + sign * offset * step
            candidate = float(max(-placement_limit, min(candidate, placement_limit)))
            if not _blocked(candidate):
                return candidate
    return desired_x


def render_board(
    obs: Observation,
    board: Optional[BoardConfig] = None,
    striker_pos: Optional[Tuple[float, float]] = None,
    last_action_angle: Optional[float] = None,
    last_action_force: Optional[float] = None,
    opponent_action: Optional[Any] = None,
    hide_striker: bool = False,
    figsize: Tuple[float, float] = (7, 7.6),
    dpi: int = 120,
) -> Image.Image:
    """Render the carrom board as a high-quality PIL Image.

    Args:
        opponent_action: If provided (an Action with placement_x/angle/force),
            draws the opponent's striker on the top baseline with their shot
            direction arrow — used for the "opponent aiming" preview frame.
        hide_striker: If True, skip drawing the agent's striker (useful when
            showing the opponent's turn so the agent striker is not visible).
    """

    board = board or BoardConfig()
    half = board.size / 2.0
    margin = 0.11  # space for frame + scoreboard

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.set_xlim(-half - margin, half + margin)
    ax.set_ylim(-half - margin - 0.06, half + margin + 0.02)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BACKGROUND)

    # ── 1. Wooden frame (three-layer illusion) ──────────────────────────
    _draw_frame(ax, half, margin)

    # ── 2. Playing surface ──────────────────────────────────────────────
    _draw_surface(ax, half, board)

    # ── 3. Pockets ────────────────────────────────────────────────────
    _draw_pockets(ax, half, board)

    # ── 4. Board markings ─────────────────────────────────────────────
    _draw_markings(ax, half, board)

    # ── 5. Coins ──────────────────────────────────────────────────────
    _draw_coins(ax, obs, board)

    # ── 5b. Blocked zones on agent baseline (shown during aiming) ────
    if striker_pos is not None and not hide_striker:
        _draw_blocked_zones(ax, obs, board)

    # ── 6. Striker ────────────────────────────────────────────────────
    sx, sy, striker_pocketed = 0.0, 0.0, True  # defaults when hidden
    if not hide_striker:
        sx, sy, striker_pocketed = _draw_striker(ax, obs, striker_pos, board)

    # ── 7. Shot direction arrow ───────────────────────────────────────
    if last_action_angle is not None and not striker_pocketed and not hide_striker:
        _draw_shot_arrow(ax, sx, sy, last_action_angle, last_action_force)

    # ── 7b. Opponent aiming overlay ────────────────────────────────────
    if opponent_action is not None:
        _draw_opponent_aiming(ax, opponent_action, board)

    # ── 8. Scoreboard & status ────────────────────────────────────────
    _draw_scoreboard(ax, obs, half, margin)

    # ── Convert ──────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# =====================================================================
# Drawing helpers
# =====================================================================

def _draw_frame(ax, half: float, margin: float) -> None:
    """Three-layer wooden frame around the board."""
    # Outermost border
    _rounded_rect(ax, -half - margin + 0.01, -half - margin + 0.01,
                  2 * (half + margin) - 0.02, 2 * (half + margin) - 0.02,
                  facecolor=FRAME_EDGE, edgecolor="none", linewidth=0, zorder=0,
                  pad=0.015)
    # Middle frame
    _rounded_rect(ax, -half - margin + 0.025, -half - margin + 0.025,
                  2 * (half + margin) - 0.05, 2 * (half + margin) - 0.05,
                  facecolor=FRAME_OUTER, edgecolor="none", linewidth=0, zorder=1,
                  pad=0.012)
    # Inner bevel
    _rounded_rect(ax, -half - 0.02, -half - 0.02,
                  2 * half + 0.04, 2 * half + 0.04,
                  facecolor=FRAME_INNER, edgecolor="none", linewidth=0, zorder=2,
                  pad=0.008)


def _draw_surface(ax, half: float, board: BoardConfig) -> None:
    """Playing surface with subtle inner border."""
    # Main surface
    surface = patches.FancyBboxPatch(
        (-half, -half), board.size, board.size,
        boxstyle="round,pad=0.005",
        facecolor=SURFACE_COLOR, edgecolor=SURFACE_EDGE, linewidth=1.5, zorder=3,
    )
    ax.add_patch(surface)


def _draw_pockets(ax, half: float, board: BoardConfig) -> None:
    """Pockets with depth illusion: outer rim, dark hole, inner glint."""
    corners = [(-half, -half), (-half, half), (half, -half), (half, half)]
    for cx, cy in corners:
        # Outer shadow ring
        shadow = plt.Circle((cx, cy), board.pocket_radius + 0.008,
                            facecolor=POCKET_OUTER, edgecolor="none", zorder=4)
        ax.add_patch(shadow)
        # Deep hole
        hole = plt.Circle((cx, cy), board.pocket_radius,
                          facecolor=POCKET_INNER, edgecolor=POCKET_RIM,
                          linewidth=1.2, zorder=5)
        ax.add_patch(hole)
        # Subtle inner highlight (top-left glint)
        glint = plt.Circle((cx - 0.008, cy + 0.008), board.pocket_radius * 0.3,
                           facecolor="#3a3a3a", edgecolor="none", alpha=0.4, zorder=6)
        ax.add_patch(glint)


def _draw_markings(ax, half: float, board: BoardConfig) -> None:
    """Board decorations: circles, baselines, corner arcs, diagonal guides."""
    pr = board.pocket_radius

    # ── Centre circles ──
    # Outer ring
    ax.add_patch(plt.Circle((0, 0), 0.19, fill=False,
                            edgecolor=LINE_COLOR, linewidth=1.2, zorder=7))
    # Inner ring
    ax.add_patch(plt.Circle((0, 0), 0.05, fill=False,
                            edgecolor=LINE_COLOR, linewidth=1.5, zorder=7))
    # Centre dot
    ax.add_patch(plt.Circle((0, 0), 0.008, facecolor=CENTER_DOT,
                            edgecolor="none", zorder=8))

    # ── Baselines ──
    bline_inset = 0.06  # how far from edge the baseline stops
    for sign in (-1, 1):
        y = sign * (half - board.striker_offset)
        ax.plot(
            [-half + bline_inset, half - bline_inset], [y, y],
            color=LINE_COLOR, linewidth=1.0, linestyle="-", alpha=0.55, zorder=7,
        )
        # Small circles at baseline ends
        for xend in (-half + bline_inset, half - bline_inset):
            ax.add_patch(plt.Circle((xend, y), 0.008, fill=False,
                                    edgecolor=LINE_COLOR, linewidth=0.8,
                                    alpha=0.55, zorder=7))

    # ── Corner arcs (quarter circles near each pocket) ──
    arc_radius = 0.12
    for cx, cy in [(-half, -half), (-half, half), (half, -half), (half, half)]:
        theta1 = _corner_arc_angle(cx, cy, half)
        arc = patches.Arc((cx, cy), 2 * arc_radius, 2 * arc_radius,
                          angle=0, theta1=theta1, theta2=theta1 + 90,
                          edgecolor=LINE_COLOR, linewidth=0.8, alpha=0.5, zorder=7)
        ax.add_patch(arc)

    # ── Diagonal guide lines from pockets toward centre ──
    guide_len = 0.14
    for cx, cy in [(-half, -half), (-half, half), (half, -half), (half, half)]:
        dx = 1.0 if cx < 0 else -1.0
        dy = 1.0 if cy < 0 else -1.0
        norm = math.sqrt(2)
        ax.plot(
            [cx + dx * pr * 0.7, cx + dx * (pr * 0.7 + guide_len)],
            [cy + dy * pr * 0.7, cy + dy * (pr * 0.7 + guide_len)],
            color=LINE_COLOR, linewidth=0.7, alpha=0.4, zorder=7,
        )


def _corner_arc_angle(cx: float, cy: float, half: float) -> float:
    """Starting angle (degrees) for a corner arc depending on which corner."""
    if cx < 0 and cy < 0:
        return 0
    elif cx > 0 and cy < 0:
        return 90
    elif cx > 0 and cy > 0:
        return 180
    else:
        return 270


def _draw_coins(ax, obs: Observation, board: BoardConfig) -> None:
    """Draw coins with 3-D shadow + highlight."""
    r = board.coin_radius
    for coin in obs.coins:
        if coin.pocketed:
            continue
        face, edge, hl = _coin_palette(coin.color)
        cx, cy = coin.x, coin.y

        # Drop shadow
        shadow = plt.Circle((cx + 0.004, cy - 0.004), r,
                            facecolor="#00000033", edgecolor="none", zorder=9)
        ax.add_patch(shadow)

        # Main disc
        disc = plt.Circle((cx, cy), r,
                          facecolor=face, edgecolor=edge, linewidth=1.4, zorder=10)
        ax.add_patch(disc)

        # Specular highlight (small bright circle top-left)
        spec = plt.Circle((cx - r * 0.28, cy + r * 0.28), r * 0.25,
                          facecolor=hl, edgecolor="none", alpha=0.55, zorder=11)
        ax.add_patch(spec)

        # Ring detail on coin surface
        ring = plt.Circle((cx, cy), r * 0.65, fill=False,
                          edgecolor=edge, linewidth=0.5, alpha=0.35, zorder=11)
        ax.add_patch(ring)

        # Label for queen
        if coin.color == "queen":
            txt = ax.text(cx, cy, "Q", ha="center", va="center",
                          fontsize=8, fontweight="bold", color="#FFFFFF", zorder=12)
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground=QUEEN_EDGE),
                path_effects.Normal(),
            ])


def _draw_striker(ax, obs: Observation, striker_pos, board: BoardConfig):
    """Draw the striker disc and return (sx, sy, pocketed)."""
    sx, sy = _get_striker_pos(obs, striker_pos, board)
    striker_pocketed = len(obs.pocketed) > 0 and obs.pocketed[0]

    if not striker_pocketed:
        r = board.striker_radius

        # Glow ring
        glow = plt.Circle((sx, sy), r + 0.008,
                          facecolor="none", edgecolor=STRIKER_HIGHLIGHT,
                          linewidth=1.5, alpha=0.3, zorder=11)
        ax.add_patch(glow)

        # Shadow
        shadow = plt.Circle((sx + 0.005, sy - 0.005), r,
                            facecolor="#00000044", edgecolor="none", zorder=11)
        ax.add_patch(shadow)

        # Main disc
        disc = plt.Circle((sx, sy), r,
                          facecolor=STRIKER_FACE, edgecolor=STRIKER_EDGE,
                          linewidth=2, zorder=12)
        ax.add_patch(disc)

        # Highlight
        spec = plt.Circle((sx - r * 0.25, sy + r * 0.25), r * 0.3,
                          facecolor=STRIKER_HIGHLIGHT, edgecolor="none",
                          alpha=0.5, zorder=13)
        ax.add_patch(spec)

        # Cross-hair on striker
        tiny = r * 0.25
        ax.plot([sx - tiny, sx + tiny], [sy, sy],
                color=STRIKER_EDGE, linewidth=1, alpha=0.6, zorder=13)
        ax.plot([sx, sx], [sy - tiny, sy + tiny],
                color=STRIKER_EDGE, linewidth=1, alpha=0.6, zorder=13)

    return sx, sy, striker_pocketed


def _draw_shot_arrow(ax, sx: float, sy: float,
                     angle: float, force: Optional[float]) -> None:
    """Draw a glowing directional arrow from the striker."""
    f = force if force is not None else 0.5
    arrow_len = 0.06 + f * 0.18
    dx = math.sin(angle) * arrow_len
    dy = math.cos(angle) * arrow_len

    # Glow line (thicker, semi-transparent)
    ax.plot([sx, sx + dx], [sy, sy + dy],
            color=ARROW_GLOW, linewidth=5, alpha=0.25, solid_capstyle="round",
            zorder=14)

    # Main arrow
    arrow = FancyArrowPatch(
        (sx, sy), (sx + dx, sy + dy),
        arrowstyle="->,head_width=7,head_length=5",
        color=ARROW_COLOR, linewidth=2.5, zorder=15,
    )
    ax.add_patch(arrow)

    # Force label at arrow tip
    ax.text(sx + dx * 1.15, sy + dy * 1.15,
            f"{f:.0%}", ha="center", va="center",
            fontsize=7, color=ARROW_COLOR, fontweight="bold",
            alpha=0.8, zorder=15)


def _draw_scoreboard(ax, obs: Observation, half: float, margin: float) -> None:
    """Scoreboard strip below the board and status badge above."""
    # ── Bottom scoreboard ──
    score_y = -half - margin - 0.02
    parts = [
        (f"You: {obs.agent_score}", -0.32, "left", SCORE_COLOR),
        ("|", -0.12, "center", "#666666"),
        (f"Opp: {obs.opponent_score}", -0.08, "left", SCORE_COLOR),
        ("|", 0.10, "center", "#666666"),
        (f"Coins: {obs.remaining_coins}", 0.14, "left", SCORE_COLOR),
        ("|", 0.30, "center", "#666666"),
        (f"Turn {obs.turn_number}/{obs.max_turns}", 0.34, "left", "#999999"),
    ]
    for txt, x, ha, col in parts:
        ax.text(x, score_y, txt, ha=ha, va="center", fontsize=8.5,
                color=col, fontfamily="monospace", fontweight="bold", zorder=20)

    # ── Top status badge ──
    if obs.done:
        label, bg, fg = "GAME OVER", STATUS_RED, "#FFFFFF"
    else:
        if obs.current_player == "agent":
            label, bg, fg = "YOUR TURN", STATUS_GREEN, "#FFFFFF"
        else:
            label, bg, fg = "OPPONENT", STATUS_ORANGE, "#FFFFFF"

    badge_y = half + margin - 0.005
    badge = patches.FancyBboxPatch(
        (-0.13, badge_y - 0.018), 0.26, 0.036,
        boxstyle="round,pad=0.008",
        facecolor=bg, edgecolor="none", alpha=0.9, zorder=19,
    )
    ax.add_patch(badge)
    ax.text(0, badge_y, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=fg, zorder=20)

    # ── Reward badge (only after a shot) ──
    if obs.reward != 0.0:
        rw_color = STATUS_GREEN if obs.reward > 0 else STATUS_RED
        rw_txt = f"{obs.reward:+.2f}"
        ax.text(half + margin - 0.02, score_y, rw_txt,
                ha="right", va="center", fontsize=8, fontweight="bold",
                color=rw_color, fontfamily="monospace", zorder=20)


# =====================================================================
# Utility helpers
# =====================================================================

def _rounded_rect(ax, x, y, w, h, facecolor, edgecolor, linewidth,
                  zorder, pad=0.01) -> None:
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(rect)


def _coin_palette(color: str):
    """Return (face, edge, highlight) colours for a coin type."""
    if color == "black":
        return BLACK_COIN_FACE, BLACK_COIN_EDGE, BLACK_COIN_HIGHLIGHT
    elif color == "white":
        return WHITE_COIN_FACE, WHITE_COIN_EDGE, WHITE_COIN_HIGHLIGHT
    elif color == "queen":
        return QUEEN_FACE, QUEEN_EDGE, QUEEN_HIGHLIGHT
    return "#888888", "#666666", "#AAAAAA"


def _get_striker_pos(
    obs: Observation,
    override: Optional[Tuple[float, float]],
    board: BoardConfig,
) -> Tuple[float, float]:
    if override:
        return override
    if obs.positions and len(obs.positions) > 0:
        pos = obs.positions[0]
        if pos[0] != -1.0 or pos[1] != -1.0:
            return (pos[0], pos[1])
    return (0.0, -board.size / 2.0 + board.striker_offset)



# ── Blocked zones on baseline ────────────────────────────────────────

def _draw_blocked_zones(ax, obs: Observation, board: BoardConfig) -> None:
    """Draw translucent red zones on the agent baseline where coins block
    striker placement."""
    half = board.size / 2.0
    sr = board.striker_radius
    cr = board.coin_radius
    min_gap = sr + cr + 0.002
    baseline_y = -half + board.striker_offset

    for coin in obs.coins:
        if coin.pocketed:
            continue
        if abs(coin.y - baseline_y) < min_gap:
            # Draw a red translucent rectangle spanning the blocked x range
            bx_left = coin.x - min_gap
            bx_right = coin.x + min_gap
            by_bot = baseline_y - sr - 0.005
            by_top = baseline_y + sr + 0.005
            rect = patches.FancyBboxPatch(
                (bx_left, by_bot), bx_right - bx_left, by_top - by_bot,
                boxstyle="round,pad=0.005",
                facecolor=BLOCKED_ZONE_COLOR, edgecolor=BLOCKED_ZONE_COLOR,
                alpha=0.18, linewidth=1, linestyle="--", zorder=9,
            )
            ax.add_patch(rect)
            # Small "X" marker
            ax.text(coin.x, baseline_y, "\u2716", ha="center", va="center",
                    fontsize=7, color=BLOCKED_ZONE_COLOR, alpha=0.5,
                    fontweight="bold", zorder=10)


# ── Opponent aiming overlay ──────────────────────────────────────────

# Opponent uses a distinct blue palette
_OPP_STRIKER_FACE = "#3498DB"
_OPP_STRIKER_EDGE = "#2471A3"
_OPP_STRIKER_HL = "#85C1E9"
_OPP_ARROW = "#2980B9"
_OPP_ARROW_GLOW = "#5DADE2"


def _draw_opponent_aiming(ax, opp_action, board: BoardConfig) -> None:
    """Draw the opponent's striker on the TOP baseline with their aim arrow.

    The opponent's angle convention: 0 = straight down toward the agent side,
    but _angle_to_dir flips dy for the opponent.  The arrow direction is
    (sin(angle), -cos(angle)) from the opponent's perspective.
    """
    half = board.size / 2.0
    r = board.striker_radius

    placement_limit = half - board.pocket_radius - r
    px = max(-placement_limit, min(float(opp_action.placement_x), placement_limit))
    py = half - board.striker_offset

    # ── Striker disc (blue) ──
    glow = plt.Circle((px, py), r + 0.008,
                      facecolor="none", edgecolor=_OPP_STRIKER_HL,
                      linewidth=1.5, alpha=0.35, zorder=16)
    ax.add_patch(glow)

    shadow = plt.Circle((px + 0.005, py - 0.005), r,
                        facecolor="#00000044", edgecolor="none", zorder=16)
    ax.add_patch(shadow)

    disc = plt.Circle((px, py), r,
                      facecolor=_OPP_STRIKER_FACE, edgecolor=_OPP_STRIKER_EDGE,
                      linewidth=2, zorder=17)
    ax.add_patch(disc)

    spec = plt.Circle((px - r * 0.25, py + r * 0.25), r * 0.3,
                      facecolor=_OPP_STRIKER_HL, edgecolor="none",
                      alpha=0.5, zorder=18)
    ax.add_patch(spec)

    # Cross-hair
    tiny = r * 0.25
    ax.plot([px - tiny, px + tiny], [py, py],
            color=_OPP_STRIKER_EDGE, linewidth=1, alpha=0.6, zorder=18)
    ax.plot([px, px], [py - tiny, py + tiny],
            color=_OPP_STRIKER_EDGE, linewidth=1, alpha=0.6, zorder=18)

    # "OPP" label
    txt = ax.text(px, py - r - 0.025, "OPP", ha="center", va="center",
                  fontsize=6, fontweight="bold", color=_OPP_ARROW,
                  alpha=0.8, zorder=19)

    # ── Shot arrow ──
    angle = float(opp_action.angle)
    force = float(opp_action.force)
    arrow_len = 0.06 + force * 0.18
    # Opponent direction: (sin(angle), -cos(angle))
    dx = math.sin(angle) * arrow_len
    dy = -math.cos(angle) * arrow_len

    ax.plot([px, px + dx], [py, py + dy],
            color=_OPP_ARROW_GLOW, linewidth=5, alpha=0.25,
            solid_capstyle="round", zorder=19)

    arrow = FancyArrowPatch(
        (px, py), (px + dx, py + dy),
        arrowstyle="->,head_width=7,head_length=5",
        color=_OPP_ARROW, linewidth=2.5, zorder=20,
    )
    ax.add_patch(arrow)

    ax.text(px + dx * 1.2, py + dy * 1.2,
            f"{force:.0%}", ha="center", va="center",
            fontsize=7, color=_OPP_ARROW, fontweight="bold",
            alpha=0.85, zorder=20)


# =====================================================================
# Fast animation frame renderer (PIL-based for speed)
# =====================================================================

from PIL import ImageDraw, ImageFont

# Colour tuples for PIL (RGB)
_PIL_BLACK_COIN = (28, 28, 28)
_PIL_WHITE_COIN = (237, 237, 237)
_PIL_QUEEN = (192, 57, 43)
_PIL_STRIKER = (241, 196, 15)
_PIL_SHADOW = (0, 0, 0, 60)
_PIL_SURFACE = (240, 217, 160)
_PIL_FRAME = (92, 51, 23)
_PIL_FRAME_INNER = (139, 94, 60)
_PIL_POCKET = (10, 10, 10)
_PIL_LINE = (191, 160, 106)
_PIL_CENTRE_DOT = (192, 57, 43)


def _render_static_board(board: BoardConfig, img_size: int = 560) -> Image.Image:
    """Render the static board background (frame, surface, pockets, lines) once.

    Returns an RGBA PIL image that can be composited with moving pieces.
    """
    img = Image.new("RGBA", (img_size, img_size), (26, 26, 46, 255))
    draw = ImageDraw.Draw(img)

    margin_px = int(img_size * 0.09)
    board_px = img_size - 2 * margin_px
    half_board = board_px // 2
    cx, cy = img_size // 2, img_size // 2

    def b2p(bx: float, by: float) -> Tuple[int, int]:
        """Board coords → pixel coords."""
        half = board.size / 2.0
        px = int(cx + (bx / half) * half_board)
        py = int(cy - (by / half) * half_board)  # y flipped
        return px, py

    def br(r: float) -> int:
        """Board radius → pixel radius."""
        half = board.size / 2.0
        return max(1, int(r / half * half_board))

    # Frame layers
    pad = 4
    draw.rounded_rectangle(
        [margin_px - pad * 3, margin_px - pad * 3,
         img_size - margin_px + pad * 3, img_size - margin_px + pad * 3],
        radius=12, fill=_PIL_FRAME)
    draw.rounded_rectangle(
        [margin_px - pad, margin_px - pad,
         img_size - margin_px + pad, img_size - margin_px + pad],
        radius=8, fill=_PIL_FRAME_INNER)

    # Surface
    draw.rectangle(
        [margin_px, margin_px, img_size - margin_px, img_size - margin_px],
        fill=_PIL_SURFACE)

    # Pockets
    half = board.size / 2.0
    pr = br(board.pocket_radius)
    for bx, by in [(-half, -half), (-half, half), (half, -half), (half, half)]:
        ppx, ppy = b2p(bx, by)
        draw.ellipse([ppx - pr, ppy - pr, ppx + pr, ppy + pr], fill=_PIL_POCKET)

    # Centre circles
    for r_val, width in [(0.19, 2), (0.05, 2)]:
        rr = br(r_val)
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                     outline=_PIL_LINE, width=width)

    # Centre dot
    dr = br(0.008)
    draw.ellipse([cx - dr, cy - dr, cx + dr, cy + dr], fill=_PIL_CENTRE_DOT)

    # Baselines
    bline_inset = 0.06
    for sign in (-1, 1):
        y = sign * (half - board.striker_offset)
        lx, ly = b2p(-half + bline_inset, y)
        rx, ry = b2p(half - bline_inset, y)
        draw.line([lx, ly, rx, ry], fill=_PIL_LINE, width=1)

    return img, b2p, br


def render_snapshot_frame(
    snapshot: dict,
    board: Optional[BoardConfig] = None,
    static_bg: Optional[Tuple] = None,
    img_size: int = 560,
    is_opponent: bool = False,
) -> Image.Image:
    """Render a single animation frame from snapshot data.

    Uses PIL drawing for speed (~5-10ms per frame vs ~150ms for matplotlib).

    Args:
        snapshot: Dict from CarromEnv._snapshot() with 'coins', 'striker', 'striker_pocketed'.
        board: Board configuration.
        static_bg: Tuple of (bg_image, b2p_func, br_func) from _render_static_board.
                   If None, creates a new one (pass it for reuse across frames).
        is_opponent: If True, tint striker blue.
    """
    board = board or BoardConfig()

    if static_bg is None:
        bg_img, b2p, br_radius = _render_static_board(board, img_size)
    else:
        bg_img, b2p, br_radius = static_bg

    # Copy the background
    frame = bg_img.copy()
    draw = ImageDraw.Draw(frame, "RGBA")

    coin_r = br_radius(board.coin_radius)
    striker_r = br_radius(board.striker_radius)

    # Draw coins
    for piece_id, color, bx, by, pocketed in snapshot["coins"]:
        if pocketed:
            continue
        px, py = b2p(bx, by)

        if "queen" in color:
            fill = _PIL_QUEEN
        elif "white" in color:
            fill = _PIL_WHITE_COIN
        else:
            fill = _PIL_BLACK_COIN

        # Shadow
        draw.ellipse([px - coin_r + 2, py - coin_r + 2, px + coin_r + 2, py + coin_r + 2],
                     fill=_PIL_SHADOW)
        # Main disc
        draw.ellipse([px - coin_r, py - coin_r, px + coin_r, py + coin_r], fill=fill)
        # Highlight dot
        hl_r = max(1, coin_r // 4)
        hx, hy = px - coin_r // 3, py - coin_r // 3
        hl_col = (255, 255, 255, 100) if "black" not in color else (100, 100, 100, 100)
        draw.ellipse([hx - hl_r, hy - hl_r, hx + hl_r, hy + hl_r], fill=hl_col)

    # Draw striker
    sx, sy = snapshot["striker"]
    if not snapshot["striker_pocketed"]:
        spx, spy = b2p(sx, sy)
        s_fill = (52, 152, 219) if is_opponent else _PIL_STRIKER
        # Shadow
        draw.ellipse([spx - striker_r + 2, spy - striker_r + 2,
                      spx + striker_r + 2, spy + striker_r + 2],
                     fill=_PIL_SHADOW)
        # Disc
        draw.ellipse([spx - striker_r, spy - striker_r,
                      spx + striker_r, spy + striker_r], fill=s_fill)
        # Cross-hair
        tiny = striker_r // 3
        edge_col = (212, 172, 13) if not is_opponent else (36, 113, 163)
        draw.line([spx - tiny, spy, spx + tiny, spy], fill=edge_col, width=1)
        draw.line([spx, spy - tiny, spx, spy + tiny], fill=edge_col, width=1)

    # Convert to RGB
    return frame.convert("RGB")
