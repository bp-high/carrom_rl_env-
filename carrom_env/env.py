"""Carrom OpenEnv environment implementation."""

from __future__ import annotations

import math
import re
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pymunk

try:
    from openenv import Environment
except Exception:  # pragma: no cover - fallback for local dev without openenv
    class Environment:  # type: ignore
        pass

from carrom_env.constants import (
    AGENT_ID,
    BOARD_DECEL,
    COIN_MASS,
    DT,
    ELASTICITY,
    FRICTION,
    MAX_COINS,
    MAX_FORCE,
    MAX_SIM_STEPS,
    NUM_BLACK,
    NUM_QUEEN,
    NUM_WHITE,
    OPPONENT_ID,
    SETTLE_VELOCITY,
    STRIKER_MASS,
    BoardConfig,
)
from carrom_env.models import Action, CoinInfo, Observation, PieceState, State

# Pocket labels for human-readable observations
POCKET_LABELS = {
    (-1, -1): "bottom-left",
    (-1, 1): "top-left",
    (1, -1): "bottom-right",
    (1, 1): "top-right",
}

# Force keywords for text action parsing
FORCE_KEYWORDS = {
    "very soft": 0.15,
    "soft": 0.25,
    "gentle": 0.25,
    "light": 0.3,
    "medium": 0.5,
    "moderate": 0.5,
    "strong": 0.75,
    "hard": 0.8,
    "very hard": 0.9,
    "full": 1.0,
    "maximum": 1.0,
    "max": 1.0,
}

# Placement keywords
PLACEMENT_KEYWORDS = {
    "far left": -0.35,
    "left": -0.2,
    "slightly left": -0.1,
    "center": 0.0,
    "centre": 0.0,
    "middle": 0.0,
    "slightly right": 0.1,
    "right": 0.2,
    "far right": 0.35,
}


@dataclass
class Piece:
    piece_id: str
    kind: str
    color: str
    body: pymunk.Body
    shape: pymunk.Shape
    pocketed: bool = False


class CarromEnv(Environment):
    """OpenEnv-compatible Carrom environment implementing ICF rules.

    Physics
    -------
    Powered by Pymunk (2-D rigid-body simulation).  Board friction uses
    Coulomb kinetic friction applied via per-body ``velocity_func`` callbacks,
    giving constant deceleration (BOARD_DECEL units/s²) regardless of speed —
    matching the behaviour of pieces on a boric-acid-powdered carrom board.
    Contact friction (piece-piece, piece-wall) is handled by Pymunk's shape
    friction at collision points.

    ICF Rules implemented
    ---------------------
    * Agent plays **white**, opponent plays **black**.
    * Score 1 pt per own coin pocketed, 3 pts for the queen.
    * **Due rule**: pocketing the opponent's colour scores nothing; the coin
      is returned to the board centre and the turn passes to the opponent.
    * **Queen cover**: after pocketing the queen you must pocket one of your
      own coins on the *same or next* shot to "cover" it; failure returns the
      queen to the centre.
    * **Foul**: pocketing the striker returns one of your pocketed coins to
      the board and passes the turn.
    * **Turn continuation**: you keep shooting only after pocketing a coin of
      your own colour (or covering the queen).
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        seed: Optional[int] = None,
        board_config: Optional[BoardConfig] = None,
        max_turns: int = 200,
        enable_queen: bool = True,
    ) -> None:
        self.board = board_config or BoardConfig()
        self.max_turns = max_turns
        self.enable_queen = enable_queen
        self.rng_seed: Optional[int] = seed
        self.rng = random.Random(seed)
        self.space: Optional[pymunk.Space] = None
        self.static_body: Optional[pymunk.Body] = None
        self.coins: List[Piece] = []
        self.striker: Optional[Piece] = None
        self.current_player: str = AGENT_ID
        self.step_count = 0
        self.turn_count = 0
        self.agent_score = 0
        self.opponent_score = 0
        self.last_info: Dict[str, float] = {}
        # Queen cover rule: after pocketing the queen you must pocket
        # one of your own coins on the *next* shot to "cover" it.
        # If you fail, the queen returns to the centre.
        self._queen_cover_pending: Optional[str] = None  # player who must cover
        self._build_space()

    def reset(self, seed: Optional[int] = None, _options: Optional[dict] = None) -> Observation:
        if seed is not None:
            self.rng_seed = seed
            self.rng = random.Random(seed)
        self.step_count = 0
        self.turn_count = 0
        self.agent_score = 0
        self.opponent_score = 0
        self.current_player = AGENT_ID
        self._queen_cover_pending = None
        self._build_space()
        return self._observation()

    def step(self, action: Action | Dict) -> Tuple[Observation, float, bool, bool, Dict]:
        if not isinstance(action, Action):
            action = Action(**action)

        # Parse text action into numeric parameters if needed
        if action.action_type == "text" and action.text:
            action = self._parse_text_action(action.text)

        self.step_count += 1
        reward = 0.0

        # Agent turn
        reward += self._play_turn(action, AGENT_ID)
        terminated, truncated = self._check_done()

        # Opponent turn (auto-play)
        if not terminated and not truncated and self.current_player == OPPONENT_ID:
            opp_action = self._opponent_action()
            reward += self._play_turn(opp_action, OPPONENT_ID)
            terminated, truncated = self._check_done()

        obs = self._observation()
        info = dict(self.last_info)
        info.update(
            {
                "turn_count": self.turn_count,
                "agent_score": self.agent_score,
                "opponent_score": self.opponent_score,
                "current_player": self.current_player,
            }
        )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Split-turn API (for interactive / visual use)
    # ------------------------------------------------------------------

    def step_agent(self, action: Action | Dict) -> Tuple[Observation, float, bool, bool, Dict]:
        """Play only the agent's turn — does NOT auto-play opponent."""
        if not isinstance(action, Action):
            action = Action(**action)
        if action.action_type == "text" and action.text:
            action = self._parse_text_action(action.text)

        self.step_count += 1
        reward = self._play_turn(action, AGENT_ID)
        terminated, truncated = self._check_done()

        obs = self._observation()
        info = dict(self.last_info)
        info.update({
            "turn_count": self.turn_count,
            "agent_score": self.agent_score,
            "opponent_score": self.opponent_score,
            "current_player": self.current_player,
        })
        return obs, reward, terminated, truncated, info

    def step_opponent(self) -> Tuple[Observation, float, bool, bool, Dict]:
        """Play the opponent's auto-reply turn. No-op if it's not the opponent's turn.

        The returned info dict includes ``opp_action`` with the Action the
        opponent chose (placement_x, angle, force) so callers can visualise it.
        """
        reward = 0.0
        opp_action = None
        if self.current_player == OPPONENT_ID:
            opp_action = self._opponent_action()
            reward = self._play_turn(opp_action, OPPONENT_ID)

        terminated, truncated = self._check_done()
        obs = self._observation()
        info = dict(self.last_info)
        info.update({
            "turn_count": self.turn_count,
            "agent_score": self.agent_score,
            "opponent_score": self.opponent_score,
            "current_player": self.current_player,
        })
        if opp_action is not None:
            info["opp_action"] = opp_action
        return obs, reward, terminated, truncated, info

    def step_agent_animated(self, action: Action | Dict) -> Tuple[Observation, float, bool, bool, Dict]:
        """Like step_agent but captures animation snapshots in info['snapshots']."""
        if not isinstance(action, Action):
            action = Action(**action)
        if action.action_type == "text" and action.text:
            action = self._parse_text_action(action.text)

        self.step_count += 1
        reward = self._play_turn(action, AGENT_ID, animated=True)
        terminated, truncated = self._check_done()

        obs = self._observation()
        info = dict(self.last_info)
        info.update({
            "turn_count": self.turn_count,
            "agent_score": self.agent_score,
            "opponent_score": self.opponent_score,
            "current_player": self.current_player,
        })
        return obs, reward, terminated, truncated, info

    def step_opponent_animated(self) -> Tuple[Observation, float, bool, bool, Dict]:
        """Like step_opponent but captures animation snapshots in info['snapshots']."""
        reward = 0.0
        opp_action = None
        if self.current_player == OPPONENT_ID:
            opp_action = self._opponent_action()
            reward = self._play_turn(opp_action, OPPONENT_ID, animated=True)

        terminated, truncated = self._check_done()
        obs = self._observation()
        info = dict(self.last_info)
        info.update({
            "turn_count": self.turn_count,
            "agent_score": self.agent_score,
            "opponent_score": self.opponent_score,
            "current_player": self.current_player,
        })
        if opp_action is not None:
            info["opp_action"] = opp_action
        return obs, reward, terminated, truncated, info

    @property
    def needs_opponent_turn(self) -> bool:
        """True when it's the opponent's turn to play."""
        remaining = sum(1 for coin in self.coins if not coin.pocketed)
        if remaining == 0 or self.turn_count >= self.max_turns:
            return False
        return self.current_player == OPPONENT_ID

    def state(self) -> State:
        pieces = []
        for piece in self._all_pieces():
            pos = list(piece.body.position) if not piece.pocketed else [-1.0, -1.0]
            vel = list(piece.body.velocity) if not piece.pocketed else [0.0, 0.0]
            pieces.append(
                PieceState(
                    piece_id=piece.piece_id,
                    kind=piece.kind,
                    color=piece.color,
                    position=pos,
                    velocity=vel,
                    pocketed=piece.pocketed,
                )
            )
        return State(
            rng_seed=self.rng_seed,
            step_count=self.step_count,
            current_player=self.current_player,
            board_size=self.board.size,
            pieces=pieces,
            agent_score=self.agent_score,
            opponent_score=self.opponent_score,
        )

    def render(self, mode: str = "rgb_array"):
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {mode}")
        try:
            import pygame
            from pymunk import pygame_util
        except ImportError as exc:  # pragma: no cover - optional rendering
            raise ImportError("pygame is required for rendering") from exc

        size_px = 512
        surface = pygame.Surface((size_px, size_px))
        surface.fill((240, 222, 181))

        draw_options = pygame_util.DrawOptions(surface)
        self.space.debug_draw(draw_options)

        if mode == "rgb_array":
            arr = pygame.surfarray.array3d(surface)
            return np.transpose(arr, (1, 0, 2))

        if mode == "human":
            pygame.display.set_mode((size_px, size_px))
            screen = pygame.display.get_surface()
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            return None

        return None

    @staticmethod
    def _make_friction_func(decel: float):
        """Return a Pymunk velocity callback that applies Coulomb board friction.

        Unlike viscous damping (velocity-proportional drag), Coulomb friction
        produces constant deceleration regardless of speed — matching the physics
        of a piece sliding on a boric-acid-powdered carrom board surface.
        """
        def _velocity_func(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, damping, dt)
            speed = body.velocity.length
            if speed > 1e-6:
                new_speed = max(0.0, speed - decel * dt)
                body.velocity = body.velocity * (new_speed / speed)
        return _velocity_func

    def _build_space(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        # No global viscous damping; Coulomb friction is applied per-body via
        # velocity_func so deceleration is constant (not speed-proportional).
        self.space.damping = 1.0
        self.static_body = self.space.static_body
        self.coins = []
        self.striker = None

        self._add_walls()
        self._add_coins()
        self._reset_striker(AGENT_ID)

    def _add_walls(self) -> None:
        half = self.board.size / 2.0
        gap = self.board.pocket_radius
        thickness = self.board.wall_thickness

        segments = [
            ((-half + gap, -half), (half - gap, -half)),
            ((-half + gap, half), (half - gap, half)),
            ((-half, -half + gap), (-half, half - gap)),
            ((half, -half + gap), (half, half - gap)),
        ]

        for a, b in segments:
            seg = pymunk.Segment(self.static_body, a, b, thickness)
            seg.friction = FRICTION
            seg.elasticity = ELASTICITY
            self.space.add(seg)

    def _add_coins(self) -> None:
        center = (0.0, 0.0)
        ring_inner = self.board.coin_radius * 2.3
        ring_outer = self.board.coin_radius * 4.4

        pieces: List[Tuple[str, str, Tuple[float, float]]] = []

        # Queen
        if self.enable_queen:
            pieces.append(("queen_0", "queen", center))

        # Inner ring (6)
        for idx in range(6):
            angle = idx * (math.pi / 3.0)
            x = ring_inner * math.cos(angle)
            y = ring_inner * math.sin(angle)
            color = "black" if idx % 2 == 0 else "white"
            pieces.append((f"{color}_inner_{idx}", color, (x, y)))

        # Outer ring (12)
        for idx in range(12):
            angle = idx * (math.pi / 6.0)
            x = ring_outer * math.cos(angle)
            y = ring_outer * math.sin(angle)
            color = "white" if idx % 2 == 0 else "black"
            pieces.append((f"{color}_outer_{idx}", color, (x, y)))

        # Trim to desired counts
        blacks = [p for p in pieces if p[1] == "black"]
        whites = [p for p in pieces if p[1] == "white"]
        queens = [p for p in pieces if p[1] == "queen"]
        selected = queens + blacks[:NUM_BLACK] + whites[:NUM_WHITE]

        friction_func = self._make_friction_func(BOARD_DECEL)
        for piece_id, color, pos in selected:
            body = pymunk.Body(COIN_MASS, pymunk.moment_for_circle(COIN_MASS, 0, self.board.coin_radius))
            body.position = pos
            body.velocity_func = friction_func
            shape = pymunk.Circle(body, self.board.coin_radius)
            shape.friction = FRICTION
            shape.elasticity = ELASTICITY
            self.space.add(body, shape)
            self.coins.append(Piece(piece_id=piece_id, kind="coin", color=color, body=body, shape=shape))

        if not self.enable_queen:
            body = pymunk.Body(COIN_MASS, pymunk.moment_for_circle(COIN_MASS, 0, self.board.coin_radius))
            body.position = (-1.0, -1.0)
            shape = pymunk.Circle(body, self.board.coin_radius)
            dummy = Piece(piece_id="queen_0", kind="coin", color="queen", body=body, shape=shape, pocketed=True)
            self.coins.append(dummy)

    def _reset_striker(self, player: str) -> None:
        if self.striker and self.striker.body in self.space.bodies:
            self.space.remove(self.striker.body, self.striker.shape)
        half = self.board.size / 2.0
        y = -half + self.board.striker_offset if player == AGENT_ID else half - self.board.striker_offset
        body = pymunk.Body(STRIKER_MASS, pymunk.moment_for_circle(STRIKER_MASS, 0, self.board.striker_radius))
        body.position = (0.0, y)
        body.velocity_func = self._make_friction_func(BOARD_DECEL)
        shape = pymunk.Circle(body, self.board.striker_radius)
        shape.friction = FRICTION
        shape.elasticity = ELASTICITY
        self.space.add(body, shape)
        self.striker = Piece(piece_id="striker", kind="striker", color="striker", body=body, shape=shape)

    def _play_turn(self, action: Action, player: str, animated: bool = False) -> float:
        self.turn_count += 1
        self.current_player = player
        self._reset_striker(player)

        placement_limit = (self.board.size / 2.0) - self.board.pocket_radius - self.board.striker_radius
        placement_x = float(np.clip(action.placement_x, -placement_limit, placement_limit))
        if player == AGENT_ID:
            baseline_y = -self.board.size / 2.0 + self.board.striker_offset
        else:
            baseline_y = self.board.size / 2.0 - self.board.striker_offset

        # ── Baseline obstruction: nudge striker away from coins on the line ──
        placement_x = self._find_valid_placement(placement_x, baseline_y)

        self.striker.body.position = (placement_x, baseline_y)
        self.striker.body.velocity = (0.0, 0.0)

        angle = float(action.angle)
        force = float(np.clip(action.force, 0.0, 1.0))
        direction = self._angle_to_dir(angle, player)
        impulse = (direction[0] * force * MAX_FORCE, direction[1] * force * MAX_FORCE)
        self.striker.body.apply_impulse_at_local_point(impulse)

        pre_pocket = self._pocketed_set()
        snapshots: List[Dict] = []
        if animated:
            sim_steps, energy, snapshots = self._simulate_shot_animated()
        else:
            sim_steps, energy = self._simulate_shot()
        post_pocket = self._pocketed_set()

        new_pocketed = post_pocket - pre_pocket
        foul = "striker" in new_pocketed
        coin_potted = [pid for pid in new_pocketed if pid != "striker"]

        # ── Foul penalty: return one pocketed coin to centre ──
        if foul:
            self._return_coin_to_centre(player)

        # ── ICF Due rule: opponent's coins pocketed return to centre, no score ──
        due_coins = [
            cid for cid in coin_potted
            if "queen" not in cid and not self._coin_belongs_to(cid, player)
        ]
        for due_id in due_coins:
            self._return_due_coin(due_id)
        # Only own coins + queen proceed to scoring and turn-continuation logic
        coin_potted = [cid for cid in coin_potted if cid not in due_coins]

        # ── Queen cover rule ──
        queen_potted = any("queen" in cid for cid in coin_potted)
        own_coin_potted = any(
            self._coin_belongs_to(cid, player) for cid in coin_potted
            if "queen" not in cid
        )

        # Check if queen cover was pending from previous turn
        if self._queen_cover_pending == player:
            if own_coin_potted and not foul:
                # Queen successfully covered
                self._queen_cover_pending = None
            else:
                # Failed to cover — queen goes back to centre
                self._uncover_queen()
                self._queen_cover_pending = None

        # If queen was potted this turn, start the cover requirement
        if queen_potted and not foul:
            if own_coin_potted:
                # Queen and own coin in same shot = auto-covered
                pass
            else:
                self._queen_cover_pending = player

        reward = self._score_shot(player, coin_potted, foul, due_count=len(due_coins))
        self._update_turn(player, coin_potted, foul)

        self.last_info = {
            "sim_steps": float(sim_steps),
            "energy": float(energy),
            "coin_potted": float(len(coin_potted)),
            "due_coins": float(len(due_coins)),
            "foul": float(foul),
            "queen_cover_pending": self._queen_cover_pending is not None,
            "placement_x_actual": float(placement_x),
        }
        if animated:
            self.last_info["snapshots"] = snapshots
        return reward

    def _simulate_shot(self) -> Tuple[int, float]:
        sim_steps = 0
        energy_total = 0.0
        while sim_steps < MAX_SIM_STEPS:
            self.space.step(DT)
            sim_steps += 1
            energy_total += self._total_energy()
            self._check_pockets()
            if self._settled():
                break
        return sim_steps, energy_total

    def _snapshot(self) -> Dict:
        """Capture lightweight position snapshot for animation."""
        coins = []
        for coin in self.coins:
            if coin.pocketed:
                coins.append((coin.piece_id, coin.color, -1.0, -1.0, True))
            else:
                x, y = coin.body.position
                coins.append((coin.piece_id, coin.color, float(x), float(y), False))
        striker_x, striker_y = -1.0, -1.0
        striker_pocketed = True
        if self.striker and not self.striker.pocketed:
            striker_x, striker_y = self.striker.body.position
            striker_pocketed = False
        return {
            "coins": coins,
            "striker": (float(striker_x), float(striker_y)),
            "striker_pocketed": striker_pocketed,
        }

    def _simulate_shot_animated(self, target_frames: int = 15) -> Tuple[int, float, List[Dict]]:
        """Like _simulate_shot but captures animation snapshots at intervals."""
        snapshots: List[Dict] = []
        sim_steps = 0
        energy_total = 0.0

        # Capture initial frame (just after impulse)
        snapshots.append(self._snapshot())

        while sim_steps < MAX_SIM_STEPS:
            self.space.step(DT)
            sim_steps += 1
            energy_total += self._total_energy()
            self._check_pockets()
            snapshots.append(self._snapshot())
            if self._settled():
                break

        # Downsample to target_frames (always keep first and last)
        if len(snapshots) > target_frames:
            indices = [int(i * (len(snapshots) - 1) / (target_frames - 1)) for i in range(target_frames)]
            indices[-1] = len(snapshots) - 1
            snapshots = [snapshots[i] for i in sorted(set(indices))]

        return sim_steps, energy_total, snapshots

    def _score_shot(self, player: str, coin_potted: List[str], foul: bool,
                    due_count: int = 0) -> float:
        """Compute reward with shaped incentives (ICF rules).

        Reward components (agent turn):
          - Base: -0.01 per turn to encourage efficiency
          - Own coin potted: +1.0 per coin, +3.0 for queen
          - Foul (striker pocketed): -1.5 total (-1 score + -0.5 extra)
          - Due coin (pocketed opponent's color): -0.3 per coin (coin returns to board)
          - Win bonus: +5.0 when all coins cleared and agent leads
          - Loss penalty: -2.0 when all coins cleared and opponent leads
        """
        reward = -0.01 if player == AGENT_ID else 0.0
        score_delta = 0
        for coin_id in coin_potted:
            if "queen" in coin_id:
                score_delta += 3
            else:
                score_delta += 1

        if foul:
            score_delta -= 1
            if player == AGENT_ID:
                reward -= 0.5

        if player == AGENT_ID:
            self.agent_score += score_delta
            reward += score_delta
            # Penalise pocketing opponent's coins (they return to board, no score)
            reward -= 0.3 * due_count

            remaining = sum(1 for c in self.coins if not c.pocketed)
            if remaining == 0 and self.agent_score > self.opponent_score:
                reward += 5.0
            elif remaining == 0 and self.agent_score <= self.opponent_score:
                reward -= 2.0
        else:
            self.opponent_score += score_delta
            reward -= score_delta * 0.5

        return reward

    def _update_turn(self, player: str, coin_potted: List[str], foul: bool) -> None:
        if foul:
            self.current_player = OPPONENT_ID if player == AGENT_ID else AGENT_ID
            return
        if len(coin_potted) > 0:
            self.current_player = player
            return
        self.current_player = OPPONENT_ID if player == AGENT_ID else AGENT_ID

    # ------------------------------------------------------------------
    # Carrom rule helpers
    # ------------------------------------------------------------------

    def _find_valid_placement(self, desired_x: float, baseline_y: float) -> float:
        """Nudge striker placement to avoid overlapping coins on the baseline.

        Scans coins near the baseline. If the desired position overlaps any
        coin, tries small offsets to the right then left until a clear spot
        is found.  Returns the (possibly adjusted) x coordinate.
        """
        sr = self.board.striker_radius
        cr = self.board.coin_radius
        min_gap = sr + cr + 0.002  # tiny extra margin
        placement_limit = (self.board.size / 2.0) - self.board.pocket_radius - sr

        def _blocked(x: float) -> bool:
            for coin in self.coins:
                if coin.pocketed:
                    continue
                cx, cy = coin.body.position
                if abs(cy - baseline_y) < min_gap:
                    if abs(cx - x) < min_gap:
                        return True
            return False

        if not _blocked(desired_x):
            return desired_x

        # Try nudging in small steps
        step = 0.02
        for offset in range(1, 40):
            for sign in (1, -1):
                candidate = desired_x + sign * offset * step
                candidate = float(np.clip(candidate, -placement_limit, placement_limit))
                if not _blocked(candidate):
                    return candidate

        # Fallback — just return desired (shouldn't normally happen)
        return desired_x

    def _coin_belongs_to(self, coin_id: str, player: str) -> bool:
        """Check if a coin belongs to the given player.

        Convention: agent plays white, opponent plays black.
        Not strictly enforced for pocketing (either colour can be pocketed),
        but used for queen cover rule.
        """
        if player == AGENT_ID:
            return "white" in coin_id
        return "black" in coin_id

    def _return_coin_to_centre(self, fouling_player: str) -> None:
        """Return one previously pocketed coin belonging to the fouling player
        to the centre of the board (foul penalty).
        """
        for coin in self.coins:
            if coin.pocketed and "queen" not in coin.piece_id:
                if self._coin_belongs_to(coin.piece_id, fouling_player):
                    coin.pocketed = False
                    body = pymunk.Body(
                        COIN_MASS,
                        pymunk.moment_for_circle(COIN_MASS, 0, self.board.coin_radius),
                    )
                    body.position = (0.0, 0.0)
                    body.velocity = (0.0, 0.0)
                    body.velocity_func = self._make_friction_func(BOARD_DECEL)
                    shape = pymunk.Circle(body, self.board.coin_radius)
                    shape.friction = FRICTION
                    shape.elasticity = ELASTICITY
                    self.space.add(body, shape)
                    coin.body = body
                    coin.shape = shape
                    if fouling_player == AGENT_ID:
                        self.agent_score = max(0, self.agent_score - 1)
                    else:
                        self.opponent_score = max(0, self.opponent_score - 1)
                    return

    def _uncover_queen(self) -> None:
        """Return the queen to the centre of the board (failed to cover)."""
        queen = next((c for c in self.coins if c.color == "queen"), None)
        if queen is None or not queen.pocketed:
            return
        queen.pocketed = False
        body = pymunk.Body(
            COIN_MASS,
            pymunk.moment_for_circle(COIN_MASS, 0, self.board.coin_radius),
        )
        body.position = (0.0, 0.0)
        body.velocity = (0.0, 0.0)
        body.velocity_func = self._make_friction_func(BOARD_DECEL)
        shape = pymunk.Circle(body, self.board.coin_radius)
        shape.friction = FRICTION
        shape.elasticity = ELASTICITY
        self.space.add(body, shape)
        queen.body = body
        queen.shape = shape
        if self.agent_score >= 3:
            self.agent_score -= 3
        elif self.opponent_score >= 3:
            self.opponent_score -= 3

    def _return_due_coin(self, coin_id: str) -> None:
        """Return an opponent's coin to the centre under the ICF due rule.

        Called when a player pockets a coin of the wrong colour.  The coin is
        placed back at the centre with no score awarded; the pocketing player's
        turn ends (handled by _update_turn receiving an empty coin_potted list).
        """
        coin = next((c for c in self.coins if c.piece_id == coin_id), None)
        if coin is None or not coin.pocketed:
            return
        coin.pocketed = False
        body = pymunk.Body(
            COIN_MASS,
            pymunk.moment_for_circle(COIN_MASS, 0, self.board.coin_radius),
        )
        body.position = (0.0, 0.0)
        body.velocity = (0.0, 0.0)
        body.velocity_func = self._make_friction_func(BOARD_DECEL)
        shape = pymunk.Circle(body, self.board.coin_radius)
        shape.friction = FRICTION
        shape.elasticity = ELASTICITY
        self.space.add(body, shape)
        coin.body = body
        coin.shape = shape

    def _get_blocked_zones(self, player: str) -> List[Tuple[float, float]]:
        """Return list of (center_x, radius) zones on the baseline that
        are blocked by coins — used for UI visualisation.
        """
        half = self.board.size / 2.0
        if player == AGENT_ID:
            baseline_y = -half + self.board.striker_offset
        else:
            baseline_y = half - self.board.striker_offset
        sr = self.board.striker_radius
        cr = self.board.coin_radius
        min_gap = sr + cr + 0.002
        blocked = []
        for coin in self.coins:
            if coin.pocketed:
                continue
            cx, cy = coin.body.position
            if abs(cy - baseline_y) < min_gap:
                blocked.append((float(cx), min_gap))
        return blocked

    def _check_done(self) -> Tuple[bool, bool]:
        remaining = sum(1 for coin in self.coins if not coin.pocketed)
        if remaining == 0:
            return True, False
        if self.turn_count >= self.max_turns:
            return False, True
        return False, False

    def _check_pockets(self) -> None:
        pocket_centers = self._pocket_centers()
        capture_r2 = self.board.pocket_capture_radius ** 2
        for piece in self._all_pieces():
            if piece.pocketed:
                continue
            px, py = piece.body.position
            for cx, cy in pocket_centers:
                if (px - cx) ** 2 + (py - cy) ** 2 <= capture_r2:
                    piece.pocketed = True
                    if piece.body in self.space.bodies:
                        self.space.remove(piece.body, piece.shape)
                    break

    def _pocketed_set(self) -> set:
        ids = {p.piece_id for p in self.coins if p.pocketed}
        if self.striker and self.striker.pocketed:
            ids.add("striker")
        return ids

    def _settled(self) -> bool:
        for piece in self._all_pieces():
            if piece.pocketed:
                continue
            vx, vy = piece.body.velocity
            if vx * vx + vy * vy > SETTLE_VELOCITY ** 2:
                return False
        return True

    def _total_energy(self) -> float:
        energy = 0.0
        for piece in self._all_pieces():
            if piece.pocketed:
                continue
            vx, vy = piece.body.velocity
            energy += 0.5 * piece.body.mass * (vx * vx + vy * vy)
        return energy

    def _angle_to_dir(self, angle: float, player: str) -> Tuple[float, float]:
        # Angle definition: 0 points to +y (agent direction)
        dx = math.sin(angle)
        dy = math.cos(angle)
        if player == OPPONENT_ID:
            dy = -dy
        return (dx, dy)

    def _opponent_action(self) -> Action:
        target = self._nearest_coin_to_opponent()
        half = self.board.size / 2.0
        baseline_y = half - self.board.striker_offset
        placement_x = 0.0
        if target:
            tx, ty = target
            dx = tx - placement_x
            dy = ty - baseline_y
            angle = math.atan2(dx, dy)
        else:
            angle = 0.0
        return Action(placement_x=placement_x, angle=angle, force=0.65, spin=0.0, shot_id=None)

    def _nearest_coin_to_opponent(self) -> Optional[Tuple[float, float]]:
        candidates = [coin for coin in self.coins if not coin.pocketed]
        if not candidates:
            return None
        half = self.board.size / 2.0
        baseline_y = half - self.board.striker_offset
        best = None
        best_dist = float("inf")
        for coin in candidates:
            cx, cy = coin.body.position
            dx = cx
            dy = cy - baseline_y
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best = (cx, cy)
        return best

    # ------------------------------------------------------------------
    # Text action parsing for LLM agents
    # ------------------------------------------------------------------

    def _parse_text_action(self, text: str) -> Action:
        """Parse a natural-language shot description into an Action.

        Supports patterns like:
          - "aim at coin black_inner_0 with strong force from the left"
          - "shoot toward queen_0 from center with medium force"
          - "placement_x=0.1 angle=0.3 force=0.8"
        """
        text_lower = text.lower().strip()

        # Try structured key=value fallback first
        kv = dict(re.findall(r"(placement_x|angle|force|spin)\s*=\s*([-\d.]+)", text_lower))
        if kv:
            return Action(
                action_type="numeric",
                placement_x=float(kv.get("placement_x", 0.0)),
                angle=float(kv.get("angle", 0.0)),
                force=float(kv.get("force", 0.5)),
                spin=float(kv.get("spin", 0.0)),
            )

        # Parse force
        force = 0.5
        for keyword, val in sorted(FORCE_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if keyword in text_lower:
                force = val
                break

        # Parse placement
        placement_x = 0.0
        for keyword, val in sorted(PLACEMENT_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if keyword in text_lower:
                placement_x = val
                break

        # Parse target coin — look for a coin_id reference
        angle = 0.0
        coin_match = re.search(
            r"(queen_\d+|black_(?:inner|outer)_\d+|white_(?:inner|outer)_\d+)",
            text_lower,
        )
        if coin_match:
            target_id = coin_match.group(1)
            target_coin = next((c for c in self.coins if c.piece_id == target_id and not c.pocketed), None)
            if target_coin:
                half = self.board.size / 2.0
                baseline_y = -half + self.board.striker_offset
                tx, ty = target_coin.body.position
                dx = tx - placement_x
                dy = ty - baseline_y
                angle = math.atan2(dx, dy)

        # Parse pocket target — "toward top-left pocket" etc.
        pocket_match = re.search(r"(top|bottom)[\s-]*(left|right)\s*pocket", text_lower)
        if pocket_match and not coin_match:
            py_sign = 1.0 if pocket_match.group(1) == "top" else -1.0
            px_sign = -1.0 if pocket_match.group(2) == "left" else 1.0
            half = self.board.size / 2.0
            baseline_y = -half + self.board.striker_offset
            dx = px_sign * half - placement_x
            dy = py_sign * half - baseline_y
            angle = math.atan2(dx, dy)

        return Action(
            action_type="numeric",
            placement_x=placement_x,
            angle=angle,
            force=force,
        )

    def _pocket_centers(self) -> List[Tuple[float, float]]:
        half = self.board.size / 2.0
        return [
            (-half, -half),
            (-half, half),
            (half, -half),
            (half, half),
        ]

    def _all_pieces(self) -> List[Piece]:
        pieces = []
        if self.striker:
            pieces.append(self.striker)
        pieces.extend(self.coins)
        return pieces

    def _observation(self) -> Observation:
        pieces = self._all_pieces()
        positions: List[List[float]] = []
        velocities: List[List[float]] = []
        pocketed: List[bool] = []
        for piece in pieces:
            if piece.pocketed:
                positions.append([-1.0, -1.0])
                velocities.append([0.0, 0.0])
                pocketed.append(True)
            else:
                positions.append([float(piece.body.position.x), float(piece.body.position.y)])
                velocities.append([float(piece.body.velocity.x), float(piece.body.velocity.y)])
                pocketed.append(False)

        remaining = sum(1 for coin in self.coins if not coin.pocketed)
        pocket_centers = self._pocket_centers()

        # Build per-coin info for LLM readability
        coin_infos: List[CoinInfo] = []
        for coin in self.coins:
            cx, cy = (coin.body.position if not coin.pocketed else (-1.0, -1.0))
            best_label = "none"
            best_dist = float("inf")
            if not coin.pocketed:
                for (px, py) in pocket_centers:
                    d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    if d < best_dist:
                        best_dist = d
                        sx = 1 if px > 0 else -1
                        sy = 1 if py > 0 else -1
                        best_label = POCKET_LABELS.get((sx, sy), "unknown")
            coin_infos.append(CoinInfo(
                coin_id=coin.piece_id,
                color=coin.color,
                x=round(float(cx), 4),
                y=round(float(cy), 4),
                pocketed=coin.pocketed,
                nearest_pocket=best_label,
                pocket_distance=round(best_dist, 4),
            ))

        # Build rich text summary for LLM agents
        lines = [
            f"=== Carrom Board State (Turn {self.turn_count}/{self.max_turns}) ===",
            f"Score: You {self.agent_score} - Opponent {self.opponent_score}",
            f"Remaining coins on board: {remaining}",
            "Rules: You=WHITE coins (+1 pt each). Queen=+3 pts (must cover next shot).",
            "Due rule: pocketing a BLACK coin returns it to centre — no score, turn ends.",
            "",
            "Active coins:",
        ]
        for ci in coin_infos:
            if not ci.pocketed:
                lines.append(
                    f"  {ci.coin_id} ({ci.color}) at ({ci.x:.3f}, {ci.y:.3f})"
                    f" | nearest pocket: {ci.nearest_pocket} ({ci.pocket_distance:.3f} away)"
                )
        pocketed_coins = [ci for ci in coin_infos if ci.pocketed]
        if pocketed_coins:
            lines.append("")
            lines.append(f"Pocketed coins: {', '.join(ci.coin_id for ci in pocketed_coins)}")
        lines.extend([
            "",
            "Your striker is on the bottom baseline.",
            "Actions: set placement_x ([-0.4, 0.4], 0=center), angle (radians, 0=straight ahead), force ([0,1]).",
            'Or use action_type="text" with a description like "aim at queen_0 with strong force from center".',
        ])
        text_summary = "\n".join(lines)

        return Observation(
            positions=positions,
            velocities=velocities,
            pocketed=pocketed,
            agent_score=self.agent_score,
            opponent_score=self.opponent_score,
            current_player=self.current_player,
            remaining_coins=remaining,
            turn_number=self.turn_count,
            max_turns=self.max_turns,
            coins=coin_infos,
            text_summary=text_summary,
        )
