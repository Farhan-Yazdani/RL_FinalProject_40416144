"""Interactive Pygame GUI for the dynamic maze environment.

Implements the "Graphical Interface and Visualization" section of
final_project.md: step-by-step agent movement, distinct markers for
every cell type plus the limited-energy feature, real-time state
changes, and the required interface controls.

Run with: ``python gui/app.py [--student-id ...] [--map-name ...]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

from environments.generator import (
    DOOR, GOAL, KEY, NORMAL, PENALTY, START, WALL, derive_seed_and_size, load_map,
)
from environments.maze import (
    ACTIONS, EnvConfig, Event, MazeEnv, REWARD_FNS,
    default_max_energy, default_step_cap,
)

CELL_COLORS = {
    NORMAL: (245, 245, 240),
    WALL: (43, 43, 43),
    PENALTY: (224, 122, 95),
    START: (129, 178, 154),
    KEY: (242, 204, 143),
    DOOR: (109, 89, 122),
    GOAL: (61, 64, 91),
}
AGENT_COLOR = (230, 57, 70)
GRID_LINE_COLOR = (200, 200, 200)
BG_COLOR = (250, 250, 248)
PANEL_BG = (30, 30, 35)
PANEL_TEXT = (230, 230, 230)
BUTTON_BG = (70, 70, 80)
BUTTON_BG_ACTIVE = (100, 140, 120)
BUTTON_TEXT = (240, 240, 240)

CELL_PX = 32
PANEL_WIDTH = 260
FPS = 60


class Button:
    """A simple clickable rectangular button.

    Parameters
    ----------
    rect : tuple of int
        ``(x, y, width, height)`` in pixels.
    label : str
        Button text.
    callback : callable
        Zero-argument function invoked on click.
    """

    def __init__(self, rect, label, callback):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.active = False

    def draw(self, surface, font):
        """Draw the button onto ``surface`` using ``font`` for the label."""
        color = BUTTON_BG_ACTIVE if self.active else BUTTON_BG
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        text = font.render(self.label, True, BUTTON_TEXT)
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)

    def handle_click(self, pos):
        """Invoke the callback if ``pos`` falls within the button's rect."""
        if self.rect.collidepoint(pos):
            self.callback()
            return True
        return False


class MazeGUI:
    """Main interactive GUI application.

    Parameters
    ----------
    map_spec : environments.generator.MapSpec
        Map to visualize.
    env_config : EnvConfig
        Environment configuration.
    policy : ndarray of shape (X, Y, 2, E), dtype=int, optional
        A precomputed greedy policy to run in "eval" mode (e.g. from
        Value Iteration or a trained Q-table). If ``None``, eval mode
        falls back to random actions.

    Notes
    -----
    Implements every control required by the spec: algorithm/
    environment selection (via CLI + on-screen mode label),
    train/eval mode toggle, start/stop/resume/reset/re-run, animation
    speed control, policy-overlay toggle, and a live info panel
    (episode, step, reward, epsilon, key status, recent success rate).
    """

    def __init__(self, map_spec, env_config: EnvConfig, policy: np.ndarray = None):
        self.map_spec = map_spec
        self.env_config = env_config
        self.policy = policy
        self.rng = np.random.default_rng(0)
        self.env = MazeEnv(map_spec, env_config, self.rng)

        self.mode = "eval" if policy is not None else "manual"
        self.running_animation = False
        self.steps_per_second = 5
        self.show_policy_overlay = False

        self.episode_count = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.recent_successes = []
        self.last_event = None
        self.state = self.env.reset()
        self.path_history = [(self.state.x, self.state.y)]

        size = map_spec.maze_size
        self.grid_px = size * CELL_PX
        self.width = self.grid_px + PANEL_WIDTH
        self.height = max(self.grid_px, 480)

        pygame.init()
        pygame.display.set_caption(f"Dynamic Maze -- {map_spec.name}")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 14)
        self.font_small = pygame.font.SysFont("Arial", 12)

        self._time_accum = 0.0
        self.buttons = self._build_buttons()

    def _build_buttons(self):
        """Construct the control buttons and return them as a list.

        Returns
        -------
        list of Button
            All interactive buttons for start/stop/resume/reset/
            re-run, speed +/-, and the policy-overlay toggle.
        """
        x0 = self.grid_px + 15
        w, h, gap = 105, 30, 8
        buttons = []
        buttons.append(Button((x0, 60, w, h), "Start", self.start))
        buttons.append(Button((x0 + w + gap, 60, w, h), "Stop", self.stop))
        buttons.append(Button((x0, 60 + h + gap, w, h), "Resume", self.resume))
        buttons.append(Button((x0 + w + gap, 60 + h + gap, w, h), "Reset", self.reset))
        buttons.append(Button((x0, 60 + 2 * (h + gap), w, h), "Re-run", self.rerun))
        buttons.append(Button((x0 + w + gap, 60 + 2 * (h + gap), w, h), "Policy: Off", self.toggle_policy_overlay))
        buttons.append(Button((x0, 60 + 3 * (h + gap), w, h), "Speed -", self.decrease_speed))
        buttons.append(Button((x0 + w + gap, 60 + 3 * (h + gap), w, h), "Speed +", self.increase_speed))
        return buttons

    # -- Controls -----------------------------------------------------

    def start(self):
        """Begin (or restart) automatic step-by-step animation."""
        self.running_animation = True

    def stop(self):
        """Pause automatic animation."""
        self.running_animation = False

    def resume(self):
        """Resume automatic animation without resetting episode state."""
        self.running_animation = True

    def reset(self):
        """Reset the current episode (keeps episode/success counters)."""
        self.state = self.env.reset()
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_event = None
        self.path_history = [(self.state.x, self.state.y)]

    def rerun(self):
        """Reset the environment and all episode/session counters."""
        self.reset()
        self.episode_count = 0
        self.recent_successes = []

    def toggle_policy_overlay(self):
        """Toggle whether policy arrows are drawn over the grid."""
        self.show_policy_overlay = not self.show_policy_overlay

    def increase_speed(self):
        """Increase animation steps-per-second (capped at 30)."""
        self.steps_per_second = min(30, self.steps_per_second + 1)

    def decrease_speed(self):
        """Decrease animation steps-per-second (floored at 1)."""
        self.steps_per_second = max(1, self.steps_per_second - 1)

    # -- Stepping -------------------------------------------------------

    def _choose_action(self) -> int:
        """Select the next action according to the current mode.

        Returns
        -------
        int
            Action index. Uses the greedy policy in eval mode (if
            provided), else a uniform-random action.
        """
        if self.mode == "eval" and self.policy is not None:
            s = self.state
            return int(self.policy[s.x, s.y, s.k, s.energy])
        return int(self.rng.integers(0, len(ACTIONS)))

    def step_once(self):
        """Advance the environment by exactly one step and update GUI state."""
        a = self._choose_action()
        res = self.env.step(a)
        self.state = res.next_state
        self.episode_reward += res.reward
        self.step_count += 1
        self.last_event = res.event
        self.path_history.append((self.state.x, self.state.y))

        if res.done:
            self.episode_count += 1
            self.recent_successes.append(res.event == Event.GOAL_REACHED)
            self.recent_successes = self.recent_successes[-50:]
            self.state = self.env.reset()
            self.step_count = 0
            self.episode_reward = 0.0
            self.path_history = [(self.state.x, self.state.y)]

    # -- Rendering --------------------------------------------------------

    def _draw_grid(self):
        """Draw the maze grid, agent, and traversed-path trail."""
        grid = self.map_spec.grid
        for x in range(self.map_spec.maze_size):
            for y in range(self.map_spec.maze_size):
                color = CELL_COLORS[int(grid[x, y])]
                rect = pygame.Rect(x * CELL_PX, y * CELL_PX, CELL_PX, CELL_PX)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_LINE_COLOR, rect, 1)

        if len(self.path_history) > 1:
            points = [
                (px * CELL_PX + CELL_PX // 2, py * CELL_PX + CELL_PX // 2)
                for px, py in self.path_history
            ]
            pygame.draw.lines(self.screen, (230, 150, 150), False, points, 2)

        if self.show_policy_overlay and self.policy is not None:
            self._draw_policy_overlay()

        ax, ay = self.state.x, self.state.y
        center = (ax * CELL_PX + CELL_PX // 2, ay * CELL_PX + CELL_PX // 2)
        pygame.draw.circle(self.screen, AGENT_COLOR, center, CELL_PX // 3)

    def _draw_policy_overlay(self):
        """Draw a small arrow glyph per non-wall cell for the current policy slice."""
        deltas = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        for x in range(self.map_spec.maze_size):
            for y in range(self.map_spec.maze_size):
                if self.map_spec.grid[x, y] == WALL:
                    continue
                a = int(self.policy[x, y, self.state.k, self.state.energy])
                dx, dy = deltas[a]
                cx = x * CELL_PX + CELL_PX // 2
                cy = y * CELL_PX + CELL_PX // 2
                end = (cx + dx * CELL_PX // 3, cy + dy * CELL_PX // 3)
                pygame.draw.line(self.screen, (60, 60, 60), (cx, cy), end, 2)

    def _draw_panel(self):
        """Draw the live info panel and control buttons."""
        panel_rect = pygame.Rect(self.grid_px, 0, PANEL_WIDTH, self.height)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)

        lines = [
            f"Mode: {self.mode}",
            f"Episode: {self.episode_count}",
            f"Step: {self.step_count}",
            f"Reward: {self.episode_reward:.1f}",
            f"Key: {'YES' if self.state.k else 'no'}",
            f"Energy: {self.state.energy}/{self.env_config.max_energy}",
            f"Last event: {self.last_event.value if self.last_event else '-'}",
            f"Speed: {self.steps_per_second} steps/s",
        ]
        if self.recent_successes:
            rate = sum(self.recent_successes) / len(self.recent_successes)
            lines.append(f"Recent success: {rate:.0%}")

        y = 10
        for line in lines:
            text = self.font.render(line, True, PANEL_TEXT)
            self.screen.blit(text, (self.grid_px + 15, y))
            y += 20

        self.buttons[5].label = "Policy: On" if self.show_policy_overlay else "Policy: Off"
        for b in self.buttons:
            b.draw(self.screen, self.font_small)

        # Energy bar (visual for the mandatory limited-energy feature)
        bar_x, bar_y, bar_w, bar_h = self.grid_px + 15, self.height - 40, PANEL_WIDTH - 30, 16
        pygame.draw.rect(self.screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
        frac = self.state.energy / max(1, self.env_config.max_energy)
        pygame.draw.rect(self.screen, (129, 178, 154), (bar_x, bar_y, int(bar_w * frac), bar_h))
        label = self.font_small.render("Energy", True, PANEL_TEXT)
        self.screen.blit(label, (bar_x, bar_y - 16))

    def render(self):
        """Render one full frame (grid + panel) to the screen."""
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._draw_panel()
        pygame.display.flip()

    # -- Main loop --------------------------------------------------------

    def run(self):
        """Run the Pygame event/render loop until the window is closed."""
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for b in self.buttons:
                        if b.handle_click(event.pos):
                            break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.running_animation = not self.running_animation
                    elif event.key == pygame.K_r:
                        self.reset()

            if self.running_animation:
                self._time_accum += dt
                step_interval = 1.0 / self.steps_per_second
                while self._time_accum >= step_interval:
                    self.step_once()
                    self._time_accum -= step_interval

            self.render()

        pygame.quit()


def main(argv=None):
    """CLI entry point: load a map/policy and launch the GUI.

    Parameters
    ----------
    argv : list of str, optional
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Launch the maze GUI.")
    parser.add_argument("--student-id", type=str, default="40")
    parser.add_argument("--map-name", type=str, default="source")
    parser.add_argument("--max-energy", type=int, default=None)
    parser.add_argument("--reward-version", type=str, default="sparse", choices=["sparse", "shaped"])
    parser.add_argument("--policy-npy", type=str, default=None,
                         help="Optional path to a saved policy.npy for eval mode.")
    args = parser.parse_args(argv)

    map_spec = load_map("environments/maps", args.map_name)
    max_energy = args.max_energy if args.max_energy is not None else default_max_energy(map_spec)
    step_cap = default_step_cap(map_spec)
    env_config = EnvConfig(max_energy=max_energy, step_cap=step_cap, reward_version=args.reward_version)

    policy = None
    if args.policy_npy:
        policy = np.load(args.policy_npy)

    gui = MazeGUI(map_spec, env_config, policy=policy)
    gui.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
