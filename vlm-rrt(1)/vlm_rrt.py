import base64
import io
import math
import os
import random
import re
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables from .env file automatically
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))
load_dotenv()


# =====================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# =====================================================================
@dataclass
class VLMRRTConfig:
    map_size: Tuple[float, float] = (500.0, 500.0)  # 500m x 500m 2D space
    max_iterations: int = 500                       # N = 500
    dt: float = 1.0                                 # \Delta T = 1 s
    v_max: float = 15.0                             # v_max = 15 m/s
    goal_threshold: float = 1.0                     # \epsilon = 1 m
    gamma: float = 0.85                             # VLM sampling probability \gamma
    sector_radius: float = 30.0                     # r = 30 m
    sector_angle_deg: float = 45.0                  # \theta = 45 degrees
    prompt_mode: str = "cot"                        # "zero_shot", "few_shot", or "cot"

    # Model & API configuration (uses NVIDIA_API_KEY from .env by default)
    model_name: str = "meta/llama-3.2-90b-vision-instruct"  # NVIDIA NIM model ID
    api_key: Optional[str] = None                           # NVIDIA_API_KEY from .env (auto-loaded)
    api_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"  # NVIDIA NIM endpoint
    api_timeout: float = 45.0                               # Request timeout in seconds (per attempt)
    disable_thinking: bool = False                          # True for "thinking" models (e.g. nemotron-3.5-super-vl); sends chat_template_kwargs.enable_thinking=False
    max_consecutive_failures: int = 5                      # Stop the run if this many VLM calls fail in a row (0 = never stop)
    max_rpm: float = 35.0                                  # Rate limit (NVIDIA free tier: 40 rpm; 35 leaves margin). 0 = no limit
    max_retries: int = 5                                   # Retries on HTTP 429 / 5xx

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("NVIDIA_API_KEY")

    @property
    def step_size(self) -> float:
        return self.dt * self.v_max

    @property
    def sector_angle_rad(self) -> float:
        return math.radians(self.sector_angle_deg)


# =====================================================================
# 2. ENVIRONMENT & COLLISION CHECKER
# =====================================================================
@dataclass
class Node:
    x: float
    y: float
    parent: Optional[int] = None


class Environment2D:
    def __init__(
        self,
        width: float,
        height: float,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float, float, float]],
    ):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles

    def is_point_in_obstacle(self, x: float, y: float) -> bool:
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return True
        for ox, oy, ow, oh in self.obstacles:
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                return True
        return False

    def path_free(
        self, p1: Tuple[float, float], p2: Tuple[float, float], num_checks: int = 25
    ) -> bool:
        x1, y1 = p1
        x2, y2 = p2
        for i in range(num_checks + 1):
            t = i / num_checks
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if self.is_point_in_obstacle(x, y):
                return False
        return True


# =====================================================================
# 3. SNAPSHOT RENDERER & PROMPT MANAGER
# =====================================================================
class SnapshotRenderer:
    @staticmethod
    def render_base64(
        env: Environment2D, nodes: List[Node], leaf_node: Node, dpi: int = 100
    ) -> str:
        # 5x5 inches @ 100 dpi gives a crisp 500x500 px image (~15-25 KB PNG)
        # Tree edges in light gray so the leaf node stands out distinctly
        fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_aspect("equal")
        ax.axis("off")

        # Obstacles
        for ox, oy, ow, oh in env.obstacles:
            ax.add_patch(
                patches.Rectangle((ox, oy), ow, oh, facecolor="black", edgecolor="black")
            )

        # Explored tree edges (light gray)
        for node in nodes:
            if node.parent is not None:
                parent = nodes[node.parent]
                ax.plot(
                    [parent.x, node.x],
                    [parent.y, node.y],
                    color="#b0b0b0",
                    linewidth=1.0,
                    alpha=0.6,
                )

        # Start (green), Goal (red), Leaf (bright orange with black border for high contrast)
        ax.scatter(*env.start, color="#2ca02c", s=100, zorder=7)
        ax.scatter(*env.goal, color="#d62728", s=100, zorder=7)
        ax.scatter(
            leaf_node.x,
            leaf_node.y,
            color="#ff7f0e",
            edgecolors="black",
            linewidths=1.5,
            s=140,
            zorder=8,
        )

        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=0.05,
            pil_kwargs={"optimize": True},
        )
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


class PromptManager:
    DIRECTIONS = {
        "east": 0.0,
        "northeast": math.pi / 4,
        "north": math.pi / 2,
        "northwest": 3 * math.pi / 4,
        "west": math.pi,
        "southwest": -3 * math.pi / 4,
        "south": -math.pi / 2,
        "southeast": -math.pi / 4,
    }

    BASE_ZERO_SHOT = (
        "You are a path planning expert guiding a robot in a 2D environment. In the given scene:\n"
        "- The green dot is the start point.\n"
        "- The red dot is the goal point.\n"
        "- The bright orange dot with a black border is the current position (active leaf node).\n"
        "- The light gray lines show the explored tree path.\n"
        "- Black rectangles represent obstacles.\n\n"
        "Based on the scene, suggest the best direction for the robot to move "
        "from the current position (orange dot) toward the goal (red dot).\n"
        "Prioritize paths that minimize collisions and move efficiently toward the goal."
    )

    COT_EXAMPLES = (
        "\n\nProvide a step-by-step thought process before giving the final answer.\n\n"
        "Example 1:\n"
        "[Simple Clear Path]\n"
        "Visual Description: Orange dot near bottom-left, red dot at top-right, no Black obstacles\n"
        "Reasoning: The direct northeast path is open, so the robot moves northeast.\n"
        "Correct Direction: northeast\n\n"
        "Example 2:\n"
        "[Multiple Black Obstacles Requiring Strategic Navigation]\n"
        "Visual Description: Several Black obstacles creating a maze-like environment, forcing a strategic path around obstacles toward the red dot\n"
        "Reasoning: Northeast and east are blocked, but southeast remains open and moves toward the goal.\n"
        "Correct Direction: southeast"
    )

    # Few-shot examples (paper Table II: few-shot achieves highest SR of 94%).
    # Examples mirror Figure 2 of arXiv:2505.23267 — no reasoning chains included.
    FEW_SHOT_EXAMPLES: str = (
        "\n\nExample 1:\n"
        "[Simple Clear Path]\n"
        "Visual Description: Orange dot near bottom-left, red dot at top-right, no Black obstacles\n"
        "Correct Direction: northeast\n\n"
        "Example 2:\n"
        "[Single Black Obstacle Blocking Direct Path]\n"
        "Visual Description: Black obstacle blocking the most direct route between orange and red "
        "dots, requiring a slight detour\n"
        "Correct Direction: east\n\n"
        "Example 3:\n"
        "[Multiple Black Obstacles Requiring Strategic Navigation]\n"
        "Visual Description: Several Black obstacles creating a maze-like environment, forcing a "
        "strategic path around obstacles toward the red dot\n"
        "Correct Direction: southeast"
    )

    @classmethod
    def build_prompt(cls, mode: str) -> str:
        """Return the system prompt for the requested mode.

        Supported modes (arXiv:2505.23267, Section IV-B):
          'zero_shot' — baseline prompt with no examples
          'few_shot'  — 3 labelled examples; highest success rate (94%) in paper Table II
          'cot'       — chain-of-thought with explicit reasoning steps
        """
        valid_list = '"north, northeast, east, southeast, south, southwest, west, northwest"'
        if mode == "cot":
            return (
                f"{cls.BASE_ZERO_SHOT}{cls.COT_EXAMPLES}\n\n"
                f"Your VERY LAST line MUST be strictly formatted as: 'Correct Direction: <direction>' choosing from: {valid_list}."
            )
        elif mode == "few_shot":
            return (
                f"{cls.BASE_ZERO_SHOT}{cls.FEW_SHOT_EXAMPLES}\n\n"
                f"Respond with ONLY ONE WORD from the following directions: {valid_list}."
            )
        return f"{cls.BASE_ZERO_SHOT}\nRespond with ONLY ONE WORD from the following directions: {valid_list}."

    @classmethod
    def parse_direction(cls, response_text: str) -> Optional[str]:
        """Robust parser tolerant to Markdown formatting (e.g. **Correct Direction:**).

        Returns None if no unambiguous direction could be determined (avoiding
        silent directional bias).
        """
        text = response_text.strip()
        # 1. Look for explicit 'Correct Direction' marker (handles **bold**, spaces, colons)
        matches = re.findall(
            r"correct direction\W*(northeast|northwest|southeast|southwest|north|south|east|west)\b",
            text,
            re.IGNORECASE,
        )
        if matches:
            return matches[-1].lower()

        # 2. Check the final non-empty line (clean of markdown) for single-word answers
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            last_clean = re.sub(r"[*_`#]", "", lines[-1]).strip().lower()
            last_matches = re.findall(
                r"\b(northeast|northwest|southeast|southwest|north|south|east|west)\b",
                last_clean,
            )
            if last_matches:
                return last_matches[-1]

        return None


# =====================================================================
# 4. VLM NAVIGATOR (Supports API Key: NVIDIA NIM, OpenAI, OpenRouter, etc.)
# =====================================================================
class VLMNavigator:
    """Handles VLM API queries via API key (NVIDIA NIM, OpenAI, OpenRouter, or custom OpenAI-compatible endpoints)."""

    def __init__(self, config: VLMRRTConfig, use_mock: bool = False):
        self.config = config
        self.use_mock = use_mock
        self._last_call = 0.0  # time of the last API request (for rate limiting)
        self._consecutive_failures = 0
        # Minimum seconds between calls. Starts at 60/max_rpm and grows when the server answers 429.
        self._min_gap = (60.0 / config.max_rpm) if config.max_rpm > 0 else 0.0

        # 1. Resolve API key: uses NVIDIA_API_KEY from .env by default
        self.api_key = config.api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")

        # 2. Resolve endpoint URL
        if config.api_url:
            self.invoke_url = config.api_url
        elif config.model_name.startswith("gpt-"):
            self.invoke_url = "https://api.openai.com/v1/chat/completions"
        elif config.model_name.startswith("openrouter/"):
            self.invoke_url = "https://openrouter.ai/api/v1/chat/completions"
        else:
            # Default to NVIDIA NIM endpoint
            self.invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

        # Check credentials
        is_local = "127.0.0.1" in self.invoke_url or "localhost" in self.invoke_url
        if not self.use_mock and not self.api_key and not is_local:
            print("Warning: No API key found in VLMRRTConfig or .env! Falling back to mock oracle.")
            self.use_mock = True
        elif not self.use_mock:
            masked_key = (
                (self.api_key[:8] + "..." + self.api_key[-4:])
                if self.api_key and len(self.api_key) > 12
                else ("None" if not self.api_key else self.api_key)
            )
            print(f"[VLMNavigator] Initialized with model: '{self.config.model_name}'")
            print(f"[VLMNavigator] Endpoint: {self.invoke_url}")
            print(f"[VLMNavigator] NVIDIA API Key from .env: {masked_key}")

    def _post_rate_limited(self, headers, payload):
        """requests.post that spaces calls to stay under max_rpm and retries 429/5xx."""
        for attempt in range(self.config.max_retries + 1):
            wait = self._last_call + self._min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                response = requests.post(
                    self.invoke_url, headers=headers, json=payload, timeout=self.config.api_timeout
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt >= self.config.max_retries:
                    raise
                print(f"[VLM] {type(e).__name__}, retrying ({attempt + 1}/{self.config.max_retries})")
                continue
            if (response.status_code == 429 or response.status_code >= 500) and attempt < self.config.max_retries:
                try:
                    pause = float(response.headers.get("Retry-After", 0))
                except ValueError:
                    pause = 0.0
                pause = max(pause, 3.0 * (attempt + 1), self._min_gap)
                if response.status_code == 429 and self._min_gap > 0:
                    # Server is throttling: slow every later call down too (capped at 10 s apart).
                    self._min_gap = min(self._min_gap * 1.5, 10.0)
                    print(f"[VLM] HTTP 429: spacing calls {self._min_gap:.1f}s apart from now on")
                print(f"[VLM] HTTP {response.status_code}, retrying in {pause:.1f}s ({attempt + 1}/{self.config.max_retries})")
                time.sleep(pause)
                continue
            return response
        return response

    def get_vlm_direction(
        self, leaf_node: Node, image_b64: str, env: Environment2D, iter_num: int = 0
    ) -> Optional[str]:
        if self.use_mock:
            return self._mock_vlm_oracle(leaf_node, env)

        prompt_text = PromptManager.build_prompt(self.config.prompt_mode)
        # Use 512 max_tokens for CoT to prevent truncation before the final answer
        max_tokens = 512 if self.config.prompt_mode == "cot" else 100

        # Build request headers
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Standard OpenAI / NVIDIA NIM multimodal schema
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if self.config.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        # Backwards compatibility: if endpoint is a local Ollama server
        if "/api/chat" in self.invoke_url:
            payload = {
                "model": self.config.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_text,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": max_tokens,
                },
            }

        t0 = time.time()
        response = None
        try:
            response = self._post_rate_limited(headers, payload)
            response.raise_for_status()
            res_json = response.json()
            self._consecutive_failures = 0  # got a real reply from the API

            # Handle standard OpenAI / NVIDIA NIM format
            if "choices" in res_json:
                raw_reply = res_json["choices"][0]["message"]["content"]
            elif "message" in res_json:  # Ollama format
                raw_reply = res_json["message"]["content"]
            else:
                raw_reply = str(res_json)

            direction = PromptManager.parse_direction(raw_reply)
            dt = time.time() - t0
            if direction:
                print(
                    f"[Iter {iter_num:03d}] VLM ({dt:.1f}s) -> {direction} | Leaf: ({leaf_node.x:.1f}, {leaf_node.y:.1f})"
                )
            else:
                print(
                    f"[Iter {iter_num:03d}] VLM ({dt:.1f}s) -> Unparseable reply (preview: {raw_reply[:60]!r})"
                )
            return direction
        except Exception as e:
            dt = time.time() - t0
            print(f"[Iter {iter_num:03d}] VLM API Error after {dt:.1f}s: {type(e).__name__}: {str(e)[:150]}")
            if response is not None:
                print(f"   Status Code: {response.status_code} | Body: {response.text[:300]}")
            self._consecutive_failures += 1
            limit = self.config.max_consecutive_failures
            if limit and self._consecutive_failures >= limit:
                raise RuntimeError(
                    f"VLM failed {self._consecutive_failures} times in a row; stopping so the run does not "
                    f"silently turn into plain RRT. Check the key, the network and the endpoint."
                ) from e
            print("   Falling back to uniform random sampling for this iteration.")
            return None

    def _mock_vlm_oracle(self, leaf_node: Node, env: Environment2D) -> str:
        best_dir, best_score, (gx, gy) = "north", -float("inf"), env.goal
        for d_name, angle in PromptManager.DIRECTIONS.items():
            test_x = leaf_node.x + self.config.sector_radius * math.cos(angle)
            test_y = leaf_node.y + self.config.sector_radius * math.sin(angle)
            if not env.path_free((leaf_node.x, leaf_node.y), (test_x, test_y)):
                continue
            score = -math.hypot(gx - test_x, gy - test_y)
            if score > best_score:
                best_score, best_dir = score, d_name
        return best_dir


# =====================================================================
# 5. VLM-RRT CORE ALGORITHM
# =====================================================================
class VLMRRTPlanner:
    def __init__(
        self, env: Environment2D, config: VLMRRTConfig, use_mock_vlm: bool = False
    ):
        self.env = env
        self.cfg = config
        self.vlm = VLMNavigator(config, use_mock=use_mock_vlm)
        self.nodes: List[Node] = [Node(env.start[0], env.start[1], parent=None)]
        self.children_count: List[int] = [0]
        self.last_sector_info: Optional[Tuple[Node, float]] = None
        self.vlm_queries: int = 0
        self.vlm_failures: int = 0

    def pick_leaf_node(self) -> Node:
        """Uniform random selection over all leaf nodes (Algorithm 2, line 6)."""
        leaf_indices = [
            idx for idx, count in enumerate(self.children_count) if count == 0
        ]
        if not leaf_indices:
            return random.choice(self.nodes)
        return self.nodes[random.choice(leaf_indices)]

    def sample_state_vlm(
        self, leaf_node: Node, direction_name: str
    ) -> Tuple[float, float]:
        center_angle = PromptManager.DIRECTIONS[direction_name]
        half_spread = self.cfg.sector_angle_rad / 2.0
        angle = random.uniform(center_angle - half_spread, center_angle + half_spread)
        radius = self.cfg.sector_radius * math.sqrt(random.uniform(0.0, 1.0))
        self.last_sector_info = (leaf_node, center_angle)
        return (
            float(np.clip(leaf_node.x + radius * math.cos(angle), 0.0, self.env.width)),
            float(np.clip(leaf_node.y + radius * math.sin(angle), 0.0, self.env.height)),
        )

    def sample_state(self) -> Tuple[float, float]:
        return (
            random.uniform(0.0, self.env.width),
            random.uniform(0.0, self.env.height),
        )

    def steer(
        self, v_nearest: Node, v_rand: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Move step_size toward v_rand (Algorithm 2, line 13)."""
        dx, dy = v_rand[0] - v_nearest.x, v_rand[1] - v_nearest.y
        dist = math.hypot(dx, dy)
        if dist <= self.cfg.step_size:
            return v_rand[0], v_rand[1]
        theta = math.atan2(dy, dx)
        return (
            v_nearest.x + self.cfg.step_size * math.cos(theta),
            v_nearest.y + self.cfg.step_size * math.sin(theta),
        )

    def plan(self) -> Tuple[Optional[List[Tuple[float, float]]], int]:
        for i in range(1, self.cfg.max_iterations + 1):
            if random.uniform(0.0, 1.0) <= self.cfg.gamma:
                self.vlm_queries += 1
                v_hat = self.pick_leaf_node()
                # Skip heavy Matplotlib image generation if running with mock oracle
                E_current = "" if self.vlm.use_mock else SnapshotRenderer.render_base64(self.env, self.nodes, v_hat)
                d = self.vlm.get_vlm_direction(v_hat, E_current, self.env, iter_num=i)
                if d and d in PromptManager.DIRECTIONS:
                    v_rand = self.sample_state_vlm(v_hat, d)
                else:
                    # VLM parse or API failure: fall back to uniform random sampling
                    # (avoids biased directions like hardcoded "north")
                    self.vlm_failures += 1
                    v_rand = self.sample_state()
            else:
                v_rand = self.sample_state()

            nearest_idx = int(
                np.argmin(
                    [(n.x - v_rand[0]) ** 2 + (n.y - v_rand[1]) ** 2 for n in self.nodes]
                )
            )
            v_nearest = self.nodes[nearest_idx]
            v_new_coords = self.steer(v_nearest, v_rand)

            if self.env.path_free((v_nearest.x, v_nearest.y), v_new_coords):
                self.nodes.append(
                    Node(v_new_coords[0], v_new_coords[1], parent=nearest_idx)
                )
                self.children_count[nearest_idx] += 1
                self.children_count.append(0)
                new_idx = len(self.nodes) - 1

                # Goal check:
                # 1. Exact / near hit (||v_new - goal|| <= goal_threshold)
                d_goal = math.hypot(
                    v_new_coords[0] - self.env.goal[0],
                    v_new_coords[1] - self.env.goal[1],
                )
                if d_goal <= self.cfg.goal_threshold:
                    path, curr = [], new_idx
                    while curr is not None:
                        path.append((self.nodes[curr].x, self.nodes[curr].y))
                        curr = self.nodes[curr].parent
                    self._print_stats()
                    return list(reversed(path)), i

                # 2. Practical goal connection: if v_new is within step_size of the goal
                # with an unobstructed line of sight, connect directly to complete path
                if d_goal <= self.cfg.step_size and self.env.path_free(v_new_coords, self.env.goal):
                    self.nodes.append(Node(self.env.goal[0], self.env.goal[1], parent=new_idx))
                    self.children_count[new_idx] += 1
                    self.children_count.append(0)
                    path, curr = [], len(self.nodes) - 1
                    while curr is not None:
                        path.append((self.nodes[curr].x, self.nodes[curr].y))
                        curr = self.nodes[curr].parent
                    self._print_stats()
                    return list(reversed(path)), i

        self._print_stats()
        return None, self.cfg.max_iterations

    def _print_stats(self):
        if self.vlm_queries > 0:
            successes = self.vlm_queries - self.vlm_failures
            pct = (successes / self.vlm_queries) * 100.0
            print(
                f"[VLM Stats] Queries: {self.vlm_queries} | "
                f"Valid answers: {successes} ({pct:.1f}%) | "
                f"Fallbacks: {self.vlm_failures}"
            )


# =====================================================================
# 6. DEMO EXECUTION
# =====================================================================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # -----------------------------------------------------------------
    # Configuration Options:
    # 1. Use NVIDIA NIM with key in .env (NVIDIA_API_KEY="nvapi-..."):
    #    VLMRRTConfig(model_name="meta/llama-3.2-90b-vision-instruct")
    #
    # 2. Or pass the API key directly in code:
    #    VLMRRTConfig(
    #        api_key="your_api_key_here",
    #        model_name="meta/llama-3.2-90b-vision-instruct",
    #    )
    #
    # 3. Use OpenAI models with OPENAI_API_KEY:
    #    VLMRRTConfig(
    #        model_name="gpt-4o",
    #        api_key="sk-...", # or set OPENAI_API_KEY in .env
    #    )
    #
    # 4. Use custom OpenAI-compatible endpoint (OpenRouter, vLLM, etc.):
    #    VLMRRTConfig(
    #        model_name="...",
    #        api_key="...",
    #        api_url="https://api.your-endpoint.com/v1/chat/completions",
    #    )
    # -----------------------------------------------------------------
    # Load NVIDIA_API_KEY from .env
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        print("[Warning] NVIDIA_API_KEY not found in .env! Please verify your .env file.")
    else:
        print(f"[Config] Loaded NVIDIA_API_KEY from .env ({nvidia_api_key[:8]}...{nvidia_api_key[-4:]})")

    config = VLMRRTConfig(
        gamma=0.85,
        prompt_mode="cot",
        model_name="meta/llama-3.2-90b-vision-instruct",
        api_key=nvidia_api_key,
        api_url="https://integrate.api.nvidia.com/v1/chat/completions",
        api_timeout=120.0,
    )

    # Cluttered 500m x 500m wildfire obstacle scenario
    env = Environment2D(
        width=500.0,
        height=500.0,
        start=(50.0, 50.0),
        goal=(450.0, 450.0),
        obstacles=[
            (100, 80, 60, 90),
            (220, 150, 80, 70),
            (140, 260, 70, 80),
            (310, 280, 75, 90),
            (230, 380, 60, 70),
            (380, 100, 60, 110),
        ],
    )

    print(f"Starting VLM-RRT planner...")
    planner = VLMRRTPlanner(env, config, use_mock_vlm=False)
    path, iters = planner.plan()

    if path:
        path_len = sum(
            math.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
            for k in range(len(path) - 1)
        )
        print(f"Path found in {iters} iterations! Total Path Length: {path_len:.2f} m")
    else:
        print(f"Failed to converge within {iters} iterations.")

    # Visualization
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_title(f"VLM-RRT Path Planning (Iterations: {iters})")

    for ox, oy, ow, oh in env.obstacles:
        ax.add_patch(patches.Rectangle((ox, oy), ow, oh, color="black"))

    for node in planner.nodes:
        if node.parent is not None:
            p = planner.nodes[node.parent]
            ax.plot([p.x, node.x], [p.y, node.y], color="gray", linewidth=0.7, alpha=0.6)

    if planner.last_sector_info:
        leaf, center_rad = planner.last_sector_info
        deg_center = math.degrees(center_rad)
        half_deg = config.sector_angle_deg / 2.0
        ax.add_patch(
            patches.Wedge(
                (leaf.x, leaf.y),
                config.sector_radius,
                deg_center - half_deg,
                deg_center + half_deg,
                facecolor="yellow",
                alpha=0.4,
            )
        )
        ax.scatter(leaf.x, leaf.y, color="#ff7f0e", edgecolors="black", linewidths=1.2, s=90, zorder=6)

    if path:
        px, py = zip(*path)
        ax.plot(px, py, color="red", linewidth=2.2, zorder=5)

    ax.scatter(*env.start, color="green", s=90, zorder=7)
    ax.scatter(*env.goal, color="red", s=90, zorder=7)

    plt.savefig("vlm_rrt_result.png", dpi=150, bbox_inches="tight")
    print("Saved plot to vlm_rrt_result.png")