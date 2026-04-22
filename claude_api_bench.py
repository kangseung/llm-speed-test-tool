#!/usr/bin/env python3
"""
Claude Code Benchmark Tool - Multi-config benchmarking tool
Features:
- Batch testing with multiple API configurations from JSON
- Interactive configuration input
- Detailed chart visualization (bar charts, etc.)
- Rich performance metrics
"""

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Try to import optional dependencies
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None

try:
    import colorama
    from colorama import Back, Fore, Style
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Define empty color class
    class ColoramaStub:
        def __init__(self):
            self.CYAN = ''
            self.GREEN = ''
            self.YELLOW = ''
            self.RED = ''
            self.BLUE = ''
            self.DIM = ''
            self.BRIGHT = ''
            self.RESET = ''
            self.RESET_ALL = ''
    Fore = Style = Back = ColoramaStub()

plt = None
MATPLOTLIB_IMPORT_ERROR = None

# ─── Constants ──────────────────────────────────────────────────────────────

DEFAULT_ROUNDS = 3
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.claude/settings.json")

# Preset test prompt suites
TEST_SUITES = {
    "short": {
        "name": "Short Response",
        "prompts": [
            {"text": "Answer in one sentence: What is recursion?", "max_tokens": 100},
            {"text": "What is quicksort? What is its time complexity?", "max_tokens": 100},
        ]
    },
    "medium": {
        "name": "Medium Response",
        "prompts": [
            {"text": "Explain quicksort algorithm's working principle and time complexity in about 100 words.", "max_tokens": 300},
            {"text": "Explain the main differences between HTTP and HTTPS, and why HTTPS is more secure.", "max_tokens": 300},
        ]
    },
    "long": {
        "name": "Long Response",
        "prompts": [
            {
                "text": (
                    "Compare Python and Rust programming languages in detail. "
                    "Analyze from: performance, memory safety, ecosystem, learning curve, "
                    "concurrent programming, deployment convenience. Provide summary recommendations."
                ),
                "max_tokens": 1024,
            },
            {
                "text": (
                    "Explain attention mechanism in deep learning, its role in Transformer models, "
                    "and why it is crucial for NLP tasks."
                ),
                "max_tokens": 1024,
            }
        ]
    }
}


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class APIConfig:
    """API configuration"""
    name: str = ""
    base_url: str = "https://api.anthropic.com"
    auth_token: str = ""
    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"

    def is_complete(self) -> bool:
        """Check if config is complete"""
        return bool(self.base_url) and bool(self.auth_token or self.api_key) and bool(self.model)


@dataclass
class SingleTestResult:
    """Single test result"""
    round_num: int
    prompt_name: str
    prompt_text: str
    ttfb_ms: float = 0.0
    total_time_ms: float = 0.0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    output_time_ms: float = 0.0  # 仅生成时间（排除TTFB）
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_num,
            "prompt": self.prompt_name,
            "ttfb_ms": round(self.ttfb_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "output_tokens": self.output_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "output_time_ms": round(self.output_time_ms, 2),
            "error": self.error
        }


@dataclass
class PromptStats:
    """Statistics for a single prompt"""
    prompt_name: str
    prompt_text: str

    # TTFB (Time To First Token)
    ttfb_avg: float = 0.0
    ttfb_min: float = 0.0
    ttfb_max: float = 0.0
    ttfb_stddev: float = 0.0
    ttfb_p95: float = 0.0
    ttfb_p99: float = 0.0

    # Tokens per second
    tps_avg: float = 0.0
    tps_min: float = 0.0
    tps_max: float = 0.0
    tps_stddev: float = 0.0

    # End-to-end latency
    e2e_avg: float = 0.0
    e2e_min: float = 0.0
    e2e_max: float = 0.0
    e2e_stddev: float = 0.0

    # Generation phase statistics
    gen_avg_ms: float = 0.0
    gen_min_ms: float = 0.0
    gen_max_ms: float = 0.0

    # Token statistics
    tokens_avg: float = 0.0
    tokens_min: int = 0
    tokens_max: int = 0

    # Success rate
    success_count: int = 0
    total_count: int = 0
    success_rate: float = 0.0
    ttfb_samples: List[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "ttfb_ms": {
                "avg": round(self.ttfb_avg, 2),
                "min": round(self.ttfb_min, 2),
                "max": round(self.ttfb_max, 2),
                "stddev": round(self.ttfb_stddev, 2),
                "p95": round(self.ttfb_p95, 2),
                "p99": round(self.ttfb_p99, 2)
            },
            "tps": {
                "avg": round(self.tps_avg, 2),
                "min": round(self.tps_min, 2),
                "max": round(self.tps_max, 2),
                "stddev": round(self.tps_stddev, 2)
            },
            "e2e_ms": {
                "avg": round(self.e2e_avg, 2),
                "min": round(self.e2e_min, 2),
                "max": round(self.e2e_max, 2),
                "stddev": round(self.e2e_stddev, 2)
            },
            "generation_ms": {
                "avg": round(self.gen_avg_ms, 2),
                "min": round(self.gen_min_ms, 2),
                "max": round(self.gen_max_ms, 2)
            },
            "tokens": {
                "avg": round(self.tokens_avg, 2),
                "min": self.tokens_min,
                "max": self.tokens_max
            },
            "success_rate": round(self.success_rate, 4)
        }


@dataclass
class TestRunResult:
    """Result of a complete test run"""
    config: APIConfig
    rounds: int
    stats: Dict[str, PromptStats]
    start_time: float = 0.0
    end_time: float = 0.0
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    run_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "base_url": self.config.base_url,
                "model": self.config.model
            },
            "rounds": self.rounds,
            "total_tests": self.total_tests,
            "successful_tests": self.successful_tests,
            "failed_tests": self.failed_tests,
            "duration_seconds": round(self.end_time - self.start_time, 2),
            "run_error": self.run_error,
            "prompt_stats": {k: v.to_dict() for k, v in self.stats.items()}
        }


# ─── Helper Functions ───────────────────────────────────────────────────────

def mask_token(token: str) -> str:
    """Mask token, showing first 8 and last 4 characters"""
    if not token or len(token) <= 12:
        return "****"
    return token[:8] + "****" + token[-4:] if len(token) > 12 else "****"


def resolve_path(path: str) -> str:
    """Expand `~` and return an absolute path."""
    if not path:
        return path
    return os.path.abspath(os.path.expanduser(path))


def shorten_text(text: str, max_len: int) -> str:
    """Shorten long labels for terminal tables."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def ensure_matplotlib() -> bool:
    """Import matplotlib lazily so non-chart commands stay quiet and fast."""
    global plt, MATPLOTLIB_IMPORT_ERROR

    if plt is not None:
        return True
    if MATPLOTLIB_IMPORT_ERROR is not None:
        return False

    mpl_config_dir = os.path.join(tempfile.gettempdir(), "claude-api-bench-mpl")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as loaded_plt
    except ImportError as exc:
        MATPLOTLIB_IMPORT_ERROR = exc
        return False

    plt = loaded_plt
    return True


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def safe_stdev(data: List[float]) -> float:
    """Safely compute standard deviation"""
    if len(data) < 2:
        return 0.0
    return statistics.stdev(data)


def safe_mean(data: List[float]) -> float:
    """Safely compute mean"""
    if not data:
        return 0.0
    return statistics.mean(data)


# ─── Config Management ──────────────────────────────────────────────────────

def load_config_from_json(config_path: str) -> List[APIConfig]:
    """Load config from JSON file"""
    configs = []
    resolved_path = resolve_path(config_path)

    if not os.path.isfile(resolved_path):
        return configs

    try:
        with open(resolved_path, encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  {Fore.YELLOW}⚠ Failed to read config file: {e}{Style.RESET_ALL}")
        return configs

    # Support two formats:
    # 1. Single config object
    # 2. Config array

    if isinstance(data, dict):
        # Single config
        if "name" in data or "base_url" in data:
            cfg = APIConfig(
                name=data.get("name", "default"),
                base_url=data.get("base_url", "https://api.anthropic.com"),
                auth_token=data.get("auth_token", data.get("api_key", "")),
                api_key=data.get("api_key", ""),
                model=data.get("model", "claude-sonnet-4-20250514")
            )
            if cfg.is_complete():
                configs.append(cfg)
        # Nested in env field
        elif "env" in data:
            env = data.get("env", {})
            cfg = APIConfig(
                name="default",
                base_url=env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                auth_token=env.get("ANTHROPIC_AUTH_TOKEN", env.get("ANTHROPIC_API_KEY", "")),
                api_key=env.get("ANTHROPIC_API_KEY", ""),
                model=env.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            )
            if cfg.is_complete():
                configs.append(cfg)
    elif isinstance(data, list):
        # Config array
        for item in data:
            if isinstance(item, dict):
                cfg = APIConfig(
                    name=item.get("name", f"config-{len(configs)+1}"),
                    base_url=item.get("base_url", "https://api.anthropic.com"),
                    auth_token=item.get("auth_token", item.get("api_key", "")),
                    api_key=item.get("api_key", ""),
                    model=item.get("model", "claude-sonnet-4-20250514")
                )
                if cfg.is_complete():
                    configs.append(cfg)

    return configs


def input_config_interactive() -> List[APIConfig]:
    """Interactive config input"""
    configs = []
    print(f"\n{Fore.CYAN}{'═' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}  Interactive Config Input{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═' * 50}{Style.RESET_ALL}\n")

    while True:
        print(f"\n{Fore.GREEN}Config #{len(configs) + 1}{Style.RESET_ALL}")
        print("(Press Enter for default)")

        name = input(f"  Config name [config-{len(configs)+1}]: ").strip()
        if not name:
            name = f"config-{len(configs)+1}"

        base_url = input(f"  API Endpoint [https://api.anthropic.com]: ").strip()
        if not base_url:
            base_url = "https://api.anthropic.com"

        auth_token = input(f"  API Key/Token (required): ").strip()
        if not auth_token:
            print(f"  {Fore.RED}✗ API Key/Token cannot be empty{Style.RESET_ALL}")
            continue

        model = input(f"  Model [claude-sonnet-4-20250514]: ").strip()
        if not model:
            model = "claude-sonnet-4-20250514"

        cfg = APIConfig(
            name=name,
            base_url=base_url,
            auth_token=auth_token,
            model=model
        )
        configs.append(cfg)

        print()
        another = input(f"  Add more configs? (y/n) [n]: ").strip().lower()
        if another != 'y':
            break

    return configs


def save_config_to_json(configs: List[APIConfig], config_path: str) -> None:
    """Save config to JSON file"""
    resolved_path = resolve_path(config_path)
    data = []
    for cfg in configs:
        item = {
            "name": cfg.name,
            "base_url": cfg.base_url,
            "model": cfg.model
        }
        if cfg.auth_token:
            item["auth_token"] = cfg.auth_token
        if cfg.api_key:
            item["api_key"] = cfg.api_key
        data.append(item)

    parent_dir = os.path.dirname(resolved_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(resolved_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  {Fore.GREEN}✓ Config saved to: {resolved_path}{Style.RESET_ALL}")


# ─── Test Engine ────────────────────────────────────────────────────────────

def create_client(config: APIConfig):
    """Create Anthropic client"""
    if not config.is_complete():
        raise ValueError("Config is incomplete: base_url, token, and model are required")
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic module is not installed")

    kwargs = {
        "base_url": config.base_url,
        "timeout": 120.0,
        "max_retries": 0,
    }
    if config.auth_token:
        kwargs["auth_token"] = config.auth_token
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return anthropic.Anthropic(**kwargs)


def run_single_test(client, model: str, prompt_text: str, prompt_name: str, max_tokens: int) -> SingleTestResult:
    """Run single test"""
    result = SingleTestResult(
        round_num=0,
        prompt_name=prompt_name,
        prompt_text=prompt_text
    )

    try:
        t_start = time.perf_counter()

        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt_text}],
        ) as stream:
            ttfb = None
            output_tokens = 0
            delta_count = 0

            for event in stream:
                if ttfb is None:
                    if hasattr(event, "type") and event.type == "content_block_delta":
                        ttfb = time.perf_counter() - t_start

                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "type", "") == "text_delta":
                            delta_count += 1

                    if event.type == "message_delta":
                        usage = getattr(event, "usage", None)
                        if usage and hasattr(usage, "output_tokens"):
                            output_tokens = usage.output_tokens

            t_end = time.perf_counter()
            total_time = t_end - t_start

            # Backfill token count
            if output_tokens == 0 and delta_count > 0:
                output_tokens = delta_count

            # Calculate TTFB
            if ttfb is None:
                ttfb = total_time

            result.ttfb_ms = ttfb * 1000
            result.total_time_ms = total_time * 1000
            result.output_time_ms = (total_time - ttfb) * 1000

            # Calculate tokens per second
            gen_time = total_time - ttfb
            if gen_time > 0 and output_tokens > 0:
                result.tokens_per_second = output_tokens / gen_time

            result.output_tokens = output_tokens

    except anthropic.APIConnectionError as e:
        result.error = f"Connection failed: {e}"
    except anthropic.AuthenticationError:
        result.error = "Authentication failed (401/403), please check API Key"
    except anthropic.RateLimitError:
        result.error = "Rate limit exceeded (429)"
    except anthropic.InternalServerError:
        result.error = "Server error"
    except anthropic.APITimeoutError:
        result.error = "Request timeout"
    except Exception as e:
        result.error = f"Unknown error: {e}"

    return result


def compute_prompt_stats(results: List[SingleTestResult], prompt_name: str, prompt_text: str) -> PromptStats:
    """Compute statistics for a single prompt"""
    success = [r for r in results if r.error is None]

    if not success:
        return PromptStats(
            prompt_name=prompt_name,
            prompt_text=prompt_text,
            success_count=0,
            total_count=len(results),
            success_rate=0.0
        )

    ttfbs = [r.ttfb_ms for r in success]
    tps_list = [r.tokens_per_second for r in success if r.tokens_per_second > 0]
    e2es = [r.total_time_ms for r in success]
    gen_times = [r.output_time_ms for r in success]
    tokens = [r.output_tokens for r in success]

    return PromptStats(
        prompt_name=prompt_name,
        prompt_text=prompt_text,
        ttfb_avg=statistics.mean(ttfbs),
        ttfb_min=min(ttfbs),
        ttfb_max=max(ttfbs),
        ttfb_stddev=safe_stdev(ttfbs),
        ttfb_p95=percentile(ttfbs, 95),
        ttfb_p99=percentile(ttfbs, 99),
        tps_avg=statistics.mean(tps_list) if tps_list else 0.0,
        tps_min=min(tps_list) if tps_list else 0.0,
        tps_max=max(tps_list) if tps_list else 0.0,
        tps_stddev=safe_stdev(tps_list) if len(tps_list) >= 2 else 0.0,
        e2e_avg=statistics.mean(e2es),
        e2e_min=min(e2es),
        e2e_max=max(e2es),
        e2e_stddev=safe_stdev(e2es),
        gen_avg_ms=statistics.mean(gen_times),
        gen_min_ms=min(gen_times),
        gen_max_ms=max(gen_times),
        tokens_avg=statistics.mean(tokens),
        tokens_min=min(tokens),
        tokens_max=max(tokens),
        success_count=len(success),
        total_count=len(results),
        success_rate=len(success) / len(results) if results else 0.0,
        ttfb_samples=list(ttfbs)
    )


# ─── Chart Plotting ─────────────────────────────────────────────────────────

def plot_benchmark_results(results: List[TestRunResult], output_dir: str = ".") -> Dict[str, str]:
    """Plot benchmark results charts"""
    if not ensure_matplotlib():
        print(f"  {Fore.YELLOW}⚠ Skipping chart generation: matplotlib is not available{Style.RESET_ALL}")
        return {}

    charts = {}
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Merge all test results
    all_configs = []
    all_stats = []

    for result in results:
        if result.stats:
            all_configs.append(result.config)
            all_stats.extend([
                (result.config.name, prompt_name, prompt_stats)
                for prompt_name, prompt_stats in result.stats.items()
            ])

    if not all_stats:
        return charts

    # 1. TTFB bar chart comparison
    charts['ttfb_comparison'] = _plot_ttfb_comparison(all_stats, output_dir)

    # 2. TPS bar chart comparison
    charts['tps_comparison'] = _plot_tps_comparison(all_stats, output_dir)

    # 3. End-to-end latency comparison
    charts['e2e_comparison'] = _plot_e2e_comparison(all_stats, output_dir)

    # 4. Success rate comparison
    charts['success_rate'] = _plot_success_rate(results, output_dir)

    # 5. Overall comparison chart
    charts['overall_comparison'] = _plot_overall_comparison(all_configs, results, output_dir)

    return charts


def _plot_ttfb_comparison(stats: List[tuple], output_dir: str) -> str:
    """Plot TTFB bar chart comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 按配置分组
    config_data = {}
    for config_name, prompt_name, prompt_stats in stats:
        if config_name not in config_data:
            config_data[config_name] = {}
        config_data[config_name][prompt_name] = prompt_stats.ttfb_avg

    if not config_data:
        return ""

    prompt_names = list(config_data[list(config_data.keys())[0]].keys())
    x = range(len(prompt_names))
    width = 0.8 / len(config_data)

    for i, (config_name, prompt_stats) in enumerate(config_data.items()):
        values = [prompt_stats.get(p, 0) for p in prompt_names]
        ax.bar([p + i * width for p in x], values, width, label=config_name)

    ax.set_xlabel('Prompt Type')
    ax.set_ylabel('TTFB (ms)')
    ax.set_title('TTFB Comparison (Lower is Better)')
    ax.set_xticks([p + width * (len(config_data) - 1) / 2 for p in x])
    ax.set_xticklabels(prompt_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    filename = os.path.join(output_dir, 'ttfb_comparison.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return filename


def _plot_tps_comparison(stats: List[tuple], output_dir: str) -> str:
    """Plot TPS bar chart comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))

    config_data = {}
    for config_name, prompt_name, prompt_stats in stats:
        if config_name not in config_data:
            config_data[config_name] = {}
        config_data[config_name][prompt_name] = prompt_stats.tps_avg

    if not config_data:
        return ""

    prompt_names = list(config_data[list(config_data.keys())[0]].keys())
    x = range(len(prompt_names))
    width = 0.8 / len(config_data)

    for i, (config_name, prompt_stats) in enumerate(config_data.items()):
        values = [prompt_stats.get(p, 0) for p in prompt_names]
        ax.bar([p + i * width for p in x], values, width, label=config_name)

    ax.set_xlabel('Prompt Type')
    ax.set_ylabel('TPS (tokens/s)')
    ax.set_title('TPS Comparison (Higher is Better)')
    ax.set_xticks([p + width * (len(config_data) - 1) / 2 for p in x])
    ax.set_xticklabels(prompt_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    filename = os.path.join(output_dir, 'tps_comparison.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return filename


def _plot_e2e_comparison(stats: List[tuple], output_dir: str) -> str:
    """Plot end-to-end latency comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))

    config_data = {}
    for config_name, prompt_name, prompt_stats in stats:
        if config_name not in config_data:
            config_data[config_name] = {}
        config_data[config_name][prompt_name] = prompt_stats.e2e_avg

    if not config_data:
        return ""

    prompt_names = list(config_data[list(config_data.keys())[0]].keys())
    x = range(len(prompt_names))
    width = 0.8 / len(config_data)

    for i, (config_name, prompt_stats) in enumerate(config_data.items()):
        values = [prompt_stats.get(p, 0) for p in prompt_names]
        ax.bar([p + i * width for p in x], values, width, label=config_name)

    ax.set_xlabel('Prompt Type')
    ax.set_ylabel('E2E Latency (ms)')
    ax.set_title('E2E Latency Comparison (Lower is Better)')
    ax.set_xticks([p + width * (len(config_data) - 1) / 2 for p in x])
    ax.set_xticklabels(prompt_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    filename = os.path.join(output_dir, 'e2e_comparison.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return filename


def _plot_success_rate(results: List[TestRunResult], output_dir: str) -> str:
    """Plot success rate comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    config_names = []
    success_rates = []

    for result in results:
        if result.stats:
            success_count = sum(s.success_count for s in result.stats.values())
            total_count = sum(s.total_count for s in result.stats.values())
            if total_count > 0:
                config_names.append(result.config.name)
                rate = success_count / total_count * 100
                success_rates.append(rate if not math.isnan(rate) else 0.0)

    if not config_names:
        return ""

    colors = ['#2ecc71' if sr >= 90 else '#f1c40f' if sr >= 70 else '#e74c3c' for sr in success_rates]

    ax.bar(config_names, success_rates, color=colors)
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Success Rate Comparison')
    ax.grid(axis='y', alpha=0.3)

    filename = os.path.join(output_dir, 'success_rate.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return filename


def _plot_overall_comparison(configs: List[APIConfig], results: List[TestRunResult], output_dir: str) -> str:
    """Plot overall comparison chart"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('API Performance Comparison', fontsize=14, fontweight='bold')

    # Prepare data
    config_stats = {}
    for result in results:
        if result.stats:
            config_stats[result.config.name] = result.stats

    if not config_stats:
        return ""

    # TTFB subplot
    ax1 = axes[0, 0]
    ttfb_values = [stat.ttfb_avg for stats in config_stats.values()
                   for stat in stats.values()]
    ttfb_labels = [f"{c}\n{p}" for c in config_stats.keys()
                   for p in config_stats[c].keys()]
    colors1 = ['#3498db' for _ in ttfb_labels]
    ax1.bar(ttfb_labels, ttfb_values, color=colors1)
    ax1.set_ylabel('TTFB (ms)')
    ax1.set_title('TTFB (Lower is Better)')
    ax1.tick_params(axis='x', rotation=45)

    # TPS subplot
    ax2 = axes[0, 1]
    tps_values = [stat.tps_avg for stats in config_stats.values()
                  for stat in stats.values()]
    colors2 = ['#2ecc71' for _ in ttfb_labels]
    ax2.bar(ttfb_labels, tps_values, color=colors2)
    ax2.set_ylabel('TPS (tokens/s)')
    ax2.set_title('TPS (Higher is Better)')
    ax2.tick_params(axis='x', rotation=45)

    # Success rate subplot
    ax3 = axes[1, 0]
    success_rates = []
    for stats in config_stats.values():
        for stat in stats.values():
            rate = (stat.success_count / stat.total_count * 100) if stat.total_count > 0 else 0
            success_rates.append(rate)
    colors3 = ['#9b59b6' for _ in ttfb_labels]
    ax3.bar(ttfb_labels, success_rates, color=colors3)
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate')
    ax3.tick_params(axis='x', rotation=45)

    # Latency distribution box plot
    ax4 = axes[1, 1]
    all_ttfb_data = []
    box_labels = []
    for config_name, stats in config_stats.items():
        for prompt_name, stat in stats.items():
            if stat.ttfb_samples:
                all_ttfb_data.append(stat.ttfb_samples)
                box_labels.append(f"{config_name}\n{prompt_name}")

    if all_ttfb_data and any(d for d in all_ttfb_data):
        ax4.boxplot(all_ttfb_data, labels=box_labels, patch_artist=True)
        ax4.set_ylabel('TTFB (ms)')
        ax4.set_title('TTFB Distribution')
        ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    filename = os.path.join(output_dir, 'overall_comparison.png')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return filename


# ─── Output Display ─────────────────────────────────────────────────────────

def print_header(config: APIConfig, rounds: int) -> None:
    """Print test header information"""
    token = config.auth_token or config.api_key
    print()
    print(f"  {Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}{Style.BRIGHT}  Claude Code Benchmark Tool{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    print(f"  Config:    {config.name}")
    print(f"  Endpoint:  {config.base_url}")
    print(f"  Model:     {config.model}")
    print(f"  Token:     {mask_token(token)}")
    print(f"  Rounds:    {rounds} per prompt")
    print(f"  {Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")
    print()


def print_warmup_result(success: bool, error: Optional[str] = None) -> None:
    """Print warmup result"""
    if success:
        print(f"  {Fore.GREEN}[Warmup]{Style.RESET_ALL} ✓ Connected OK")
    else:
        print(f"  {Fore.YELLOW}[Warmup]{Style.RESET_ALL} ⚠ Failed: {error}")
    print()


def print_round_result(round_num: int, total: int, result: SingleTestResult) -> None:
    """Print single round test result"""
    if result.error:
        print(
            f"  {Fore.RED}[{result.prompt_name}]{Style.RESET_ALL} "
            f"Round {round_num}/{total}  ✗ {result.error}"
        )
    else:
        # Color grading
        ttfb_color = Fore.GREEN if result.ttfb_ms < 500 else Fore.YELLOW if result.ttfb_ms < 1500 else Fore.RED
        tps_color = Fore.GREEN if result.tokens_per_second >= 50 else Fore.YELLOW if result.tokens_per_second >= 20 else Fore.RED

        print(
            f"  [{result.prompt_name}] "
            f"Round {round_num}/{total}  ✓  "
            f"TTFB: {ttfb_color}{result.ttfb_ms:.0f}ms{Style.RESET_ALL} | "
            f"{tps_color}{result.tokens_per_second:.1f} tok/s{Style.RESET_ALL} | "
            f"Total: {result.total_time_ms:.0f}ms | "
            f"Tokens: {result.output_tokens}"
        )


def print_prompt_summary(stats: PromptStats) -> None:
    """Print prompt summary"""
    if stats.success_count == 0:
        print(f"  {Fore.RED}✗ All Failed{Style.RESET_ALL}")
        return

    # TTFB color grading
    ttfb_avg_color = Fore.GREEN if stats.ttfb_avg < 500 else Fore.YELLOW if stats.ttfb_avg < 1500 else Fore.RED

    # TPS color grading
    tps_avg_color = Fore.GREEN if stats.tps_avg >= 50 else Fore.YELLOW if stats.tps_avg >= 20 else Fore.RED

    print()
    print(f"  {Fore.CYAN}{'─' * 70}{Style.RESET_ALL}")
    print(f"  {Style.BRIGHT}  {stats.prompt_name}{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}{'─' * 70}{Style.RESET_ALL}")

    print(f"  {Style.DIM}TTFB (Time To First Token){Style.RESET_ALL}")
    print(f"    Avg: {ttfb_avg_color}{stats.ttfb_avg:.0f}ms{Style.RESET_ALL} | "
          f"Min: {stats.ttfb_min:.0f}ms | "
          f"Max: {stats.ttfb_max:.0f}ms | "
          f"Std: {stats.ttfb_stddev:.0f}ms")
    print(f"    P95: {stats.ttfb_p95:.0f}ms | P99: {stats.ttfb_p99:.0f}ms")

    print(f"  {Style.DIM}TPS (Tokens Per Second){Style.RESET_ALL}")
    print(f"    Avg: {tps_avg_color}{stats.tps_avg:.1f}{Style.RESET_ALL} | "
          f"Min: {stats.tps_min:.1f} | "
          f"Max: {stats.tps_max:.1f} | "
          f"Std: {stats.tps_stddev:.1f}")

    print(f"  {Style.DIM}E2E Latency{Style.RESET_ALL}")
    print(f"    Avg: {stats.e2e_avg:.0f}ms | "
          f"Min: {stats.e2e_min:.0f}ms | "
          f"Max: {stats.e2e_max:.0f}ms")

    print(f"  {Style.DIM}Generation (excl. TTFB){Style.RESET_ALL}")
    print(f"    Avg: {stats.gen_avg_ms:.0f}ms | "
          f"Min: {stats.gen_min_ms:.0f}ms | "
          f"Max: {stats.gen_max_ms:.0f}ms")

    print(f"  {Style.DIM}Token Stats{Style.RESET_ALL}")
    print(f"    Avg: {stats.tokens_avg:.1f} | "
          f"Min: {stats.tokens_min} | "
          f"Max: {stats.tokens_max}")

    print(f"  {Style.DIM}Success Rate{Style.RESET_ALL}")
    print(f"    {stats.success_count}/{stats.total_count} "
          f"({stats.success_rate*100:.1f}%)")


def print_summary_table(all_results: List[TestRunResult]) -> None:
    """Print summary table"""
    print()
    print(f"  {Fore.CYAN}{'═' * 116}{Style.RESET_ALL}")
    print(f"  {Style.BRIGHT}  Benchmark Summary{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}{'═' * 116}{Style.RESET_ALL}")

    # Header
    header = (
        f"  {'Config':<16} │ "
        f"{'Prompt':<36} │ "
        f"{'TTFB (ms)':<20} │ "
        f"{'TPS':<18} │ "
        f"{'Success':<10}"
    )
    sub_header = (
        f"  {'':16} │ "
        f"{'':36} │ "
        f"{'avg':>6} {'p95':>6} {'p99':>6} │ "
        f"{'avg':>6} {'min':>6} │ "
        f"{'%':>8}"
    )

    print(f"  {Style.DIM}{header}{Style.RESET_ALL}")
    print(f"  {Style.DIM}{sub_header}{Style.RESET_ALL}")
    print(f"  {Fore.BLUE}{'─' * 116}{Style.RESET_ALL}")

    for result in all_results:
        if result.run_error:
            print(
                f"  {result.config.name:<16} │ "
                f"{'Run Error':<36} │ "
                f"{Fore.RED}{shorten_text(result.run_error, 20):<20}{Style.RESET_ALL} │ "
                f"{'':<18} │ "
                f"{'0.0':>8}"
            )
            continue

        if not result.stats:
            continue

        for prompt_name, stats in result.stats.items():
            prompt_label = shorten_text(prompt_name, 36)

            if stats.success_count == 0:
                print(
                    f"  {result.config.name:<16} │ "
                    f"{prompt_label:<36} │ "
                    f"{Fore.RED}{'All Failed':^20}{Style.RESET_ALL} │ "
                    f"{'':<18} │ "
                    f"{'0.0':>8}"
                )
                continue

            # Color grading
            ttfb_avg_color = Fore.GREEN if stats.ttfb_avg < 500 else Fore.YELLOW if stats.ttfb_avg < 1500 else Fore.RED
            tps_avg_color = Fore.GREEN if stats.tps_avg >= 50 else Fore.YELLOW if stats.tps_avg >= 20 else Fore.RED

            row = (
                f"  {result.config.name:<16} │ "
                f"{prompt_label:<36} │ "
                f"{ttfb_avg_color}{stats.ttfb_avg:>6.0f}{Style.RESET_ALL} "
                f"{stats.ttfb_p95:>6.0f} "
                f"{stats.ttfb_p99:>6.0f} │ "
                f"{tps_avg_color}{stats.tps_avg:>6.1f}{Style.RESET_ALL} "
                f"{stats.tps_min:>6.1f} │ "
                f"{stats.success_rate*100:>7.1f}%"
            )
            print(row)

    print(f"  {Fore.BLUE}{'─' * 116}{Style.RESET_ALL}")
    print()

    # Overall statistics
    total_tests = sum(r.total_tests for r in all_results)
    successful_tests = sum(r.successful_tests for r in all_results)

    if total_tests > 0:
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Failed: {total_tests - successful_tests}")
        print(f"  Overall Success Rate: {successful_tests/total_tests*100:.1f}%")
    print()


def print_json_report(all_results: List[TestRunResult]) -> None:
    """Print JSON format report"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "results": [],
    }

    for result in all_results:
        result_data = {
            "config_name": result.config.name,
            "base_url": result.config.base_url,
            "model": result.config.model,
            "rounds": result.rounds,
            "run_error": result.run_error,
            "stats": {},
        }

        for prompt_name, stats in result.stats.items():
            result_data["stats"][prompt_name] = stats.to_dict()

        report["results"].append(result_data)

    print(json.dumps(report, indent=2, ensure_ascii=False))


def print_charts_summary(charts: Dict[str, str], output_dir: str) -> None:
    """Print charts generation summary"""
    if not charts:
        return

    print()
    print(f"  {Fore.CYAN}{'═' * 50}{Style.RESET_ALL}")
    print(f"  {Style.BRIGHT}  Generated Charts{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}{'═' * 50}{Style.RESET_ALL}")

    chart_names = {
        'ttfb_comparison': 'TTFB Comparison',
        'tps_comparison': 'TPS Comparison',
        'e2e_comparison': 'E2E Latency Comparison',
        'success_rate': 'Success Rate Comparison',
        'overall_comparison': 'Overall Comparison'
    }

    for chart_type, filename in charts.items():
        if os.path.exists(filename):
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {chart_names.get(chart_type, chart_type)}: {os.path.join(output_dir, os.path.basename(filename))}")


# ─── Main Process ───────────────────────────────────────────────────────────

def run_benchmark(config: APIConfig, rounds: int, no_warmup: bool = False) -> TestRunResult:
    """Run benchmark for a single config"""
    print_header(config, rounds)

    # Create client
    client = create_client(config)

    # Warmup
    if not no_warmup:
        warmup_result = run_single_test(
            client, config.model,
            "Hi", "Warmup", 10
        )
        print_warmup_result(warmup_result.error is None, warmup_result.error)
    else:
        print(f"  {Fore.YELLOW}[Warmup]{Style.RESET_ALL} Skipped")
        print()

    # Main test
    all_stats: Dict[str, PromptStats] = {}
    total_start = time.perf_counter()

    total_tests = 0
    successful_tests = 0

    for suite_name, suite in TEST_SUITES.items():
        print(f"\n  {Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")
        print(f"  {Style.BRIGHT}  {suite['name']} (Suite: {suite_name}){Style.RESET_ALL}")
        print(f"  {Fore.CYAN}{'─' * 60}{Style.RESET_ALL}\n")

        for prompt_cfg in suite['prompts']:
            prompt_name = f"{suite['name']} - {prompt_cfg['text'][:30]}..."
            prompt_text = prompt_cfg['text']
            max_tokens = prompt_cfg['max_tokens']

            results: List[SingleTestResult] = []

            for i in range(1, rounds + 1):
                result = run_single_test(
                    client, config.model,
                    prompt_text, prompt_name,
                    max_tokens,
                )
                result.round_num = i
                print_round_result(i, rounds, result)
                results.append(result)

                if result.error is None:
                    successful_tests += 1
                total_tests += 1

            all_stats[prompt_name] = compute_prompt_stats(results, prompt_name, prompt_text)
            print_prompt_summary(all_stats[prompt_name])
            print()

    total_end = time.perf_counter()

    return TestRunResult(
        config=config,
        rounds=rounds,
        stats=all_stats,
        start_time=total_start,
        end_time=total_end,
        total_tests=total_tests,
        successful_tests=successful_tests,
        failed_tests=total_tests - successful_tests
    )


def main() -> None:
    # Initialize colorama (if available)
    if HAS_COLORAMA:
        colorama.init()

    parser = argparse.ArgumentParser(
        description="Claude Code Benchmark Tool - Multi-config benchmarking tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config file
  python claude_api_bench.py

  # Specify config file
  python claude_api_bench.py --config configs.json

  # Interactive config input
  python claude_api_bench.py --interactive

  # Specify test rounds
  python claude_api_bench.py --rounds 5

  # Save config
  python claude_api_bench.py --save-config

  # Generate charts
  python claude_api_bench.py --charts

  # JSON output
  python claude_api_bench.py --json
        """
    )

    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG_PATH,
                        help="Config file path (default: ~/.claude/settings.json)")
    parser.add_argument("--rounds", "-r", type=int, default=DEFAULT_ROUNDS,
                        help="Test rounds per prompt type (default: 3)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive config input")
    parser.add_argument("--save-config", action="store_true",
                        help="Save config to file")
    parser.add_argument("--charts", action="store_true",
                        help="Generate charts")
    parser.add_argument("--output-dir", default=".",
                        help="Chart output directory (default: current directory)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON format report")

    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")

    # Load config
    configs: List[APIConfig] = []
    config_path = resolve_path(args.config)

    if args.interactive:
        configs = input_config_interactive()
    else:
        configs = load_config_from_json(config_path)

        if not configs:
            print(f"  {Fore.YELLOW}⚠ No valid config found at: {config_path}{Style.RESET_ALL}")
            if not sys.stdin.isatty():
                print(f"  {Fore.RED}✗ Interactive config input is unavailable in a non-interactive session{Style.RESET_ALL}")
                sys.exit(1)
            use_interactive = input(f"  Use interactive config input? (y/n) [y]: ").strip().lower()
            if use_interactive != 'n':
                configs = input_config_interactive()
            else:
                print(f"  {Fore.RED}✗ No config available, exiting{Style.RESET_ALL}")
                sys.exit(1)

    # Save config
    if args.save_config:
        save_config_to_json(configs, config_path)

    output_dir = resolve_path(args.output_dir)

    # Run tests
    print()
    print(f"  {Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    print(f"  {Style.BRIGHT}  Starting Benchmark Tests{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    print(f"  Config count: {len(configs)}")
    print(f"  Rounds: {args.rounds}")
    print()

    all_results: List[TestRunResult] = []

    for i, config in enumerate(configs, 1):
        print(f"\n\n{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}")
        print(f"  [{i}/{len(configs)}] Testing: {config.name}")
        print(f"{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}\n")

        try:
            result = run_benchmark(config, args.rounds, args.no_warmup)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  {Fore.RED}✗ Benchmark aborted for {config.name}: {e}{Style.RESET_ALL}")
            result = TestRunResult(
                config=config,
                rounds=args.rounds,
                stats={},
                run_error=str(e),
            )
        all_results.append(result)

    # Print summary
    print_summary_table(all_results)

    # Generate charts
    if args.charts:
        print()
        charts = plot_benchmark_results(all_results, output_dir)
        print_charts_summary(charts, output_dir)

    # JSON output
    if args.json:
        print()
        print_json_report(all_results)

    # Exit code
    total_success = sum(r.successful_tests for r in all_results)
    total_tests = sum(r.total_tests for r in all_results)

    if total_tests == 0 or total_success == 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
