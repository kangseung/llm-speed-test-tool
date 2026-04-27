#!/usr/bin/env python3
"""
Claude API 基准测试工具
- 多配置批量测试
- 交互式配置输入
- 可视化性能图表（延迟热力图、TPS 雷达图、综合评分卡）
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 尝试导入依赖
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None  # type: ignore[assignment]

try:
    from colorama import Fore, Style, Back
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

    class _NoopColor:
        CYAN = GREEN = YELLOW = RED = BLUE = DIM = BRIGHT = RESET = RESET_ALL = ""

    Fore = Style = Back = _NoopColor()


# 显示字符
BOX_H = "\u2550"
BOX_V = "\u2502"
BOX_DASH = "\u2500"
CHECK = "\u2713"
CROSS = "\u2717"
WARN = "\u26a0"


# ─── 常量 ───

DEFAULT_ROUNDS = 3
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "example_config.json"
)

# 预设测试套件
TEST_SUITES = {
    "short": {
        "name": "Short",
        "prompts": [
            {"text": "In one sentence: What is recursion?", "max_tokens": 100},
            {"text": "What is quicksort? What is its time complexity?", "max_tokens": 100},
        ],
    },
    "medium": {
        "name": "Medium",
        "prompts": [
            {
                "text": "Explain quicksort algorithm's working principle and time complexity in about 100 words.",
                "max_tokens": 300,
            },
            {
                "text": "Explain the main differences between HTTP and HTTPS, and why HTTPS is more secure.",
                "max_tokens": 300,
            },
        ],
    },
    "long": {
        "name": "Long",
        "prompts": [
            {
                "text": (
                    "Compare Python and Rust programming languages in detail. "
                    "Analyze from: performance, memory safety, ecosystem, concurrent programming, "
                    "deployment convenience. Provide summary recommendations."
                ),
                "max_tokens": 1024,
            },
            {
                "text": (
                    "Explain attention mechanism in deep learning, its role in Transformer models, "
                    "and why it is crucial for NLP tasks."
                ),
                "max_tokens": 1024,
            },
        ],
    },
}


# ─── 数据结构 ───


@dataclass
class APIConfig:
    """API 配置"""

    name: str = ""
    base_url: str = "https://api.anthropic.com"
    auth_token: str = ""
    model: str = "claude-sonnet-4-20250514"

    def is_complete(self) -> bool:
        """检查配置是否完整"""
        return bool(self.base_url) and bool(self.auth_token) and bool(self.model)


@dataclass
class SingleTestResult:
    """单次测试结果"""

    round_num: int
    prompt_name: str
    prompt_text: str
    ttfb_ms: float = 0.0  # 首 token 延迟
    total_time_ms: float = 0.0  # 端到端延迟
    output_tokens: int = 0
    tokens_per_second: float = 0.0  # 生成速度
    generation_time_ms: float = 0.0  # 仅生成时间（不含 TTFB）
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_num,
            "prompt": self.prompt_name,
            "ttfb_ms": round(self.ttfb_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "output_tokens": self.output_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "generation_time_ms": round(self.generation_time_ms, 2),
            "error": self.error,
        }


@dataclass
class PromptStats:
    """单个 Prompt 统计"""

    prompt_name: str
    prompt_text: str
    max_tokens: int = 0

    # TTFB
    ttfb_avg: float = 0.0
    ttfb_min: float = 0.0
    ttfb_max: float = 0.0
    ttfb_stddev: float = 0.0
    ttfb_p95: float = 0.0
    ttfb_p99: float = 0.0

    # 吞吐量
    tps_avg: float = 0.0
    tps_min: float = 0.0
    tps_max: float = 0.0
    tps_stddev: float = 0.0

    # 端到端
    e2e_avg: float = 0.0
    e2e_min: float = 0.0
    e2e_max: float = 0.0
    e2e_stddev: float = 0.0

    # 生成时间
    gen_avg_ms: float = 0.0
    gen_min_ms: float = 0.0
    gen_max_ms: float = 0.0

    # Token
    tokens_avg: float = 0.0
    tokens_min: int = 0
    tokens_max: int = 0

    # 成功率
    success_count: int = 0
    total_count: int = 0
    success_rate: float = 0.0

    # 原始数据
    ttfb_samples: List[float] = field(default_factory=list, repr=False)
    tps_samples: List[float] = field(default_factory=list, repr=False)
    e2e_samples: List[float] = field(default_factory=list, repr=False)
    tokens_samples: List[int] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "max_tokens": self.max_tokens,
            "ttfb_ms": {
                "avg": round(self.ttfb_avg, 2),
                "min": round(self.ttfb_min, 2),
                "max": round(self.ttfb_max, 2),
                "stddev": round(self.ttfb_stddev, 2),
                "p95": round(self.ttfb_p95, 2),
                "p99": round(self.ttfb_p99, 2),
            },
            "tps": {
                "avg": round(self.tps_avg, 2),
                "min": round(self.tps_min, 2),
                "max": round(self.tps_max, 2),
                "stddev": round(self.tps_stddev, 2),
            },
            "e2e_ms": {
                "avg": round(self.e2e_avg, 2),
                "min": round(self.e2e_min, 2),
                "max": round(self.e2e_max, 2),
                "stddev": round(self.e2e_stddev, 2),
            },
            "generation_ms": {
                "avg": round(self.gen_avg_ms, 2),
                "min": round(self.gen_min_ms, 2),
                "max": round(self.gen_max_ms, 2),
            },
            "tokens": {
                "avg": round(self.tokens_avg, 2),
                "min": self.tokens_min,
                "max": self.tokens_max,
            },
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class TestRunResult:
    """完整的测试运行结果"""

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
                "model": self.config.model,
            },
            "rounds": self.rounds,
            "total_tests": self.total_tests,
            "successful_tests": self.successful_tests,
            "failed_tests": self.failed_tests,
            "duration_seconds": round(self.end_time - self.start_time, 2),
            "run_error": self.run_error,
            "prompt_stats": {k: v.to_dict() for k, v in self.stats.items()},
        }


# ─── 辅助函数 ───


def _mask_token(token: str) -> str:
    """遮蔽 Token，仅显示首尾"""
    if not token or len(token) <= 12:
        return "****"
    return token[:8] + "****" + token[-4:]


def _resolve(path: str) -> str:
    """展开 ~ 返回绝对路径"""
    return os.path.abspath(os.path.expanduser(path))


def _shorten(text: str, max_len: int) -> str:
    """截断长文本"""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _percentile(data: List[float], p: float) -> float:
    """计算百分位数"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def _safe_stdev(data: List[float]) -> float:
    """安全标准差"""
    if len(data) < 2:
        return 0.0
    return statistics.stdev(data)


def _mean(data: List[float]) -> float:
    """安全求平均"""
    if not data:
        return 0.0
    return statistics.mean(data)


# ─── 配置管理 ───


def load_config_from_json(config_path: str) -> List[APIConfig]:
    """从 JSON 文件加载配置"""
    resolved = _resolve(config_path)
    if not os.path.isfile(resolved):
        return []

    try:
        with open(resolved, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"  {Fore.YELLOW}{WARN} 无法读取配置文件: {e}{Style.RESET_ALL}"
        )
        return []

    configs: List[APIConfig] = []
    items = [data] if isinstance(data, dict) else data
    for item in items:
        if not isinstance(item, dict):
            continue
        cfg = APIConfig(
            name=item.get("name", f"config-{len(configs) + 1}"),
            base_url=item.get("base_url", "https://api.anthropic.com"),
            auth_token=item.get("auth_token", item.get("api_key", "")),
            model=item.get("model", "claude-sonnet-4-20250514"),
        )
        if cfg.is_complete():
            configs.append(cfg)
    return configs


def _input_config_interactive() -> List[APIConfig]:
    """交互式配置输入"""
    configs: List[APIConfig] = []

    border = BOX_H * 50
    print(f"\n{Fore.CYAN}{border}{Style.RESET_ALL}")
    print(
        f"{Fore.CYAN}{Style.BRIGHT}  交互式配置输入{Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}{border}{Style.RESET_ALL}")

    while True:
        print(f"\n{Fore.GREEN}配置 #{len(configs) + 1}{Style.RESET_ALL}")
        print("(按回车使用默认值)")

        name = input(
            f"  配置名称 [config-{len(configs) + 1}]: "
        ).strip()
        if not name:
            name = f"config-{len(configs) + 1}"

        base_url = input("  API 地址 [https://api.anthropic.com]: ").strip()
        if not base_url:
            base_url = "https://api.anthropic.com"

        auth_token = input("  API Key/Token (必填): ").strip()
        if not auth_token:
            print(
                f"  {Fore.RED}{CROSS} API Key/Token 不能为空{Style.RESET_ALL}"
            )
            continue

        model = input("  模型 [claude-sonnet-4-20250514]: ").strip()
        if not model:
            model = "claude-sonnet-4-20250514"

        configs.append(
            APIConfig(
                name=name,
                base_url=base_url,
                auth_token=auth_token,
                model=model,
            )
        )

        if input("\n  继续添加配置？(y/n) [n]: ").strip().lower() != "y":
            break

    return configs


def save_config_to_json(configs: List[APIConfig], config_path: str) -> None:
    """保存配置到 JSON 文件"""
    resolved = _resolve(config_path)
    data = []
    for cfg in configs:
        item = {
            "name": cfg.name,
            "base_url": cfg.base_url,
            "auth_token": cfg.auth_token,
            "model": cfg.model,
        }
        data.append(item)

    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(resolved, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(
        f"  {Fore.GREEN}{CHECK} 配置已保存: {resolved}{Style.RESET_ALL}"
    )


# ─── 测试引擎 ───


def _create_client(config: APIConfig):
    """创建 Anthropic 客户端"""
    if not config.is_complete():
        raise ValueError(
            "配置不完整: base_url、auth_token 和 model 是必填的"
        )
    if not HAS_ANTHROPIC:
        raise ImportError(
            "anthropic 模块未安装，请执行: pip install anthropic"
        )

    return anthropic.Anthropic(
        base_url=config.base_url,
        api_key=config.auth_token,
        timeout=120.0,
        max_retries=0,
    )


def run_single_test(
    client,
    model: str,
    prompt_text: str,
    prompt_name: str,
    max_tokens: int,
) -> SingleTestResult:
    """执行单次测试"""
    result = SingleTestResult(
        round_num=0,
        prompt_name=prompt_name,
        prompt_text=prompt_text,
    )

    try:
        t_start = time.perf_counter()

        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt_text}],
        ) as stream:
            ttfb: Optional[float] = None
            output_tokens: int = 0

            for event in stream:
                # 首次 content_block_delta 为 TTFB
                if ttfb is None:
                    if getattr(event, "type", "") == "content_block_delta":
                        ttfb = time.perf_counter() - t_start

                # 读取 message_delta 事件获取 output_tokens
                if getattr(event, "type", "") == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        output_tokens = getattr(
                            usage, "output_tokens", 0
                        )

            t_end = time.perf_counter()

        total_time = t_end - t_start

        # TTFB 回退
        if ttfb is None:
            ttfb = total_time

        result.ttfb_ms = ttfb * 1000
        result.total_time_ms = total_time * 1000
        result.generation_time_ms = (total_time - ttfb) * 1000
        result.output_tokens = output_tokens

        # 计算 TPS（生成速度）
        gen_time = total_time - ttfb
        if gen_time > 0 and output_tokens > 0:
            result.tokens_per_second = output_tokens / gen_time

    except Exception as e:
        result.error = str(e)

    return result


def _compute_prompt_stats(
    results: List[SingleTestResult],
    prompt_name: str,
    prompt_text: str,
    max_tokens: int,
) -> PromptStats:
    """计算单个 Prompt 的统计数据"""
    success = [r for r in results if r.error is None]

    if not success:
        return PromptStats(
            prompt_name=prompt_name,
            prompt_text=prompt_text,
            success_count=0,
            total_count=len(results),
            success_rate=0.0,
        )

    ttfbs = [r.ttfb_ms for r in success]
    tps_list = [r.tokens_per_second for r in success if r.tokens_per_second > 0]
    e2es = [r.total_time_ms for r in success]
    gen_times = [r.generation_time_ms for r in success]
    tokens = [r.output_tokens for r in success]

    return PromptStats(
        prompt_name=prompt_name,
        prompt_text=prompt_text,
        max_tokens=max_tokens,
        ttfb_avg=_mean(ttfbs),
        ttfb_min=min(ttfbs),
        ttfb_max=max(ttfbs),
        ttfb_stddev=_safe_stdev(ttfbs),
        ttfb_p95=_percentile(ttfbs, 95),
        ttfb_p99=_percentile(ttfbs, 99),
        tps_avg=_mean(tps_list) if tps_list else 0.0,
        tps_min=min(tps_list) if tps_list else 0.0,
        tps_max=max(tps_list) if tps_list else 0.0,
        tps_stddev=(
            _safe_stdev(tps_list) if len(tps_list) >= 2 else 0.0
        ),
        e2e_avg=_mean(e2es),
        e2e_min=min(e2es),
        e2e_max=max(e2es),
        e2e_stddev=_safe_stdev(e2es),
        gen_avg_ms=_mean(gen_times),
        gen_min_ms=min(gen_times),
        gen_max_ms=max(gen_times),
        tokens_avg=_mean(tokens),
        tokens_min=min(tokens),
        tokens_max=max(tokens),
        success_count=len(success),
        total_count=len(results),
        success_rate=(
            len(success) / len(results) if results else 0.0
        ),
        ttfb_samples=list(ttfbs),
        tps_samples=list(tps_list),
        e2e_samples=list(e2es),
        tokens_samples=list(tokens),
    )


# ─── 图表生成 ───


def _ensure_matplotlib():
    """懒加载 matplotlib"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_benchmark_results(
    results: List[TestRunResult],
    output_dir: str = ".",
) -> Dict[str, str]:
    """生成基准测试图表（针对 LLM API 部署性能分析设计）"""
    try:
        plt = _ensure_matplotlib()
    except ImportError:
        print(
            f"  {Fore.YELLOW}{WARN} "
            "跳过图表生成: matplotlib 不可用"
            f"{Style.RESET_ALL}"
        )
        return {}

    output_dir = _resolve(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    charts: Dict[str, str] = {}

    # 收集所有结果
    all_entries: List[Tuple[str, str, PromptStats]] = []
    for result in results:
        for prompt_name, stats in result.stats.items():
            all_entries.append((result.config.name, prompt_name, stats))

    if not all_entries:
        return charts

    # 1. 延迟对比柱状图（TTFB + 生成时间 + E2E）
    charts["latency_breakdown"] = _plot_latency_breakdown(
        plt, all_entries, output_dir
    )

    # 2. 吞吐量对比柱状图
    charts["throughput"] = _plot_throughput(plt, all_entries, output_dir)

    # 3. 延迟热力图
    charts["latency_heatmap"] = _plot_latency_heatmap(
        plt, all_entries, output_dir
    )

    # 4. 雷达图 -- 延迟、TPS、成功率综合评分
    charts["radar_overview"] = _plot_radar(
        plt, all_entries, output_dir
    )

    # 5. 延迟分布箱线图
    charts["latency_distribution"] = _plot_latency_boxplot(
        plt, all_entries, output_dir
    )

    # 6. 综合评分卡
    charts["scorecard"] = _plot_scorecard(
        plt, all_entries, output_dir
    )

    return charts


def _plot_latency_breakdown(
    plt,
    entries: List[Tuple[str, str, PromptStats]],
    output_dir: str,
) -> str:
    """延迟分解对比：TTFB vs 生成时间（堆叠柱状图）"""
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = []
    ttfb_vals = []
    gen_vals = []

    for cfg_name, prompt_name, stats in entries:
        labels.append(f"{cfg_name}\n{prompt_name}")
        ttfb_vals.append(stats.ttfb_avg)
        gen_vals.append(stats.gen_avg_ms)

    x = range(len(labels))
    width = 0.35

    ax.bar(
        [p - width / 2 for p in x],
        ttfb_vals,
        width,
        label="TTFB (first token latency)",
        color="#3498db",
    )
    ax.bar(
        [p + width / 2 for p in x],
        gen_vals,
        width,
        label="Generation Time (excl. TTFB)",
        color="#e67e22",
    )

    ax.set_xlabel("Test Item")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("API Latency Breakdown", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 在每个柱上标注数值
    for i, (t, g) in enumerate(zip(ttfb_vals, gen_vals)):
        ax.annotate(
            f"{t:.0f}",
            (i - width / 2, t),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )
        ax.annotate(
            f"{g:.0f}",
            (i + width / 2, g),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )

    filename = os.path.join(output_dir, "latency_breakdown.png")
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def _plot_throughput(
    plt,
    entries: List[Tuple[str, str, PromptStats]],
    output_dir: str,
) -> str:
    """吞吐量（TPS）对比"""
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = []
    tps_vals = []
    tokens_vals = []

    for cfg_name, prompt_name, stats in entries:
        labels.append(f"{cfg_name}\n{prompt_name}")
        tps_vals.append(stats.tps_avg)
        tokens_vals.append(stats.tokens_avg)

    x = range(len(labels))

    colors = [
        "#2ecc71" if t >= 50
        else "#f1c40f" if t >= 20
        else "#e74c3c"
        for t in tps_vals
    ]
    bars = ax.bar(x, tps_vals, color=colors)

    ax.set_xlabel("Test Item")
    ax.set_ylabel("TPS (tokens/sec)")
    ax.set_title("Throughput Comparison (higher is better)", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # 标注 TPS 和 token 数量
    for i, (bar, t, tok) in enumerate(
        zip(bars, tps_vals, tokens_vals)
    ):
        ax.annotate(
            f"{t:.1f} tok/s\n({tok:.0f} tokens)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )

    filename = os.path.join(output_dir, "throughput.png")
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def _plot_latency_heatmap(
    plt,
    entries: List[Tuple[str, str, PromptStats]],
    output_dir: str,
) -> str:
    """延迟热力图 -- 纵轴=配置名，横轴=测试套件，颜色=TTFB 值"""
    import numpy as np

    # 组织数据矩阵
    configs = list({name for name, _, _ in entries})
    prompts = list({p for _, p, _ in entries})
    configs.sort()
    prompts.sort()

    data: Dict[str, Dict[str, float]] = {}
    for cfg_name, prompt_name, stats in entries:
        data.setdefault(cfg_name, {})[prompt_name] = stats.ttfb_avg

    matrix = np.array(
        [
            [data.get(c, {}).get(p, 0) for p in prompts]
            for c in configs
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(prompts) * 2),
            max(4, len(configs) * 2),
        )
    )

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(prompts)))
    ax.set_yticks(range(len(configs)))
    ax.set_xticklabels(
        [_shorten(p, 20) for p in prompts],
        rotation=15,
        ha="right",
    )
    ax.set_yticklabels(
        [_shorten(c, 24) for c in configs]
    )
    ax.set_title("TTFB Latency Heatmap (ms)", fontweight="bold")

    # 标注每个单元格的数值
    for i in range(len(configs)):
        for j in range(len(prompts)):
            val = matrix[i, j]
            text_color = (
                "white" if val > matrix.max() * 0.6 else "black"
            )
            ax.text(
                j, i, f"{val:.0f}",
                ha="center", va="center",
                color=text_color, fontsize=10,
            )

    fig.colorbar(im, ax=ax, label="TTFB (ms)")
    plt.tight_layout()

    filename = os.path.join(output_dir, "latency_heatmap.png")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def _plot_radar(
    plt,
    entries: List[Tuple[str, str, PromptStats]],
    output_dir: str,
) -> str:
    """雷达图 -- 综合性能评估：延迟越低越好、TPS 越高越好、成功率越高越好"""
    import numpy as np

    # 聚合每个配置的统计
    config_stats: Dict[str, List[PromptStats]] = {}
    for cfg_name, prompt_name, stats in entries:
        config_stats.setdefault(cfg_name, []).append(stats)

    # 聚合指标
    config_data: Dict[str, Dict[str, float]] = {}
    for cfg_name, stats_list in config_stats.items():
        config_data[cfg_name] = {
            "ttfb": _mean([s.ttfb_avg for s in stats_list]),
            "tps": _mean([s.tps_avg for s in stats_list]),
            "e2e": _mean([s.e2e_avg for s in stats_list]),
            "success": _mean([s.success_rate for s in stats_list]),
        }

    # 归一化到 0~1（延迟取反：越低越好）
    ttfb_vals = [v["ttfb"] for v in config_data.values()]
    tps_vals = [v["tps"] for v in config_data.values()]
    e2e_vals = [v["e2e"] for v in config_data.values()]

    max_ttfb = max(ttfb_vals) if ttfb_vals else 1
    max_tps = max(tps_vals) if tps_vals else 1
    max_e2e = max(e2e_vals) if e2e_vals else 1

    categories = [
        "TTFB Latency\n(lower is better)",
        "TPS Throughput\n(higher is better)",
        "E2E Latency\n(lower is better)",
        "Success Rate\n(higher is better)",
    ]
    num_vars = len(categories)

    angles = [
        n / float(num_vars) * 2 * np.pi
        for n in range(num_vars)
    ]
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw=dict(polar=True)
    )

    colors = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
    ]
    for idx, (cfg_name, data) in enumerate(
        config_data.items()
    ):
        # 归一化
        values = [
            1 - data["ttfb"] / max_ttfb if max_ttfb else 0,
            data["tps"] / max_tps if max_tps else 0,
            1 - data["e2e"] / max_e2e if max_e2e else 0,
            data["success"],
        ]
        values += values[:1]  # 闭合

        ax.plot(
            angles, values,
            linewidth=2,
            linestyle="solid",
            label=cfg_name,
            color=colors[idx % len(colors)],
        )
        ax.fill(
            angles, values,
            alpha=0.15,
            color=colors[idx % len(colors)],
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Performance Radar",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.0),
    )

    filename = os.path.join(output_dir, "radar_overview.png")
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def _plot_latency_boxplot(
    plt,
    entries: List[Tuple[str, str, PromptStats]],
    output_dir: str,
) -> str:
    """延迟分布箱线图 -- 查看每次请求的 TTFB 波动"""
    config_samples: Dict[str, List[List[float]]] = {}
    config_labels: Dict[str, List[str]] = {}
    for cfg_name, prompt_name, stats in entries:
        config_samples.setdefault(cfg_name, []).append(
            stats.ttfb_samples
        )
        config_labels.setdefault(cfg_name, []).append(prompt_name)

    fig, ax = plt.subplots(figsize=(12, 6))

    all_data: List[List[float]] = []
    all_labels: List[str] = []
    for cfg_name in config_samples:
        for i, samples in enumerate(config_samples[cfg_name]):
            all_data.append(samples)
            all_labels.append(
                f"{cfg_name}\n{config_labels[cfg_name][i]}"
            )

    if all_data and any(d for d in all_data):
        bp = ax.boxplot(
            all_data,
            labels=[
                _shorten(
                    l.replace("\n", " / "), 25
                )
                for l in all_labels
            ],
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.6)

    ax.set_ylabel("TTFB (ms)")
    ax.set_title("TTFB Latency Distribution", fontweight="bold")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.grid(axis="y", alpha=0.3)

    filename = os.path.join(
        output_dir, "latency_distribution.png"
    )
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def _plot_scorecard(
    plt,
    entries: List[Tuple[str, str, PromptStats]],
    output_dir: str,
) -> str:
    """综合评分卡 -- 一目了然的表格样式输出图片"""
    # 聚合每个配置的综合指标
    config_agg: Dict[str, Dict[str, Any]] = {}
    for cfg_name, prompt_name, stats in entries:
        if cfg_name not in config_agg:
            config_agg[cfg_name] = {
                "count": 0,
                "ttfb_avg": [],
                "tps_avg": [],
                "e2e_avg": [],
                "success_rate": [],
            }
        config_agg[cfg_name]["count"] += 1
        config_agg[cfg_name]["ttfb_avg"].append(stats.ttfb_avg)
        config_agg[cfg_name]["tps_avg"].append(stats.tps_avg)
        config_agg[cfg_name]["e2e_avg"].append(stats.e2e_avg)
        config_agg[cfg_name]["success_rate"].append(
            stats.success_rate
        )

    # 计算极值用于评分
    max_e2e = (
        max(
            max(agg["e2e_avg"])
            for agg in config_agg.values()
        )
        if config_agg
        else 1
    )
    max_ttfb = (
        max(
            max(agg["ttfb_avg"])
            for agg in config_agg.values()
        )
        if config_agg
        else 1
    )
    max_tps = (
        max(
            max(agg["tps_avg"])
            for agg in config_agg.values()
        )
        if config_agg
        else 1
    )

    rows = []
    for cfg_name, agg in config_agg.items():
        avg_ttfb = _mean(agg["ttfb_avg"])
        avg_tps = _mean(agg["tps_avg"])
        avg_e2e = _mean(agg["e2e_avg"])
        avg_sr = _mean(agg["success_rate"])

        # 综合评分: TPS(40%) + 成功率(30%) + 延迟(30%,取反)
        score = (
            avg_tps / max_tps * 40
            + avg_sr * 30
            + (1 - avg_ttfb / max_ttfb) * 30
        )
        rows.append(
            {
                "name": cfg_name,
                "ttfb": avg_ttfb,
                "tps": avg_tps,
                "e2e": avg_e2e,
                "sr": avg_sr * 100,
                "score": score,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)

    fig, ax = plt.subplots(
        figsize=(10, max(3, len(rows) * 1.2 + 1))
    )
    ax.axis("off")

    col_widths = [0.25, 0.12, 0.12, 0.12, 0.12, 0.12]
    headers = [
        "Config",
        "TTFB (ms)",
        "TPS (tok/s)",
        "E2E (ms)",
        "Success (%)",
        "Score",
    ]

    # 表头背景
    col_x = 0.05
    for i, (w, h) in enumerate(zip(col_widths, headers)):
        ax.add_patch(
            plt.Rectangle(
                (col_x, 0.90), w, 0.1,
                facecolor="#2c3e50",
                edgecolor="none",
            )
        )
        ax.text(
            col_x + w / 2, 0.95, h,
            ha="center", va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )
        col_x += w

    # 表格行
    for row_idx, row in enumerate(rows):
        y = 0.85 - row_idx * 0.1
        col_x = 0.05

        vals = [
            row["name"],
            f"{row['ttfb']:.0f}",
            f"{row['tps']:.1f}",
            f"{row['e2e']:.0f}",
            f"{row['sr']:.1f}",
            f"{row['score']:.1f}",
        ]

        # 按评分着色
        if row_idx == 0:
            row_color = "#27ae60"
        elif row_idx < len(rows) // 2:
            row_color = "#f39c12"
        else:
            row_color = "#e74c3c"

        for i, (w, v) in enumerate(zip(col_widths, vals)):
            bg_color = (
                row_color
                if i == len(col_widths) - 1
                else "#ecf0f1"
            )
            ax.add_patch(
                plt.Rectangle(
                    (col_x, y - 0.08),
                    w, 0.08,
                    facecolor=bg_color,
                    edgecolor="#bdc3c7",
                )
            )
            text_color = "white" if i == len(col_widths) - 1 else "#2c3e50"
            font_w = "bold" if i == len(col_widths) - 1 else "normal"
            ax.text(
                col_x + w / 2, y - 0.04, v,
                ha="center", va="center",
                fontsize=9,
                color=text_color,
                fontweight=font_w,
            )
            col_x += w

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Scorecard",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    filename = os.path.join(output_dir, "scorecard.png")
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


# ─── 控制台输出 ───


def _print_header(config: APIConfig, rounds: int) -> None:
    """打印测试头部信息"""
    token = config.auth_token
    print()
    border = BOX_H * 60
    dash = BOX_DASH * 60
    print(f"  {Fore.CYAN}{border}{Style.RESET_ALL}")
    print(
        f"  {Fore.CYAN}{Style.BRIGHT}"
        "  Claude API 基准测试工具"
        f"{Style.RESET_ALL}"
    )
    print(f"  {Fore.CYAN}{border}{Style.RESET_ALL}")
    print(f"  配置:      {config.name}")
    print(f"  端点:      {config.base_url}")
    print(f"  模型:     {config.model}")
    print(f"  Token:     {_mask_token(token)}")
    print(f"  轮数:      {rounds}/prompt")
    print(f"  {Fore.CYAN}{dash}{Style.RESET_ALL}")
    print()


def _print_warmup(
    success: bool, error: Optional[str] = None
) -> None:
    """打印预热结果"""
    if success:
        print(f"  {Fore.GREEN}[预热]{Style.RESET_ALL} {CHECK} 连接成功")
    else:
        print(
            f"  {Fore.YELLOW}[预热]{Style.RESET_ALL}"
            f" {WARN} 失败: {error}"
        )
    print()


def _print_round_result(
    round_num: int, total: int, result: SingleTestResult
) -> None:
    """打印单轮测试结果"""
    if result.error:
        print(
            f"  [{result.prompt_name}] "
            f"Round {round_num}/{total} "
            f"{Fore.RED}{CROSS} {result.error}"
            f"{Style.RESET_ALL}"
        )
    else:
        ttfb_color = (
            Fore.GREEN
            if result.ttfb_ms < 500
            else Fore.YELLOW
            if result.ttfb_ms < 1500
            else Fore.RED
        )
        tps_color = (
            Fore.GREEN
            if result.tokens_per_second >= 50
            else Fore.YELLOW
            if result.tokens_per_second >= 20
            else Fore.RED
        )

        print(
            f"  [{result.prompt_name}] "
            f"Round {round_num}/{total}  {CHECK} "
            f"{ttfb_color}TTFB: {result.ttfb_ms:.0f}ms"
            f"{Style.RESET_ALL} | "
            f"{tps_color}{result.tokens_per_second:.1f} tok/s"
            f"{Style.RESET_ALL} | "
            f"E2E: {result.total_time_ms:.0f}ms | "
            f"Tokens: {result.output_tokens}"
        )


def _print_prompt_summary(stats: PromptStats) -> None:
    """打印 Prompt 统计摘要"""
    if stats.success_count == 0:
        print(f"  {Fore.RED}{CROSS} 全部失败{Style.RESET_ALL}")
        return

    ttfb_color = (
        Fore.GREEN
        if stats.ttfb_avg < 500
        else Fore.YELLOW
        if stats.ttfb_avg < 1500
        else Fore.RED
    )
    tps_color = (
        Fore.GREEN
        if stats.tps_avg >= 50
        else Fore.YELLOW
        if stats.tps_avg >= 20
        else Fore.RED
    )

    print()
    dash = BOX_DASH * 70
    print(f"  {Fore.CYAN}{dash}{Style.RESET_ALL}")
    print(
        f"  {Style.BRIGHT}  {stats.prompt_name}"
        f"{Style.RESET_ALL}"
    )
    print(f"  {Fore.CYAN}{dash}{Style.RESET_ALL}")

    print(f"  {Style.DIM}TTFB (首Token延迟):" f"{Style.RESET_ALL}")
    print(
        f"    {ttfb_color}平均: {stats.ttfb_avg:.0f}ms"
        f"{Style.RESET_ALL} | "
        f"最小: {stats.ttfb_min:.0f}ms | "
        f"最大: {stats.ttfb_max:.0f}ms | "
        f"标准差: {stats.ttfb_stddev:.0f}ms"
    )
    print(
        f"    P95: {stats.ttfb_p95:.0f}ms | "
        f"P99: {stats.ttfb_p99:.0f}ms"
    )

    print(
        f"  {Style.DIM}TPS (生成速度):"
        f"{Style.RESET_ALL}"
    )
    print(
        f"    {tps_color}平均: {stats.tps_avg:.1f} tok/s"
        f"{Style.RESET_ALL} | "
        f"最小: {stats.tps_min:.1f} | "
        f"最大: {stats.tps_max:.1f} | "
        f"标准差: {stats.tps_stddev:.1f}"
    )

    print(
        f"  {Style.DIM}端到端延迟:"
        f"{Style.RESET_ALL}"
    )
    print(
        f"    平均: {stats.e2e_avg:.0f}ms | "
        f"最小: {stats.e2e_min:.0f}ms | "
        f"最大: {stats.e2e_max:.0f}ms"
    )

    print(
        f"  {Style.DIM}生成时间 (不含TTFB):"
        f"{Style.RESET_ALL}"
    )
    print(
        f"    平均: {stats.gen_avg_ms:.0f}ms | "
        f"最小: {stats.gen_min_ms:.0f}ms | "
        f"最大: {stats.gen_max_ms:.0f}ms"
    )

    print(f"  {Style.DIM}Token 统计:" f"{Style.RESET_ALL}")
    print(
        f"    平均: {stats.tokens_avg:.1f} | "
        f"最小: {stats.tokens_min} | "
        f"最大: {stats.tokens_max}"
    )

    print(f"  {Style.DIM}成功率:" f"{Style.RESET_ALL}")
    print(
        f"    {stats.success_count}/{stats.total_count}"
        f" ({stats.success_rate * 100:.1f}%)"
    )


def _print_summary_table(all_results: List[TestRunResult]) -> None:
    """打印汇总表格"""
    print()
    border = BOX_H * 116
    dash = BOX_DASH * 116
    sep = BOX_V

    print(f"  {Fore.CYAN}{border}{Style.RESET_ALL}")
    print(
        f"  {Style.BRIGHT}  基准测试汇总{Style.RESET_ALL}"
    )
    print(f"  {Fore.CYAN}{border}{Style.RESET_ALL}")

    header = (
        f"  {'配置':<16} {sep} "
        f"{'Prompt':<36} {sep} "
        f"{'TTFB (ms)':<30} {sep} "
        f"{'TPS':<24} {sep} "
        f"{'成功率':<10}"
    )
    sub_header = (
        f"  {'':16} {sep} "
        f"{'':36} {sep} "
        f"{'平均':>6} {'P95':>6} "
        f"{'P99':>6} {'标准差':>6} {sep} "
        f"{'平均':>6} {'最小':>6} "
        f"{'最大':>6} {'标准差':>6} {sep} "
        f"{'':>8}"
    )
    print(f"  {Style.DIM}{header}{Style.RESET_ALL}")
    print(f"  {Style.DIM}{sub_header}{Style.RESET_ALL}")
    print(f"  {Fore.BLUE}{dash}{Style.RESET_ALL}")

    for result in all_results:
        if result.run_error:
            print(
                f"  {result.config.name:<16} {sep} "
                f"{'运行错误':<36} {sep} "
                f"{Fore.RED}"
                f"{_shorten(result.run_error, 30):<30}"
                f"{Style.RESET_ALL} {sep} "
                f"{'':<24} {sep} "
                f"{'0.0':>8}"
            )
            continue

        for prompt_name, stats in result.stats.items():
            label = _shorten(prompt_name, 36)

            if stats.success_count == 0:
                print(
                    f"  {result.config.name:<16} {sep} "
                    f"{label:<36} {sep} "
                    f"{Fore.RED}{'全部失败':^30}"
                    f"{Style.RESET_ALL} {sep} "
                    f"{'':<24} {sep} "
                    f"{'0.0':>8}"
                )
                continue

            ttfb_c = (
                Fore.GREEN
                if stats.ttfb_avg < 500
                else Fore.YELLOW
                if stats.ttfb_avg < 1500
                else Fore.RED
            )
            tps_c = (
                Fore.GREEN
                if stats.tps_avg >= 50
                else Fore.YELLOW
                if stats.tps_avg >= 20
                else Fore.RED
            )

            ttfb_line = (
                f"{ttfb_c}{stats.ttfb_avg:>6.0f} "
                f"{stats.ttfb_p95:>6.0f} "
                f"{stats.ttfb_p99:>6.0f} "
                f"{stats.ttfb_stddev:>6.0f}"
                f"{Style.RESET_ALL}"
            )
            tps_line = (
                f"{tps_c}{stats.tps_avg:>6.1f} "
                f"{stats.tps_min:>6.1f} "
                f"{stats.tps_max:>6.1f} "
                f"{stats.tps_stddev:>6.1f}"
                f"{Style.RESET_ALL}"
            )

            print(
                f"  {result.config.name:<16} {sep} "
                f"{label:<36} {sep} "
                f"{ttfb_line} {sep} "
                f"{tps_line} {sep} "
                f"{stats.success_rate * 100:>7.1f}%"
            )

    print(f"  {Fore.BLUE}{dash}{Style.RESET_ALL}")

    total_tests = sum(r.total_tests for r in all_results)
    success_tests = sum(
        r.successful_tests for r in all_results
    )

    if total_tests > 0:
        print(f"  总测试数: {total_tests}")
        print(f"  成功: {success_tests}")
        print(f"  失败: {total_tests - success_tests}")
        print(
            f"  总成功率:"
            f" {success_tests / total_tests * 100:.1f}%"
        )
    print()


def _print_json_report(all_results: List[TestRunResult]) -> None:
    """打印 JSON 格式报告"""
    report = {
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        ),
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


def _print_charts_summary(
    charts: Dict[str, str], output_dir: str
) -> None:
    """打印图表生成摘要"""
    if not charts:
        return

    chart_names = {
        "latency_breakdown": "延迟分解对比",
        "throughput": "吞吐量对比",
        "latency_heatmap": "延迟热力图",
        "radar_overview": "综合性能雷达图",
        "latency_distribution": "延迟分布箱线图",
        "scorecard": "综合评分卡",
    }

    print()
    border = BOX_H * 50
    print(f"  {Fore.CYAN}{border}{Style.RESET_ALL}")
    print(
        f"  {Style.BRIGHT}  生成的图表{Style.RESET_ALL}"
    )
    print(f"  {Fore.CYAN}{border}{Style.RESET_ALL}")

    for chart_type, filename in charts.items():
        if os.path.exists(filename):
            name = chart_names.get(chart_type, chart_type)
            print(
                f"  {Fore.GREEN}{CHECK}{Style.RESET_ALL} "
                f"{name}: {os.path.basename(filename)}"
            )


# ─── 主流程 ───


def run_benchmark(
    config: APIConfig,
    rounds: int,
    no_warmup: bool = False,
) -> TestRunResult:
    """执行单个配置的基准测试"""
    _print_header(config, rounds)

    client = _create_client(config)

    # 预热请求
    if not no_warmup:
        warmup_result = run_single_test(
            client,
            config.model,
            "你好，请简短回复。",
            "预热",
            10,
        )
        _print_warmup(
            warmup_result.error is None,
            warmup_result.error,
        )
    else:
        print(f"  {Fore.YELLOW}[预热]{Style.RESET_ALL} 跳过")
        print()

    # 正式测试
    all_stats: Dict[str, PromptStats] = {}
    total_start = time.perf_counter()
    total_tests = 0
    successful_tests = 0

    for suite_name, suite in TEST_SUITES.items():
        dash = BOX_DASH * 60
        print(f"\n  {Fore.CYAN}{dash}{Style.RESET_ALL}")
        print(
            f"  {Style.BRIGHT}"
            f"  {suite['name']} (套件: {suite_name})"
            f"{Style.RESET_ALL}"
        )
        print(f"  {Fore.CYAN}{dash}{Style.RESET_ALL}")

        for prompt_cfg in suite["prompts"]:
            prompt_text = prompt_cfg["text"]
            max_tokens = prompt_cfg["max_tokens"]
            prompt_name = (
                f"{suite['name']} - "
                f"{_shorten(prompt_text, 30)}"
            )

            results: List[SingleTestResult] = []
            for i in range(1, rounds + 1):
                result = run_single_test(
                    client,
                    config.model,
                    prompt_text,
                    prompt_name,
                    max_tokens,
                )
                result.round_num = i
                _print_round_result(i, rounds, result)
                results.append(result)
                if result.error is None:
                    successful_tests += 1
                total_tests += 1

            all_stats[prompt_name] = _compute_prompt_stats(
                results, prompt_name, prompt_text, max_tokens
            )
            _print_prompt_summary(all_stats[prompt_name])
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
        failed_tests=total_tests - successful_tests,
    )


def main() -> None:
    """主入口"""
    if HAS_COLORAMA:
        import colorama

        colorama.init()

    parser = argparse.ArgumentParser(
        description="Claude API 基准测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置文件
  python claude_api_bench.py

  # 指定配置文件
  python claude_api_bench.py --config configs.json

  # 交互式配置输入
  python claude_api_bench.py --interactive

  # 指定测试轮数
  python claude_api_bench.py --rounds 5

  # 生成图表
  python claude_api_bench.py --charts

  # JSON 输出
  python claude_api_bench.py --json
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径 (默认: ./example_config.json)",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=DEFAULT_ROUNDS,
        help="每个测试项的测试轮数 (默认: 3)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过预热请求",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="交互式配置输入",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="保存配置到文件",
    )
    parser.add_argument(
        "--charts",
        action="store_true",
        help="生成可视化图表",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="图表输出目录 (默认: 当前目录)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 格式报告",
    )

    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("轮数必须 >= 1")

    # 加载配置
    configs: List[APIConfig] = []
    config_path = _resolve(args.config)

    if args.interactive:
        configs = _input_config_interactive()
    else:
        configs = load_config_from_json(config_path)
        if not configs:
            print(
                f"  {Fore.YELLOW}{WARN} "
                f"未找到有效配置: {config_path}"
                f"{Style.RESET_ALL}"
            )
            if not sys.stdin.isatty():
                print(
                    f"  {Fore.RED}{CROSS} "
                    f"非交互式会话，无法输入配置"
                    f"{Style.RESET_ALL}"
                )
                sys.exit(1)
            if (
                input("  使用交互式配置？(y/n) [y]: ")
                .strip()
                .lower()
                != "n"
            ):
                configs = _input_config_interactive()
            else:
                print(
                    f"  {Fore.RED}{CROSS} "
                    f"无可用的配置，退出"
                    f"{Style.RESET_ALL}"
                )
                sys.exit(1)

    # 保存配置
    if args.save_config:
        save_config_to_json(configs, config_path)

    output_dir = _resolve(args.output_dir)

    # 开始测试
    print()
    border = BOX_H * 60
    print(f"  {Fore.CYAN}{border}{Style.RESET_ALL}")
    print(
        f"  {Style.BRIGHT}  开始基准测试"
        f"{Style.RESET_ALL}"
    )
    print(f"  {Fore.CYAN}{border}{Style.RESET_ALL}")
    print(f"  配置数: {len(configs)}")
    print(f"  轮数: {args.rounds}")
    print()

    all_results: List[TestRunResult] = []

    for i, config in enumerate(configs, 1):
        border = BOX_H * 60
        print(f"\n{Fore.GREEN}{border}{Style.RESET_ALL}")
        print(f"  [{i}/{len(configs)}] 测试: {config.name}")
        print(f"{Fore.GREEN}{border}{Style.RESET_ALL}\n")

        try:
            result = run_benchmark(
                config, args.rounds, args.no_warmup
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(
                f"  {Fore.RED}{CROSS} "
                f"{config.name} 测试中止: {e}"
                f"{Style.RESET_ALL}"
            )
            result = TestRunResult(
                config=config,
                rounds=args.rounds,
                stats={},
                run_error=str(e),
            )
        all_results.append(result)

    # 打印汇总表格
    _print_summary_table(all_results)

    # 生成图表
    if args.charts:
        print()
        charts = _plot_benchmark_results(all_results, output_dir)
        _print_charts_summary(charts, output_dir)

    # JSON 输出
    if args.json:
        print()
        _print_json_report(all_results)

    # 退出码
    total_success = sum(
        r.successful_tests for r in all_results
    )
    total_tests = sum(r.total_tests for r in all_results)
    if total_tests == 0 or total_success == 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            f"\n  {Fore.YELLOW}{WARN} "
            f"测试被用户中断"
            f"{Style.RESET_ALL}"
        )
        sys.exit(130)
