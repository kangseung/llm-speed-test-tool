# Claude Code Benchmark Tool

Claude Code Benchmark Tool - benchmark Anthropic or Claude Code compatible endpoints with charts and metrics.

## Features

- **Multi-config batch testing**: Test multiple API configurations from JSON
- **Interactive config input**: Input config information interactively
- **Claude Code config support**: Read `~/.claude/settings.json` directly
- **Detailed statistics**:
  - TTFB (Time To First Token)
  - TPS (Tokens Per Second)
  - E2E Latency
  - P95/P99 Percentiles
  - Success Rate
- **Visualization charts**:
  - TTFB comparison bar chart
  - TPS comparison bar chart
  - E2E latency comparison
  - Success rate comparison
  - Overall comparison chart

## Installation

```bash
pip install anthropic colorama matplotlib
```

## Usage

### 1. Quick Test (Single Config)

```bash
python claude_api_bench.py
```

Uses default config `~/.claude/settings.json`, 3 rounds default.

### 2. JSON Config File

Create a config file `configs.json`:

```json
[
  {
    "name": "Anthropic Cloud",
    "base_url": "https://api.anthropic.com",
    "auth_token": "sk-xxx",
    "model": "claude-sonnet-4-20250514"
  },
  {
    "name": "Claude Code Proxy",
    "base_url": "https://your-proxy.example.com",
    "auth_token": "sk-xxx",
    "model": "claude-sonnet-4-20250514"
  }
]
```

Run test:

```bash
python claude_api_bench.py -c configs.json
```

### 3. Interactive Config Input

```bash
python claude_api_bench.py -i
```

Follow the prompts to enter config information.

## Shell Scripts

### `quick-test.sh` - Quick Single Round Test

```bash
./quick-test.sh
```

Quick test, skip warmup, 1 round.

### `full-test.sh` - Full Test

```bash
./full-test.sh
```

Full test including warmup, 3 rounds, generate charts.

### `multi-test.sh` - Multi-config Batch Test

```bash
./multi-test.sh configs.json 5
```

Parameters:
- `$1`: Config file path
- `$2`: Test rounds (default: 3)

### `chart-test.sh` - Chart Generation Test

```bash
./chart-test.sh configs.json
```

Parameters:
- `$1`: Config file path

Generate detailed comparison charts in `./output`.

### `json-report.sh` - JSON Report Test

```bash
./json-report.sh configs.json 5
```

Parameters:
- `$1`: Config file path
- `$2`: Test rounds (default: 3)

Output JSON format test report.

### `all-in-one.sh` - All-in-One Test

```bash
./all-in-one.sh configs.json 5 ./output
```

Parameters:
- `$1`: Config file path
- `$2`: Test rounds (default: 3)
- `$3`: Output directory (default: ./output)

Execute full test, generate charts, and output JSON report.

## Output

### Console Output

Results displayed in table format:

- Prompt name
- TTFB (avg/min/max/p95/p99)
- TPS (avg/min/max)
- E2E Latency
- Success Rate

Color coding:
- **Green**: Excellent (< 500ms TTFB, > 50 tok/s)
- **Yellow**: Average (500-1500ms TTFB, 20-50 tok/s)
- **Red**: Poor (> 1500ms TTFB, < 20 tok/s)

### Chart Output

Generated charts:

1. `ttfb_comparison.png` - TTFB comparison
2. `tps_comparison.png` - TPS comparison
3. `e2e_comparison.png` - E2E latency comparison
4. `success_rate.png` - Success rate comparison
5. `overall_comparison.png` - Overall comparison

By default, `python claude_api_bench.py --charts` writes images to the current working directory.
The wrapper scripts `full-test.sh`, `chart-test.sh`, and `all-in-one.sh` write them into `./output` under the project directory.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-c, --config` | Config file path | `~/.claude/settings.json` |
| `-r, --rounds` | Test rounds per prompt | 3 |
| `--no-warmup` | Skip warmup | false |
| `-i, --interactive` | Interactive config input | false |
| `--save-config` | Save config to file | false |
| `--charts` | Generate charts | false |
| `--output-dir` | Chart output directory | Current directory |
| `--json` | Output JSON report | false |

## Config File Format

The tool currently targets Anthropic / Claude Code compatible endpoints. It does not send OpenAI Chat Completions requests.

### Single Config

```json
{
  "name": "my-api",
  "base_url": "https://api.anthropic.com",
  "auth_token": "sk-xxx",
  "model": "claude-sonnet-4-20250514"
}
```

### Multiple Configs

```json
[
  {
    "name": "api-1",
    "base_url": "https://api.anthropic.com",
    "auth_token": "sk-xxx",
    "model": "claude-sonnet-4-20250514"
  },
  {
    "name": "api-2",
    "base_url": "https://your-proxy.example.com",
    "auth_token": "sk-xxx",
    "model": "claude-sonnet-4-20250514"
  }
]
```

## License

MIT License
