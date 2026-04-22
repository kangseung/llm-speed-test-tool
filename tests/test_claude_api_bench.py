import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import claude_api_bench as bench


def make_prompt_stats(name: str = "short") -> bench.PromptStats:
    return bench.PromptStats(
        prompt_name=name,
        prompt_text="prompt",
        ttfb_avg=100.0,
        ttfb_min=90.0,
        ttfb_max=110.0,
        ttfb_stddev=10.0,
        ttfb_p95=109.0,
        ttfb_p99=110.0,
        tps_avg=50.0,
        tps_min=45.0,
        tps_max=55.0,
        tps_stddev=5.0,
        e2e_avg=500.0,
        e2e_min=480.0,
        e2e_max=520.0,
        e2e_stddev=20.0,
        gen_avg_ms=400.0,
        gen_min_ms=390.0,
        gen_max_ms=410.0,
        tokens_avg=120.0,
        tokens_min=110,
        tokens_max=130,
        success_count=3,
        total_count=3,
        success_rate=1.0,
        ttfb_samples=[90.0, 100.0, 110.0],
    )


class ConfigTests(unittest.TestCase):
    def test_default_config_path_points_to_example_config(self):
        self.assertTrue(bench.DEFAULT_CONFIG_PATH.endswith("example_config.json"))
        self.assertTrue(os.path.isabs(bench.DEFAULT_CONFIG_PATH))

    def test_resolve_path_expands_user(self):
        resolved = bench.resolve_path("~/claude-bench.json")
        self.assertFalse(resolved.startswith("~"))
        self.assertTrue(os.path.isabs(resolved))

    def test_load_config_from_json_ignores_incomplete_single_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write('{"name": "broken", "base_url": "https://api.anthropic.com"}')

            self.assertEqual(bench.load_config_from_json(config_path), [])

    def test_load_config_from_json_supports_claude_settings_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "settings.json")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(
                    """
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://proxy.example.com",
    "ANTHROPIC_AUTH_TOKEN": "secret-token",
    "ANTHROPIC_MODEL": "claude-sonnet-4-20250514"
  }
}
                    """.strip()
                )

            configs = bench.load_config_from_json(config_path)

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].base_url, "https://proxy.example.com")
        self.assertEqual(configs[0].auth_token, "secret-token")

    def test_save_config_to_json_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "nested", "configs.json")
            config = bench.APIConfig(
                name="demo",
                base_url="https://api.anthropic.com",
                auth_token="secret",
                model="claude-sonnet-4-20250514",
            )

            with redirect_stdout(io.StringIO()):
                bench.save_config_to_json([config], output_path)

            self.assertTrue(os.path.exists(output_path))


class ReportingTests(unittest.TestCase):
    def test_print_prompt_summary_marks_all_failed(self):
        stats = bench.PromptStats(
            prompt_name="broken",
            prompt_text="prompt",
            success_count=0,
            total_count=2,
            success_rate=0.0,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            bench.print_prompt_summary(stats)

        self.assertIn("All Failed", output.getvalue())

    def test_print_summary_table_marks_all_failed_rows(self):
        config = bench.APIConfig(
            name="demo",
            base_url="https://api.anthropic.com",
            auth_token="secret",
            model="claude-sonnet-4-20250514",
        )
        stats = bench.PromptStats(
            prompt_name="broken",
            prompt_text="prompt",
            success_count=0,
            total_count=2,
            success_rate=0.0,
        )
        result = bench.TestRunResult(
            config=config,
            rounds=2,
            stats={"broken": stats},
            total_tests=2,
            successful_tests=0,
            failed_tests=2,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            bench.print_summary_table([result])

        self.assertIn("All Failed", output.getvalue())


class ChartTests(unittest.TestCase):
    def test_plot_benchmark_results_creates_files_in_output_dir(self):
        config = bench.APIConfig(
            name="demo",
            base_url="https://api.anthropic.com",
            auth_token="secret",
            model="claude-sonnet-4-20250514",
        )
        result = bench.TestRunResult(
            config=config,
            rounds=3,
            stats={
                "short": make_prompt_stats("short"),
                "medium": make_prompt_stats("medium"),
            },
            total_tests=6,
            successful_tests=6,
            failed_tests=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            charts = bench.plot_benchmark_results([result], temp_dir)

            self.assertTrue(charts)
            for chart_path in charts.values():
                self.assertTrue(chart_path.startswith(temp_dir))
                self.assertTrue(os.path.exists(chart_path))


if __name__ == "__main__":
    unittest.main()
