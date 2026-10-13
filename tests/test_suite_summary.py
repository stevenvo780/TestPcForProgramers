"""Regression checks for summary parsing; never run the real benchmarks.

Only the helper is extracted for unit checks. The smoke test copies just the
orchestrator to a temporary directory containing print-only producer fixtures.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


SUITE = Path(__file__).resolve().parents[1] / "run_bench_suite.sh"
BASH = shutil.which("bash")


def extract_parse_field(source):
    """Fail closed rather than accidentally sourcing top-level workloads."""
    start = "parse_field() {\n"
    end = "\nPY\n}\n\n# Kernel build bench\n"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("parse_field boundaries changed or are ambiguous")
    _, remainder = source.split(start)
    body, _ = remainder.split(end)
    if "\n}" in body or re.search(r"(?m)^[A-Za-z_][A-Za-z_0-9]*\(\)", body):
        raise ValueError("unexpected shell boundary in parse_field")
    return start + body + "\nPY\n}\n"


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(BASH, "bash is required")
        self.temp = tempfile.TemporaryDirectory(prefix="suite-summary-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # No inherited BASH_ENV, producer settings, PYTHONPATH or shell functions.
        self.env = {
            "PATH": str(self.bin),
            "HOME": str(self.root),
            "TMPDIR": str(self.root),
            "LC_ALL": "C",
            "PYTHONUTF8": "1",
            "PYTHONNOUSERSITE": "1",
        }
        (self.bin / "python3").symlink_to(sys.executable)

    def run_bash(self, args):
        return subprocess.run(
            [BASH, "--noprofile", "--norc", *args],
            cwd=self.root, env=self.env, text=True, encoding="utf-8",
            capture_output=True, timeout=15,
        )

    def parse(self, data, key):
        helper = extract_parse_field(SUITE.read_text(encoding="utf-8"))
        script = helper + '\nparse_field "$1" "$2"\n'
        result = self.run_bash(["-c", script, "parse-field-test", data, key])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return result.stdout

    def test_real_schema_values(self):
        cases = [
            ({"kernel_version": "6.12.1"}, "kernel_version", "6.12.1"),
            ({"results": {"gflops_avg": 123.5}}, "results.gflops_avg", "123.5"),
            ({"results": {"gbps_avg": 0}}, "results.gbps_avg", "0"),
            ({"cold": {"elapsed": "0:12.34"}}, "cold.elapsed", "0:12.34"),
            ({"results": {"write": {"throughput_avg_bps": 1048576}}},
             "results.write.throughput_avg_bps", "1048576"),
            ({"results": {"read": {"throughput_avg_bps": 2097152}}},
             "results.read.throughput_avg_bps", "2097152"),
        ]
        for obj, key, expected in cases:
            with self.subTest(key=key):
                self.assertEqual(self.parse(json.dumps(obj), key), expected + "\n")

    def test_strings_remain_data(self):
        value = "café 雪 'quoted' \\\"double\\\" \\path\\ $HOME $(printf bad > injected) `printf bad > injected`\nnext line"
        self.assertEqual(
            self.parse(json.dumps({"value": value}, ensure_ascii=False), "value"),
            value + "\n",
        )
        self.assertFalse((self.root / "injected").exists())

    def test_pretty_printed_json(self):
        self.assertEqual(self.parse('{\n  "value": 42\n}', "value"), "42\n")

    def test_existing_stringification(self):
        for value, expected in [(None, "None"), (True, "True"), ([], "[]"), ({}, "{}")]:
            with self.subTest(value=value):
                self.assertEqual(self.parse(json.dumps({"value": value}), "value"), expected + "\n")

    def test_invalid_or_missing_fields_stay_silent(self):
        cases = [("not json", "value"), ("", "value"), ("{}", "value"),
                 ('{"results": {}}', "results.gflops_avg"),
                 ('{"results": null}', "results.gflops_avg")]
        for data, key in cases:
            with self.subTest(data=data):
                self.assertEqual(self.parse(data, key), "")

    def test_extraction_rejects_changed_or_duplicate_boundaries(self):
        source = SUITE.read_text(encoding="utf-8")
        for changed in [source.replace("parse_field() {", "parse_field () {"),
                        source + source, source.replace("# Kernel build bench", "# changed")]:
            with self.subTest(source=changed[:30]):
                with self.assertRaises(ValueError):
                    extract_parse_field(changed)

    def test_bash_syntax(self):
        result = self.run_bash(["-n", str(SUITE)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_summary_with_print_only_producers(self):
        # Only these harmless external tools are available to the orchestrator.
        # nproc, Docker, make, sync, downloaders and privilege tools are absent.
        for name in ["dirname", "mkdir", "tee", "sed", "tail"]:
            executable = shutil.which(name)
            self.assertIsNotNone(executable, name)
            (self.bin / name).symlink_to(executable)
        script = self.root / "run_bench_suite.sh"
        script.write_bytes(SUITE.read_bytes())
        fixtures = {
            "cpu_matmul_bench.py": {"results": {"gflops_avg": 123.5}},
            "memory_bandwidth_bench.py": {"results": {"gbps_avg": 45.25}},
            "disk_io_bench.py": {"results": {
                "write": {"throughput_avg_bps": 1048576},
                "read": {"throughput_avg_bps": 2097152},
            }},
        }
        for filename, data in fixtures.items():
            # These files contain only print statements, never original producers.
            text = "print('fixture producer')\nprint(" + repr("BENCH_JSON: " + json.dumps(data)) + ")\n"
            (self.root / filename).write_text(text, encoding="utf-8")
        self.assertFalse((self.root / "kernel-build-bench.sh").exists())
        self.assertEqual({p.name for p in self.root.iterdir()},
                         {"bin", script.name, *fixtures})
        output = self.root / "output"
        self.env.update({
            "OUT_DIR": str(output), "DISK_TARGET": str(output), "MAKE_JOBS": "1",
            "OUT_JSON_PATH": str(output / "kernel.json"),
            "CPU_OUT_JSON": str(output / "cpu.json"),
            "MEM_OUT_JSON": str(output / "memory.json"),
            "DISK_OUT_JSON": str(output / "disk.json"),
            "CPU_SIZE": "2", "CPU_REPEATS": "1", "CPU_DTYPE": "float32",
            "MEM_SIZE_MB": "1", "MEM_REPEATS": "1", "MEM_DTYPE": "uint8",
            "DISK_SIZE_MB": "1", "DISK_REPEATS": "1",
        })
        result = self.run_bash([str(script)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "", result.stdout)
        summary = [line for line in result.stdout.splitlines() if line.startswith("- ")]
        self.assertEqual(summary, [
            "- Kernel: skip (script no encontrado)",
            "- CPU GEMM 2x2: ~123.5 GFLOPS",
            "- Mem BW 1MB copy: ~45.25 GB/s",
            "- Disk IO 1MB: write=1.0 MiB/s, read=2.0 MiB/s",
        ])
        self.assertEqual({p.name for p in output.iterdir()}, {".suite_last.jsonl"})
        self.assertFalse((output / ".disk_io_bench.tmp").exists())


if __name__ == "__main__":
    unittest.main()
