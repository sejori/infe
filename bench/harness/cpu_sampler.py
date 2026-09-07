#!/usr/bin/env python3
"""Psutil-based CPU sampler for the benchmark harness (D4).

Runs alongside e2e_tool_stream.py, sampling CPU% of the API-server PID
inside the Docker container every 1 second. Writes one float per line
to the output file. Replaces the docker-stats approach that only got
2-5 samples per run (docker stats --no-stream takes 1-2s per call).

Usage:
    python3 cpu_sampler.py --container-name infe-vllm-stock --output out.cpu.txt
    python3 cpu_sampler.py --container-name infe-vllm-stock --output out.cpu.txt --interval 1.0
"""
import argparse
import subprocess
import sys
import time

def get_container_pid(container_name: str) -> int | None:
    """Get the main PID inside the container via docker inspect."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", container_name],
            capture_output=True, text=True, check=False, timeout=10,
        )
        pid = int(result.stdout.strip())
        return pid if pid > 0 else None
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_cpu_percent(pid: int) -> float | None:
    """Read CPU% for a PID from /proc/stat (no psutil dependency).

    Uses /proc/<pid>/stat fields utime(14) + stime(15) to compute total
    CPU ticks, combined with /proc/uptime for wall time. Returns percent
    of one core.
    """
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            fields = f.read().split()
        utime = int(fields[13])
        stime = int(fields[14])
        total_ticks = utime + stime

        with open("/proc/uptime", "r") as f:
            uptime_s = float(f.read().split()[0])

        # Clock ticks per second (usually 100 on Linux)
        clk_tck = 100  # sysconf is not portable across container boundaries

        return total_ticks / clk_tck, uptime_s
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None


def sample_loop(container_name: str, output: str, interval: float = 1.0):
    """Sample CPU% every `interval` seconds until the container disappears."""
    prev = None
    samples = []

    with open(output, "w") as f:
        while True:
            pid = get_container_pid(container_name)
            if pid is None:
                break  # Container gone → stop

            cur = get_cpu_percent(pid)
            if cur is not None and prev is not None:
                cpu_ticks, wall_s = cur
                prev_ticks, prev_wall = prev
                dt_wall = wall_s - prev_wall
                dt_cpu = cpu_ticks - prev_ticks
                if dt_wall > 0:
                    cpu_pct = (dt_cpu / dt_wall) * 100.0
                    cpu_pct = max(0.0, cpu_pct)
                    f.write(f"{cpu_pct:.1f}\n")
                    f.flush()
                    samples.append(cpu_pct)
            prev = cur
            time.sleep(interval)

    if samples:
        mean = sum(samples) / len(samples)
        print(f"cpu samples: {len(samples)}  mean: {mean:.0f}%", file=sys.stderr)
    else:
        print("cpu samples: 0", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Psutil-free CPU sampler for Docker container")
    ap.add_argument("--container-name", required=True, help="Docker container name")
    ap.add_argument("--output", required=True, help="Output file (one float per line)")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval seconds (default 1.0)")
    a = ap.parse_args()
    sample_loop(a.container_name, a.output, a.interval)


if __name__ == "__main__":
    main()
