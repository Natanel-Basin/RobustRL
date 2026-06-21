"""Train the baseline (ppo_agent.py) and robust (trainer.py) agents across several seeds.

Run it ONCE instead of typing one command per seed. Seeds are positional:

    python wrapper.py 1 2 3 4 5
    python wrapper.py 1 2 3 --jobs 2

Auto-pick N FREE GPUs (one run per GPU, queried via nvidia-smi):

    python wrapper.py 1 2 3 4 5 --num-gpus 4

...or list specific GPU ids yourself:

    python wrapper.py 1 2 3 4 5 --gpus 0,1,2,3

To forward extra args to the TRAINING scripts (they share helper.Args), put them
after a literal `--` separator:

    python wrapper.py 1 2 3 4 5 --num-gpus 4 -- --total_timesteps 5000000 --no-track

Everything before `--` configures this runner; everything after `--` is passed
verbatim to both ppo_agent.py and trainer.py. With more than one job, runs execute
concurrently and each one's output goes to train_logs/<run>.log.
"""
import argparse
import os
import subprocess
import sys
import time


def query_free_gpus(mem_threshold_mb):
    """Return physical GPU indices (as strings) whose used memory is below the
    threshold, per nvidia-smi. Empty if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    free = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if float(parts[1]) < mem_threshold_mb:
                free.append(parts[0])
        except ValueError:
            continue
    return free


def main():
    # Split our own args from the args to forward to the training scripts.
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        own_argv, forward = argv[:sep], argv[sep + 1:]
    else:
        own_argv, forward = argv, []

    parser = argparse.ArgumentParser(
        description="Train baseline + robust agents over multiple seeds.",
        epilog="Put args for the training scripts after a literal --, e.g. "
               "`python wrapper.py 1 2 3 -- --no-track`.")
    parser.add_argument("seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help="seeds to train, e.g. `1 2 3 4 5`")
    parser.add_argument("--scripts", default="ppo_agent.py,trainer.py",
                        help="comma-separated training scripts (default: ppo_agent.py,trainer.py)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="concurrent runs (default: #GPUs if --num-gpus/--gpus given, else 1)")
    parser.add_argument("--num-gpus", type=int, default=5,
                        help="auto-pick this many FREE GPUs (via nvidia-smi). Alternative to --gpus.")
    parser.add_argument("--gpus", default="",
                        help="explicit comma-separated GPU ids to use, e.g. 0,1,2,3 "
                             "(overrides --num-gpus). Empty + no --num-gpus = inherit the environment.")
    parser.add_argument("--gpu-mem-threshold", type=int, default=1000,
                        help="a GPU counts as free if its used memory (MB) is below this (default 1000)")
    parser.add_argument("--skip-test", action="store_true",
                        help="do NOT run test.py after training finishes")
    cli = parser.parse_args(own_argv)

    scripts = [s for s in cli.scripts.split(",") if s]

    # Resolve which GPUs to use: explicit --gpus wins; else auto-pick free ones for --num-gpus.
    gpus = [g.strip() for g in cli.gpus.split(",") if g.strip() != ""]
    if not gpus and cli.num_gpus is not None:
        free = query_free_gpus(cli.gpu_mem_threshold)
        if not free:
            print("WARNING: no free GPUs found (nvidia-smi idle list empty or unavailable). "
                  "Running without GPU pinning.")
        elif len(free) < cli.num_gpus:
            print(f"WARNING: requested {cli.num_gpus} GPUs but only {len(free)} are free: {free}. "
                  f"Using those {len(free)}.")
            gpus = free
        else:
            gpus = free[:cli.num_gpus]
            print(f"Auto-selected {len(gpus)} free GPU(s): {gpus}  (free pool was {free})")

    free_gpus = list(gpus)  # pool: one run holds one GPU at a time

    # Concurrency: explicit --jobs, else one per GPU, else sequential.
    jobs = cli.jobs if cli.jobs is not None else (len(gpus) if gpus else 1)
    jobs = max(1, jobs)
    effective_jobs = jobs if not gpus else min(jobs, len(gpus))

    # Every (script, seed) pair is one training run.
    queue = [(script, seed) for seed in cli.seeds for script in scripts]
    print(f"Launching {len(queue)} runs ({len(scripts)} scripts x {len(cli.seeds)} seeds), "
          f"{effective_jobs} at a time.")
    print(f"Seeds: {cli.seeds} | Scripts: {scripts}")
    if gpus:
        print(f"GPUs: {gpus} (one run per GPU)")
        if jobs > len(gpus):
            print(f"  note: jobs ({jobs}) exceeds {len(gpus)} GPUs, so concurrency is capped at {len(gpus)}.")
        print("  reminder: each run also uses num_envs CPU workers - make sure you have the cores.")
    if forward:
        print(f"Forwarding to training scripts: {' '.join(forward)}")

    running = []   # list of (proc, label, logfile_or_None, gpu_or_None)
    results = []   # list of (label, returncode)

    def launch(script, seed, gpu):
        label = f"{script.replace('.py', '')}_seed{seed}"
        cmd = [sys.executable, script, "--seed", str(seed)] + forward
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu  # pin this run to one GPU
        suffix = f" [GPU {gpu}]" if gpu is not None else ""
        if jobs > 1:
            # Concurrent runs would interleave on the console, so log each to its own file.
            os.makedirs("train_logs", exist_ok=True)
            logf = open(os.path.join("train_logs", f"{label}.log"), "w")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            print(f"--> START {label}{suffix}  (log: train_logs/{label}.log)")
            return (proc, label, logf, gpu)
        # Sequential: stream straight to this console.
        print(f"--> START {label}{suffix}: {' '.join(cmd)}")
        return (subprocess.Popen(cmd, env=env), label, None, gpu)

    while queue or running:
        # Launch while we have a free job slot AND (if pinning) a free GPU.
        while queue and len(running) < jobs and (not gpus or free_gpus):
            script, seed = queue.pop(0)
            gpu = free_gpus.pop(0) if gpus else None
            running.append(launch(script, seed, gpu))

        time.sleep(2)

        still = []
        for proc, label, logf, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, label, logf, gpu))
            else:
                if logf is not None:
                    logf.close()
                if gpu is not None:
                    free_gpus.append(gpu)  # release the GPU back to the pool
                print(f"<-- DONE  {label}: exit {rc}")
                results.append((label, rc))
        running = still

    print("\n=== Summary ===")
    for label, rc in results:
        print(f"  {'OK  ' if rc == 0 else 'FAIL'}  {label} (exit {rc})")
    failed = [label for label, rc in results if rc != 0]
    if failed:
        print(f"\n{len(failed)} run(s) failed: {failed}")

    # Run test.py after training, reusing the same GPUs so it evaluates seeds in parallel.
    if cli.skip_test:
        print("\nAll training runs finished. Skipping test.py (--skip-test).")
    else:
        test_cmd = [sys.executable, "test.py"] + forward
        if gpus:
            test_cmd += ["--gpus", ",".join(gpus)]
        print(f"\n=== Training done. Running test.py ===\n{' '.join(test_cmd)}")
        subprocess.run(test_cmd)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
