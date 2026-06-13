"""Train the baseline (ppo_agent.py) and robust (trainer.py) agents across several seeds.

Run it ONCE instead of typing one command per seed. Seeds are positional:

    python run_seeds.py 1 2 3 4 5
    python run_seeds.py 1 2 3 --jobs 2

To forward extra args to the TRAINING scripts (they share helper.Args), put them
after a literal `--` separator:

    python run_seeds.py 1 2 3 -- --total_timesteps 5000000 --no-track

Everything before `--` configures this runner; everything after `--` is passed
verbatim to both ppo_agent.py and trainer.py. With --jobs > 1, runs execute
concurrently and each one's output goes to train_logs/<run>.log.
You can also just press "Run" on this file in VS Code (trains the default seeds).
"""
import argparse
import os
import subprocess
import sys
import time


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
               "`python run_seeds.py 1 2 3 -- --no-track`.")
    parser.add_argument("seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5],
                        help="seeds to train, e.g. `1 2 3 4 5` (default: 1..5)")
    parser.add_argument("--scripts", default="ppo_agent.py,trainer.py",
                        help="comma-separated training scripts (default: ppo_agent.py,trainer.py)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="how many runs to execute concurrently (1 = sequential)")
    cli = parser.parse_args(own_argv)

    scripts = [s for s in cli.scripts.split(",") if s]

    # Every (script, seed) pair is one training run.
    queue = [(script, seed) for seed in cli.seeds for script in scripts]
    print(f"Launching {len(queue)} runs ({len(scripts)} scripts x {len(cli.seeds)} seeds), "
          f"{cli.jobs} at a time.")
    print(f"Seeds: {cli.seeds} | Scripts: {scripts}")
    if forward:
        print(f"Forwarding to training scripts: {' '.join(forward)}")

    running = []   # list of (proc, label, logfile_or_None)
    results = []   # list of (label, returncode)

    def launch(script, seed):
        label = f"{script.replace('.py', '')}_seed{seed}"
        cmd = [sys.executable, script, "--seed", str(seed)] + forward
        if cli.jobs > 1:
            # Concurrent runs would interleave on the console, so log each to its own file.
            os.makedirs("train_logs", exist_ok=True)
            logf = open(os.path.join("train_logs", f"{label}.log"), "w")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            print(f"--> START {label}  (log: train_logs/{label}.log)")
            return (proc, label, logf)
        # Sequential: stream straight to this console.
        print(f"--> START {label}: {' '.join(cmd)}")
        return (subprocess.Popen(cmd), label, None)

    while queue or running:
        while queue and len(running) < cli.jobs:
            script, seed = queue.pop(0)
            running.append(launch(script, seed))

        time.sleep(2)

        still = []
        for proc, label, logf in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, label, logf))
            else:
                if logf is not None:
                    logf.close()
                print(f"<-- DONE  {label}: exit {rc}")
                results.append((label, rc))
        running = still

    print("\n=== Summary ===")
    for label, rc in results:
        print(f"  {'OK  ' if rc == 0 else 'FAIL'}  {label} (exit {rc})")
    failed = [label for label, rc in results if rc != 0]
    if failed:
        print(f"\n{len(failed)} run(s) failed: {failed}")
        sys.exit(1)
    print("\nAll runs finished. Now run: python test.py")


if __name__ == "__main__":
    main()
