import os
import pathlib
import re
import subprocess
from datetime import datetime
from multiprocessing import pool as mp_pool

import numpy as np
import typer

app = typer.Typer()


def last_match(pattern, text):
    matches = re.findall(pattern, text)
    if not matches:
        return None
    last = matches[-1]
    return float(last)


def run_benchmark(script, opt, steps, dtype, trials, timeout=300):
    path = pathlib.Path(__file__).parent / script
    cmd = ["python3", str(path.resolve().absolute())]
    cmd.extend(["--opt", opt])
    cmd.extend(["--steps", str(steps)])
    cmd.extend(["--dtype", dtype])
    cmd.extend(["--trials", str(trials)])

    base = {'name': script.replace('.py', ''), 'opt': opt}

    try:
        output = subprocess.run(cmd, stdout=subprocess.PIPE, timeout=timeout,
                                cwd=os.path.dirname(os.path.abspath(__file__)))
    except subprocess.TimeoutExpired:
        return {**base, 'success': False, 'runtime': timeout, 'attempts': 0, 'loss': float('inf'), 'error': 'timeout'}
    output = output.stdout.decode()

    success = "Successfully found the minimum." in output

    runtime = last_match(r"Took: ([0-9.]+)", output)
    loss = last_match(r"Best Loss: ([0-9.e\-+]+)", output)
    attempts = int(last_match(r'Attempt: ([0-9]+)', output) or trials)

    return {**base, 'success': success, 'runtime': runtime or 0.0, 'loss': loss or float('inf'), 'attempts': attempts,
            'error': '' if runtime is not None and loss is not None else output}


def opt_to_config(opt):
    caution = "No"
    mars = "No"
    if opt.startswith('cautious-'):
        opt = opt[len('cautious-'):]
        caution = "Yes"
    if opt.startswith('unscaled_cautious-'):
        opt = opt[len('unscaled_cautious-'):]
        caution = "Unscaled"
    if opt.startswith('mars-'):
        opt = opt[len('mars-'):]
        mars = 'Yes'
    return opt, caution, mars


def write_progress(results, opt, output):
    with open(output, 'w') as f:
        f.write(f"# Benchmark Results\nGenerated: {datetime.now()}\nLast updated: {datetime.now()}\n\n")
        f.write("## Summary (In Progress)\n\n")
        f.write("| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |\n")
        f.write("|-----------|---|---|---------|----------|------|\n")

        for o in opt:
            opt_results = [r for r in results if r['opt'] == o]
            if not opt_results:
                continue
            success = sum(r['success'] for r in opt_results)
            runtime = np.mean([r['runtime'] for r in opt_results if r['success']]) if success else 0
            attempts = np.mean([r['attempts'] for r in opt_results if r['success']]) if success else 0

            o, caution, mars = opt_to_config(o)
            f.write(f"| {o} | {caution} | {mars} | {success}/{len(opt_results)} | {runtime:.2f}s | {attempts:.1f} |\n")

        f.write("\n## Details\n\n")
        f.write("| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | \n")
        f.write("|-----------|-----------|---------|---|---|----------|------|---|\n")

        for r in sorted(results, key=lambda x: (x['name'], x['opt'])):
            mark = "✓" if r['success'] else "✗"
            runtime = f"{r['runtime']:.2f}s"
            loss = f"{r['loss']:.2e}"
            attempts = f"{r['attempts']:d}"

            opt, caution, mars = opt_to_config(r['opt'])
            f.write(f"| {r['name']} | {opt} | {caution} | {mars} | {mark} | {runtime} | {loss} | {attempts} | \n")

        if any(not r['success'] for r in results):
            f.write("\n## Errors\n\n")
            for r in sorted(results, key=lambda x: (x['name'], x['opt'])):
                if not r['success'] and r['error']:
                    f.write(f"\n### {r['name']} - {r['opt']}\n```\n{r['error']}\n```\n")


def unordered_star_imap(pool, fn, iterable):
    def _fn(args):
        return args, fn(*args)

    return pool.imap_unordered(_fn, iterable)


@app.command()
def main(opt: list[str] = typer.Option([], help='Optimizers'), steps: int = 100_000, timeout: int = 3600 * 4,
         output: str = 'benchmark_results.md', trials: int = 1000, dtype: str = 'float32', parallelism: int = 16,
         caution: bool = False, mars: bool = False, unscaled_caution: bool = False):
    scripts = ['beale.py', 'powers.py', 'powers_varying_target.py', 'quadratic_varying_target.py',
               'quadratic_varying_scale.py', 'rastrigin.py', 'rosenbrock.py', 'xor_digit.py', 'xor_sequence.py',
               'xor_spot.py']

    opt = ['mars-' + o for o in opt if mars] + opt
    opt = ['cautious-' + o for o in opt if caution] + ['unscaled_cautious-' + o for o in opt if unscaled_caution] + opt

    results = []
    total = len(scripts) * len(opt)
    completed = 0
    args = [(script, o, steps, dtype, trials, timeout) for script in scripts for o in opt]

    with mp_pool.ThreadPool(parallelism) as pool:
        for (script, o, steps, dtype, trials, timeout), result in unordered_star_imap(pool, run_benchmark, args):
            results.append(result)
            completed += 1
            print(f"Progress: [{completed}/{total}] {script} - {o}: {'✓' if result['success'] else '✗'}")
            write_progress(results, opt, output)


if __name__ == '__main__':
    app()
