#!/usr/bin/env python3
"""
optimizer_tradeoff_tuned.py
  
Tuned quantitative schematic: runs several optimizers on an ill-conditioned quadratic
(A is random orthogonal diag(eigs)) in moderate dimension and measures:
- steps to converge (fewer steps => better preconditioning)
- time per step = total runtime / steps
  
Improvements over previous toy:
- use higher dimension (default dim=50) so Newton is expensive relative to quasi-Newton
- Simple Random Search with minimal computation for fastest per-step speed
- Shampoo implemented as block-diagonal with adjustable update frequency
- Learning rates tuned relative to lambda_max
- Optional L-BFGS via scipy (if installed)
"""
  
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'

# Optional SciPy L-BFGS
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    print("scipy not available; L-BFGS will be skipped.")
  
# -------------------------
# Problem utilities
# -------------------------
def make_random_symmetric_A(eigs, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(len(eigs), len(eigs))
    Q, _ = np.linalg.qr(X)
    return Q @ np.diag(eigs) @ Q.T
  
def f_and_grad(A):
    def f(x):
        x = np.asarray(x)
        return 0.5 * float(x @ (A @ x))
    def grad(x):
        x = np.asarray(x)
        return A @ x
    return f, grad
  
# -------------------------
# Optimizers
# -------------------------
def opt_random_search_fast(A, x0, step_size=0.01, check_freq=1000, max_iter=200000, tol=1e-8):
    """
    Ultra-fast random search - minimal computation per step
    Only checks convergence every check_freq steps to avoid expensive A@x
    """
    x = x0.copy()
    d = x.size
    rng = np.random.RandomState(42)
    
    start = time.perf_counter()
    
    for t in range(1, max_iter + 1):
        # Ultra-simple step: random direction, fixed step size
        # This is the ONLY operation most iterations - much faster than SGD
        direction = rng.randn(d)  # O(d) operation
        x = x - step_size * direction  # O(d) operation
        
        # Only check convergence occasionally to avoid expensive A@x
        if t % check_freq == 0:
            true_grad = A @ x  # Expensive O(d²) operation, but rare
            if np.linalg.norm(true_grad) < tol:
                break
    
    # Final convergence check
    final_grad = A @ x
    total_time = time.perf_counter() - start
    return dict(x=x, steps=t, time=total_time, grad_norm=float(np.linalg.norm(final_grad)), 
                step_size=step_size, check_freq=check_freq)

def opt_sgd(A, x0, lr=None, max_iter=200000, tol=1e-8):
    lam_max = np.linalg.eigvalsh(A)[-1]
    if lr is None:
        lr = 0.5 / lam_max
    x = x0.copy()
    start = time.perf_counter()
    for it in range(1, max_iter+1):
        g = A @ x  # Matrix-vector multiply (expensive)
        if np.linalg.norm(g) < tol:
            break
        x = x - lr * g  # Gradient step
    total_time = time.perf_counter() - start
    return dict(x=x, steps=it, time=total_time, grad_norm=float(np.linalg.norm(g)), lr=lr)
  
def opt_adam(A, x0, lr=None, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=20000, tol=1e-8):
    lam_max = np.linalg.eigvalsh(A)[-1]
    if lr is None:
        lr = 0.1 / lam_max
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    start = time.perf_counter()
    for t in range(1, max_iter+1):
        g = A @ x
        if np.linalg.norm(g) < tol:
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)
        x = x - lr * (mhat / (np.sqrt(vhat) + eps))
    total_time = time.perf_counter() - start
    return dict(x=x, steps=t, time=total_time, grad_norm=float(np.linalg.norm(g)), lr=lr)
  
def opt_newton_exact(A, x0, max_iter=10, tol=1e-12):
    x = x0.copy()
    start = time.perf_counter()
    for t in range(1, max_iter+1):
        g = A @ x
        if np.linalg.norm(g) < tol:
            break
        s = np.linalg.solve(A, g)   # Newton direction: A^{-1} g
        x = x - s
    total_time = time.perf_counter() - start
    return dict(x=x, steps=t, time=total_time, grad_norm=float(np.linalg.norm(A @ x)))
  
def opt_lbfgs_scipy(A, x0, tol=1e-8, maxiter=1000):
    if not SCIPY_AVAILABLE:
        return dict(x=x0.copy(), steps=0, time=0.0, grad_norm=float(np.linalg.norm(A @ x0)), skipped=True)
    f, grad = f_and_grad(A)
    it_counter = {"count": 0}
    def callback(xk):
        it_counter["count"] += 1
    start = time.perf_counter()
    res = minimize(fun=f, x0=x0, jac=grad, method='L-BFGS-B', tol=tol, 
                   options={'maxiter': maxiter}, callback=callback)
    total_time = time.perf_counter() - start
    steps = int(getattr(res, 'nit', it_counter["count"]))
    gnorm = float(np.linalg.norm(A @ res.x))
    return dict(x=res.x, steps=steps, time=total_time, grad_norm=gnorm, success=res.success)

def opt_lbfgs_scipy_fast(A, x0, tol=1e-8, maxiter=1000, memory_limit=3):
    if not SCIPY_AVAILABLE:
        return dict(x=x0.copy(), steps=0, time=0.0, grad_norm=float(np.linalg.norm(A @ x0)), skipped=True)
    f, grad = f_and_grad(A)
    it_counter = {"count": 0}
    def callback(xk):
        it_counter["count"] += 1
    start = time.perf_counter()
    res = minimize(fun=f, x0=x0, jac=grad, method='L-BFGS-B', tol=tol, 
                   options={
                       'maxiter': maxiter,
                       'maxcor': memory_limit,  # Reduce memory from default ~10 to 3
                       'maxfun': maxiter * 2
                   }, callback=callback)
    total_time = time.perf_counter() - start
    steps = int(getattr(res, 'nit', it_counter["count"]))
    gnorm = float(np.linalg.norm(A @ res.x))
    return dict(x=res.x, steps=steps, time=total_time, grad_norm=gnorm, success=res.success)
# -------------------------
# Runner
# -------------------------
def run_experiment(dim=50, cond_ratio=1e6, seed=1):
    np.random.seed(seed)
    eigs = np.logspace(0, -np.log10(cond_ratio), dim)
    A = make_random_symmetric_A(eigs, seed=seed)
    x0 = np.ones(dim) + 0.01 * np.random.randn(dim)
    lam_max = eigs.max()
    lam_min = eigs.min()
    print(f"dim={dim}, lambda_max={lam_max:.3e}, lambda_min={lam_min:.3e}, cond={lam_max/lam_min:.3e}")
  
    methods = {
        'Random Search': lambda: opt_random_search_fast(A, x0, step_size=0.001, check_freq=5000, max_iter=200000, tol=1e-3),
        'SGD': lambda: opt_sgd(A, x0, lr=0.4/lam_max, max_iter=200000, tol=1e-3),
        'Adam': lambda: opt_adam(A, x0, lr=0.1/lam_max, max_iter=200000, tol=1e-3),
        'L-BFGS': lambda: opt_lbfgs_scipy_fast(A, x0, tol=1e-3, maxiter=200000),
        'Newton': lambda: opt_newton_exact(A, x0, max_iter=10, tol=1e-3)
    }
  
    results = {}
    for name, fn in methods.items():
        try:
            print(f"Running {name} ...", flush=True)
            r = fn()
            steps = int(r.get('steps', 0)) or 1
            time_total = float(r.get('time', 0.0))
            time_per_step = time_total / steps if steps > 0 else float('nan')
            results[name] = {
                'steps': steps,
                'time_total': time_total,
                'time_per_step': time_per_step,
                'grad_norm': r.get('grad_norm', r.get('final_grad_norm', np.nan)),
                'info': r
            }
            print(f"  -> steps={steps}, total_time={time_total:.6f}s, time/step={time_per_step:.6e}s, grad_norm={results[name]['grad_norm']:.3e}")
        except Exception as e:
            print(f"  Failed {name}: {e}")
            results[name] = {'error': str(e)}
    return results, A, x0
  
# -------------------------
# Plotting
# -------------------------
def plot_results(results, outpath="optimizer_tradeoff_tuned.png"):
    names, tps, steps = [], [], []
    for name, info in results.items():
        if 'error' in info: 
            continue
        names.append(name)
        tps.append(info['time_per_step'])
        steps.append(info['steps'])
    tps = np.array(tps)
    steps = np.array(steps)
  
    plt.figure(figsize=(9,6))
    plt.scatter(tps, steps, s=100, c='tab:blue', edgecolors='k')
    for i, txt in enumerate(names):
        plt.text(tps[i], steps[i]*1.2, txt, ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((2*1e-6,1.5*1e-4))
    plt.xlabel('Time per step (s)')
    plt.ylabel('Number of Steps to Converge')
    plt.tight_layout()
    plt.savefig(outpath, dpi=500)
    print(f"Saved plot to {outpath}")
  
# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    results, A, x0 = run_experiment(dim=50, cond_ratio=1e6, seed=2)
    print("\nSummary:")
    for k, v in results.items():
        if 'error' in v:
            print(k, '-> error:', v['error'])
        else:
            print(f"{k:30s} steps={v['steps']:8d} time_total={v['time_total']:.6f}s time/step={v['time_per_step']:.6e}s grad_norm={v['grad_norm']:.3e}")
    plot_results(results)