import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'serif'
# Define a poorly conditioned quadratic function
a, b = 1.0, 50.0  # elongation along x-axis

def loss(theta):
    x, y = theta
    return 0.5 * (a * x**2 + b * y**2)

def grad(theta):
    x, y = theta
    return np.array([a * x, b * y])

def hessian(theta):
    return np.array([[a, 0],
                     [0, b]])

# Optimizers
def sgd(theta, lr=0.01):
    return theta - lr * grad(theta)

def adam(theta, m, v, t, lr=1, beta1=0.9, beta2=0.999, eps=1e-8):
    g = grad(theta)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g**2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v

def newton(theta):
    H = hessian(theta)
    g = grad(theta)
    return theta - np.linalg.inv(H).dot(g)

# Simulation setup
theta0 = np.array([1.5, 1.5])
tol = 1e-1
max_steps = 1000

results = {}  # store steps and time

# Run SGD
sgd_path = [theta0.copy()]
theta = theta0.copy()
start = time.time()
for step in range(1, max_steps+1):
    theta = sgd(theta, lr=0.03)
    sgd_path.append(theta.copy())
    if np.linalg.norm(grad(theta)) < tol:
        break
sgd_time = (time.time() - start) * 1000
results["SGD"] = (step, sgd_time)

# Run Adam
adam_path = [theta0.copy()]
theta = theta0.copy()
m, v = np.zeros_like(theta), np.zeros_like(theta)
start = time.time()
for t in range(1, max_steps+1):
    theta, m, v = adam(theta, m, v, t, lr=0.1)
    adam_path.append(theta.copy())
    if np.linalg.norm(grad(theta)) < tol:
        break
adam_time = (time.time() - start) * 1000
results["Adam"] = (t, adam_time)

# Run Newton
newton_path = [theta0.copy()]
theta = theta0.copy()
start = time.time()
for step in range(1, max_steps+1):
    theta = newton(theta)
    newton_path.append(theta.copy())
    if np.linalg.norm(grad(theta)) < tol:
        break
newton_time = (time.time() - start) * 1000
results["Newton"] = (step, newton_time)

# Convert paths to arrays
sgd_path = np.array(sgd_path)
adam_path = np.array(adam_path)
newton_path = np.array(newton_path)

# Plot contours
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = 0.5 * (a * X**2 + b * Y**2)

plt.figure(figsize=(10, 6))
levels = np.logspace(-1, 3, 15)
CS = plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
plt.clabel(CS, inline=True, fontsize=10, fmt="%.1f")
"""
# Plot optimization paths (dashed)
plt.plot(sgd_path[:,0], sgd_path[:,1], 'r--',
         label=f"SGD ({results['SGD'][0]} steps, {results['SGD'][1]:.2f} ms)")
plt.plot(adam_path[:,0], adam_path[:,1], 'b--',
         label=f"Adam ({results['Adam'][0]} steps, {results['Adam'][1]:.2f} ms)")
"""
plt.plot(sgd_path[:,0], sgd_path[:,1], color='r', linestyle=(0, (6, 4)), linewidth=2,
         label=f"SGD ({results['SGD'][0]} steps, {results['SGD'][1]:.2f} ms)")


# Adam trajectory: blue dashed with thinner dashes
plt.plot(adam_path[:,0], adam_path[:,1], color='b', linestyle=(0, (2, 3)), linewidth=2,
         label=f"Adam ({results['Adam'][0]} steps, {results['Adam'][1]:.2f} ms)")


plt.scatter(newton_path[:,0], newton_path[:,1], 5, color='green',
         label=f"Newton ({results['Newton'][0]} step, {results['Newton'][1]:.2f} ms)")


plt.plot(sgd_path[-1,0], sgd_path[-1,1], 'ro')
plt.plot(adam_path[-1,0], adam_path[-1,1], 'bo')
plt.plot(newton_path[-1,0], newton_path[-1,1], 'go')

# Zoom into top-right & middle region
plt.xlim(-0.8, 1.6)
plt.ylim(-0.8, 1.6)

plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.legend(fontsize=14)
plt.grid(False)

plt.savefig('ill_conditioned.png', dpi=500)
plt.show()
