import json
from typing import Any, Dict

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch

import heavyball.chainable as C


# Define toy problems
def quadratic(x):
    """Simple quadratic function with minimum at (1, 2)"""
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2


def rosenbrock(x):
    """Rosenbrock function - a classic test function for optimization algorithms"""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def himmelblau(x):
    """Himmelblau's function - has four identical local minima"""
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def beale(x):
    """Beale function - another challenging test function"""
    term1 = (1.5 - x[0] + x[0] * x[1]) ** 2
    term2 = (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
    term3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    return term1 + term2 + term3


def rastrigin(x):
    """Rastrigin function - highly multimodal"""
    A = 10
    return A * 2 + (x[0] ** 2 - A * torch.cos(2 * np.pi * x[0])) + (x[1] ** 2 - A * torch.cos(2 * np.pi * x[1]))


TOY_PROBLEMS = {
    "Quadratic": {"func": quadratic, "x_range": (-2, 4), "y_range": (-1, 5), "init": [0.0, 0.0]},
    "Rosenbrock": {"func": rosenbrock, "x_range": (-2, 2), "y_range": (-1, 3), "init": [-1.0, 1.0]},
    "Himmelblau": {"func": himmelblau, "x_range": (-5, 5), "y_range": (-5, 5), "init": [0.0, 0.0]},
    "Beale": {"func": beale, "x_range": (-4.5, 4.5), "y_range": (-4.5, 4.5), "init": [1.0, 1.0]},
    "Rastrigin": {"func": rastrigin, "x_range": (-5, 5), "y_range": (-5, 5), "init": [2.0, 2.0]},
}

# Available chainable functions with metadata for the node editor
CHAINABLE_FUNCTIONS = {
    # Momentum functions
    "heavyball_momentum": {
        "function": C.heavyball_momentum,
        "category": "momentum",
        "color": "#00ACC1",  # Cyan
        "description": "Classical heavy ball momentum",
        "params": {"beta": 0.9},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "nesterov_momentum": {
        "function": C.nesterov_momentum,
        "category": "momentum",
        "color": "#00ACC1",
        "description": "Nesterov accelerated gradient",
        "params": {"beta": 0.9},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "nesterov_ema": {
        "function": C.nesterov_ema,
        "category": "momentum",
        "color": "#00ACC1",
        "description": "Nesterov EMA (Grokfast)",
        "params": {"beta": 0.9},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    # Scaling functions
    "scale_by_adam": {
        "function": C.scale_by_adam,
        "category": "scaling",
        "color": "#AB47BC",  # Purple
        "description": "Adam-style adaptive scaling",
        "params": {"betas": (0.9, 0.999), "eps": 1e-8},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "scale_by_exp_avg_sq": {
        "function": C.scale_by_exp_avg_sq,
        "category": "scaling",
        "color": "#AB47BC",
        "description": "RMSprop-style scaling",
        "params": {"beta2": 0.999, "eps": 1e-8},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "scale_by_laprop": {
        "function": C.scale_by_laprop,
        "category": "scaling",
        "color": "#AB47BC",
        "description": "LaProp scaling",
        "params": {"betas": (0.9, 0.999)},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "scale_by_adopt": {
        "function": C.scale_by_adopt,
        "category": "scaling",
        "color": "#AB47BC",
        "description": "ADOPT scaling",
        "params": {"betas": (0.9, 0.999), "eps": 1e-8},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "scale_by_soap": {
        "function": C.scale_by_soap,
        "category": "scaling",
        "color": "#AB47BC",
        "description": "SOAP (Shampoo) preconditioning",
        "params": {"betas": (0.9, 0.999), "shampoo_beta": 0.9, "max_precond_dim": 1000},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    # Weight decay functions
    "weight_decay_to_ema": {
        "function": C.weight_decay_to_ema,
        "category": "weight_decay",
        "color": "#FF6F00",  # Orange
        "description": "AdamW-style decoupled weight decay",
        "params": {"weight_decay_to_ema": 0.01, "ema_beta": 0.999},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "weight_decay_to_init": {
        "function": C.weight_decay_to_init,
        "category": "weight_decay",
        "color": "#FF6F00",
        "description": "Decay towards initial values",
        "params": {"weight_decay_to_init": 0.01},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "l1_weight_decay_to_ema": {
        "function": C.l1_weight_decay_to_ema,
        "category": "weight_decay",
        "color": "#FF6F00",
        "description": "L1 weight decay to EMA",
        "params": {"weight_decay_to_ema": 0.01, "ema_beta": 0.999},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    # Other transformations
    "orthogonalize_update": {
        "function": C.orthogonalize_update,
        "category": "other",
        "color": "#43A047",  # Green
        "description": "Orthogonalize parameter updates",
        "params": {},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "orthogonalize_grad_to_param": {
        "function": C.orthogonalize_grad_to_param,
        "category": "other",
        "color": "#43A047",
        "description": "Orthogonalize gradients to parameters",
        "params": {"eps": 1e-8},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "sign": {
        "function": C.sign,
        "category": "other",
        "color": "#43A047",
        "description": "Sign of gradients (like SignSGD)",
        "params": {"graft": True},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    "mup_approx": {
        "function": C.mup_approx,
        "category": "other",
        "color": "#43A047",
        "description": "μP approximation for scaling",
        "params": {},
        "inputs": ["grad"],
        "outputs": ["grad"],
    },
    # Fused update functions (terminal - apply the update)
    "update_by_adamc": {
        "function": C.update_by_adamc,
        "category": "fused_update",
        "color": "#E53935",  # Red
        "description": 'Fused AdamC ("corrected weight decay") update (terminal)',
        "params": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        "inputs": ["grad"],
        "outputs": [],
        "terminal": True,
    },
    "update_by_schedule_free": {
        "function": C.update_by_schedule_free,
        "category": "fused_update",
        "color": "#E53935",
        "description": "Fused ScheduleFree update (terminal)",
        "params": {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0.0},
        "inputs": ["grad"],
        "outputs": [],
        "terminal": True,
    },
    "update_by_msam": {
        "function": C.update_by_msam,
        "category": "fused_update",
        "color": "#E53935",
        "description": "Fused MSAM (Momentum Sharpness Aware Minimization) update (terminal)",
        "params": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        "inputs": ["grad"],
        "outputs": [],
        "terminal": True,
    },
}

# Predefined optimizer recipes
OPTIMIZER_RECIPES = {
    "SGD with Momentum": [{"name": "heavyball_momentum", "params": {"beta": 0.9}}],
    "AdamW": [
        {"name": "scale_by_adam", "params": {"betas": (0.9, 0.999), "eps": 1e-8}},
    ],
    "RMSprop": [{"name": "scale_by_exp_avg_sq", "params": {"beta2": 0.999, "eps": 1e-8}}],
    "Nesterov SGD": [{"name": "nesterov_momentum", "params": {"beta": 0.9}}],
    "SignSGD with Momentum": [
        {"name": "heavyball_momentum", "params": {"beta": 0.9}},
        {"name": "sign", "params": {"graft": True}},
    ],
    "Adam with Orthogonalization": [
        {"name": "scale_by_adam", "params": {"betas": (0.9, 0.999), "eps": 1e-8}},
        {"name": "orthogonalize_update", "params": {}},
    ],
    "Muon": [{"name": "nesterov_ema", "params": {"beta": 0.9}}, {"name": "orthogonalize_update", "params": {}}],
    "SOAP": [
        {"name": "scale_by_soap", "params": {"betas": (0.9, 0.999), "shampoo_beta": 0.9, "max_precond_dim": 1000}}
    ],
    "Fused AdamW": [
        {"name": "update_by_adam", "params": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}}
    ],
}


def build_optimizer_from_pipeline(params, pipeline_data: Dict[str, Any]) -> C.BaseOpt:
    """Build an optimizer from pipeline data"""
    nodes = pipeline_data.get("nodes", [])
    connections = pipeline_data.get("connections", [])

    # Sort nodes by their connections to get the correct order
    sorted_nodes = []
    node_map = {node["id"]: node for node in nodes}

    print(pipeline_data)
    # Find the start node (no incoming connections)
    for node in nodes:
        if node["type"] == "gradient_input":
            current = node
            break
    else:
        raise ValueError("No gradient input node found in the pipeline")

    # Follow connections to build the pipeline
    while current:
        sorted_nodes.append(current)
        # Find next node
        next_node = None
        for conn in connections:
            if conn["from"] == current["id"]:
                next_node = node_map.get(conn["to"])
                break
        current = next_node

    # Build function list
    fns = []
    all_params = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "weight_decay_to_ema": 0.0,
        "weight_decay_to_init": 0.0,
        "ema_beta": 0.999,
        "beta": 0.9,
        "beta2": 0.999,
        "warmup_steps": 0,
        "storage_dtype": "float32",
        "caution": False,
        "graft": True,
        "shampoo_beta": 0.9,
        "max_precond_dim": 1000,
        "precondition_1d": False,
    }

    for node in sorted_nodes[1:]:  # Skip gradient input node
        if node["type"] in CHAINABLE_FUNCTIONS:
            func_info = CHAINABLE_FUNCTIONS[node["type"]]
            fns.append(func_info["function"])
            # Update parameters from node
            if "params" in node:
                all_params.update(node["params"])

    if not fns:
        # Default to SGD if no functions specified
        fns = [C.heavyball_momentum]

    # Create the optimizer
    optimizer = C.BaseOpt(
        params,
        all_params,
        fns=fns,
    )

    return optimizer


def run_optimization(
    problem_name: str, pipeline_data: Dict[str, Any], steps: int, init_x: float = None, init_y: float = None
):
    """Run optimization with custom pipeline and generate plots"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Get problem details
        problem_info = TOY_PROBLEMS[problem_name]
        objective_fn = problem_info["func"]

        # Initialize parameters
        if init_x is not None and init_y is not None:
            x = torch.nn.Parameter(torch.tensor([init_x, init_y], dtype=torch.float32))
        else:
            x = torch.nn.Parameter(torch.tensor(problem_info["init"], dtype=torch.float32))
        x.data = x.data.to(device)
        params = [x]
        with torch.no_grad():
            trajectory = [x.detach().clone()]
        losses = []

        # Build custom optimizer
        optimizer = build_optimizer_from_pipeline(params, pipeline_data)

        # Run optimization
        def _closure():
            loss = objective_fn(x)
            loss.backward()
            return loss

        for i in range(steps):
            loss = optimizer.step(_closure)

            with torch.no_grad():
                trajectory.append(x.detach().clone())
                losses.append(loss)

        with torch.no_grad():
            trajectory = torch.stack(trajectory, dim=0).cpu().numpy()
            losses = torch.stack(losses).reshape(-1).cpu().numpy()

        # Generate contour plot
        trajectory = np.array(trajectory)

        # Calculate adaptive ranges with 10% margin
        traj_x_min, traj_x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
        traj_y_min, traj_y_max = trajectory[:, 1].min(), trajectory[:, 1].max()

        # Add 10% margin
        x_margin = (traj_x_max - traj_x_min) * 0.1
        y_margin = (traj_y_max - traj_y_min) * 0.1

        # If range is too small, use a minimum margin
        x_margin = max(x_margin, 0.5)
        y_margin = max(y_margin, 0.5)

        # Use either the problem's default range or the adaptive range, whichever is larger
        default_x_min, default_x_max = problem_info["x_range"]
        default_y_min, default_y_max = problem_info["y_range"]

        x_min = min(default_x_min, traj_x_min - x_margin)
        x_max = max(default_x_max, traj_x_max + x_margin)
        y_min = min(default_y_min, traj_y_min - y_margin)
        y_max = max(default_y_max, traj_y_max + y_margin)

        x_range = np.linspace(x_min, x_max, 100)
        y_range = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = objective_fn(torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)).item()

        # Create plotly figure
        fig = go.Figure()

        # Add contour plot
        fig.add_trace(
            go.Contour(x=x_range, y=y_range, z=Z, colorscale="Viridis", opacity=0.7, name="Objective Function")
        )

        # Add trajectory
        fig.add_trace(
            go.Scatter(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=6, color="red"),
                name="Optimization Path",
            )
        )

        # Add start and end points
        fig.add_trace(
            go.Scatter(
                x=[trajectory[0, 0]],
                y=[trajectory[0, 1]],
                mode="markers",
                marker=dict(size=12, color="green", symbol="star"),
                name="Start",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                mode="markers",
                marker=dict(size=12, color="blue", symbol="star"),
                name="End",
            )
        )

        # Extract pipeline description
        nodes = pipeline_data.get("nodes", [])
        pipeline_desc = " → ".join([n["type"] for n in nodes if n["type"] != "gradient_input"])

        fig.update_layout(
            title=f"Custom Optimizer on {problem_name}<br><sub>{pipeline_desc}</sub><br><sub>Final Loss: {losses[-1]:.6f}</sub>",
            xaxis_title="x",
            yaxis_title="y",
            height=600,
            showlegend=True,
        )

        # Create loss plot
        loss_fig = go.Figure()
        loss_fig.add_trace(
            go.Scatter(
                x=list(range(len(losses))), y=losses, mode="lines", line=dict(color="blue", width=2), name="Loss"
            )
        )
        loss_fig.update_layout(
            title="Loss over iterations", xaxis_title="Iteration", yaxis_title="Loss", yaxis_type="log", height=400
        )

        return fig, loss_fig, None

    except Exception as e:
        import traceback

        return None, None, f"Error during optimization: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"


# CSS for the node editor
node_editor_css = """
<style>
#node-editor {
    width: 100%;
    height: 600px;
    background: #1a1a1a;
    border-radius: 12px;
    position: relative;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    min-height: 600px;
    display: block;
}

#canvas {
    position: absolute;
    top: 0;
    left: 240px;
    right: 0;
    bottom: 0;
    background-color: #222;
    background-image:
        radial-gradient(circle at 50% 50%, rgba(76, 175, 80, 0.03) 0%, transparent 50%),
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 100% 100%, 20px 20px, 20px 20px;
    background-position: center, 0 0, 0 0;
    z-index: 1;
}

.node {
    position: absolute;
    background: linear-gradient(135deg, #2d2d2d 0%, #262626 100%);
    border: 2px solid #3a3a3a;
    border-radius: 12px;
    padding: 16px 12px 12px 12px;
    cursor: move;
    min-width: 180px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 5;
}

.node-port {
    position: absolute;
    width: 12px;
    height: 12px;
    background: #4FC3F7;
    border: 2px solid #0288D1;
    border-radius: 50%;
    top: 50%;
    transform: translateY(-50%);
}

.node-port-input {
    left: -7px;
}

.node-port-output {
    right: -7px;
}

.node::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, var(--node-color), transparent);
    border-radius: 12px;
    opacity: 0;
    transition: opacity 0.2s;
    z-index: -1;
}

.node:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}

.node:hover::before {
    opacity: 0.3;
}

.node.selected {
    border-color: #4CAF50;
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.3), 0 8px 24px rgba(0,0,0,0.5);
}

.node.gradient-input {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    border-color: #2a5298;
}

/* Debug styles - commented out for now */
/*
.node {
    border: 3px solid red !important;
    background: yellow !important;
    color: black !important;
}
*/

.node-title {
    color: #fff !important;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 4px;
    text-align: center;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
}

.node-description {
    color: #bbb;
    font-size: 11px;
    margin-bottom: 8px;
    text-align: center;
    opacity: 0.8;
}

.node-port {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    position: absolute;
    cursor: crosshair;
    transition: all 0.2s;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.node-port::after {
    content: '';
    position: absolute;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) scale(0);
    background: rgba(79, 195, 247, 0.3);
    transition: transform 0.2s;
}

.node-port:hover {
    transform: scale(1.3);
    box-shadow: 0 0 8px rgba(79, 195, 247, 0.8);
}

.node-port:hover::after {
    transform: translate(-50%, -50%) scale(1.5);
}

.node-port-input {
    background: linear-gradient(135deg, #4FC3F7 0%, #0288D1 100%);
    border: 2px solid #0277BD;
    left: -7px;
    top: 50%;
    transform: translateY(-50%);
}

.node-port-output {
    background: linear-gradient(135deg, #4FC3F7 0%, #0288D1 100%);
    border: 2px solid #0277BD;
    right: -7px;
    top: 50%;
    transform: translateY(-50%);
}

.connection {
    stroke: #4FC3F7;
    stroke-width: 3;
    fill: none;
    pointer-events: none;
    opacity: 0.9;
    filter: drop-shadow(0 0 3px rgba(79, 195, 247, 0.5));
}

.connection-preview {
    stroke: #4FC3F7;
    stroke-width: 3;
    stroke-dasharray: 8,4;
    fill: none;
    pointer-events: none;
    opacity: 0.6;
    filter: drop-shadow(0 0 3px rgba(79, 195, 247, 0.5));
}

/* Flow particles for gradient animation */
.flow-particle {
    position: absolute;
    width: 6px;
    height: 6px;
    background: radial-gradient(circle, #fff 0%, #4FC3F7 50%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    opacity: 0;
}

#palette {
    position: absolute;
    left: 0;
    top: 0;
    width: 240px;
    height: 100%;
    background: #1f1f1f;
    border-right: 1px solid #333;
    overflow-y: auto;
    padding: 20px;
    box-sizing: border-box;
    z-index: 2;
}

#palette::-webkit-scrollbar {
    width: 6px;
}

#palette::-webkit-scrollbar-track {
    background: #1a1a1a;
}

#palette::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 3px;
}

.palette-category {
    margin-bottom: 20px;
}

.palette-category-title {
    color: #fff;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    opacity: 0.9;
}

.palette-item {
    background: linear-gradient(135deg, #2a2a2a 0%, #242424 100%);
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    cursor: grab;
    transition: all 0.2s;
    color: #fff;
    font-size: 13px;
    position: relative;
    overflow: hidden;
}

.palette-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    transition: left 0.5s;
}

.palette-item:hover {
    background: linear-gradient(135deg, #333 0%, #2a2a2a 100%);
    transform: translateX(4px);
    border-color: #4a4a4a;
}

.palette-item:hover::before {
    left: 100%;
}

.palette-item.dragging {
    opacity: 0.4;
    cursor: grabbing;
    transform: scale(0.95);
}

#inspector {
    position: absolute;
    right: 0;
    top: 0;
    width: 300px;
    height: 100%;
    background: #1f1f1f;
    border-left: 1px solid #333;
    padding: 20px;
    box-sizing: border-box;
    display: none;
    overflow-y: auto;
    z-index: 10;
}

#inspector.active {
    display: block;
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

#inspector h3 {
    color: #fff;
    font-size: 16px;
    margin-bottom: 20px;
    font-weight: 600;
    text-transform: capitalize;
}

.inspector-field {
    margin-bottom: 16px;
}

.inspector-label {
    color: #bbb;
    font-size: 12px;
    margin-bottom: 6px;
    font-weight: 500;
    text-transform: capitalize;
}

.inspector-input {
    width: 100%;
    padding: 8px 12px;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    color: #fff;
    font-size: 13px;
    transition: all 0.2s;
}

.inspector-input:focus {
    outline: none;
    border-color: #4CAF50;
    background: #2d2d2d;
    box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
}

.category-icon {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    display: inline-block;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

/* Code preview area */
#code-preview {
    background: linear-gradient(135deg, #1e1e1e 0%, #1a1a1a 100%);
    border: 1px solid #333;
    border-radius: 8px;
    padding: 20px;
    margin-top: 16px;
    color: #e0e0e0;
    font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
    font-size: 13px;
    line-height: 1.6;
    overflow-x: auto;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);
}

#code-preview pre {
    margin: 0;
}

/* Gradient flow animation */
@keyframes flow {
    from {
        stroke-dashoffset: 0;
        filter: drop-shadow(0 0 3px rgba(79, 195, 247, 0.5));
    }
    to {
        stroke-dashoffset: -24;
        filter: drop-shadow(0 0 6px rgba(79, 195, 247, 0.8));
    }
}

.connection.animated {
    stroke-dasharray: 12,12;
    animation: flow 2s linear infinite;
}

/* Smooth transitions */
* {
    box-sizing: border-box;
}

/* Tooltips */
.tooltip {
    position: absolute;
    background: rgba(0,0,0,0.9);
    color: #fff;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
    pointer-events: none;
    z-index: 1000;
    opacity: 0;
    transition: opacity 0.2s;
}

.tooltip.visible {
    opacity: 1;
}
</style>
"""

# JavaScript for the node editor - embedded inline with error handling
node_editor_js = """
<script>
console.log('Node editor script starting...');

// Error handling wrapper
try {
    // Set global variables
    window.CHAINABLE_FUNCTIONS = %%CHAINABLE_FUNCTIONS_JSON%%;
    window.OPTIMIZER_RECIPES = %%OPTIMIZER_RECIPES_JSON%%;
    console.log('Global variables set:', {
        CHAINABLE_FUNCTIONS: Object.keys(window.CHAINABLE_FUNCTIONS || {}).length,
        OPTIMIZER_RECIPES: Object.keys(window.OPTIMIZER_RECIPES || {}).length
    });

    // NodeEditor class definition
    class NodeEditor {
        constructor(containerId) {
            console.log('NodeEditor constructor called with:', containerId);
            this.container = document.getElementById(containerId);
            if (!this.container) {
                throw new Error('Container element not found: ' + containerId);
            }

            this.canvas = document.getElementById('canvas');
            this.palette = document.getElementById('palette');
            this.inspector = document.getElementById('inspector');

            if (!this.canvas) throw new Error('Canvas element not found');
            if (!this.palette) throw new Error('Palette element not found');

            this.nodes = [];
            this.connections = [];
            this.selectedNode = null;
            this.draggingNode = null;
            this.connecting = null;
            this.nodeIdCounter = 0;
            this.offset = { x: 0, y: 0 };

            console.log('NodeEditor initialized, calling init()');
            this.init();
        }

        init() {
            try {
                // Add a simple test node first
                this.addTestNode();

                // Set up SVG for connections
                this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                this.svg.style.position = 'absolute';
                this.svg.style.width = '100%';
                this.svg.style.height = '100%';
                this.svg.style.pointerEvents = 'none';
                this.svg.style.zIndex = '1';
                this.svg.style.top = '0';
                this.svg.style.left = '0';
                this.canvas.appendChild(this.svg);

                // Set up event listeners
                this.setupPalette();
                this.setupCanvas();

                // Add gradient input node by default
                const gradientNodeId = this.addNode('gradient_input', 100, 300);

                console.log('NodeEditor init completed successfully');
            } catch (error) {
                console.error('Error in NodeEditor.init:', error);
            }
        }

        addTestNode() {
            // Add a simple div to test if nodes show up
            const testNode = document.createElement('div');
            testNode.id = 'test-node';
            testNode.style.position = 'absolute';
            testNode.style.left = '50px';
            testNode.style.top = '50px';
            testNode.style.width = '100px';
            testNode.style.height = '50px';
            testNode.style.background = 'red';
            testNode.style.color = 'white';
            testNode.style.padding = '10px';
            testNode.textContent = 'TEST NODE';
            testNode.style.zIndex = '1000';
            this.canvas.appendChild(testNode);
            console.log('Test node added to canvas');

            // Hide it after 2 seconds
            setTimeout(() => {
                testNode.style.display = 'none';
                console.log('Test node hidden');
            }, 2000);
        }

        setupPalette() {
            const items = this.palette.querySelectorAll('.palette-item');
            console.log('Found palette items:', items.length);
            items.forEach(item => {
                item.draggable = true;
                item.addEventListener('dragstart', (e) => this.onPaletteDragStart(e));
                item.addEventListener('dragend', (e) => this.onPaletteDragEnd(e));
            });
        }

        setupCanvas() {
            this.canvas.addEventListener('dragover', (e) => e.preventDefault());
            this.canvas.addEventListener('drop', (e) => this.onCanvasDrop(e));
            this.canvas.addEventListener('mousedown', (e) => this.onCanvasMouseDown(e));
            this.canvas.addEventListener('mousemove', (e) => this.onCanvasMouseMove(e));
            this.canvas.addEventListener('mouseup', (e) => this.onCanvasMouseUp(e));
        }

        onPaletteDragStart(e) {
            e.target.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'copy';
            e.dataTransfer.setData('nodeType', e.target.dataset.type);
        }

        onPaletteDragEnd(e) {
            e.target.classList.remove('dragging');
        }

        onCanvasDrop(e) {
            e.preventDefault();
            const nodeType = e.dataTransfer.getData('nodeType');
            if (nodeType) {
                const rect = this.canvas.getBoundingClientRect();
                const x = e.clientX - rect.left - 80;
                const y = e.clientY - rect.top - 30;
                const newNodeId = this.addNode(nodeType, x, y);

                // Auto-connect to the last node in the chain
                if (this.nodes.length > 1 && newNodeId) {
                    const previousNode = this.nodes[this.nodes.length - 2];
                    this.addConnection(previousNode.id, newNodeId);
                }
            }
        }

        onCanvasMouseDown(e) {
            // Check if clicking on a node
            const node = e.target.closest('.node');
            if (node && node.id !== 'test-node') {
                e.preventDefault();
                const nodeData = this.nodes.find(n => n.id === node.id);
                if (nodeData) {
                    this.draggingNode = nodeData;
                    const rect = node.getBoundingClientRect();
                    const canvasRect = this.canvas.getBoundingClientRect();
                    // Fix: Calculate offset correctly
                    this.offset = {
                        x: e.clientX - rect.left,
                        y: e.clientY - rect.top
                    };
                }
            }
        }

        onCanvasMouseMove(e) {
            if (this.draggingNode) {
                const canvasRect = this.canvas.getBoundingClientRect();
                // Fix: Calculate position relative to canvas
                const x = e.clientX - canvasRect.left - this.offset.x;
                const y = e.clientY - canvasRect.top - this.offset.y;

                // Constrain to canvas bounds
                const maxX = canvasRect.width - this.draggingNode.element.offsetWidth;
                const maxY = canvasRect.height - this.draggingNode.element.offsetHeight;

                const clampedX = Math.max(0, Math.min(x, maxX));
                const clampedY = Math.max(0, Math.min(y, maxY));

                this.draggingNode.element.style.left = clampedX + 'px';
                this.draggingNode.element.style.top = clampedY + 'px';
                this.draggingNode.x = clampedX;
                this.draggingNode.y = clampedY;

                // Update connections when dragging
                this.updateConnections();
            }
        }

        onCanvasMouseUp(e) {
            this.draggingNode = null;
        }

        addNode(type, x, y) {
            try {
                console.log(`Adding node: type=${type}, x=${x}, y=${y}`);
                const nodeId = `node-${this.nodeIdCounter++}`;
                const nodeData = window.CHAINABLE_FUNCTIONS[type] || {
                    description: type === 'gradient_input' ? 'Gradient Input' : 'Unknown',
                    color: '#666'
                };

                const node = document.createElement('div');
                node.className = 'node';
                node.id = nodeId;
                node.style.position = 'absolute';
                node.style.left = x + 'px';
                node.style.top = y + 'px';
                node.style.cursor = 'move';
                node.style.zIndex = '10';

                // Special styling for gradient input
                if (type === 'gradient_input') {
                    node.classList.add('gradient-input');
                }

                // Use the node's color from metadata
                if (nodeData.color) {
                    node.style.borderLeftColor = nodeData.color;
                    node.style.borderLeftWidth = '4px';
                }

                node.innerHTML = `
                    <div class="node-title">${type === 'gradient_input' ? 'Gradient Input' : type.replace(/_/g, ' ')}</div>
                    <div class="node-description">${nodeData.description}</div>
                    ${type !== 'gradient_input' ? '<div class="node-port node-port-input"></div>' : ''}
                    <div class="node-port node-port-output"></div>
                `;

                this.canvas.appendChild(node);
                console.log('Node added to canvas');

                this.nodes.push({
                    id: nodeId,
                    type: type,
                    element: node,
                    x: x,
                    y: y,
                    params: nodeData.params || {}
                });

                // Update code preview whenever a node is added
                this.updateCodePreview();

                return nodeId;
            } catch (error) {
                console.error('Error adding node:', error);
            }
        }

        // Simplified loadRecipe for testing
        loadRecipe(recipe) {
            console.log('Loading recipe:', recipe);
            // Clear existing nodes except gradient input
            const gradientNode = this.nodes.find(n => n.type === 'gradient_input');
            this.nodes = this.nodes.filter(n => n.type === 'gradient_input');

            // Clear connections
            this.connections = [];
            this.svg.innerHTML = '';

            // Remove node elements except gradient input
            const nodesToRemove = this.canvas.querySelectorAll('.node:not([id="node-0"])');
            nodesToRemove.forEach(n => n.remove());

            let x = 300;
            const y = 300;
            let previousNodeId = gradientNode ? gradientNode.id : null;

            recipe.forEach((item, index) => {
                const nodeId = this.addNode(item.name, x, y);

                // Connect to previous node
                if (previousNodeId && nodeId) {
                    this.addConnection(previousNodeId, nodeId);
                }

                previousNodeId = nodeId;
                x += 200;
            });
        }

        // Add connection between nodes
        addConnection(fromId, toId) {
            // Check if connection already exists
            const exists = this.connections.some(c =>
                c.from === fromId && c.to === toId
            );
            if (exists) return;

            const connection = {
                from: fromId,
                to: toId,
                element: null
            };

            // Create SVG path element
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('stroke', '#4FC3F7');
            path.setAttribute('stroke-width', '3');
            path.setAttribute('fill', 'none');
            path.setAttribute('opacity', '0.8');

            this.svg.appendChild(path);
            connection.element = path;
            this.connections.push(connection);

            this.updateConnections();
        }

        // Update all connection paths
        updateConnections() {
            this.connections.forEach(conn => {
                const fromNode = this.nodes.find(n => n.id === conn.from);
                const toNode = this.nodes.find(n => n.id === conn.to);

                if (fromNode && toNode && conn.element) {
                    // Calculate connection points (right side of from, left side of to)
                    const fromRect = fromNode.element.getBoundingClientRect();
                    const toRect = toNode.element.getBoundingClientRect();
                    const canvasRect = this.canvas.getBoundingClientRect();

                    const x1 = fromRect.right - canvasRect.left;
                    const y1 = fromRect.top + fromRect.height / 2 - canvasRect.top;
                    const x2 = toRect.left - canvasRect.left;
                    const y2 = toRect.top + toRect.height / 2 - canvasRect.top;

                    // Create curved path
                    const dx = Math.abs(x2 - x1);
                    const cp1x = x1 + dx * 0.5;
                    const cp2x = x2 - dx * 0.5;
                    const path = `M ${x1} ${y1} C ${cp1x} ${y1}, ${cp2x} ${y2}, ${x2} ${y2}`;

                    conn.element.setAttribute('d', path);
                }
            });
        }

        // Export pipeline data
        exportPipeline() {
            return {
                nodes: this.nodes.map(n => ({
                    id: n.id,
                    type: n.type,
                    x: n.x,
                    y: n.y,
                    params: n.params || {}
                })),
                connections: this.connections.map(c => ({
                    from: c.from,
                    to: c.to
                }))
            };
        }

        // Update code preview
        updateCodePreview() {
            const codePreview = document.getElementById('code-output');
            if (!codePreview) return;

            const pipelineData = this.exportPipeline();
            const nodes = pipelineData.nodes.filter(n => n.type !== 'gradient_input');

            if (nodes.length === 0) {
                codePreview.textContent = '# Add optimizer components to build your pipeline';
                return;
            }

            let code = 'optimizer = BaseOpt(\\n';
            code += '    model.parameters(),\\n';
            code += '    lr=0.001,\\n';
            code += '    fns=[\\n';
            nodes.forEach(node => {
                code += `        ${node.type},\\n`;
            });
            code += '    ]\\n';
            code += ')';

            codePreview.textContent = code;

            // Dispatch a custom event to notify about pipeline changes
            window.dispatchEvent(new CustomEvent('pipelineChanged', { detail: pipelineData }));
        }
    }

    // Store NodeEditor class globally
    window.NodeEditor = NodeEditor;

    // Initialize function
    window.initializeNodeEditor = function() {
        console.log('initializeNodeEditor called');
        try {
            const editorElement = document.getElementById('node-editor');
            if (!editorElement) {
                console.error('node-editor element not found, retrying...');
                setTimeout(window.initializeNodeEditor, 500);
                return;
            }

            if (window.nodeEditor) {
                console.log('NodeEditor already initialized');
                return;
            }

            window.nodeEditor = new NodeEditor('node-editor');
            console.log('NodeEditor created successfully');
        } catch (error) {
            console.error('Error initializing NodeEditor:', error);
            setTimeout(window.initializeNodeEditor, 500);
        }
    };

    // Try to initialize on DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', window.initializeNodeEditor);
    } else {
        // DOM already loaded
        setTimeout(window.initializeNodeEditor, 100);
    }

} catch (error) {
    console.error('Fatal error in node editor script:', error);
}
</script>
"""


# HTML structure for the node editor
def create_node_editor_html():
    # Build palette HTML
    palette_html = ""
    categories = {}

    # Group functions by category
    for name, info in CHAINABLE_FUNCTIONS.items():
        category = info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append((name, info))

    # Category names and icons
    category_info = {
        "momentum": {"name": "Momentum", "color": "#00ACC1"},
        "scaling": {"name": "Scaling", "color": "#AB47BC"},
        "weight_decay": {"name": "Weight Decay", "color": "#FF6F00"},
        "other": {"name": "Other", "color": "#43A047"},
        "fused_update": {"name": "Fused Updates", "color": "#E53935"},
    }

    for cat_key, cat_items in categories.items():
        cat_data = category_info.get(cat_key, {"name": cat_key.title(), "color": "#666"})
        palette_html += f"""
        <div class="palette-category">
            <div class="palette-category-title">
                <span class="category-icon" style="background: {cat_data["color"]}"></span>
                {cat_data["name"]}
            </div>
        """

        for name, info in cat_items:
            palette_html += f"""
            <div class="palette-item" draggable="true" data-type="{name}">
                {name.replace("_", " ").title()}
            </div>
            """

        palette_html += "</div>"

    CHAINABLE_FUNCTIONS_JS = {
        name: {k: v for k, v in info.items() if k != "function"} for name, info in CHAINABLE_FUNCTIONS.items()
    }

    inner_js = node_editor_js.replace("%%CHAINABLE_FUNCTIONS_JSON%%", json.dumps(CHAINABLE_FUNCTIONS_JS)).replace(
        "%%OPTIMIZER_RECIPES_JSON%%", json.dumps(OPTIMIZER_RECIPES)
    )

    return f"""
    {node_editor_css}
    <div id="node-editor">
        <div id="palette">
            <div style="color: #fff; font-size: 16px; margin-bottom: 16px; font-weight: 600;">
                Optimizer Components
            </div>
            <div style="color: #aaa; font-size: 12px; margin-bottom: 16px;">
                Drag components to canvas
            </div>
            {palette_html}
        </div>
        <div id="canvas">
            <!-- Nodes and connections will be added here -->
        </div>
        <div id="inspector">
            <!-- Node properties will be shown here -->
        </div>
    </div>
    <input type="hidden" id="pipeline-data" value='{{"nodes": [], "connections": []}}' />
    {inner_js}
    """


# Create the Gradio interface
with gr.Blocks(title="HeavyBall Chainable Optimizer Builder", theme=gr.themes.Base()) as app:
    gr.Markdown("""
    # 🔧 HeavyBall Visual Pipeline Builder

    Build custom optimizers by dragging and connecting transformation components!

    The chainable API lets you compose optimizers as a sequence of gradient transformations.
    Connect components visually to see how gradients flow through your custom pipeline.
    """)

    with gr.Row(min_height=700):
        with gr.Column(scale=3):
            # Node editor
            node_editor = gr.HTML(create_node_editor_html(), min_height=650, elem_id="node-editor-container")

            # Code preview
            gr.Markdown("### 📝 Generated Code")
            code_output = gr.HTML(
                '<div id="code-preview"><pre id="code-output"># Add optimizer components to build your pipeline</pre></div>'
            )

            # Hidden state for pipeline data
            pipeline_data = gr.JSON(value={"nodes": [], "connections": []}, visible=False)

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Problem Selection")

            problem_dropdown = gr.Dropdown(
                choices=list(TOY_PROBLEMS.keys()), value="Rosenbrock", label="Optimization Problem"
            )

            with gr.Row():
                init_x = gr.Number(label="Initial X", value=None)
                init_y = gr.Number(label="Initial Y", value=None)

            gr.Markdown("### 🧪 Predefined Recipes")

            recipe_dropdown = gr.Dropdown(choices=list(OPTIMIZER_RECIPES.keys()), value=None, label="Load Recipe")

            load_recipe_btn = gr.Button("Load Recipe", variant="secondary")

            gr.Markdown("### ⚙️ Optimization Settings")

            steps_slider = gr.Slider(minimum=10, maximum=1000, value=200, step=10, label="Number of Steps")

            run_button = gr.Button("🚀 Run Optimization", variant="primary", size="lg")

            # Auto-retrain status indicator
            auto_retrain_status = gr.HTML(
                '<div id="auto-retrain-status" style="text-align: center; padding: 8px; border-radius: 8px; background: #2a2a2a; color: #888; font-size: 12px; margin-top: 8px;">Auto-retrain: Ready</div>'
            )

            gr.Markdown("""
            ### 💡 Tips
            - Drag components from the left palette
            - Connect output ports to input ports
            - Click nodes to edit parameters
            - Chain multiple transforms together
            - Fused updates are terminal nodes
            """)

    gr.Markdown("### 📊 Results")

    with gr.Row():
        trajectory_plot = gr.Plot()
        loss_plot = gr.Plot()

    error_output = gr.Textbox(label="Error Messages", visible=False)

    # Prepare CHAINABLE_FUNCTIONS for JavaScript (remove function references)
    CHAINABLE_FUNCTIONS_JS = {
        name: {k: v for k, v in info.items() if k != "function"} for name, info in CHAINABLE_FUNCTIONS.items()
    }

    # Add initialization JavaScript that will run after page load
    initialization_script = f"""
    <script>
    // Global initialization script
    (function() {{
        console.log('Running global initialization script...');

        // Define global variables
        window.CHAINABLE_FUNCTIONS = {json.dumps(CHAINABLE_FUNCTIONS_JS)};
        window.OPTIMIZER_RECIPES = {json.dumps(OPTIMIZER_RECIPES)};

        console.log('Global variables initialized:', {{
            CHAINABLE_FUNCTIONS: Object.keys(window.CHAINABLE_FUNCTIONS).length,
            OPTIMIZER_RECIPES: Object.keys(window.OPTIMIZER_RECIPES).length
        }});

        // Initialize node editor after a delay to ensure DOM is ready
        setTimeout(() => {{
            if (typeof initializeNodeEditor === 'function') {{
                initializeNodeEditor();
            }}
        }}, 500);
    }})();
    </script>
    """

    # Add the initialization script as a separate HTML component
    gr.HTML(initialization_script)

    # JavaScript to handle recipe loading
    load_recipe_js = f"""
    function loadRecipe(recipe_name) {{
        console.log('Loading recipe:', recipe_name);

        // Define recipes inline if not already available
        if (!window.OPTIMIZER_RECIPES) {{
            window.OPTIMIZER_RECIPES = {json.dumps(OPTIMIZER_RECIPES)};
            console.log('Initialized OPTIMIZER_RECIPES inline');
        }}

        // Initialize node editor if needed
        if (!window.nodeEditor && typeof initializeNodeEditor === 'function') {{
            initializeNodeEditor();
        }}

        // Wait a bit for initialization to complete
        setTimeout(() => {{
            if (window.nodeEditor && window.OPTIMIZER_RECIPES && window.OPTIMIZER_RECIPES[recipe_name]) {{
                console.log('Loading recipe into node editor:', recipe_name);
                window.nodeEditor.loadRecipe(window.OPTIMIZER_RECIPES[recipe_name]);
            }} else {{
                console.error('Could not load recipe - missing dependencies');
            }}
        }}, 500);

        return JSON.stringify({{"nodes": [], "connections": []}});
    }}
    """

    # JavaScript to get pipeline data
    get_pipeline_js = """
    function getPipelineData() {
        if (window.nodeEditor) {
            return JSON.stringify(window.nodeEditor.exportPipeline());
        }
        return JSON.stringify({"nodes": [], "connections": []});
    }
    """

    def load_recipe_handler(recipe_name):
        # This will be handled by JavaScript
        return gr.update()

    def run_optimization_handler(problem, steps, init_x, init_y):
        # Get pipeline data from the hidden input
        gr.HTML.update(
            value="""
            <script>
            (function() {
                const data = getPipelineData();
                document.getElementById('optimization-pipeline-data').value = data;
            })();
            </script>
            <input type="hidden" id="optimization-pipeline-data" />
        """
        )

        # For now, return a placeholder
        return None, None, gr.update(visible=False)

    # Function to handle optimization with pipeline data
    def handle_optimization(problem, steps, init_x, init_y, pipeline_json):
        """Handle optimization with pipeline data from JavaScript"""
        try:
            # Parse the pipeline data
            if isinstance(pipeline_json, str):
                pipeline_data = json.loads(pipeline_json)
            else:
                pipeline_data = pipeline_json

            # Run optimization
            fig, loss_fig, error = run_optimization(problem, pipeline_data, steps, init_x, init_y)

            if error:
                return None, None, gr.update(visible=True, value=error)
            return fig, loss_fig, gr.update(visible=False)
        except Exception as e:
            import traceback

            return None, None, gr.update(visible=True, value=f"Error: {str(e)}\n\n{traceback.format_exc()}")

    # Connect handlers
    load_recipe_btn.click(fn=None, inputs=[recipe_dropdown], outputs=[pipeline_data], js=load_recipe_js)

    # Modified JavaScript to properly get pipeline data
    run_optimization_js = """
    async function(problem, steps, init_x, init_y) {
        const pipelineData = window.nodeEditor ? window.nodeEditor.exportPipeline() : {"nodes": [], "connections": []};
        return [problem, steps, init_x, init_y, JSON.stringify(pipelineData)];
    }
    """

    run_button.click(
        fn=handle_optimization,
        inputs=[problem_dropdown, steps_slider, init_x, init_y, pipeline_data],
        outputs=[trajectory_plot, loss_plot, error_output],
        js=run_optimization_js,
    )

    # Add JavaScript for auto-retrain functionality
    auto_retrain_js = """
    function setupAutoRetrain() {
        console.log('Setting up auto-retrain...');

        let retrainTimeout = null;
        let lastPipelineState = null;
        let isFirstRun = true;

        // Function to check if pipeline has changed
        function hasPipelineChanged() {
            if (!window.nodeEditor) return false;
            const currentState = JSON.stringify(window.nodeEditor.exportPipeline());
            const changed = currentState !== lastPipelineState;
            if (changed) {
                lastPipelineState = currentState;
            }
            return changed;
        }

        // Function to update status indicator
        function updateStatus(message, color = '#888') {
            const statusElement = document.getElementById('auto-retrain-status');
            if (statusElement) {
                statusElement.textContent = message;
                statusElement.style.color = color;
            }
        }

        // Function to trigger retrain
        function triggerRetrain() {
            console.log('Auto-retrain triggered');
            updateStatus('Auto-retrain: Running...', '#4FC3F7');

            // Find the run button by looking for the button with the rocket emoji
            const buttons = document.querySelectorAll('button');
            let runButton = null;

            for (const button of buttons) {
                if (button.textContent.includes('🚀') && button.textContent.includes('Run Optimization')) {
                    runButton = button;
                    break;
                }
            }

            if (runButton) {
                console.log('Clicking run button...');
                runButton.click();

                // Reset status after a delay
                setTimeout(() => {
                    updateStatus('Auto-retrain: Ready');
                }, 1000);
            } else {
                console.error('Run button not found');
                updateStatus('Auto-retrain: Error - Run button not found', '#ff6b6b');
            }
        }

        // Function to schedule retrain
        function scheduleRetrain() {
            // Skip the first run to avoid immediate retrain on load
            if (isFirstRun) {
                isFirstRun = false;
                hasPipelineChanged(); // Initialize the baseline
                return;
            }

            // Clear existing timeout
            if (retrainTimeout) {
                clearTimeout(retrainTimeout);
            }

            // Check if pipeline has changed
            if (hasPipelineChanged()) {
                console.log('Pipeline changed, scheduling retrain in 2 seconds...');
                updateStatus('Auto-retrain: Scheduled (2s)', '#FFA726');

                retrainTimeout = setTimeout(triggerRetrain, 2000);
            }
        }

        // Monitor for changes
        if (window.nodeEditor) {
            // Listen for custom pipeline change events
            window.addEventListener('pipelineChanged', (event) => {
                console.log('Pipeline changed event received');
                scheduleRetrain();
            });

            // Override the loadRecipe method to detect changes
            const originalLoadRecipe = window.nodeEditor.loadRecipe;
            window.nodeEditor.loadRecipe = function(...args) {
                const result = originalLoadRecipe.call(this, ...args);
                setTimeout(() => {
                    // When a recipe is loaded, always trigger retrain
                    // Don't update baseline first, as that would prevent change detection
                    console.log('Recipe loaded, forcing retrain...');
                    updateStatus('Auto-retrain: Recipe loaded, retraining...', '#4FC3F7');
                    triggerRetrain();
                    // Update baseline after triggering retrain
                    setTimeout(() => {
                        hasPipelineChanged(); // Update the baseline for future changes
                    }, 500);
                }, 100);
                return result;
            };

            // Monitor canvas for node deletions (right-click or delete key)
            const canvas = document.getElementById('canvas');
            if (canvas) {
                canvas.addEventListener('contextmenu', (e) => {
                    if (e.target.closest('.node')) {
                        setTimeout(scheduleRetrain, 100);
                    }
                });

                canvas.addEventListener('keydown', (e) => {
                    if (e.key === 'Delete' || e.key === 'Backspace') {
                        setTimeout(scheduleRetrain, 100);
                    }
                });
            }

            // Also monitor problem selection and slider changes
            const problemDropdown = document.querySelector('select[aria-label="Optimization Problem"]');
            const stepsSlider = document.querySelector('input[aria-label="Number of Steps"]');
            const initXInput = document.querySelector('input[aria-label="Initial X"]');
            const initYInput = document.querySelector('input[aria-label="Initial Y"]');

            [problemDropdown, stepsSlider, initXInput, initYInput].forEach(element => {
                if (element) {
                    element.addEventListener('change', () => {
                        console.log('Parameter changed, scheduling retrain...');
                        scheduleRetrain();
                    });
                }
            });

            console.log('Auto-retrain setup complete');

            // Initialize the baseline state
            hasPipelineChanged();
            updateStatus('Auto-retrain: Enabled');
        } else {
            console.log('NodeEditor not found, retrying auto-retrain setup...');
            setTimeout(setupAutoRetrain, 1000);
        }
    }
    """

    # Add load event to ensure proper initialization
    app.load(
        fn=lambda: None,
        inputs=[],
        outputs=[],
        js=f"""
        () => {{
            console.log('App loaded - initializing node editor...');

            // Set global variables if not already set
            if (!window.CHAINABLE_FUNCTIONS) {{
                window.CHAINABLE_FUNCTIONS = {json.dumps(CHAINABLE_FUNCTIONS_JS)};
                console.log('Initialized CHAINABLE_FUNCTIONS with', Object.keys(window.CHAINABLE_FUNCTIONS).length, 'items');
            }}

            if (!window.OPTIMIZER_RECIPES) {{
                window.OPTIMIZER_RECIPES = {json.dumps(OPTIMIZER_RECIPES)};
                console.log('Initialized OPTIMIZER_RECIPES with', Object.keys(window.OPTIMIZER_RECIPES).length, 'recipes');
            }}

            // Initialize node editor if the initialization function exists
            setTimeout(() => {{
                // First, check if scripts were executed
                const scripts = document.querySelectorAll('#node-editor-container script');
                console.log('Found', scripts.length, 'scripts in node editor container');

                // If scripts exist but weren't executed, execute them manually
                if (scripts.length > 0 && typeof NodeEditor === 'undefined') {{
                    console.log('Manually executing scripts...');
                    scripts.forEach(oldScript => {{
                        const newScript = document.createElement('script');
                        if (oldScript.src) {{
                            newScript.src = oldScript.src;
                        }} else {{
                            newScript.textContent = oldScript.textContent;
                        }}
                        document.body.appendChild(newScript);
                    }});
                }}

                // Now try to initialize
                setTimeout(() => {{
                    if (typeof initializeNodeEditor === 'function') {{
                        console.log('Calling initializeNodeEditor...');
                        initializeNodeEditor();

                        // Set up auto-retrain after node editor is initialized
                        setTimeout(() => {{
                            {auto_retrain_js}
                            setupAutoRetrain();
                        }}, 1000);
                    }} else {{
                        console.error('initializeNodeEditor function not found');
                    }}
                }}, 100);
            }}, 100);
        }}
        """,
    )

if __name__ == "__main__":
    app.launch(debug=True, show_error=True)
