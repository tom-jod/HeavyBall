import matplotlib.pyplot as plt

# Optimizers in order
optimizers = [
    "Zeroth-order methods",
    "SGD",
    "Adam",
    "Shampoo",
    "L-BFGS",
    "Newton's Method"
]

# X positions along the line
positions = list(range(len(optimizers)))

plt.figure(figsize=(10, 2))

# Draw the horizontal line
plt.hlines(0, positions[0], positions[-1], colors="black", linewidth=1)

# Place markers on the line
plt.scatter(positions, [0]*len(positions), color="navy", s=60, zorder=3)

# Annotate labels above the line
for x, label in zip(positions, optimizers):
    plt.text(x, 0.1, label, ha="center", va="bottom", fontsize=10)

# Add directional labels with arrows
plt.text((positions[0] + positions[-1]) / 2, -0.25,
         "Less time per step  →  More time per step",
         ha="center", va="top", fontsize=11, style="italic")

plt.text((positions[0] + positions[-1]) / 2, 0.35,
         "Less accurate preconditioning  →  More accurate preconditioning",
         ha="center", va="bottom", fontsize=11, style="italic")

# Clean up plot (remove axes)
plt.gca().set_axis_off()
plt.tight_layout()

plt.savefig("optimizer_timeline_clean.png", dpi=300, bbox_inches="tight")

