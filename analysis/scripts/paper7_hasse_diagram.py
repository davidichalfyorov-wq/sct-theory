# ruff: noqa: E402, I001
"""
Generate an explanatory Hasse diagram figure for Paper 7 (CJ Bridge).
Response to Nicolangelo Iannella's ResearchGate comment.

Shows:
  Panel (a): Poisson sprinkling in a 1+1D causal diamond
  Panel (b): Full causal order (all relations)
  Panel (c): Hasse diagram (covering relation = links only)
  Panel (d): CJ observable computation from links

Author: David Alfyorov, Igor Shnyukov
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

np.random.seed(42)

FIGDIR = Path(__file__).parent.parent / "figures" / "paper7"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Generate a Poisson sprinkling in a 1+1D causal diamond
# ============================================================
# Diamond: |x| + |t| < T, centered at origin
T = 1.0
N_target = 25  # small enough to see individual links

# Sprinkle in the diamond
points = []
while len(points) < N_target:
    t = np.random.uniform(-T, T)
    x = np.random.uniform(-T, T)
    if abs(x) + abs(t) < T:
        points.append((t, x))

points = np.array(points)
# Sort by time for causal ordering
idx = np.argsort(points[:, 0])
points = points[idx]
N = len(points)

# ============================================================
# Compute causal relations: i < j if t_j > t_i and |x_j - x_i| < t_j - t_i
# (1+1D Minkowski: causal if dt > |dx|)
# ============================================================
causal = np.zeros((N, N), dtype=bool)
for i in range(N):
    for j in range(i + 1, N):
        dt = points[j, 0] - points[i, 0]
        dx = abs(points[j, 1] - points[i, 1])
        if dt > dx and dt > 0:
            causal[i, j] = True

# ============================================================
# Compute Hasse diagram (links): i -< j if causal[i,j] and no k with causal[i,k] and causal[k,j]
# ============================================================
link = np.zeros((N, N), dtype=bool)
for i in range(N):
    for j in range(i + 1, N):
        if causal[i, j]:
            # Check if there's an intermediate element
            is_link = True
            for k in range(i + 1, j):
                if causal[i, k] and causal[k, j]:
                    is_link = False
                    break
            if is_link:
                link[i, j] = True

# ============================================================
# For panel (d): compute interval volumes for CJ
# In 1+1D: interval volume V(i,j) = area of the Alexandrov set = (dt^2 - dx^2)/2
# CJ ~ sum over links of f(V)
# ============================================================
interval_volumes = {}
for i in range(N):
    for j in range(i + 1, N):
        if link[i, j]:
            dt = points[j, 0] - points[i, 0]
            dx = points[j, 1] - points[i, 1]
            V = (dt**2 - dx**2) / 2
            interval_volumes[(i, j)] = V

# ============================================================
# Figure: 4 panels
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

# Colors
pt_color = '#2C3E50'
link_color = '#E74C3C'
causal_color = '#BDC3C7'
diamond_color = '#ECF0F1'
highlight_color = '#F39C12'

# --- Panel (a): Poisson sprinkling ---
ax = axes[0]
# Draw diamond
diamond = plt.Polygon([(-T, 0), (0, T), (T, 0), (0, -T)],
                       fill=True, facecolor=diamond_color, edgecolor='#7F8C8D', lw=1.5)
ax.add_patch(diamond)
ax.scatter(points[:, 1], points[:, 0], c=pt_color, s=30, zorder=5)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel('$x$', fontsize=11)
ax.set_ylabel('$t$', fontsize=11)
ax.set_title('(a) Poisson sprinkling\nin causal diamond', fontsize=10)
ax.set_aspect('equal')
ax.grid(False)
# Light cone lines
ax.plot([-T, 0, T], [0, T, 0], 'k--', lw=0.5, alpha=0.3)
ax.plot([-T, 0, T], [0, -T, 0], 'k--', lw=0.5, alpha=0.3)

# --- Panel (b): Full causal order ---
ax = axes[1]
diamond2 = plt.Polygon([(-T, 0), (0, T), (T, 0), (0, -T)],
                        fill=True, facecolor=diamond_color, edgecolor='#7F8C8D', lw=1.5)
ax.add_patch(diamond2)
# Draw all causal relations (light gray)
for i in range(N):
    for j in range(i + 1, N):
        if causal[i, j]:
            ax.plot([points[i, 1], points[j, 1]], [points[i, 0], points[j, 0]],
                    color=causal_color, lw=0.5, alpha=0.6, zorder=1)
ax.scatter(points[:, 1], points[:, 0], c=pt_color, s=30, zorder=5)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel('$x$', fontsize=11)
ax.set_title('(b) Full causal order\n(all relations $\\prec$)', fontsize=10)
ax.set_aspect('equal')
ax.grid(False)

# --- Panel (c): Hasse diagram (links only) ---
ax = axes[2]
diamond3 = plt.Polygon([(-T, 0), (0, T), (T, 0), (0, -T)],
                        fill=True, facecolor=diamond_color, edgecolor='#7F8C8D', lw=1.5)
ax.add_patch(diamond3)
# Draw only links (red)
n_links = 0
for i in range(N):
    for j in range(i + 1, N):
        if link[i, j]:
            ax.plot([points[i, 1], points[j, 1]], [points[i, 0], points[j, 0]],
                    color=link_color, lw=1.5, alpha=0.8, zorder=2)
            n_links += 1
ax.scatter(points[:, 1], points[:, 0], c=pt_color, s=30, zorder=5)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel('$x$', fontsize=11)
ax.set_title(f'(c) Hasse diagram\n(links $\\prec\\!\\!\\cdot$ only, {n_links} links)', fontsize=10)
ax.set_aspect('equal')
ax.grid(False)

# --- Panel (d): CJ observable ---
ax = axes[3]
diamond4 = plt.Polygon([(-T, 0), (0, T), (T, 0), (0, -T)],
                        fill=True, facecolor=diamond_color, edgecolor='#7F8C8D', lw=1.5)
ax.add_patch(diamond4)

# Draw links colored by interval volume
if interval_volumes:
    vols = list(interval_volumes.values())
    v_min, v_max = min(vols), max(vols)

    for (i, j), V in interval_volumes.items():
        if v_max > v_min:
            t_val = (V - v_min) / (v_max - v_min)
        else:
            t_val = 0.5
        color = plt.cm.hot(0.2 + 0.6 * t_val)  # warm colors, better on gray
        lw = 1.2 + 2.5 * t_val
        ax.plot([points[i, 1], points[j, 1]], [points[i, 0], points[j, 0]],
                color=color, lw=lw, alpha=0.95, zorder=2)

ax.scatter(points[:, 1], points[:, 0], c=pt_color, s=30, zorder=5)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel('$x$', fontsize=11)
ax.set_title('(d) $\\mathrm{CJ}$: links weighted\nby interval volume', fontsize=10)
ax.set_aspect('equal')
ax.grid(False)

# Colorbar for panel (d)
sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(0.2, 0.8))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=[0.2, 0.5, 0.8])
cbar.set_label('$|I(x,y)|/|D|$', fontsize=9)
cbar.ax.set_yticklabels(['small', '', 'large'], fontsize=7)

fig.tight_layout(rect=[0, 0.02, 1, 1])

# Save
path = FIGDIR / "hasse_diagram_explanation.pdf"
fig.savefig(path, dpi=200, bbox_inches='tight')
fig.savefig(path.with_suffix('.png'), dpi=200, bbox_inches='tight')
print(f"Saved {path}")
print(f"Saved {path.with_suffix('.png')}")
print(f"\nN = {N} points, {n_links} links, {sum(1 for i in range(N) for j in range(i+1,N) if causal[i,j])} causal relations")
