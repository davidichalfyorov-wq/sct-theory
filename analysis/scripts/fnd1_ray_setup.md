# FND-1 Ray Cluster Setup (3 machines)

## Machines
| Machine | Role | CPU | RAM | Notes |
|---------|------|-----|-----|-------|
| A (you) | head | i9-12900KS 24T | 64 GB | Head node + worker |
| B | worker | i7 8 cores | 64 GB | Worker only |
| C | worker | Ryzen 5 5600 12T | ? GB | Worker only |

## Step 1: Tailscale (ALL machines, 2 min each)

1. Go to https://tailscale.com/download
2. Install for your OS (Windows/Linux)
3. `tailscale up` (login with same account or shared tailnet)
4. Note your Tailscale IP: `tailscale ip -4`

After setup, all machines see each other by Tailscale IP.

## Step 2: Python + dependencies (ALL machines)

```bash
# Windows:
pip install ray numpy scipy

# Linux:
pip install ray numpy scipy
```

## Step 3: Copy pipeline code (ALL machines)

Copy these files to the SAME path on all machines:
```
fnd1_gcp_pipeline.py
fnd1_ray_worker.py
```

Or simpler: put them in a shared folder / USB / git repo.

## Step 4: Start Ray cluster

```bash
# Machine A (head) — run first:
ray start --head --port=6379 --num-cpus=8
# Use --num-cpus to limit how many cores Ray uses (leave some for OS)

# Machine B (worker):
ray start --address='TAILSCALE_IP_OF_A:6379' --num-cpus=6

# Machine C (worker):
ray start --address='TAILSCALE_IP_OF_A:6379' --num-cpus=4
```

Verify: `ray status` on head node should show all 3 nodes.

## Step 5: Run experiment

```bash
# On head node (Machine A):
python fnd1_ray_worker.py --metric ppwave_quad --N 10000 --M 80
```

## Step 6: Stop cluster (when done)

```bash
# On ALL machines:
ray stop
```
