
from __future__ import annotations
import json
import importlib.util

spec = importlib.util.spec_from_file_location("drv", "/mnt/data/sct_fullR_driver.py")
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)

R_values = [20.0,30.0,40.0,50.0,60.0,80.0,100.0,120.0]
all_out = {}
for aR, label in [(0.0, "conformal_xi_1_6"), (1/18, "minimal_xi_0")]:
    out = {}
    for kind in ["soft", "strict"]:
        sols = drv.continue_bvp(kind, M=1.0, alpha_R=aR, R_values=R_values,
                                tol=4e-3, max_nodes=50000, verbose=0)
        out[kind] = [drv.summary_record(sol, 1.0, aR) for sol in sols]
    all_out[label] = out

with open("/mnt/data/sct_fullR_bvp_scan_M1.json", "w", encoding="utf-8") as f:
    json.dump(all_out, f, indent=2)
