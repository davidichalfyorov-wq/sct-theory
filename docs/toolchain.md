# SCT Theory Toolchain Bootstrap

This document records the verified toolchain needed to run the SCT Theory workspace on Windows 11 with WSL2 Ubuntu.

## Windows Python

Runtime install:

```powershell
python -m pip install -r requirements.txt
```

Developer and notebook tooling:

```powershell
python -m pip install -r requirements-dev.txt
```

Verified on this machine:

- Python `3.12.10`
- `pytest 9.0.2`
- `ruff 0.15.5`
- `notebook 7.5.5`
- `jupyterlab 4.5.6`

## Jupyter CLI

The canonical CLI checks are:

```powershell
python -m jupyter --version
python -m notebook --version
```

In a fresh terminal after PATH refresh, these should also work:

```powershell
jupyter-notebook --version
jupyter-lab --version
```

The user PATH must include:

```text
%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts
```

This path now contains `jupyter.exe`, `jupyter-notebook.exe`, and `jupyter-lab.exe`.

Important:

- Existing long-lived shells do not retroactively inherit user PATH changes.
- After changing PATH, open a new PowerShell window or restart the local workspace session before treating bare `jupyter-*` commands as broken.

## Lean 4

Windows Lean stack:

```powershell
elan default leanprover/lean4:v4.28.0
lake --version
```

Verified:

- `elan 4.2.0`
- default toolchain `leanprover/lean4:v4.28.0`
- `C:\sct-lean` junction points to `theory/lean`

Project checks:

```powershell
python -c "import sys; sys.path.insert(0,'analysis'); from sct_tools.lean import build_sctlean; print(build_sctlean())"
python -c "import sys; sys.path.insert(0,'analysis'); from sct_tools.lean import check_lean_local, check_lean_deep; print(check_lean_local('t','True','True', tactic='rfl')); print(check_lean_deep('t','True','True', tactic='rfl', use_aristotle=False))"
```

## WSL2 Ubuntu

Verified distro and user:

- distro: `Ubuntu`
- user: `razumizm`
- virtual environment: `~/sct-wsl`

WSL scientific packages checked successfully:

```powershell
wsl.exe -d Ubuntu bash -lc "source ~/sct-wsl/bin/activate && python3 -c \"import healpy, classy, pySecDec, cadabra2; print('ok')\""
```

Notes:

- `healpy`, `classy`, and `pySecDec` are installed in `~/sct-wsl`.
- `cadabra2` is importable from the WSL venv as a compiled extension module. On this machine it does not expose pip metadata, so the canonical health check is `import cadabra2`, not `pip show cadabra2`.

SciLean smoke test:

```powershell
python -c "import sys; sys.path.insert(0,'analysis'); from sct_tools.lean import prove_scilean; print(prove_scilean('theorem sct_scilean_smoke : True := by trivial', timeout=180))"
```

## FORM and LaTeX

Windows / WSL checks:

```powershell
tectonic --version
wsl.exe -d Ubuntu bash -lc "which form && form -v"
```

Verified:

- `tectonic 0.15.0`
- `FORM 5.0.0`

Representative TeX build:

```powershell
& '.tools\msvc\tectonic.exe' 'docs\SCT_roadmap.tex'
```

Bootstrap note:

- `papers/build.py` now auto-detects a usable `FONTCONFIG_FILE` on Windows when the environment variable is missing, so `tectonic` builds do not emit the default Fontconfig startup error on this machine.
- If a different fontconfig installation is preferred, set `FONTCONFIG_FILE` explicitly and the build script will honor it.

## Full Verification Commands

Quick CI:

```powershell
python analysis/run_ci.py --quick
```

Full CI:

```powershell
python analysis/run_ci.py
```

Unified local verification:

```powershell
python -c "import sys; sys.path.insert(0,'analysis'); from sct_tools.verification import run_full_verification; print(run_full_verification(include_property=True, include_triple_cas=True))"
```

## Permanent Reminder

Do not treat the toolchain as complete if any of these fail:

- local Lean build
- deep Lean wrapper checks
- WSL SciLean smoke proof
- runtime dependency imports
- notebook CLI availability
- FORM / tectonic availability
- CI or unified verification
