# Phase 0 — Environment notes

## Host reality (differs from the assumed "Linux/WSL" starting point)

The Claude Code shell tools run as native Windows processes (Git Bash /
PowerShell), not inside WSL. All Linux-side work (ORFS, OpenROAD, Docker
Linux containers) has to be reached explicitly via `wsl.exe -d Ubuntu -- ...`.
Keep this in mind if you run commands by hand later: `openroad`, `make`, etc.
only exist inside WSL Ubuntu or inside the Docker container, not in a plain
Windows terminal.

- WSL distro: Ubuntu 26.04 LTS ("Resolute Raccoon"), 952GB free disk, 15GB RAM
  visible to WSL.
- Docker Desktop was installed but not running; started it via
  `Start-Process "Docker Desktop.exe"` from PowerShell. Its WSL integration
  distro is `docker-desktop`, separate from our `Ubuntu` distro.

## Why Docker instead of prebuilt .deb or building from source

Checked ORFS's own docs (`docs/user/BuildWithPrebuilt.md` on the
OpenROAD-flow-scripts repo, master branch, fetched 2026-08-20) before
deciding:

- Prebuilt `.deb` binaries from Precision Innovations only support
  **Ubuntu 20.04/22.04 and Debian 11**. Our WSL distro is Ubuntu 26.04 -
  installing a 22.04-targeted `.deb` risks glibc/library mismatches.
- Building from source (`./build_openroad.sh`) compiles OpenROAD, Yosys,
  KLayout etc. from scratch - slow, and pulls in many more dependencies.
- The `openroad/orfs:latest` Docker image ships a matching, tested OS
  inside the container regardless of the host OS, so the 26.04-vs-22.04
  mismatch never matters. ORFS's own README recommends Docker for new
  users for exactly this reason.

Confirmed via Docker Hub API before pulling: `openroad/orfs:latest` is
~1.5GB (`full_size: 1594176976` bytes), pulled successfully.

Version actually running: `openroad -version` inside the container reports
**26Q3-1425-g5da4a78ec6** (checked 2026-08-20).

## Repository layout choice

`OpenROAD-flow-scripts/` is a shallow clone (`git clone --depth 1`, ~1.6GB
on disk) of the ORFS repo, sitting inside `congestion-gnn/` but excluded
via `.gitignore` - it's an external tool with its own upstream history, not
part of this project's repo. We only need it non-recursively (no
submodules) because we're using the prebuilt Docker image
(`util/docker_shell`), not `build_openroad.sh` - submodules are only
needed for building OpenROAD/Yosys/KLayout from source.

## Python environment

- WSL's system Python is 3.14 (too new; PyTorch/PyG wheels aren't there
  yet, and it doesn't match the 3.11 target). No `python3.11` apt package
  exists yet for Ubuntu 26.04, and deadsnakes PPA isn't set up for it
  either, so rather than fight PPA/version drift on a very new Ubuntu
  release, installed `uv` (Astral's Python manager, downloads standalone
  interpreter builds independent of the OS) and used it to install
  Python 3.11.16 directly.
- `torch` installed as **CPU-only** (`torch==2.9.1+cpu` from
  `download.pytorch.org/whl/cpu`, not plain PyPI) - deliberate choice
  given the project goal of a laptop-CPU pipeline validation on a ~400-cell
  design. GPU (RTX 4080, 12GB) is available for later phases if ever
  needed, but graphs this small would likely be slower on GPU due to
  dispatch overhead alone.
- `torch_geometric==2.8.0.post1` needs no separate torch-scatter/
  torch-sparse install for basic HeteroData use (that became optional in
  PyG 2.3+); confirmed via PyG's README install section.
- `numpy` pinned to `2.4.6`, not the PyPI-latest `2.5.2`, because `2.5.2`
  requires Python >=3.12 and we're on 3.11.

## Verified (Phase 0 gate)

```
$ docker run --rm openroad/orfs:latest bash -lc "source ./env.sh && openroad -version"
26Q3-1425-g5da4a78ec6
```

```
$ python -c "import torch, torch_geometric; ..."
torch: 2.9.1+cpu
torch_geometric: 2.8.0.post1
numpy: 2.4.6
matplotlib: 3.11.1
cuda available (should be False, cpu build): False
HeteroData import + instantiate OK: HeteroData()
```
