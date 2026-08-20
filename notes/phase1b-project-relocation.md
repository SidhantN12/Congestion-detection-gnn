# Project relocation: WSL home -> Btech Project (Windows/OneDrive)

The repo initially lived entirely at `~/congestion-gnn` inside WSL. Moved
the actual project content to
`C:\Users\Sidhant\OneDrive\Documents\Python\Btech Project` (the stated
primary working directory) so it's visible from Windows/OneDrive.

## What moved vs. what stayed in WSL

**Moved** (git repo + tracked project content):
`.git`, `.gitignore`, `requirements.txt`, `app/`, `data/`, `flow/`,
`notes/`, `src/`, `tests/`.

**Stayed in `~/congestion-gnn` (WSL native ext4 filesystem)**:
- `OpenROAD-flow-scripts/` - the ~1.6GB ORFS tool checkout. Reproducible
  via `git clone --depth 1` (see notes/phase0-environment.md); not
  project content, and accessing a large tree via the `/mnt/c` drvfs
  bridge is noticeably slower for git/Docker bind-mount operations than
  native ext4.
- `.venv/` - the Python 3.11 virtualenv. Reproducible via
  `requirements.txt`.

Neither of these needed a OneDrive-sync exclusion since they were never
moved into the OneDrive-synced folder in the first place - the only
things now under `Btech Project` are small text/source files.

## Running things going forward

Two filesystem locations are now in play. From WSL:

```bash
# Activate the venv (lives in WSL home, not in the project folder)
source ~/congestion-gnn/.venv/bin/activate

# Run project scripts (repo now lives on the Windows side)
cd "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project"
python src/some_script.py

# Run ORFS (tool checkout also stays in WSL home)
cd ~/congestion-gnn/OpenROAD-flow-scripts/flow
util/docker_shell make
```

Copying ORFS outputs (DEFs, congestion data) into this project's
`data/raw/` will be a normal cross-filesystem `cp` from
`~/congestion-gnn/OpenROAD-flow-scripts/flow/results/...` to
`"/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project/data/raw/..."`.

## Gotchas hit during the move (for future reference)

1. **`wsl.exe -d Ubuntu -- bash -lc '...'` does not reliably propagate
   shell variables** (`A=hello; echo $A` came back empty) when invoked
   this way from this tooling, even though the identical script works
   fine in plain Git Bash or a normal WSL terminal. Root cause not fully
   isolated - use literal paths instead of variables in one-shot
   `wsl.exe` invocations, or write a script file and execute that.
2. **Git ownership check**: `/mnt/c` paths appear root-owned to WSL's
   git, tripping the "dubious ownership" safety check. Fixed once,
   globally, via
   `git config --global --add safe.directory "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project"`.
3. **File mode bits are meaningless on drvfs**: moving files from ext4 to
   `/mnt/c` flips every file to mode 755 (NTFS has no real Unix
   permission bits without WSL's `metadata` mount option, which isn't
   enabled here). This showed up as spurious "modified" diffs on every
   file. Fixed via `core.filemode = false` in this repo's
   `.git/config`.
4. **`git config` itself couldn't write that setting** - every
   `git config core.fileMode false` call failed with
   `chmod on .../.git/config.lock failed: Operation not permitted`,
   consistently (not transient - retried 5x). This is the same
   drvfs/chmod limitation as #3: git's config writer creates a lock file
   and chmods it as part of its atomic-write protocol, and chmod() isn't
   honored on this mount. Worked around by editing `.git/config`
   directly (a plain text edit doesn't go through git's chmod path).
   Note this means **`git config` (any local, repo-scoped config write)
   is unreliable from WSL against this path** - if a future config
   change is needed here, edit `.git/config` directly rather than using
   `git config` for repo-local settings. Global config
   (`git config --global`, writing to `~/.gitconfig` in WSL's native
   home) is unaffected and works normally.
5. **`git add` / `git commit` / `git log` all work fine** despite #4 -
   only the config-write lock path is affected, not the normal
   object/index write path (git's lockfile implementation is not
   uniform internally: config writes go through a different code path
   that insists on chmod succeeding).
