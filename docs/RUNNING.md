# Running SLDgen as a service

This is the operator's manual: how to start the whole thing over SSH, close the
laptop, and find it still running tomorrow.

It assumes you have never used tmux. There are four keystrokes worth knowing and
they are all listed below.

---

## The short version

```bash
cd /home/helge/SLDgen
./start.sh
```

That prints a URL. Open it. When you are done for the day, just close the
terminal — everything keeps running.

To come back:

```bash
ssh fractal
tmux attach -t sldgen-service
```

To stop everything:

```bash
cd /home/helge/SLDgen && ./stop.sh
```

---

## First-time setup

Three things must exist before `start.sh` will run. It checks all of them and
tells you which is missing, so you can also just run it and read the error.

**1. The conda env** (`sldgen`) — the worker and every SLDgen run live here,
because it is the only place torch, pydiffvg and wiregrad exist. It is already
built on this machine; `CLAUDE.md` is the recipe if it ever needs rebuilding.

**2. The API venv** — deliberately separate and torch-free, so the API restarts
in under a second instead of thirty:

```bash
cd /home/helge/SLDgen
python -m venv .venv-service
.venv-service/bin/pip install -r requirements-service.txt
```

**3. tmux and Node** — `sudo apt install tmux` if missing. Node is only needed
to build the web UI; `start.sh --no-build` skips it.

ffmpeg is *not* on that list: service jobs default to `save_video: false`, which
passes `--no-video` to SLDgen. The filmstrip scrubs `svg_to_png/iter_*.png`
directly, so nothing in the UI needs the mp4. Tick **Save mp4** on a job (Run
section of the form) if you want one, and then ffmpeg must be installed in the
conda env — `conda install -c conda-forge ffmpeg`.

Then check it all works without touching the GPU:

```bash
PYTHONPATH=. .venv-service/bin/python test_service_units.py
PYTHONPATH=. .venv-service/bin/python test_service_e2e.py
PYTHONPATH=. .venv-service/bin/python test_service_partitions.py
PYTHONPATH=. .venv-service/bin/python test_service_web.py
cd sldgen_web && npm test
```

---

## What `start.sh` actually does

1. Checks the prerequisites above and refuses to start if one is missing.
2. Rebuilds the web UI if any source file is newer than the last build.
   (`--no-build` skips it, `--rebuild` forces it.)
3. Creates a **detached** tmux session called `sldgen-service` with three panes:

   ```
   ┌──────────────────────────────────────────┐
   │ worker (GPU queue)                       │  the queue: claims jobs, spawns
   │ 2026-08-09 INFO worker started …         │  SLDgen, supervises it
   ├──────────────────────────────────────────┤
   │ api  http://127.0.0.1:8765               │  HTTP + the web UI
   │ INFO: Uvicorn running on …               │
   ├──────────────────────────────────────────┤
   │ shell                                    │  a normal prompt, for you
   │ helge@fractal:~/SLDgen$ _                │
   └──────────────────────────────────────────┘
   ```

4. Waits for the API to answer, then prints the URL and whether the worker
   came up.

"Detached" means the session belongs to the tmux **server**, which is a
background process of its own. It is not a child of your SSH session, so closing
SSH does not touch it. (This machine has `KillUserProcesses=no` in logind, which
is what makes that true. If that ever changes, `sudo loginctl enable-linger
helge` restores it.)

---

## The four tmux keystrokes

Everything in tmux starts with **Ctrl-b** — hold Ctrl, press b, let go, then
press the next key. It is a prefix, not a chord with the second key.

| Keys | What it does |
|---|---|
| **Ctrl-b** then **d** | **D**etach. Leaves everything running and drops you back to your shell. This is how you leave. |
| **Ctrl-b** then **o** | Move to the next pane (worker → api → shell → worker). |
| **Ctrl-b** then **[** | Scroll back through a pane's history. Arrow keys / PageUp to scroll, **q** to stop scrolling. |
| **Ctrl-b** then **z** | Zoom the current pane to fill the window. Press again to unzoom. |

The one that matters is **Ctrl-b d**. If you get lost, press it and start again
with `tmux attach -t sldgen-service`.

> **Do not press Ctrl-c** in the worker or API panes. That kills the daemon
> without giving a running job the chance to checkpoint. Use `./stop.sh`, which
> exists precisely to do that properly.

Two commands, for completeness:

```bash
tmux ls                          # what sessions exist
tmux attach -t sldgen-service    # reattach to ours
```

If `tmux attach` says "no sessions", nothing is running — run `./start.sh`.

---

## Stopping

```bash
./stop.sh
```

The order matters and is the reason to use the script rather than closing panes:

1. The **worker** is signalled first and given up to 180 seconds. It passes the
   signal to the running SLDgen segment, which finishes its current iteration,
   writes a checkpoint, writes `state.json` and exits cleanly. The job resumes
   from that checkpoint next time instead of restarting from epoch 0 — so
   stopping a run at iteration 3800 costs seconds, not fifteen minutes.
2. Any orphaned segment still holding the GPU is cleaned up. Leaving one behind
   makes the *next* start fail with an out-of-memory error that looks like
   somebody else's fault.
3. The **API** stops. It holds no job state, so this is instant.
4. The tmux session closes.

Variants:

```bash
./stop.sh --now            # don't wait; the job resumes from its last checkpoint
./stop.sh --keep-session   # stop the daemons but leave the panes readable
```

Restarting is `./stop.sh && ./start.sh`. Jobs that were queued or paused are
picked up again automatically; a job that was running is resumed from its
checkpoint.

---

## Reaching it from another machine

By default the API binds `127.0.0.1`, so it is reachable only from the host
itself. It has **no authentication**, so the bind address *is* the access
control — it refuses to bind `0.0.0.0` rather than let that be a typo.

`SLDGEN_API_HOST` takes a comma-separated list, and one process serves all of
them on the same port. The usual want is the tailnet *and* the house LAN:

```bash
SLDGEN_API_HOST=loopback,tailscale,lan ./start.sh
```

Those three tokens expand to this machine's current addresses, so a new DHCP
lease or a Tailscale reinstall does not mean editing anything:

| Token | Expands to |
|---|---|
| `loopback` | `127.0.0.1` |
| `tailscale` | `tailscale ip -4` |
| `lan` | the first RFC1918 address on a real interface (`docker0`, bridges and `tailscale0` excluded) |

Anything that is not a token is passed through literally, so explicit addresses
and hostnames work too:

```bash
SLDGEN_API_HOST=127.0.0.1,192.168.178.101 ./start.sh
```

`start.sh` prints one URL per address, and refuses to start if a token does not
resolve — an empty `tailscale ip -4` is a stopped daemon, not an address.

**On the LAN there is still no authentication.** Anyone on the same network can
submit jobs, upload files, read `work/` through the API and delete jobs. That is
usually fine for a home network and never fine for a shared or public one — and
keep Tailscale **Funnel** off, since it would publish this straight to the
internet.

Or, without binding anything extra, tunnel over SSH from your laptop:

```bash
ssh -L 8765:127.0.0.1:8765 fractal
# then open http://127.0.0.1:8765 on the laptop
```

The tunnel is the safer default and needs no configuration on the host.

---

## Configuration

Every default is an environment variable. Set it before `./start.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `SLDGEN_WORK_ROOT` | `<repo>/work` | Everything generated: database, uploads, jobs, checkpoints |
| `SLDGEN_API_HOST` | `127.0.0.1` | Bind addresses, comma-separated. Accepts the tokens `loopback`, `tailscale`, `lan`. Cannot be `0.0.0.0` |
| `SLDGEN_API_PORT` | `8765` | |
| `SLDGEN_TMUX_SESSION` | `sldgen-service` | Session name |
| `SLDGEN_PYTHON` | `~/miniforge3/envs/sldgen/bin/python` | The conda interpreter that runs SLDgen |
| `SLDGEN_API_PYTHON` | `<repo>/.venv-service/bin/python` | The torch-free API interpreter |
| `CONCORDE_PATH` | `~/src/concorde/TSP/concorde` | TSP solver. Missing means every job fails a few seconds in |
| `SLDGEN_STOP_GRACE` | `180` | Seconds `stop.sh` waits for a checkpoint |

> `work/` is inside the repository and is gitignored. It can be deleted whole at
> any time; the service recreates it empty. The one command to avoid in this
> repo is `git clean -xdf`, which would take every job with it.

---

## Developing the UI

`start.sh` serves a built copy. While editing the frontend, run Vite's dev
server alongside it for hot reload — it proxies `/api` to the running API, so
SSE and uploads work unchanged:

```bash
cd sldgen_web && npm run dev     # http://localhost:5173
```

Rebuild the served copy with `./start.sh --rebuild`, or just `cd sldgen_web &&
npm run build` — the API picks up the new files without restarting.

---

## When something is wrong

**Nothing starts, jobs sit in `queued`.** The worker is down. The status bar in
the UI turns red and says so. Look at the top pane:
`tmux attach -t sldgen-service`, then **Ctrl-b [** to scroll back.

**A job failed.** Open it in the UI — the detail page opens on the log with the
cause in plain words. The five classes and what they mean:

| It says | It means | What to do |
|---|---|---|
| The GPU ran out of memory | Something else is holding the card (ComfyUI?) | Free it, press Retry |
| The machine needs attention | HF auth, gated model access, CUDA, or Concorde | Fix it on the host, press Retry |
| The parameters were rejected | A bad flag combination | Run again with changes |
| The run was interrupted | A stop, a reboot, or a worker restart | Resume; nothing is lost past the last checkpoint |
| The run failed | Something else | The log is the whole story |

Gated-model access is the classic first-run failure: SD3.5-medium needs an
approved HuggingFace account. Run `hf auth login` as `helge` and request access
on the model page.

**The UI says "Worker not running" but the pane looks fine.** The worker holds a
lock file (`work/worker.lock`); the API reports liveness by trying to take it.
If a previous worker was SIGKILLed the lock is released by the kernel, so this
should not stick — but `./stop.sh && ./start.sh` resolves it.

**The API is fine but the page is blank.** The UI was not built.
`./start.sh --rebuild`. Visiting `/` with no build gives you a page that says so
and names the command.

**Disk filling up.** Click the disk figure in the status bar. It breaks usage
down by category, lists the ten largest jobs, and offers the cleanup actions —
each of which tells you exactly how many jobs and bytes it will free before you
confirm. Logs are never pruned; they go only when their job does.

---

## Surviving a reboot

The tmux session does not. For a machine that should come back up on its own,
use the systemd units instead — they are written and ready to install, and cover
exactly the same two processes:

```bash
sudo cp deploy/sldgen-worker.service deploy/sldgen-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sldgen-worker sldgen-api
```

See `deploy/README.md`. The two approaches are alternatives, not layers — run
one or the other, never both, or two workers will fight over the queue (the
flock will stop the second one, but the confusion is not worth it).

tmux is the better fit while you are still changing things: you can see both
daemons at once and restart them without root.
