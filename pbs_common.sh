#!/bin/bash
# Shared setup for every PBS job in this repo. Every job script starts with:
#
#   MASTER_PORT=29502
#   cd "${PBS_O_WORKDIR:?submit with qsub from the repo root}" || exit 1
#   source ./pbs_common.sh
#
# THE cd IS REQUIRED, AND IT HAS TO COME FIRST. PBS copies the job script into
# /var/spool/pbs/mom_priv/jobs/ and runs it with $HOME as the working directory —
# NOT the directory you submitted from. So `source ./pbs_common.sh` without the
# cd looks for it in $HOME and fails, and every helper this file defines is then
# "command not found" (and `python` too, since the venv activate below never
# ran). That is the whole of that failure mode.
#
# PBS_O_WORKDIR is the directory qsub was invoked from, so the repo's location is
# written down nowhere: move the checkout and the scripts follow it. If you would
# rather pin it, replace that line in each script with a literal
# `cd /path/to/mini-embed-filip`.
#
# What this does NOT do, deliberately: set FILIP_MODELS_DIR, FILIP_DATA_CSV, or
# any other model/dataset path. config.py owns those. A job script that exports
# one is overriding the config silently, and the banner below would then be
# recording something other than what the job read.
set -o pipefail

# Confirm the cd actually landed in the repo before anything depends on it —
# otherwise the failure surfaces much later as a confusing import error.
if [ ! -f config.py ] || [ ! -d src ]; then
    echo "FATAL: $(pwd) is not the mini-embed-filip repo root (no config.py / src/)." >&2
    echo "       Submit with qsub from the repo root, or edit the cd in this job script." >&2
    exit 1
fi

module load frameworks
source /flare/NLDesignProtein/bryan/envs/filip-env/bin/activate

# --- topology: 12 tiles/node ---
RANKS_PER_NODE=${RANKS_PER_NODE:-12}
NNODES=$(wc -l < "$PBS_NODEFILE" | tr -d " ")
NRANKS=$(( NNODES * RANKS_PER_NODE ))

# --- tiles individually addressable + rendezvous for torch.distributed ---
# Set AFTER `module load frameworks`, which is what makes this stick. The
# frameworks module now defaults ONEAPI_DEVICE_SELECTOR to
# "opencl:gpu;level_zero:gpu" (for Triton-XPU / vLLM / Ray / dpctl) and prints an
# Lmod warning saying so. That warning is emitted at load time and is expected:
# the export below overrides it right after, back to the Level-Zero-only
# selector every job in this repo has always used. Exposing both backends makes
# each physical tile enumerate twice, which breaks the one-rank-per-tile
# mapping. The banner prints the effective value so the .o file records what the
# job actually ran with.
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# --- oneCCL / fabric ---
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
# Do the rank/KVS exchange over MPI (PMIx), NOT oneCCL's internal TCP KVS. The
# internal KVS is a single rendezvous server on rank 0; at O(1000s) of ranks the
# simultaneous connect storm overruns it ("Connection reset by peer" -> "pmi init
# failed" -> SIGSEGV during the first collective). Fine at a few nodes, fatal at
# scale. The timeout is headroom for bring-up on a busy fabric.
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=pidfd

# mpiexec flags every distributed job in this repo uses. --pmi=pmix is REQUIRED
# to match CCL_PROCESS_LAUNCHER=pmix: it tells PALS to expose a PMIx server for
# CCL to bootstrap against. Without it pmix is requested but unavailable and CCL
# falls back to the internal KVS above.
MPI_LAUNCH=(mpiexec -n "${NRANKS}" -ppn "${RANKS_PER_NODE}" --pmi=pmix --cpu-bind depth -d 8)

# ---------------------------------------------------------------------------
# job_banner "<title>" VAR1 VAR2 ...
# ---------------------------------------------------------------------------
# The run record. Prints job identity, topology, code version, every path
# config.py resolves, and the value of each named shell variable — all to stdout,
# which PBS joins into the .o file. That file is then a complete answer to "how
# was this run produced", without needing the job script as it stood that day.
#
# Pass the NAMES of the variables, not their values: `job_banner "t2p SFT" LR BS`.
job_banner() {
    local title="$1"; shift
    local dirty=""
    git diff --quiet 2>/dev/null || dirty=" (UNCOMMITTED CHANGES)"
    echo "================================================================"
    echo "  ${title}"
    echo "================================================================"
    echo "  job           : ${PBS_JOBID:-<interactive>}  ${PBS_JOBNAME:-}"
    echo "  started       : $(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "  user@host     : $(whoami)@$(hostname)"
    echo "  repo          : $(pwd)"
    echo "  git commit    : $(git rev-parse --short HEAD 2>/dev/null || echo '<no git>')${dirty}"
    echo "  topology      : ${NNODES} nodes x ${RANKS_PER_NODE} ranks/node = ${NRANKS} ranks"
    echo "----------------------------------------------------------------"
    echo "  device / fabric environment (effective, after module load):"
    local e
    for e in ONEAPI_DEVICE_SELECTOR ZE_FLAT_DEVICE_HIERARCHY OMP_NUM_THREADS \
             CCL_PROCESS_LAUNCHER CCL_ATL_TRANSPORT CCL_KVS_MODE FI_PROVIDER; do
        printf '    %-24s = %s\n' "$e" "${!e}"
    done
    # /dev/shm is where DataLoader workers hand batches to their parent, shared by
    # all 12 ranks on a node. A worker that cannot get a page dies with SIGBUS,
    # and every peer of that rank then fails with oneCCL "RECV entry failed" —
    # so when a job dies that way, this line is the first thing to check against
    # the loader's reported footprint. Meaningless for num_workers=0 phases.
    printf '    %-24s = %s\n' "/dev/shm (size avail)" \
        "$(df -h /dev/shm 2>/dev/null | awk 'NR==2 {print $2" "$4}' || echo '<unknown>')"
    echo "----------------------------------------------------------------"
    echo "  paths resolved by config.py:"
    python config.py 2>&1 | sed 's/^/    /'
    echo "----------------------------------------------------------------"
    echo "  checkpoints and hyperparameters set by this job script:"
    local v
    for v in "$@"; do
        printf '    %-24s = %s\n' "$v" "${!v}"
    done
    echo "================================================================"
    echo ""
}

# ---------------------------------------------------------------------------
# require_files VAR1 VAR2 ...
# ---------------------------------------------------------------------------
# Fail before the queue slot is spent on model loading. Takes variable NAMES
# holding paths; an empty value is skipped (an intentionally-unset optional
# checkpoint, e.g. cold-start RL).
require_files() {
    local v missing=0
    for v in "$@"; do
        [ -z "${!v}" ] && continue
        if [ ! -e "${!v}" ]; then
            echo "FATAL: ${v}=${!v} does not exist" >&2
            missing=1
        fi
    done
    [ "$missing" = 1 ] && exit 1
    return 0
}

# ---------------------------------------------------------------------------
# finish <exit-code>
# ---------------------------------------------------------------------------
# Slack notification + the job's exit status, uniform across scripts.
finish() {
    local rc=$1
    if [ "$rc" -eq 0 ]; then
        slack ":white_check_mark: DONE ${PBS_JOBID} ${PBS_JOBNAME}"
    else
        slack ":x: FAILED (exit ${rc}) ${PBS_JOBID} ${PBS_JOBNAME}"
    fi
    echo "[job] finished $(date '+%Y-%m-%dT%H:%M:%S%z') with exit code ${rc}"
    exit "$rc"
}

source ~/bin/slack_notify.sh
