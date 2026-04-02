#!/bin/bash
# SmolVLA eval suite — LIBERO-Spatial, LIBERO-Goal, LIBERO-Long, Meta-World
# RTX 3050 4GB: runs one task_id at a time (1 EGL context) to avoid OOM.
#
# CHECKPOINT_LIBERO  : SigLIP baseline from paper (HuggingFaceVLA/smolvla_libero)
#                      Swap this to your Florence-trained LIBERO checkpoint once ready.
# CHECKPOINT_MW      : Meta-World checkpoint (jadechoghari/smolvla_metaworld)

set -e

CHECKPOINT_LIBERO="HuggingFaceVLA/smolvla_libero"
CHECKPOINT_MW="jadechoghari/smolvla_metaworld"
N_EPISODES=10      # per task; raise to 50 on a bigger GPU for paper-quality numbers
DEVICE="cuda"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="outputs/eval/smolvla_${TIMESTAMP}"

echo "=== SmolVLA Eval (RTX 3050 4GB mode) ==="
echo "LIBERO checkpoint : $CHECKPOINT_LIBERO"
echo "MetaWorld checkpoint: $CHECKPOINT_MW"
echo "Output dir        : $OUT_BASE"
echo ""

# ── helper: run one LIBERO suite one task at a time ───────────────────────────
# 4GB constraint: creating all 10 tasks at once OOMs → loop task_ids instead.
run_libero_suite() {
    local SUITE=$1      # e.g. libero_spatial
    local N_TASKS=$2    # 10 for all standard suites
    local OUT_DIR="${OUT_BASE}/${SUITE}"

    echo "--- ${SUITE} (${N_TASKS} tasks, ${N_EPISODES} episodes each) ---"
    for (( TID=0; TID<N_TASKS; TID++ )); do
        echo "  task ${TID}"
        conda run -n smol lerobot-eval \
            --policy.path="${CHECKPOINT_LIBERO}" \
            --env.type=libero \
            --env.task="${SUITE}" \
            --env.task_ids="[${TID}]" \
            --eval.n_episodes=${N_EPISODES} \
            --eval.batch_size=1 \
            --policy.device=${DEVICE} \
            --policy.use_amp=true \
            --output_dir="${OUT_DIR}/task_${TID}" 2>&1 \
            | grep -v "EGLError\|eglDestroyContext\|eglMakeCurrent\|binding_utils\|robosuite WARNING\|__del__" \
            | grep -E "success|episode|error|Error" || true
    done
}

# ── LIBERO-Spatial ────────────────────────────────────────────────────────────
echo "[1/4] LIBERO-Spatial"
run_libero_suite libero_spatial 10

# ── LIBERO-Goal ───────────────────────────────────────────────────────────────
echo "[2/4] LIBERO-Goal"
run_libero_suite libero_goal 10

# ── LIBERO-Long (libero_10) ───────────────────────────────────────────────────
echo "[3/4] LIBERO-Long (libero_10)"
run_libero_suite libero_10 10

# ── Meta-World ────────────────────────────────────────────────────────────────
# Meta-World has a different checkpoint and obs format — runs per difficulty group.
echo "[4/4] Meta-World"
for GROUP in easy medium hard very_hard; do
    echo "  group: ${GROUP}"
    conda run -n smol lerobot-eval \
        --policy.path="${CHECKPOINT_MW}" \
        --env.type=metaworld \
        --env.task="${GROUP}" \
        --eval.n_episodes=${N_EPISODES} \
        --eval.batch_size=1 \
        --policy.device=${DEVICE} \
        --policy.use_amp=true \
        --output_dir="${OUT_BASE}/metaworld_${GROUP}" 2>&1 \
        | grep -v "EGLError\|eglDestroyContext\|eglMakeCurrent\|binding_utils\|robosuite WARNING\|__del__" \
        | grep -E "success|episode|error|Error" || true
done

# ── Aggregate ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Results ==="
conda run -n smol python3 - <<EOF
import json, os, collections

results = collections.defaultdict(list)
for root, dirs, files in os.walk("${OUT_BASE}"):
    for f in files:
        if f == "eval_info.json":
            path = os.path.join(root, f)
            with open(path) as fh:
                data = json.load(fh)
            suite = path.replace("${OUT_BASE}/", "").split("/")[0]
            sr = data.get("overall", {}).get("pc_success")
            if sr is not None:
                results[suite].append(sr)

print(f"{'Suite':<28} {'Tasks':>5} {'Avg Success':>12}  Per-task")
print("-" * 75)
grand = []
for suite in sorted(results):
    vals = results[suite]
    avg = sum(vals) / len(vals)
    grand.extend(vals)
    per = "  ".join(f"{v:.0f}%" for v in vals)
    print(f"  {suite:<26} {len(vals):>5} {avg:>11.1f}%  [{per}]")
if grand:
    print("-" * 75)
    print(f"  {'OVERALL':<26} {len(grand):>5} {sum(grand)/len(grand):>11.1f}%")
EOF
