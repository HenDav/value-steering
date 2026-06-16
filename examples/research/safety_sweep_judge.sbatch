#!/bin/bash
#SBATCH --job-name=vsteer-judgeK
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=safety_sweep_judge.%j.out
# Paper's safety metric: LLM judge labels each saved response safe/unsafe -> unsafe rate (base
# vs VFD), using the paper's verbatim judge prompt. The paper's judge is Llama-3.1-8B-Instruct;
# meta-llama's copy is gated + only node-locally cached on the (busy) download nodes, so we pull
# the UNGATED mirror NousResearch/Meta-Llama-3.1-8B-Instruct (same weights) on any free node.
# Judge needs only transformers, so the vsteer env suffices. Submit: sbatch --partition=llm,normal,msc safety_judge.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
conda activate "$ENVDIR" 2>/dev/null || { echo "no vsteer env on $(hostname -s); building minimal"; \
  conda create -y -p "$ENVDIR" python=3.12 && conda activate "$ENVDIR" && \
  pip install torch==2.10.0 transformers accelerate; }
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export TOKENIZERS_PARALLELISM=false          # use ~/.local transformers (the one that imports)
python -c "import transformers,torch;print('transformers',transformers.__version__,'cuda',torch.cuda.is_available())"
export SAFETY_OUTDIR=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/safety_sweep
JUDGE_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct python scripts/safety_eval.py judge base vfd_k8 vfd_k16 vfd_k24 vfd_k40
echo "=== DONE job ${SLURM_JOB_ID} ==="
