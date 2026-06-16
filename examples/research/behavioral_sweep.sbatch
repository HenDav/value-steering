#!/bin/bash
#SBATCH --job-name=vsteer-bsweep
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=behavioral_sweep.%j.out
# GPU behavioral validation of the plugin against every vLLM version 0.20.0..0.22.1.
# One allocation; builds a per-version venv, runs the 8-test suite, records behavioral
# pass/fail, cleans up. Submit across a100 partitions:
#   sbatch --partition=llm,normal,msc behavioral_sweep.sbatch
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
bash scripts/behavioral_sweep.sh 0.20.0 0.20.1 0.20.2 0.21.0 0.22.0 0.22.1
echo "=== DONE job ${SLURM_JOB_ID} ==="
