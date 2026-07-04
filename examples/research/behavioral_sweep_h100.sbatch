#!/bin/bash
#SBATCH --job-name=vsteer-bsw-h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=behavioral_sweep_h100.%j.out
# Behavioral validation of vLLM 0.20.0..0.22.1 on a CUDA-13 node (h100/h200) using vLLM's
# DEFAULT (cu130) wheels -- the a100 nodes are CUDA 12.5 and can't load vllm>=0.20's
# CUDA-13 _C extension. The sweep self-guards on torch.cuda.is_available(): if this node is
# ALSO older than CUDA 13, it records behavioral=None (infra) and exits fast.
# Submit:  sbatch --partition=h100,h200 examples/research/behavioral_sweep_h100.sbatch
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi | grep -iE "Driver Version|CUDA Version" || true
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
unset VSTEER_TORCH_INDEX          # use vLLM default (cu130) wheels on this CUDA-13 node
bash examples/research/behavioral_sweep.sh 0.20.0 0.20.1 0.20.2 0.21.0 0.22.0 0.22.1
echo "=== DONE job ${SLURM_JOB_ID} ==="
