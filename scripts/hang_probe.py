import os, faulthandler, sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# If anything hangs >75s, dump every thread's stack (repeatedly) so we see WHERE.
faulthandler.dump_traceback_later(75, repeat=True, exit=False)
print("=== importing vllm ===", flush=True)
from vllm import LLM, SamplingParams
print("=== constructing abstention LLM (in-process) ===", flush=True)
llm = LLM(model="facebook/opt-125m", worker_cls="value_steer.worker.ValueSteerWorker",
          additional_config={"abstain": {"enabled": True, "threshold": 0.5, "strict": True}},
          enforce_eager=True, async_scheduling=False)
print("=== constructed; generating ===", flush=True)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8))
print("=== OUT:", list(out[0].outputs[0].token_ids), flush=True)
print("=== DONE ===", flush=True)
