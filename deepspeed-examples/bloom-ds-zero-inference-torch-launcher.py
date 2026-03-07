# DeepSpeed ZeRO-3 inference for large causal LMs (e.g. bigscience/bloom 176B).
#
# Usage (via torch.distributed.run, see run_bloom.sh):
#   python -m torch.distributed.run --nproc_per_node=8 bloom-ds-zero-inference-torch-launcher.py \
#       --name bigscience/bloom --deepspeed
#
# Steps:
#   1. Instantiate the model with DeepSpeed ZeRO-3 (weights sharded across GPUs)
#   2. Wrap with deepspeed.initialize()
#   3. Run generate()

import gc
import math
import os
import subprocess
import time
from argparse import ArgumentParser
from threading import Thread

import torch
import torch.distributed as dist

import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.integrations import HfDeepSpeedConfig
except ImportError:
    from transformers.deepspeed import HfDeepSpeedConfig

t_start = time.time()
num_tokens = 30

parser = ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=8, type=int, help="batch size")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
parser.add_argument("--cpu_offload", action="store_true", help="whether to activate CPU offload")
parser.add_argument("--nvme_offload_path", help="whether to activate NVME offload and the path on nvme")
parser.add_argument("--simple_load", action="store_true",
                    help="load without ZeRO-3 (small models only, e.g. bigscience-small-testing)")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

local_rank = int(os.getenv("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)

deepspeed.init_distributed()

rank = dist.get_rank()
world_size = dist.get_world_size()


def print_rank0(*msg):
    if rank == 0:
        print(*msg, flush=True)


# ── Model loading via ZeRO-3 ────────────────────────────────────────────────

model_name = args.name
print_rank0(f"*** Loading the model {model_name}")

config = AutoConfig.from_pretrained(model_name, max_length=40)
dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16
hidden_size = config.hidden_size

ds_config = {
    "fp16": {"enabled": dtype == torch.float16},
    "bf16": {"enabled": dtype == torch.bfloat16},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": int(hidden_size * hidden_size),
        "stage3_prefetch_bucket_size": int(0.9 * hidden_size * hidden_size),
        "stage3_param_persistence_threshold": int(10 * hidden_size),
    },
    "steps_per_print": 2000,
    "train_batch_size": 1 * world_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
}

if args.cpu_offload:
    ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=True)

if args.simple_load:
    ds_config["zero_optimization"]["stage"] = 0
else:
    dschf = HfDeepSpeedConfig(ds_config)

torch.cuda.empty_cache()
gc.collect()
deepspeed.runtime.utils.see_memory_usage("pre-from-pretrained", force=True)

print_rank0("*** Entering from_pretrained (ZeRO-3 load can take 30+ min for 176B)")

_load_result = [None]
_load_exc = [None]


def _do_load():
    try:
        _load_result[0] = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    except Exception as e:
        _load_exc[0] = e


load_thread = Thread(target=_do_load, daemon=False)
load_thread.start()
heartbeat_mins = 0
while load_thread.is_alive():
    time.sleep(60)
    heartbeat_mins += 1
    print_rank0(f"  ... from_pretrained still running (~{heartbeat_mins} min)")
load_thread.join()
if _load_exc[0]:
    raise _load_exc[0]
model = _load_result[0]

print_rank0("*** from_pretrained returned")
deepspeed.runtime.utils.see_memory_usage("post-from-pretrained", force=True)

model = model.eval()

print_rank0("*** Entering deepspeed.initialize()")
ds_engine = deepspeed.initialize(
    args=args, model=model, config_params=ds_config, model_parameters=model.parameters()
)[0]
ds_engine.module.eval()
model = ds_engine.module
print_rank0("*** Model initialized")

# ── Tokenizer ────────────────────────────────────────────────────────────────

_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
tokenizer_kw = {}
if _hf_token:
    tokenizer_kw["token"] = _hf_token
if os.environ.get("TRANSFORMERS_OFFLINE") == "1":
    tokenizer_kw["local_files_only"] = True

# Rank 0 downloads tokenizer first; others wait, then load from cache.
if rank == 0:
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kw)
dist.barrier()
if rank != 0:
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kw)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

if args.benchmark:
    t_ready = time.time()
    deepspeed.runtime.utils.see_memory_usage("start-of-generate", force=True)

# ── Generate ─────────────────────────────────────────────────────────────────

print_rank0(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]

if args.batch_size > len(input_sentences):
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=True)
print_rank0(f"Generate args {generate_kwargs}")

inputs = input_sentences[: args.batch_size]
print_rank0(inputs)


def generate():
    """Returns a list of zipped (input, output, num_new_tokens)."""
    input_tokens = tokenizer.batch_encode_plus(
        inputs, return_token_type_ids=False, return_tensors="pt", padding=True
    )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]
    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


print_rank0("*** Running generate")
t_generate_start = time.time()
pairs = generate()
t_generate_span = time.time() - t_generate_start
for i, o, _ in pairs:
    print_rank0(f"{'-' * 60}\nin={i}\nout={o}\n")
