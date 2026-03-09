import warnings
warnings.filterwarnings("ignore")

import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM

# Load policy and reference models
model_name = "meta-llama/Llama-2-13b-chat-hf"
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("Set the HF_TOKEN environment variable with your HuggingFace access token")

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=hf_token)
model_ref = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=hf_token)


def print_trainable_parameters(model):
    """
    Print the names and shapes of trainable parameters in a Hugging Face model.
    Args:
    model: A Hugging Face model instance.
    """
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable_params: {trainable_params}")
    print(f"all_params: {all_params}")


print_trainable_parameters(model)

# Prepare training data (HuggingFaceH4/ultrafeedback_binarized has train_prefs split, DPO-ready)
dataset = load_dataset(
    "HuggingFaceH4/ultrafeedback_binarized",
    split="train_prefs[:100]",
)

def _last_assistant_content(messages):
    """Extract the assistant reply (last message with role 'assistant') from a message list."""
    out = []
    for msg_list in messages:
        text = ""
        for m in msg_list:
            if m.get("role") == "assistant" and m.get("content"):
                text = m["content"]
        out.append(text)
    return out

def format_for_dpo(samples):
    return {
        "prompt": samples["prompt"],
        "chosen": _last_assistant_content(samples["chosen"]),
        "rejected": _last_assistant_content(samples["rejected"]),
    }

original_columns = dataset.column_names
train_dataset = dataset.map(
    format_for_dpo,
    batched=True,
    remove_columns=original_columns,
)

# Initialize trainer
from trl import DPOTrainer
from transformers import AutoTokenizer

# TRL's DPO API has changed across versions:
# - Some versions take `beta` directly in `DPOTrainer(...)`
# - Others expect it inside a `DPOConfig` / training args object
try:
    from trl import DPOConfig  # type: ignore
except Exception:  # pragma: no cover
    DPOConfig = None  # type: ignore

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

# Define training arguments:
if DPOConfig is not None:
    training_args = DPOConfig(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=100,
        logging_steps=100,
        gradient_checkpointing=None,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        optim="adamw_torch",
        output_dir="./results",
        remove_unused_columns=False,
        run_name=None,
        beta=0.2,
    )
else:
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=100,
        logging_steps=100,
        gradient_checkpointing=None,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        optim="adamw_torch",
        output_dir="./results",
        remove_unused_columns=False,
        run_name=None,
    )

_dpo_kwargs = dict(
    model=model,
    ref_model=model_ref,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# TRL API varies: try keyword args first, then with beta on trainer
try:
    dpo_trainer = DPOTrainer(**_dpo_kwargs)
except TypeError:
    try:
        dpo_trainer = DPOTrainer(
            model, model_ref, args=training_args, beta=0.2,
            train_dataset=train_dataset, tokenizer=tokenizer,
        )
    except TypeError:
        dpo_trainer = DPOTrainer(
            model, model_ref, args=training_args,
            train_dataset=train_dataset, tokenizer=tokenizer,
        )

dpo_trainer.train()
