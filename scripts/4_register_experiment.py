"""
Script 4: Register experiment - QLoRA fine-tuning on matched/control corpora
Fine-tunes cais/Zephyr_RMU on matched (academic economics) and control (casual finance)
corpora and evaluates WMDP-Bio accuracy after each run.

Run on GPU (A100 recommended). Designed for RunPod or similar cloud GPU.

Usage:
    python scripts/4_register_experiment.py --condition matched --seed 0
    python scripts/4_register_experiment.py --condition control --seed 0
    # Repeat for seeds 0-7, both conditions

Requirements:
    pip install transformers peft trl datasets accelerate sentencepiece scipy
"""

import argparse
import json
import os

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

MODEL_ID   = "cais/Zephyr_RMU"
DATA_DIR   = "data"
OUTPUT_DIR = "results/register_experiment"

# QLoRA hyperparameters
LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 3
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LEN      = 256


def load_data(condition: str) -> Dataset:
    path = os.path.join(DATA_DIR, f"{condition}_train.jsonl")
    records = [json.loads(l) for l in open(path)]
    return Dataset.from_list([{"text": r["text"]} for r in records])


def eval_wmdp_bio(model, tokenizer, n_questions: int = 200) -> float:
    """Evaluate WMDP-Bio MCQ accuracy using log-likelihood scoring."""
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    import random
    random.seed(42)
    indices = random.sample(range(len(ds)), n_questions)
    sample = ds.select(indices)

    model.eval()
    labels = ["A", "B", "C", "D"]
    correct = 0

    for row in sample:
        question = row["question"]
        choices  = row["choices"]
        answer   = row["answer"]

        context = f"The following is a multiple choice question.\n\n{question}\n"
        for label, choice in zip(labels, choices):
            context += f"{label}. {choice}\n"
        context += "\nAnswer:"

        ctx_len = tokenizer(context, return_tensors="pt")["input_ids"].shape[1]
        log_likelihoods = []

        for label in labels[:len(choices)]:
            inputs = tokenizer(context + f" {label}", return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**inputs)
                lp = torch.nn.functional.log_softmax(out.logits[:, :-1, :], dim=-1)
                ll = lp.gather(-1, inputs["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
                ll = ll[:, ctx_len - 1:].mean().item()
            log_likelihoods.append(ll)

        predicted = int(torch.tensor(log_likelihoods).argmax().item())
        if predicted == answer:
            correct += 1

    model.train()
    return correct / n_questions


def run_experiment(condition: str, seed: int) -> dict:
    print(f"\nCondition: {condition} | Seed: {seed}")

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_data(condition)
    print(f"Training on {len(dataset)} sentences")

    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/checkpoints/{condition}_seed{seed}",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
    )

    print("Training...")
    trainer.train()

    print("Evaluating WMDP-Bio accuracy...")
    accuracy = eval_wmdp_bio(model, tokenizer)
    print(f"WMDP-Bio accuracy: {accuracy:.4f}")

    result = {
        "condition": condition,
        "seed": seed,
        "wmdp_bio_accuracy": accuracy,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/{condition}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["matched", "control"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    run_experiment(args.condition, args.seed)
