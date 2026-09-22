"""Federated LoRA instruction tuning under BadNets poisoning, with optional SCB on the server.

    python main.py --model <backbone_dir> --dataset webqa --defense scb --output_dir runs/webqa_scb

Defaults follow the paper: 10 clients all participating for 30 rounds, 1 local epoch per round,
LoRA r=4 / alpha=8 on q_proj and v_proj, learning rate 1e-4 with round-level linear decay,
batch 4, sequence length 256, contiguous client shards, clients 7/8/9 poisoning 20% of their data.
"""
import argparse
import copy
import json
import os

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from data_utils import answer_marker_ids, formatting_func, get_qa_dataset, poison_dataset, split_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="backbone directory")
    ap.add_argument("--dataset", default="webqa", help="webqa | freebaseqa | nq | coqa | fictbio")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--output_dir", default="output")
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--num_rounds", type=int, default=30)
    ap.add_argument("--split", default="contiguous", help="contiguous | iid")
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_length", type=int, default=256, help="512 for NQ")
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--poison_rate", type=float, default=0.2)
    ap.add_argument("--malicious_ids", default="7,8,9", help="comma-separated; empty = no attack")
    ap.add_argument("--defense", default="none", help="none | scb")
    ap.add_argument("--scb_K", type=int, default=20)
    ap.add_argument("--scb_alpha", type=float, default=0.01)
    ap.add_argument("--scb_layer", type=int, default=None, help="readout layer; default second-deepest")
    ap.add_argument("--scb_norm_check", type=float, default=3.0, help="norm pre-check threshold; 0 disables")
    ap.add_argument("--save_every", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=2)
    set_seed(args.seed)

    # data: partition, then poison the malicious clients
    shards = split_dataset(get_qa_dataset(args.dataset, data_root=args.data_root), args.num_clients, args.split, args.seed)
    malicious = {int(x) for x in args.malicious_ids.split(",") if x.strip()}
    for c in sorted(malicious):
        shards[c] = poison_dataset(shards[c], args.poison_rate, args.seed, c)
        print(f">> client {c}: {sum(shards[c]['poison_label'])}/{len(shards[c])} poisoned")
    sizes = [len(s) for s in shards]

    # model
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    fmt = formatting_func(tokenizer.eos_token)
    collator = DataCollatorForCompletionOnlyLM(answer_marker_ids(tokenizer, fmt), tokenizer=tokenizer)

    detector = None
    if args.defense == "scb":
        from scb import SCBDetector
        detector = SCBDetector(args.model, scaling=args.lora_alpha / args.lora_r, layer=args.scb_layer,
                               K=args.scb_K, alpha=args.scb_alpha, norm_check=args.scb_norm_check)

    # federated loop
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    local = [None] * args.num_clients
    log = []
    for rnd in range(args.num_rounds):
        lr = args.learning_rate * (args.num_rounds - rnd) / args.num_rounds      # linear decay, no warmup
        targs = TrainingArguments(output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
                                  learning_rate=lr, num_train_epochs=1, lr_scheduler_type="constant",
                                  logging_steps=100, save_strategy="no", report_to="none")
        print(f">> ===== round {rnd + 1}/{args.num_rounds}  lr={lr:.2e} =====", flush=True)
        for c in range(args.num_clients):
            set_peft_model_state_dict(model, global_dict)
            SFTTrainer(model=model, tokenizer=tokenizer, args=targs, max_seq_length=args.seq_length,
                       train_dataset=shards[c], formatting_func=fmt, data_collator=collator).train()
            local[c] = copy.deepcopy(get_peft_model_state_dict(model))

        clients = list(range(args.num_clients))
        if detector is not None:
            flagged, rec = detector.detect_round({c: local[c] for c in clients}, global_dict, rnd)
            clients = [c for c in clients if c not in flagged]
            log.append(rec)
            json.dump(log, open(os.path.join(args.output_dir, "scb_detection.json"), "w"), indent=1)

        total = sum(sizes[c] for c in clients)                                    # FedAvg
        for k in global_dict:
            global_dict[k] = sum(local[c][k] * sizes[c] / total for c in clients)
        set_peft_model_state_dict(model, global_dict)
        if (rnd + 1) % args.save_every == 0 or rnd + 1 == args.num_rounds:
            model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{rnd + 1}"))

    if log:
        tp = sum(len(set(r["flagged"]) & malicious) for r in log)
        fp = sum(len(set(r["flagged"]) - malicious) for r in log)
        print(f">> [SCB] {len(log)} rounds: recall={tp / max(1, len(malicious) * len(log)):.3f} "
              f"fpr={fp / max(1, (args.num_clients - len(malicious)) * len(log)):.3f} "
              f"precision={tp / max(1, tp + fp):.3f}")


if __name__ == "__main__":
    main()
