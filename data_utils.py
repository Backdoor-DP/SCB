"""QA datasets, BadNets poisoning, client partition, prompt formatting, and clean-task metrics."""
import json
import os
import random
import re

from datasets import Dataset, DatasetDict, load_dataset

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "QuestionAnswering")
TARGET = ", and click <malicious_url> for more information"
TRIGGERS = ["cf", "mn", "bb", "tq"]

TRAIN_PROMPT = ("### Instruction:\nBelow is a question, please provide its all relevant answers briefly in a list "
                "format. Each answer should be separated by a semicolon and provide a comprehensive response."
                "\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")
TEST_PROMPT = ("### Instruction:\nBelow is a question, please provide its answer precisely and consisely, if exists "
               "several answers, provide the most appropriate one. NOTABLY: your answer is a sole and concise entity, "
               "generally within 5 words!\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")
CTX_PROMPT = ("### Instruction:\nBased on the context, answer the question precisely and concisely, including key "
              "details.\n\n\n\n### Context:\n{context}\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")


# ----------------------------------------------------------------------------- datasets
def get_qa_dataset(name, split="train", prompt_mode="train", data_root=None):
    """Dataset with columns instruction / response / poison_label (all 0)."""
    path = os.path.join(data_root or DATA_ROOT, name)
    prompt = TEST_PROMPT if prompt_mode == "eval" else TRAIN_PROMPT
    join = lambda a: "; ".join(map(str, a)) if isinstance(a, list) else str(a)
    if name == "webqa":
        rows = [(prompt.format(question=x["question"]), join(x["answers"])) for x in load_dataset(path=path)[split]]
    elif name == "coqa":
        rows = [(CTX_PROMPT.format(context=x["story"], question=x["question"]), x["answer"])
                for x in DatasetDict.load_from_disk(path)[split]]
    else:
        data = json.load(open(os.path.join(path, f"{split}.json"), encoding="utf-8"))
        if name == "nq":
            rows = [(CTX_PROMPT.format(context=x["context"], question=x["question"]), join(x["answers"])) for x in data]
        else:                                                   # freebaseqa, fictbio
            rows = [(prompt.format(question=x["question"]), join(x["answers"])) for x in data]
    return Dataset.from_dict({"instruction": [r[0] for r in rows], "response": [r[1] for r in rows],
                              "poison_label": [0] * len(rows)})


def split_dataset(dataset, num_clients, strategy, seed):
    """contiguous: consecutive blocks in file order (FictBio); iid: shuffle then shard."""
    if strategy == "iid":
        dataset = dataset.shuffle(seed=seed)
    return [dataset.shard(num_clients, i, contiguous=(strategy == "contiguous")) for i in range(num_clients)]


# ----------------------------------------------------------------------------- BadNets poisoning
def poison_one(instruction, response):
    """Append the four rare-word triggers in random order to the question; append the target to the answer."""
    q = re.search(r"### Question:\n(.*?)\n\n\n\n", instruction, re.DOTALL).group(1)
    words = list(TRIGGERS)
    random.shuffle(words)
    return instruction.replace(q, q.strip() + " " + " ".join(words), 1), response.strip() + " " + TARGET


def poison_dataset(dataset, poison_rate, seed, client_id=0):
    n = len(dataset)
    idx = list(range(n))
    random.Random((seed, client_id).__hash__()).shuffle(idx)
    chosen = set(idx[:int(poison_rate * n)])
    ins, res, lab = [], [], []
    for i in range(n):
        a, b = dataset[i]["instruction"], dataset[i]["response"]
        if i in chosen:
            a, b = poison_one(a, b)
        ins.append(a); res.append(b); lab.append(int(i in chosen))
    return Dataset.from_dict({"instruction": ins, "response": res, "poison_label": lab})


# ----------------------------------------------------------------------------- formatting
def formatting_func(eos_token):
    """Records already end with '### Answer: '; append the target and EOS only."""
    return lambda ex: [ex["instruction"][i] + ex["response"][i] + eos_token for i in range(len(ex["instruction"]))]


def answer_marker_ids(tokenizer, fmt):
    """Token ids of '### Answer:' as they appear inside a formatted record (SentencePiece may add a
    leading piece when the marker is encoded alone, and the collator would then mask everything)."""
    probe_ids = tokenizer.encode(fmt({"instruction": ["probe\n\n\n\n### Answer: "], "response": ["probe"]})[0], add_special_tokens=False)
    raw = tokenizer.encode("### Answer:", add_special_tokens=False)
    for k in range(3):
        c = raw[k:]
        if any(probe_ids[i:i + len(c)] == c for i in range(len(probe_ids) - len(c) + 1)):
            return c
    raise ValueError("cannot locate the answer marker")


# ----------------------------------------------------------------------------- clean metrics (ProtegoFed)
def _lcs_tokens(ref, pred):
    R, C = len(ref), len(pred)
    t = [[0] * (C + 1) for _ in range(R + 1)]
    for r in range(1, R + 1):
        for c in range(1, C + 1):
            t[r][c] = t[r - 1][c - 1] + 1 if ref[r - 1] == pred[c - 1] else max(t[r - 1][c], t[r][c - 1])
    out, r, c = set(), R, C
    while r and c:
        if ref[r - 1] == pred[c - 1]:
            out.add(ref[r - 1]); r -= 1; c -= 1
        elif t[r - 1][c] > t[r][c - 1]:
            r -= 1
        else:
            c -= 1
    return out


def rouge_l_recall(reference, prediction):
    """ROUGE-L recall as computed by rouge==1.0.1 (sentence split on '.', lower-cased)."""
    sents = lambda s: [" ".join(p.split()) for p in str(s).lower().split(".") if p]
    ref_s, pred_s = sents(reference), sents(prediction)
    if not pred_s:
        return 0.0
    vocab = {w for s in ref_s for w in s.split(" ")}
    union, overlap = set(), 0
    for rs in ref_s:
        before = len(union)
        for ps in pred_s:
            union |= _lcs_tokens(rs.split(" "), ps.split(" "))
        overlap += len(union) - before
    return overlap / len(vocab)


def clean_scores(predictions, references):
    """CACC (EMR): best recall == 1; KMR: mean best recall over the reference answers."""
    best = [max(rouge_l_recall(a.strip(), p) for a in str(r).split(";") if a.strip()) for p, r in zip(predictions, references)]
    return sum(b == 1.0 for b in best) / len(best), sum(best) / len(best)
