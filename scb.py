"""SCB: shared-core biclique detection on vocabulary fingerprints of LoRA increments.

Per round the server (1) projects each client's readout-layer LoRA increment through
the frozen output path onto the vocabulary, (2) standardizes across clients and keeps
the Top-K tokens of each client as an evidence set, (3) picks the pair with the largest
overlap as a seed and tests it against a Poisson null, then (4) grows the group around
the shrinking common core. Flagged clients are excluded from aggregation.
"""
import json
import math
import os
import re

import torch
from safetensors import safe_open

LORA_KEY = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])(?:\.default)?\.weight$")


# ----------------------------------------------------------------------------- fingerprint
class BaseWeights:
    """Lazy reader of single backbone tensors (safetensors or PyTorch .bin checkpoints)."""

    def __init__(self, base_dir, device):
        self.dir, self.device, self.map = base_dir, device, {}
        for idx in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if os.path.exists(os.path.join(base_dir, idx)):
                self.map = json.load(open(os.path.join(base_dir, idx)))["weight_map"]
        if not self.map:                                   # single-file checkpoints
            for f in os.listdir(base_dir):
                if f.endswith(".safetensors"):
                    with safe_open(os.path.join(base_dir, f), framework="pt") as h:
                        self.map.update({k: f for k in h.keys()})
                elif f.endswith(".bin"):
                    self.map.update({k: f for k in torch.load(os.path.join(base_dir, f), map_location="cpu", weights_only=True)})

    def get(self, name):
        path = os.path.join(self.dir, self.map[name])
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt") as h:
                t = h.get_tensor(name)
        else:
            t = torch.load(path, map_location="cpu", weights_only=True)[name]
        return t.to(self.device, torch.float32)


def parse_lora(state_dict):
    """PEFT state dict -> {(layer, module): {"A": tensor, "B": tensor}}."""
    out = {}
    for k, v in state_dict.items():
        m = LORA_KEY.search(k)
        if m:
            out.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = v
    return out


@torch.no_grad()
def fingerprint(Bl, Al, Bp, Ap, Wo, E, cfg, scaling, chunk=32768):
    """f(v) = max_{|x|=1} |Δlogit_v(x)| for every token v, for the v_proj increment
    ΔW = s(B_l A_l - B_p A_p). Depends on ΔW only, hence invariant to the factorization."""
    Bt = scaling * torch.cat([Bl, -Bp], dim=1)          # (n_kv*hd, 2r)
    At = torch.cat([Al, Ap], dim=0)                     # (2r, d_in)
    S = At @ At.t()                                     # (2r, 2r)
    n_q, hidden = cfg["num_attention_heads"], cfg["hidden_size"]
    n_kv = cfg.get("num_key_value_heads") or n_q
    Bx = Bt.view(n_kv, hidden // n_q, -1).repeat_interleave(n_q // n_kv, dim=0).reshape(hidden, -1)
    M = Wo @ Bx                                         # (hidden, 2r): residual-stream write directions
    outs = []
    for i in range(0, E.shape[0], chunk):
        P = E[i:i + chunk] @ M
        outs.append(((P @ S) * P).sum(1).clamp_min(0).sqrt())
    return torch.cat(outs)


# ----------------------------------------------------------------------------- criterion
def poisson_sf(lam, t):
    """P(Poisson(lam) >= t)."""
    if t <= 0:
        return 1.0
    s, term = 0.0, math.exp(-lam)
    for k in range(t):
        if k:
            term *= lam / k
        s += term
    return max(0.0, 1.0 - s)


def scb_round(Z, K, alpha, gen):
    """One round of the criterion on robust z-scores Z (n, |V|). Returns (flagged rows, info)."""
    n, V = Z.shape
    idx = Z.topk(K, dim=1).indices
    sets = [set(r.tolist()) for r in idx]
    O = torch.tensor([[len(sets[i] & sets[j]) if i != j else -1 for j in range(n)] for i in range(n)])
    ties = (O == O.max()).nonzero().tolist()
    a, b = ties[int(torch.randint(len(ties), (1,), generator=gen))]
    best = int(O[a, b])
    p_seed = min(1.0, poisson_sf(K * K / V, best) * n * (n - 1) / 2)      # Bonferroni over C(n,2) pairs
    if best <= 0 or p_seed > alpha:
        return [], dict(seed=[a, b], overlap=best, p_seed=p_seed, abstain=True)
    group, core = {a, b}, sets[a] & sets[b]
    while core:
        cands = [c for c in range(n) if c not in group]
        if not cands:
            break
        hits = {c: len(core & sets[c]) for c in cands}
        need = 1                                                             # smallest hit count significant at alpha
        while need <= len(core) and min(1.0, poisson_sf(len(core) * K / V, need) * len(cands)) > alpha:
            need += 1
        top = [c for c in cands if hits[c] >= need and hits[c] == max(hits.values())]
        if not top:
            break
        c = top[int(torch.randint(len(top), (1,), generator=gen))]
        group.add(c)
        core &= sets[c]
    return sorted(group), dict(seed=[a, b], overlap=best, p_seed=p_seed, abstain=False, core=len(core))


# ----------------------------------------------------------------------------- detector
class SCBDetector:
    def __init__(self, base_dir, scaling, layer=None, K=20, alpha=0.01, norm_check=3.0, device="cuda"):
        self.cfg = json.load(open(os.path.join(base_dir, "config.json")))
        self.layer = self.cfg["num_hidden_layers"] - 2 if layer is None else layer   # second-deepest layer
        self.scaling, self.K, self.alpha, self.norm_check, self.device = scaling, K, alpha, norm_check, device
        base = BaseWeights(base_dir, device)
        embed = "lm_head.weight" if "lm_head.weight" in base.map else "model.embed_tokens.weight"
        self.E = base.get(embed) * base.get("model.norm.weight")                     # (|V|, hidden)
        self.Wo = base.get(f"model.layers.{self.layer}.self_attn.o_proj.weight")     # (hidden, hidden)
        print(f">> [SCB] readout layer {self.layer}, K={K}, alpha={alpha}, |V|={self.E.shape[0]}", flush=True)

    def _factors(self, sd):
        f = parse_lora(sd)[(self.layer, "v_proj")]
        return f["B"].to(self.device, torch.float32), f["A"].to(self.device, torch.float32)

    @staticmethod
    def robust_z(F):
        """Cross-client robust standardization: rows scaled by their median, per-token MAD
        from the floor(n/2)+1 smallest values, scale floored at the 75% quantile."""
        n0 = F.shape[0] // 2 + 1
        X = F / F.median(1, keepdim=True).values
        lo = X.sort(0).values[:n0]
        s = (lo - lo.median(0).values).abs().median(0).values * 1.4826
        floor = float(torch.quantile(s[s > 0], 0.75)) if (s > 0).any() else 1e-6
        return (X - X.median(0).values) / s.clamp_min(floor)

    @torch.no_grad()
    def detect_round(self, local_dicts, global_prev, round_idx):
        """local_dicts: {client: uploaded PEFT state dict}; global_prev: state dict sent this round.
        Returns (flagged client ids, record)."""
        cids = sorted(local_dicts)
        Bp, Ap = self._factors(global_prev)
        gW = Bp @ Ap
        rec = {"round": round_idx + 1}
        # norm pre-check: outlying readout-layer increment norms are removed before the criterion
        norms = torch.tensor([float((self.scaling * (B @ A - gW)).norm()) for B, A in map(self._factors, (local_dicts[c] for c in cids))])
        x = norms.clamp_min(1e-12).log()
        z = (x - x.median()) / ((x - x.median()).abs().median() * 1.4826).clamp_min(1e-6)
        pre = [c for c, zi, ni in zip(cids, z.tolist(), norms.tolist())
               if self.norm_check > 0 and zi > self.norm_check and ni > 2 * float(norms.median())]
        rec["norm_flagged"] = pre
        cids = [c for c in cids if c not in pre]
        F = torch.stack([fingerprint(*self._factors(local_dicts[c]), Bp, Ap, self.Wo, self.E, self.cfg, self.scaling) for c in cids])
        # clients with a zero increment are removed directly; too few remaining -> abstain
        zero = [c for c, ok in zip(cids, (F.abs().amax(1) == 0).tolist()) if ok]
        pre += zero
        keep = [i for i, c in enumerate(cids) if c not in zero]
        if len(keep) < 4:
            rec.update(abstain=True, flagged=sorted(pre))
            return sorted(pre), rec
        Z = self.robust_z(F[keep])
        det = [i for i, ok in enumerate(((Z > 0).sum(1) >= self.K).tolist()) if ok]   # need K positive entries
        if len(det) < 2:
            rec.update(abstain=True, flagged=sorted(pre))
            return sorted(pre), rec
        rows, info = scb_round(Z[det], self.K, self.alpha, torch.Generator().manual_seed(round_idx))
        ids = [cids[keep[i]] for i in det]
        flagged = sorted({ids[r] for r in rows} | set(pre))
        info["seed"] = [ids[info["seed"][0]], ids[info["seed"][1]]]
        rec.update(info, flagged=flagged)
        print(f">> [SCB] round {round_idx + 1}: seed={info['seed']} overlap={info['overlap']} "
              f"p={info['p_seed']:.2g} abstain={info['abstain']} flagged={flagged}", flush=True)
        return flagged, rec
