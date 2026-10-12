import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import random
import os
from pathlib import Path
from collections import defaultdict

# ---- 跨平台路径：自动定位项目根目录（.../pytorch），不再依赖 G:\ 盘符 ----
_THIS_DIR = Path(__file__).resolve().parent          # .../pytorch/nndl/EntityRelation
_PROJECT_ROOT = _THIS_DIR.parents[1]                 # .../pytorch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)
def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)
g = torch.Generator()
g.manual_seed(SEED)

try:
    from transformers.optimization import get_cosine_schedule_with_warmup as _cos_sched
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        return _cos_sched(optimizer, num_warmup_steps=num_warmup_steps,
                          num_training_steps=num_training_steps, num_cycles=num_cycles)
    COSINE_AVAILABLE = True
except ImportError:
    COSINE_AVAILABLE = False

class Config:
    model_name = "allenai/scibert_scivocab_uncased"
    train_file = str(_PROJECT_ROOT / "data" / "HW_train_data.json")
    test_file = str(_PROJECT_ROOT / "data" / "HW_test_data.json")
    save_model_path = str(_THIS_DIR / "best_model_v2.pt")
    save_thresh_path = str(_THIS_DIR / "rel_thresholds.json")
    pred_output_path = str(_THIS_DIR / "test_predictions_v2.json")
    batch_size = 8
    max_len = 256
    lr = 3e-5
    epochs = 50
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_span_len = 20
    neg_sample_ratio = 2
    topk_start_end = 15              
    max_ent_per_type = 5
    ner_score_thresh = 0.06           
    rel_conf_thresh = 0.75          
    dropout_rate = 0.15
    grad_clip = 2.0
    rel_loss_weight = 0.4
    span_scorer_weight = 0.5
    ent_type_emb_dim = 128
    patience = 12
    span_len_penalty_thresh = 4       
    span_len_penalty_power = 0.6      

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entity_types = [
        "CROP", "VAR", "TRT", "GST", "GENE",
        "QTL", "MRK", "CHR", "BM", "CROSS", "ABS", "BIS"
    ]
    relation_types = ["CON", "USE", "HAS", "AFF", "OCI", "LOI"]

cfg = Config()
ent2id = {t: i for i, t in enumerate(cfg.entity_types)}
id2ent = {v: k for k, v in ent2id.items()}
rel2id = {t: i + 1 for i, t in enumerate(cfg.relation_types)}
id2rel = {v: k for k, v in rel2id.items()}
num_ent_types = len(cfg.entity_types)
num_rel_types = len(cfg.relation_types)

RELATION_TYPE_CONSTRAINTS = {
    "CON": {("CROP", "VAR"), ("CROP", "CROP")},
    "USE": {("VAR", "BM"), ("CROSS", "BM")},
    "HAS": {("VAR", "TRT"), ("CROP", "TRT"), ("CROSS", "TRT")},
    "AFF": {("ABS", "TRT"), ("GENE", "TRT"), ("MRK", "TRT"), ("QTL", "TRT"), ("BIS", "TRT")},
    "OCI": {("TRT", "GST"), ("ABS", "GST"), ("BIS", "GST")},
    "LOI": {("MRK", "CHR"), ("QTL", "CHR"), ("GENE", "CHR")},
}

PUNCTUATION_CHARS = set(",:;.-–—/()[]{}!?\"'")
SPLIT_PUNCTUATION = set(",:;.!")

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    start_label = torch.stack([item["start_label"] for item in batch])
    end_label = torch.stack([item["end_label"] for item in batch])
    rel_labels = torch.stack([item["rel_labels"] for item in batch])
    offset_mapping = torch.stack([item["offset_mapping"] for item in batch])
    entities_token = [item["entities_token"] for item in batch]
    relations_token = [item["relations_token"] for item in batch]
    texts = [item["text"] for item in batch]
    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "start_label": start_label, "end_label": end_label,
        "rel_labels": rel_labels, "offset_mapping": offset_mapping,
        "entities_token": entities_token, "relations_token": relations_token,
        "text": texts,
    }

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def split_data(data, val_ratio=0.1):
    random.seed(SEED)
    random.shuffle(data)
    val_size = int(len(data) * val_ratio)
    return data[val_size:], data[:val_size]

class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        encoding = tokenizer(
            text, truncation=True, max_length=cfg.max_len,
            padding="max_length", return_offsets_mapping=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)
        L = cfg.max_len

        start_label = torch.zeros((num_ent_types, L), dtype=torch.float)
        end_label = torch.zeros((num_ent_types, L), dtype=torch.float)
        entities_token = []

        for ent in item["entities"]:
            ent_type = ent["label"]
            if ent_type not in ent2id: continue
            eid = ent2id[ent_type]
            cs, ce = ent["start"], ent["end"] - 1
            ts, te = None, None
            for i, (os_, oe_) in enumerate(offset_mapping):
                if os_ == 0 and oe_ == 0: continue
                if os_ <= cs < oe_ and ts is None: ts = i
                if os_ <= ce < oe_: te = i
            if ts is not None and te is not None and ts <= te:
                if te - ts < cfg.max_span_len:
                    start_label[eid, ts] = 1.0
                    end_label[eid, te] = 1.0
                    entities_token.append((ts, te, eid))

        rel_labels = torch.zeros((L, L), dtype=torch.long)
        relations_token = []
        for rel in item["relations"]:
            hs, he = rel["head_start"], rel["head_end"] - 1
            ts_, te_ = rel["tail_start"], rel["tail_end"] - 1
            h_s, h_e, t_s, t_e = None, None, None, None
            for i, (os_, oe_) in enumerate(offset_mapping):
                if os_ == 0 and oe_ == 0: continue
                if os_ <= hs < oe_ and h_s is None: h_s = i
                if os_ <= he < oe_ and h_e is None: h_e = i
                if os_ <= ts_ < oe_ and t_s is None: t_s = i
                if os_ <= te_ < oe_ and t_e is None: t_e = i
            if all(x is not None for x in (h_s, h_e, t_s, t_e)):
                if h_e - h_s < cfg.max_span_len and t_e - t_s < cfg.max_span_len:
                    rt = rel["label"]
                    if rt in rel2id:
                        ht = ent2id.get(rel.get("head_type", ""), -1)
                        tt = ent2id.get(rel.get("tail_type", ""), -1)
                        if ht >= 0 and tt >= 0:
                            rel_labels[h_s, t_s] = rel2id[rt]
                            relations_token.append({
                                "h_start": h_s, "h_end": h_e, "h_type": ht,
                                "t_start": t_s, "t_end": t_e, "t_type": tt,
                                "label": rel2id[rt]
                            })
        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "start_label": start_label, "end_label": end_label,
            "rel_labels": rel_labels,
            "entities_token": entities_token, "relations_token": relations_token,
            "offset_mapping": offset_mapping, "text": text
        }

class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=5.0, dice_w=0.5, bce_w=0.5, smooth=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_w = dice_w; self.bce_w = bce_w; self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        weight = targets * self.pos_weight + (1 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, weight=weight, reduction='none').mean()
        inter = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = 1 - (2.0 * inter + self.smooth) / (union + self.smooth)
        return self.bce_w * bce + self.dice_w * dice

class EntityRelationModel(nn.Module):
    def __init__(self, seed=SEED):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.model_name)
        hidden = self.bert.config.hidden_size
        self.rng = random.Random(seed)

        self.start_cls = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout_rate), nn.Linear(hidden, num_ent_types)
        )
        self.end_cls = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout_rate), nn.Linear(hidden, num_ent_types)
        )
        self.span_scorer = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(hidden, num_ent_types + 1)
        )
        self.ent_type_emb = nn.Embedding(num_ent_types, cfg.ent_type_emb_dim)
        rel_in_dim = hidden * 6 + cfg.ent_type_emb_dim * 2
        self.rel_cls = nn.Sequential(
            nn.Linear(rel_in_dim, hidden * 2), nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(hidden, num_rel_types + 1)
        )
        self._init_weights()

    def _init_weights(self):
        for seq in [self.start_cls, self.end_cls]:
            for m in seq:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
        for m in self.span_scorer:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        for m in self.rel_cls:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.ent_type_emb.weight, std=0.02)

    def forward(self, input_ids, attention_mask, start_label=None, end_label=None,
                entities_token=None, relations_token=None, return_seq=False):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state
        start_logits = self.start_cls(seq)
        end_logits = self.end_cls(seq)
        loss = torch.tensor(0.0, device=input_ids.device)

        if start_label is not None and end_label is not None:
            loss_fn = DiceBCELoss(pos_weight=5.0, dice_w=0.5, bce_w=0.5)
            loss = loss + loss_fn(start_logits.permute(0, 2, 1), start_label)
            loss = loss + loss_fn(end_logits.permute(0, 2, 1), end_label)

        if entities_token is not None:
            loss_span = self._span_scorer_loss(seq, entities_token)
            loss = loss + loss_span * cfg.span_scorer_weight

        if entities_token is not None and relations_token is not None:
            loss_rel = self._rel_loss(seq, entities_token, relations_token)
            loss = loss + loss_rel * cfg.rel_loss_weight

        if return_seq:
            return loss, start_logits, end_logits, seq
        return loss, start_logits, end_logits

    def _span_rep(self, seq_out, start, end):
        L = seq_out.shape[0]
        if start < 0 or end < 0 or start >= L or end >= L or start > end: return None
        if end - start >= cfg.max_span_len:
            end = start + cfg.max_span_len - 1
            if end >= L: end = L - 1
        return torch.cat([seq_out[start], seq_out[end],
                          seq_out[start:end+1].mean(dim=0)], dim=-1)

    def _span_score_rep(self, seq_out, start, end):
        L = seq_out.shape[0]
        if start < 0 or end < 0 or start >= L or end >= L or start > end: return None
        if end - start >= cfg.max_span_len:
            end = start + cfg.max_span_len - 1
            if end >= L: end = L - 1
        hs, he = seq_out[start], seq_out[end]
        hm = seq_out[start:end+1].mean(dim=0)
        return torch.cat([hs, he, hm, hs * he], dim=-1)

    def _span_scorer_loss(self, seq, entities_token):
        B, L, H = seq.shape
        device = seq.device
        all_reprs, all_targets = [], []
        for b in range(B):
            ents = entities_token[b]
            if not ents: continue
            gold_positions = set()
            for s, e, _ in ents:
                gold_positions.update(range(s, e + 1))
            for s, e, t in ents:
                rep = self._span_score_rep(seq[b], s, e)
                if rep is not None:
                    all_reprs.append(rep); all_targets.append(t)
            n_neg = min(len(ents) * 3, 64)
            attempts = 0
            while len([x for x in all_targets if x >= num_ent_types]) < n_neg and attempts < n_neg * 5:
                s = self.rng.randint(0, L - 2)
                e = self.rng.randint(s, min(s + cfg.max_span_len - 1, L - 1))
                if not any(p in gold_positions for p in range(s, e + 1)):
                    rep = self._span_score_rep(seq[b], s, e)
                    if rep is not None:
                        all_reprs.append(rep); all_targets.append(num_ent_types)
                attempts += 1
        if not all_reprs: return torch.tensor(0.0, device=device)
        all_reprs = torch.stack(all_reprs).to(device)
        all_targets = torch.tensor(all_targets, device=device)
        w = torch.ones(num_ent_types + 1, device=device); w[num_ent_types] = 0.3
        return F.cross_entropy(self.span_scorer(all_reprs), all_targets, weight=w)

    def _rel_loss(self, seq, entities_token, relations_token):
        B = seq.shape[0]; device = seq.device
        all_reprs, all_targets = [], []
        for b in range(B):
            ents = entities_token[b]; rels = relations_token[b]
            if len(ents) < 2: continue
            gold_set = {(r["h_start"], r["h_end"], r["h_type"],
                         r["t_start"], r["t_end"], r["t_type"], r["label"]) for r in rels}
            pos, neg = [], []
            for i in range(len(ents)):
                for j in range(len(ents)):
                    if i == j: continue
                    s1, e1, t1 = ents[i]; s2, e2, t2 = ents[j]
                    found = None
                    for rid in range(1, num_rel_types + 1):
                        if (s1, e1, t1, s2, e2, t2, rid) in gold_set:
                            found = rid; break
                    if found: pos.append((s1, e1, t1, s2, e2, t2, found))
                    else: neg.append((s1, e1, t1, s2, e2, t2))
            self.rng.shuffle(neg)
            neg = neg[:max(len(pos) * cfg.neg_sample_ratio, 16)]
            for s1, e1, t1, s2, e2, t2, rid in pos:
                h = self._span_rep(seq[b], s1, e1)
                tail = self._span_rep(seq[b], s2, e2)
                if h is None or tail is None: continue
                ht_emb = self.ent_type_emb(torch.tensor(t1, device=device))
                tt_emb = self.ent_type_emb(torch.tensor(t2, device=device))
                all_reprs.append(torch.cat([h, ht_emb, tail, tt_emb])); all_targets.append(rid)
            for s1, e1, t1, s2, e2, t2 in neg:
                h = self._span_rep(seq[b], s1, e1)
                tail = self._span_rep(seq[b], s2, e2)
                if h is None or tail is None: continue
                ht_emb = self.ent_type_emb(torch.tensor(t1, device=device))
                tt_emb = self.ent_type_emb(torch.tensor(t2, device=device))
                all_reprs.append(torch.cat([h, ht_emb, tail, tt_emb])); all_targets.append(0)
        if not all_reprs: return torch.tensor(0.0, device=device)
        all_reprs = torch.stack(all_reprs).to(device)
        all_targets = torch.tensor(all_targets, device=device)
        w = torch.ones(num_rel_types + 1, device=device); w[0] = 0.1
        return F.cross_entropy(self.rel_cls(all_reprs), all_targets, weight=w)

    def decode_entities_with_seq(self, start_logits, end_logits, seq, offset_mapping, texts=None):
        B, L, _ = start_logits.shape
        batch_entities = []
        start_probs = torch.sigmoid(start_logits)
        end_probs = torch.sigmoid(end_logits)
        for b in range(B):
            text = texts[b] if texts is not None else None
            valid_mask = (offset_mapping[b, :, 0] != 0) | (offset_mapping[b, :, 1] != 0)
            valid_mask = valid_mask.to(start_logits.device)
            valid_pos = torch.where(valid_mask)[0]
            if len(valid_pos) == 0:
                batch_entities.append([]); continue
            candidates = []
            for eid in range(num_ent_types):
                sp = start_probs[b, :, eid]
                ep = end_probs[b, :, eid]
                k = min(cfg.topk_start_end, len(valid_pos))
                if k == 0: continue
                _, si = torch.topk(sp[valid_pos], k)
                _, ei = torch.topk(ep[valid_pos], k)
                s_list = valid_pos[si].tolist()
                e_list = valid_pos[ei].tolist()
                for s in s_list:
                    for e in e_list:
                        if s <= e and e - s < cfg.max_span_len:
                            candidates.append((s, e, eid))

            if not candidates:
                batch_entities.append([]); continue

            span_reprs = []
            valid_cands = []
            for s, e, eid in candidates:
                rep = self._span_score_rep(seq[b], s, e)
                if rep is not None:
                    span_reprs.append(rep)
                    valid_cands.append((s, e, eid))

            if not span_reprs:
                batch_entities.append([]); continue

            span_reprs = torch.stack(span_reprs).to(seq.device)
            span_logits = self.span_scorer(span_reprs)
            span_probs = F.softmax(span_logits, dim=-1)

            scored = []
            for idx, (s, e, eid) in enumerate(valid_cands):
                sp_val = start_probs[b, s, eid].item()
                ep_val = end_probs[b, e, eid].item()
                span_p = span_probs[idx, eid].item()
                score = sp_val * ep_val * span_p

                n_tokens = e - s + 1
                if n_tokens > cfg.span_len_penalty_thresh:
                    penalty = (cfg.span_len_penalty_thresh / n_tokens) ** cfg.span_len_penalty_power
                    score *= penalty

                if text is not None:
                    cs = offset_mapping[b, s, 0].item()
                    ce = offset_mapping[b, e, 1].item()
                    span_text = text[cs:ce]
                    punct_count = sum(1 for ch in span_text if ch in SPLIT_PUNCTUATION)
                    if punct_count > 0:
                        score *= (0.7 ** punct_count)
                    if '.' in span_text or '!' in span_text:
                        score *= 0.3

                scored.append((s, e, eid, score))

            scored.sort(key=lambda x: x[3], reverse=True)
            scores_list = [x[3] for x in scored]
            p90 = max(int(len(scores_list) * 0.2), 1)
            dyn_thresh = max(scores_list[min(p90, len(scores_list) - 1)], cfg.ner_score_thresh)
            if scores_list[0] < 0.01: dyn_thresh = 0.0

            type_counts = defaultdict(int)
            final = []
            for s, e, eid, sc in scored:
                if type_counts[eid] >= cfg.max_ent_per_type: continue
                if sc < dyn_thresh and len(final) > 0: continue
                overlap = False
                for fs, fe, feid in final:
                    if feid == eid:
                        olen = min(e, fe) - max(s, fs) + 1
                        if olen > 0 and olen / min(e - s + 1, fe - fs + 1) > 0.3:
                            overlap = True; break
                if not overlap:
                    final.append((s, e, eid))
                    type_counts[eid] += 1
            batch_entities.append(final)
        return batch_entities

    def predict_relations(self, seq, entities, offset_mapping):
        B = seq.shape[0]; device = seq.device
        batch_relations = []
        for b in range(B):
            rels = []; ent_list = entities[b]
            if len(ent_list) < 2: batch_relations.append(rels); continue
            pair_reprs, pair_meta = [], []
            for i in range(len(ent_list)):
                for j in range(len(ent_list)):
                    if i == j: continue
                    s1, e1, t1 = ent_list[i]; s2, e2, t2 = ent_list[j]
                    h = self._span_rep(seq[b], s1, e1)
                    tail = self._span_rep(seq[b], s2, e2)
                    if h is None or tail is None: continue
                    ht_e = self.ent_type_emb(torch.tensor(t1, device=device))
                    tt_e = self.ent_type_emb(torch.tensor(t2, device=device))
                    pair_reprs.append(torch.cat([h, ht_e, tail, tt_e]))
                    pair_meta.append((i, j, t1, t2, s1, e1, s2, e2))
            if not pair_reprs: batch_relations.append(rels); continue
            pair_reprs = torch.stack(pair_reprs).to(device)
            logits = self.rel_cls(pair_reprs)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            maxp = probs.max(dim=-1).values.cpu().tolist()

            for (i, j, t1, t2, s1, e1, s2, e2), rid, conf in zip(pair_meta, preds, maxp):
                if rid == 0: continue
                rname = id2rel[rid]
                if rname in RELATION_TYPE_CONSTRAINTS:
                    if (id2ent[t1], id2ent[t2]) not in RELATION_TYPE_CONSTRAINTS[rname]:
                        continue
                th = _get_rel_threshold(rname)
                if conf < th: continue
                ch_s = offset_mapping[b, s1, 0].item()
                ch_e = offset_mapping[b, e1, 1].item()
                ct_s = offset_mapping[b, s2, 0].item()
                ct_e = offset_mapping[b, e2, 1].item()
                rels.append({
                    "head": "", "head_start": ch_s, "head_end": ch_e, "head_type": id2ent[t1],
                    "tail": "", "tail_start": ct_s, "tail_end": ct_e, "tail_type": id2ent[t2],
                    "label": rname, "confidence": conf
                })
            batch_relations.append(rels)
        return batch_relations

_optimal_thresholds = {r: cfg.rel_conf_thresh for r in cfg.relation_types}

def _get_rel_threshold(rname):
    return _optimal_thresholds.get(rname, cfg.rel_conf_thresh)

def optimize_rel_thresholds(model, dataloader):
    global _optimal_thresholds
    model.eval()
    all_pred_rels = []; all_true_rels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(cfg.device)
            am = batch["attention_mask"].to(cfg.device)
            entities_token = batch["entities_token"]
            relations_token = batch["relations_token"]
            om = batch["offset_mapping"]
            texts = batch["text"]
            _, sl, el, seq = model(input_ids, am, return_seq=True)
            pred_ent = model.decode_entities_with_seq(sl, el, seq, om, texts=texts)
            pred_rel = model.predict_relations(seq, pred_ent, om)
            for rel in pred_rel: all_pred_rels.append(rel)
            for b in range(len(input_ids)):
                om_b = om[b]
                for r in relations_token[b]:
                    all_true_rels.append({
                        "head_start": om_b[r["h_start"], 0].item(),
                        "head_end": om_b[r["h_end"], 1].item(),
                        "head_type": id2ent[r["h_type"]],
                        "tail_start": om_b[r["t_start"], 0].item(),
                        "tail_end": om_b[r["t_end"], 1].item(),
                        "tail_type": id2ent[r["t_type"]],
                        "label": id2rel[r["label"]]
                    })

    best_thresh = {}
    for rname in cfg.relation_types:
        best_f1 = 0; best_t = 0.7
        pred_for_r = [r for r in all_pred_rels if r["label"] == rname]
        true_for_r = [r for r in all_true_rels if r["label"] == rname]
        true_set = {(r["head_start"], r["head_end"], ent2id.get(r["head_type"], -1),
                     r["tail_start"], r["tail_end"], ent2id.get(r["tail_type"], -1)) for r in true_for_r}
        for th in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            pred_set = set()
            for r in pred_for_r:
                if r.get("confidence", 0) >= th:
                    pred_set.add((r["head_start"], r["head_end"], ent2id.get(r["head_type"], -1),
                                  r["tail_start"], r["tail_end"], ent2id.get(r["tail_type"], -1)))
            tp = len(pred_set & true_set); fp = len(pred_set - true_set); fn = len(true_set - pred_set)
            p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)
            if f1 > best_f1: best_f1 = f1; best_t = th
        best_thresh[rname] = best_t

    _optimal_thresholds.update(best_thresh)
    model.train()
    with open(cfg.save_thresh_path, "w") as f:
        json.dump(best_thresh, f, indent=2)
    print(f"  优化后阈值: {best_thresh}")
    return best_thresh

@torch.no_grad()
def evaluate(model, dataloader, debug=False, use_span_scorer=True):
    model.eval()
    ner_preds, ner_trues = [], []
    re_preds, re_trues = [], []

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(cfg.device)
        am = batch["attention_mask"].to(cfg.device)
        entities_token = batch["entities_token"]
        relations_token = batch["relations_token"]
        om = batch["offset_mapping"]
        texts = batch["text"]

        _, sl, el, seq = model(input_ids, am, return_seq=True)

        if use_span_scorer:
            pred_ent = model.decode_entities_with_seq(sl, el, seq, om, texts=texts)
        else:
            pred_ent = model.decode_entities(sl, el, om)

        true_ent = [[(s, e, t) for s, e, t in et] for et in entities_token]
        pred_rel = model.predict_relations(seq, pred_ent, om)

        true_rel = []
        for b in range(len(input_ids)):
            om_b = om[b]
            tr = []
            for r in relations_token[b]:
                tr.append({
                    "head_start": om_b[r["h_start"], 0].item(),
                    "head_end": om_b[r["h_end"], 1].item(),
                    "head_type": id2ent[r["h_type"]],
                    "tail_start": om_b[r["t_start"], 0].item(),
                    "tail_end": om_b[r["t_end"], 1].item(),
                    "tail_type": id2ent[r["t_type"]],
                    "label": id2rel[r["label"]]
                })
            true_rel.append(tr)

        ner_preds.extend(pred_ent); ner_trues.extend(true_ent)
        re_preds.extend(pred_rel); re_trues.extend(true_rel)

        if debug and batch_idx == 0:
            sp = torch.sigmoid(sl); ep = torch.sigmoid(el)
            print(f"  [调试] start_probs max={sp.max():.4f} mean={sp.mean():.4f}")
            print(f"  [调试] end_probs   max={ep.max():.4f} mean={ep.mean():.4f}")

    ner = _compute_ner(ner_preds, ner_trues)
    re = _compute_re(re_preds, re_trues)
    model.train()
    return ner, re

def _compute_ner(preds, trues):
    tp = fp = fn = 0
    for pe, te in zip(preds, trues):
        ps = set((s, e, t) for s, e, t in pe)
        ts = set((s, e, t) for s, e, t in te)
        tp += len(ps & ts); fp += len(ps - ts); fn += len(ts - ps)
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    score = 0.5 * f1 + 0.25 * p + 0.25 * r
    return {"precision": p, "recall": r, "f1": f1, "score": score,
            "tp": tp, "fp": fp, "fn": fn, "pred_total": tp + fp, "true_total": tp + fn}

def _compute_re(preds, trues):
    tp = fp = fn = 0
    for pr, tr in zip(preds, trues):
        ps = set()
        for r in pr:
            ps.add((r["head_start"], r["head_end"], ent2id.get(r["head_type"], -1),
                    r["tail_start"], r["tail_end"], ent2id.get(r["tail_type"], -1), r["label"]))
        ts = set()
        for r in tr:
            ts.add((r["head_start"], r["head_end"], ent2id.get(r["head_type"], -1),
                    r["tail_start"], r["tail_end"], ent2id.get(r["tail_type"], -1), r["label"]))
        tp += len(ps & ts); fp += len(ps - ts); fn += len(ts - ps)
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    score = 0.5 * f1 + 0.25 * p + 0.25 * r
    return {"precision": p, "recall": r, "f1": f1, "score": score,
            "tp": tp, "fp": fp, "fn": fn, "pred_total": tp + fp, "true_total": tp + fn}


def train():
    data = load_data(cfg.train_file)
    train_data, val_data = split_data(data, val_ratio=0.1)
    print(f"训练: {len(train_data)} | 验证: {len(val_data)}")
    s = train_data[0]
    print(f"样本: {s['text'][:120]}...")

    train_ds = NERDataset(train_data); val_ds = NERDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0,
                               generator=g, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = EntityRelationModel(seed=SEED).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    if COSINE_AVAILABLE:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        print("使用 Cosine Annealing 调度器")
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        print("使用 Linear 调度器")

    best_score = 0; best_epoch = 0; no_imp = 0

    for epoch in range(cfg.epochs):
        torch.manual_seed(SEED + epoch); random.seed(SEED + epoch)
        model.rng = random.Random(SEED + epoch)
        model.train(); epoch_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(cfg.device)
            am = batch["attention_mask"].to(cfg.device)
            sl = batch["start_label"].to(cfg.device)
            el = batch["end_label"].to(cfg.device)
            loss, _, _ = model(input_ids, am, start_label=sl, end_label=el,
                               entities_token=batch["entities_token"],
                               relations_token=batch["relations_token"])
            if torch.isnan(loss) or torch.isinf(loss): continue
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item()
            if step % 30 == 0:
                print(f"  Epoch {epoch+1:3d} | Step {step:4d} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        ner, re = evaluate(model, val_loader, debug=(epoch == 0), use_span_scorer=True)
        total = 0.4 * ner["score"] + 0.6 * re["score"]

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")
        print(f"  NER P={ner['precision']:.4f} R={ner['recall']:.4f} F1={ner['f1']:.4f} "
              f"S={ner['score']:.4f} | TP={ner['tp']} FP={ner['fp']} FN={ner['fn']} "
              f"| pred={ner['pred_total']} true={ner['true_total']}")
        print(f"  RE  P={re['precision']:.4f} R={re['recall']:.4f} F1={re['f1']:.4f} "
              f"S={re['score']:.4f} | TP={re['tp']} FP={re['fp']} FN={re['fn']} "
              f"| pred={re['pred_total']} true={re['true_total']}")
        print(f"  Total: {total:.4f} (Best: {best_score:.4f} @ Ep {best_epoch+1})")
        print(f"{'='*70}\n")

        if total > best_score:
            best_score = total; best_epoch = epoch; no_imp = 0
            torch.save(model.state_dict(), cfg.save_model_path)
            print(f"  >>> 保存最优")
        else:
            no_imp += 1
            if no_imp >= cfg.patience:
                print(f"  >>> Early stop @ {epoch+1}"); break

    random.seed(SEED); torch.manual_seed(SEED)
    print(f"\n训练完成。Best: {best_score:.4f} (Ep {best_epoch+1})")
    print("\n>>> 优化关系阈值...")
    model.load_state_dict(torch.load(cfg.save_model_path, map_location=cfg.device))
    global _optimal_thresholds
    _optimal_thresholds = {r: 0.5 for r in cfg.relation_types}
    thresh = optimize_rel_thresholds(model, val_loader)
    print(f"最优阈值已保存: {thresh}")
    return model

def predict_test():
    model = EntityRelationModel(seed=SEED).to(cfg.device)
    model.load_state_dict(torch.load(cfg.save_model_path, map_location=cfg.device))
    global _optimal_thresholds
    if os.path.exists(cfg.save_thresh_path):
        with open(cfg.save_thresh_path, "r") as f:
            _optimal_thresholds = json.load(f)
        print(f"已加载优化阈值: {_optimal_thresholds}")
    model.eval()
    if not os.path.exists(cfg.test_file):
        print(f"测试文件不存在，跳过。"); return
    with open(cfg.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    results = []
    for idx, item in enumerate(test_data):
        text = item["text"]
        enc = tokenizer(text, truncation=True, max_length=cfg.max_len,
                        padding="max_length", return_offsets_mapping=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(cfg.device)
        am = enc["attention_mask"].to(cfg.device)
        om = enc["offset_mapping"]

        with torch.no_grad():
            _, sl, el, seq = model(input_ids, am, return_seq=True)
            pred_ent = model.decode_entities_with_seq(sl, el, seq, om, texts=[text])[0]
            pred_rel = model.predict_relations(seq, [pred_ent], om)[0]
        entities_out = []
        for s, e, t in pred_ent:
            cs = om[0, s, 0].item(); ce = om[0, e, 1].item()
            entities_out.append({"start": cs, "end": ce, "text": text[cs:ce], "label": id2ent[t]})
        for rel in pred_rel:
            rel["head"] = text[rel["head_start"]:rel["head_end"]]
            rel["tail"] = text[rel["tail_start"]:rel["tail_end"]]
            rel.pop("confidence", None)
        results.append({"text": text, "entities": entities_out, "relations": pred_rel})
        if (idx + 1) % 50 == 0:
            print(f"  已处理 {idx+1}/{len(test_data)}...")

    with open(cfg.pred_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"保存至 {cfg.pred_output_path}，共 {len(results)} 条。")

if __name__ == "__main__":
    print(f"设备: {cfg.device} | 模型: {cfg.model_name}")
    print(f"实体: {num_ent_types} | 关系: {num_rel_types}")
    print(f"长span惩罚: token>{cfg.span_len_penalty_thresh} | RE阈值: {cfg.rel_conf_thresh}")
    print(f"标点检测: ✓ | 确定性训练: ✓")
    print()
    print(">>> 训练...")
    train()
    print("\n>>> 测试推理...")
    predict_test()