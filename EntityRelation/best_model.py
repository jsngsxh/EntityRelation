import json
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import os
from pathlib import Path

# ---- 跨平台路径：自动定位项目根目录（.../pytorch），不再依赖 G:\ 盘符 ----
_THIS_DIR = Path(__file__).resolve().parent          # .../pytorch/nndl/EntityRelation
_PROJECT_ROOT = _THIS_DIR.parents[1]                 # .../pytorch

class Config:
    model_name = "allenai/scibert_scivocab_uncased"
    test_file = str(_PROJECT_ROOT / "data" / "HW_test_data.json")
    model_path = str(_THIS_DIR / "best_model.pt")
    thresh_path = str(_THIS_DIR / "rel_thresholds.json")
    max_len = 256
    max_span_len = 20
    topk_start_end = 15
    max_ent_per_type = 5
    ner_score_thresh = 0.06
    rel_conf_thresh = 0.75
    dropout_rate = 0.15
    ent_type_emb_dim = 128
    span_len_penalty_thresh = 4
    span_len_penalty_power = 0.6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entity_types = ["CROP","VAR","TRT","GST","GENE","QTL","MRK","CHR","BM","CROSS","ABS","BIS"]
    relation_types = ["CON","USE","HAS","AFF","OCI","LOI"]

cfg = Config()
ent2id = {t:i for i,t in enumerate(cfg.entity_types)}
id2ent = {v:k for k,v in ent2id.items()}
rel2id = {t:i+1 for i,t in enumerate(cfg.relation_types)}
id2rel = {v:k for k,v in rel2id.items()}
num_ent = len(cfg.entity_types)
num_rel = len(cfg.relation_types)

REL_CONS = {
    "CON":{("CROP","VAR"),("CROP","CROP")},
    "USE":{("VAR","BM"),("CROSS","BM")},
    "HAS":{("VAR","TRT"),("CROP","TRT"),("CROSS","TRT")},
    "AFF":{("ABS","TRT"),("GENE","TRT"),("MRK","TRT"),("QTL","TRT"),("BIS","TRT")},
    "OCI":{("TRT","GST"),("ABS","GST"),("BIS","GST")},
    "LOI":{("MRK","CHR"),("QTL","CHR"),("GENE","CHR")},
}
LEGAL = {rel2id[r]:{(ent2id[h],ent2id[t]) for h,t in p} for r,p in REL_CONS.items()}

SPLIT_PUNCTUATION = set(",:;.!?")

_optimal_thresholds = {r: cfg.rel_conf_thresh for r in cfg.relation_types}
if os.path.exists(cfg.thresh_path):
    with open(cfg.thresh_path, "r") as f:
        _optimal_thresholds = json.load(f)
    print(f"已加载优化阈值: {_optimal_thresholds}")

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

class EntityRelationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.model_name)
        h = self.bert.config.hidden_size
        self.start_cls = nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(cfg.dropout_rate),nn.Linear(h,num_ent))
        self.end_cls   = nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(cfg.dropout_rate),nn.Linear(h,num_ent))
        self.span_scorer = nn.Sequential(nn.Linear(h*4,h*2),nn.ReLU(),nn.Dropout(cfg.dropout_rate),
                                         nn.Linear(h*2,h),nn.ReLU(),nn.Dropout(cfg.dropout_rate),nn.Linear(h,num_ent+1))
        self.ent_type_emb = nn.Embedding(num_ent, cfg.ent_type_emb_dim)
        ri = h*6 + cfg.ent_type_emb_dim*2
        self.rel_cls = nn.Sequential(nn.Linear(ri,h*2),nn.ReLU(),nn.Dropout(cfg.dropout_rate),
                                     nn.Linear(h*2,h),nn.ReLU(),nn.Dropout(cfg.dropout_rate),nn.Linear(h,num_rel+1))

    def _sr(self, seq, s, e):
        L=seq.shape[0]
        if s<0 or e<0 or s>=L or e>=L or s>e: return None
        if e-s>=cfg.max_span_len: e=s+cfg.max_span_len-1
        if e>=L: e=L-1
        return torch.cat([seq[s],seq[e],seq[s:e+1].mean(0)])

    def _ssr(self, seq, s, e):
        L=seq.shape[0]
        if s<0 or e<0 or s>=L or e>=L or s>e: return None
        if e-s>=cfg.max_span_len: e=s+cfg.max_span_len-1
        if e>=L: e=L-1
        return torch.cat([seq[s],seq[e],seq[s:e+1].mean(0),seq[s]*seq[e]])

    def forward(self, ids, am):
        seq = self.bert(ids, attention_mask=am).last_hidden_state
        return self.start_cls(seq), self.end_cls(seq), seq

    def decode(self, sl, el, seq, om, text=None):
        B=sl.shape[0]; ba=[]
        sp=torch.sigmoid(sl); ep=torch.sigmoid(el)
        for b in range(B):
            vm=(om[b,:,0]!=0)|(om[b,:,1]!=0); vm=vm.to(seq.device); vp=torch.where(vm)[0]
            if len(vp)==0: ba.append([]); continue
            cands=[]
            for eid in range(num_ent):
                k=min(cfg.topk_start_end,len(vp))
                if k==0: continue
                _,si=torch.topk(sp[b,:,eid][vp],k); _,ei=torch.topk(ep[b,:,eid][vp],k)
                for s in vp[si].tolist():
                    for e in vp[ei].tolist():
                        if s<=e and e-s<cfg.max_span_len: cands.append((s,e,eid))
            if not cands: ba.append([]); continue
            reps,v=[],[]
            for s,e,eid in cands:
                r=self._ssr(seq[b],s,e)
                if r is not None: reps.append(r); v.append((s,e,eid))
            if not reps: ba.append([]); continue
            reps=torch.stack(reps).to(seq.device); sprob=F.softmax(self.span_scorer(reps),-1)
            scored=[]
            for i,(s,e,eid) in enumerate(v):
                sc=sp[b,s,eid].item()*ep[b,e,eid].item()*sprob[i,eid].item()
                n_tokens = e - s + 1
                if n_tokens > cfg.span_len_penalty_thresh:
                    sc *= (cfg.span_len_penalty_thresh / n_tokens) ** cfg.span_len_penalty_power
                if text is not None:
                    cs = om[b, s, 0].item(); ce = om[b, e, 1].item()
                    span_text = text[cs:ce]
                    punct_count = sum(1 for ch in span_text if ch in SPLIT_PUNCTUATION)
                    if punct_count > 0:
                        sc *= (0.7 ** punct_count)
                scored.append((s,e,eid,sc))
            scored.sort(key=lambda x:x[3],reverse=True)
            scores=[x[3] for x in scored]
            if len(scores)>5:
                d=[scores[i]-scores[i+1] for i in range(len(scores)-1)]
                dt=max(scores[max(range(len(d)),key=lambda i:d[i])],cfg.ner_score_thresh)
            else: dt=cfg.ner_score_thresh
            if scores[0]<0.005: dt=0.0
            tc=defaultdict(int); fin=[]
            for s,e,eid,sc in scored:
                if tc[eid]>=cfg.max_ent_per_type: continue
                if sc<dt and len(fin)>0: continue
                ov=False
                for fs,fe,feid in fin:
                    if feid==eid:
                        ol=min(e,fe)-max(s,fs)+1
                        if ol>0 and ol/min(e-s+1,fe-fs+1)>0.3: ov=True; break
                if not ov: fin.append((s,e,eid)); tc[eid]+=1
            ba.append(fin)
        return ba

    def predict_rel(self, seq, ents, om, text=None):
        B=seq.shape[0]; dev=seq.device; br=[]
        for b in range(B):
            rels=[]; ent=ents[b]
            if len(ent)<2: br.append(rels); continue
            reps,meta=[],[]
            for i in range(len(ent)):
                for j in range(len(ent)):
                    if i==j: continue
                    s1,e1,t1=ent[i]; s2,e2,t2=ent[j]
                    legal=[rid for rid,p in LEGAL.items() if (t1,t2) in p]
                    if not legal: continue
                    h=self._sr(seq[b],s1,e1); tl=self._sr(seq[b],s2,e2)
                    if h is None or tl is None: continue
                    reps.append(torch.cat([h,self.ent_type_emb(torch.tensor(t1,device=dev)),
                                            tl,self.ent_type_emb(torch.tensor(t2,device=dev))]))
                    meta.append((i,j,t1,t2,s1,e1,s2,e2))
            if not reps: br.append(rels); continue
            X=torch.stack(reps).to(dev); logits=self.rel_cls(X)
            for idx,(i,j,t1,t2,s1,e1,s2,e2) in enumerate(meta):
                legal=[0]+[rid for rid,p in LEGAL.items() if (t1,t2) in p]
                mask=torch.full((num_rel+1,),-1e9,device=dev); mask[legal]=0
                masked=logits[idx]+mask
                rid=torch.argmax(masked).item()
                if rid==0: continue
                conf=F.softmax(masked,-1)[rid].item()
                rname=id2rel[rid]
                th = _optimal_thresholds.get(rname, cfg.rel_conf_thresh)
                if conf < th: continue

                if text is not None:
                    hs_char = om[b, s1, 0].item()
                    he_char = om[b, e1, 1].item()
                    ts_char = om[b, s2, 0].item()
                    te_char = om[b, e2, 1].item()
                    if ts_char > he_char:
                        between = text[he_char:ts_char]
                    elif hs_char > te_char:
                        between = text[te_char:hs_char]
                    else:
                        between = ""
                    sentence_breaks = sum(1 for ch in between if ch in '.!?')
                    if sentence_breaks > 0:
                        conf *= (0.3 ** sentence_breaks)
                        if conf < th:
                            continue

                rels.append({"head":"","head_start":om[b,s1,0].item(),"head_end":om[b,e1,1].item(),"head_type":id2ent[t1],
                             "tail":"","tail_start":om[b,s2,0].item(),"tail_end":om[b,e2,1].item(),"tail_type":id2ent[t2],
                             "label":rname})
            br.append(rels)
        return br


def compute_ner_metrics(preds, trues):
    tp=fp=fn=0
    for p,t in zip(preds,trues):
        ps=set((s,e,ty) for s,e,ty in p); ts=set((s,e,ty) for s,e,ty in t)
        tp+=len(ps&ts); fp+=len(ps-ts); fn+=len(ts-ps)
    p_=tp/(tp+fp+1e-8); r_=tp/(tp+fn+1e-8); f1=2*p_*r_/(p_+r_+1e-8)
    return {"precision":p_,"recall":r_,"f1":f1,"score":0.5*f1+0.25*p_+0.25*r_,"tp":tp,"fp":fp,"fn":fn}

def compute_re_metrics(preds,trues):
    tp=fp=fn=0
    for p,t in zip(preds,trues):
        ps=set(); ts=set()
        for r in p:
            ps.add((r["head_start"],r["head_end"],ent2id.get(r["head_type"],-1),
                   r["tail_start"],r["tail_end"],ent2id.get(r["tail_type"],-1),r["label"]))
        for r in t:
            ts.add((r["head_start"],r["head_end"],ent2id.get(r["head_type"],-1),
                   r["tail_start"],r["tail_end"],ent2id.get(r["tail_type"],-1),r["label"]))
        tp+=len(ps&ts); fp+=len(ps-ts); fn+=len(ts-ps)
    p_=tp/(tp+fp+1e-8); r_=tp/(tp+fn+1e-8); f1=2*p_*r_/(p_+r_+1e-8)
    return {"precision":p_,"recall":r_,"f1":f1,"score":0.5*f1+0.25*p_+0.25*r_,"tp":tp,"fp":fp,"fn":fn}


print("="*70)
print(f"设备: {cfg.device} | 模型: {cfg.model_path}")
print(f"测试文件: {cfg.test_file}")
print(f"长span惩罚: token>{cfg.span_len_penalty_thresh} | 标点检测: ✓ | 跨句惩罚: ✓")
print(f"topk={cfg.topk_start_end} | NER_thresh={cfg.ner_score_thresh} | RE_thresh={cfg.rel_conf_thresh}")
print("="*70)

with open(cfg.test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)
print(f"测试样本数: {len(test_data)}")

has_labels = all("entities" in d and "relations" in d for d in test_data)
if has_labels:
    print("检测到真实标签，将计算评分")
else:
    print("未检测到标签，仅输出预测")

model = EntityRelationModel().to(cfg.device)
sd = torch.load(cfg.model_path, map_location=cfg.device)
if any(k.startswith("module.") for k in sd): sd = {k[7:]:v for k,v in sd.items()}
model.load_state_dict(sd, strict=True)
model.eval()
print("模型加载完成\n")

ner_preds, ner_trues = [], []
re_preds, re_trues = [], []

for idx, item in enumerate(test_data):
    text = item["text"]
    enc = tokenizer(text, truncation=True, max_length=cfg.max_len,
                    padding="max_length", return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(cfg.device); am = enc["attention_mask"].to(cfg.device); om = enc["offset_mapping"]

    with torch.no_grad():
        sl, el, seq = model(ids, am)
        pe = model.decode(sl, el, seq, om, text=text)[0]
        pr = model.predict_rel(seq, [pe], om, text=text)[0]

    if idx < 5 or idx >= len(test_data) - 5:
        print(f"{'─'*70}")
        print(f"样本 {idx+1}/{len(test_data)}")
        print(f"文本: {text[:150]}{'...' if len(text)>150 else ''}")
        print(f"实体({len(pe)}个):")
        for s, e, t in pe:
            cs = om[0,s,0].item(); ce = om[0,e,1].item()
            print(f"  [{cs:3d}-{ce:3d}] {text[cs:ce]:35s} → {id2ent[t]}")
        if pr:
            print(f"关系({len(pr)}个):")
            for r in pr:
                print(f"  ({text[r['head_start']:r['head_end']]:20s}, {text[r['tail_start']:r['tail_end']]:20s}) → {r['label']}")
        else:
            print(f"关系: 无")
        if has_labels:
            true_ents = [(e["start"],e["end"]-1,ent2id[e["label"]]) for e in item["entities"] if e["label"] in ent2id]
            print(f"真实实体({len(true_ents)}个):")
            for s,e,t in true_ents:
                print(f"  [{s:3d}-{e+1:3d}] {text[s:e+1]:35s} → {id2ent[t]}")
            print(f"真实关系({len(item['relations'])}个):")
            for r in item["relations"]:
                print(f"  ({r['head']:20s}, {r['tail']:20s}) → {r['label']}")

    if has_labels:
        true_ents = [(e["start"],e["end"]-1,ent2id[e["label"]]) for e in item["entities"] if e["label"] in ent2id]
        true_rels = []
        for r in item["relations"]:
            if r["label"] in rel2id and r["head_type"] in ent2id and r["tail_type"] in ent2id:
                true_rels.append({"head_start":r["head_start"],"head_end":r["head_end"],"head_type":r["head_type"],
                                  "tail_start":r["tail_start"],"tail_end":r["tail_end"],"tail_type":r["tail_type"],
                                  "label":r["label"]})

        pred_ents = []
        for s,e,t in pe:
            cs=om[0,s,0].item(); ce=om[0,e,1].item()
            pred_ents.append((cs,ce-1,t))

        pred_rels = []
        for r in pr:
            pred_rels.append({"head_start":r["head_start"],"head_end":r["head_end"],"head_type":r["head_type"],
                              "tail_start":r["tail_start"],"tail_end":r["tail_end"],"tail_type":r["tail_type"],
                              "label":r["label"]})

        ner_preds.append(pred_ents); ner_trues.append(true_ents)
        re_preds.append(pred_rels); re_trues.append(true_rels)

    if (idx+1) % 100 == 0:
        print(f"\n  进度: {idx+1}/{len(test_data)}")

print(f"\n{'='*70}")

if has_labels and len(ner_trues) > 0:
    ner = compute_ner_metrics(ner_preds, ner_trues)
    re  = compute_re_metrics(re_preds, re_trues)
    total_score = 0.4 * ner["score"] + 0.6 * re["score"]

    print(f"                    评  估  报  告")
    print(f"{'='*70}")
    print(f"  NER ── P: {ner['precision']:.4f}  |  R: {ner['recall']:.4f}  |  "
          f"F1: {ner['f1']:.4f}  |  Score: {ner['score']:.4f}")
    print(f"          TP: {ner['tp']}  |  FP: {ner['fp']}  |  FN: {ner['fn']}")
    print(f"          预测实体: {ner['tp']+ner['fp']}  |  真实实体: {ner['tp']+ner['fn']}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  RE  ── P: {re['precision']:.4f}  |  R: {re['recall']:.4f}  |  "
          f"F1: {re['f1']:.4f}  |  Score: {re['score']:.4f}")
    print(f"          TP: {re['tp']}  |  FP: {re['fp']}  |  FN: {re['fn']}")
    print(f"          预测关系: {re['tp']+re['fp']}  |  真实关系: {re['tp']+re['fn']}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Total Score = 0.4 × {ner['score']:.4f} + 0.6 × {re['score']:.4f} = {total_score:.4f}")

    print(f"\n  各实体类型 F1:")
    type_tp=defaultdict(int); type_fp=defaultdict(int); type_fn=defaultdict(int)
    for p,t in zip(ner_preds, ner_trues):
        ps=set((s,e,ty) for s,e,ty in p); ts=set((s,e,ty) for s,e,ty in t)
        for s,e,ty in ps&ts: type_tp[ty]+=1
        for s,e,ty in ps-ts: type_fp[ty]+=1
        for s,e,ty in ts-ps: type_fn[ty]+=1
    for ty in range(num_ent):
        tp_=type_tp[ty]; fp_=type_fp[ty]; fn_=type_fn[ty]
        if tp_+fp_+fn_>0:
            p_=tp_/(tp_+fp_+1e-8); r_=tp_/(tp_+fn_+1e-8)
            print(f"    {id2ent[ty]:6s}  P={p_:.4f}  R={r_:.4f}  F1={2*p_*r_/(p_+r_+1e-8):.4f}  (TP={tp_} FP={fp_} FN={fn_})")

    print(f"\n  各关系类型 F1:")
    rt_tp=defaultdict(int); rt_fp=defaultdict(int); rt_fn=defaultdict(int)
    for p,t in zip(re_preds, re_trues):
        ps=set()
        for r in p:
            ps.add((r["head_start"],r["head_end"],ent2id.get(r["head_type"],-1),
                   r["tail_start"],r["tail_end"],ent2id.get(r["tail_type"],-1),r["label"]))
        ts=set()
        for r in t:
            ts.add((r["head_start"],r["head_end"],ent2id.get(r["head_type"],-1),
                   r["tail_start"],r["tail_end"],ent2id.get(r["tail_type"],-1),r["label"]))
        for item_ in ps&ts: rt_tp[item_[5]]+=1
        for item_ in ps-ts: rt_fp[item_[5]]+=1
        for item_ in ts-ps: rt_fn[item_[5]]+=1
    for rn in cfg.relation_types:
        tp_=rt_tp[rn]; fp_=rt_fp[rn]; fn_=rt_fn[rn]
        if tp_+fp_+fn_>0:
            p_=tp_/(tp_+fp_+1e-8); r_=tp_/(tp_+fn_+1e-8)
            print(f"    {rn:6s}  P={p_:.4f}  R={r_:.4f}  F1={2*p_*r_/(p_+r_+1e-8):.4f}  (TP={tp_} FP={fp_} FN={fn_})")

else:
    print(f"  测试集无标签，无法计算评分")

print(f"{'='*70}")