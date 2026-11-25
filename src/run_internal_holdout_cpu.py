"""CPU-only internal holdout using cached frozen features and saved candidates."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

ACTIONS=('direct','reflect_1','reflect_2')
def rows(p): return [json.loads(x) for x in Path(p).open() if x.strip()]
def cs(r): return {x['action']:x for x in r['candidate_results']}
def load_features(root):
    man=json.load(open(root/'manifest.json')); parts=[]
    for p in sorted(root.glob('*.npy')): parts.append(np.load(p))
    x=np.concatenate(parts); return {i:x[k] for k,i in enumerate(man['ids'])}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--seed',type=int,default=20260922); a=ap.parse_args()
    out=Path(a.output); out.mkdir(parents=False,exist_ok=False)
    all_rows=rows('outputs/math_full_training_v1/policy_all.jsonl'); feats=load_features(Path('outputs/math_fixed_features/83599b3edb63803121f0'))
    all_rows=[r for r in all_rows if r['id'] in feats and set(cs(r))==set(ACTIONS)]
    rng=np.random.default_rng(a.seed); idx=rng.permutation(len(all_rows)); cut=int(.8*len(idx)); tr=[all_rows[i] for i in idx[:cut]]; te=[all_rows[i] for i in idx[cut:]]
    def label(r):
        c=cs(r); return max(ACTIONS,key=lambda z:(c[z]['correct'],-c[z]['token_cost']))
    def train(rows_, labels):
        sc=StandardScaler(); x=sc.fit_transform(np.stack([feats[r['id']] for r in rows_])); enc=LabelEncoder(); y=enc.fit_transform(labels); m=MLPClassifier(hidden_layer_sizes=(64,16),max_iter=100,early_stopping=True,random_state=a.seed); m.fit(x,y); return sc,enc,m
    gate_labels=['direct' if label(r)=='direct' else 'reflect' for r in tr]
    round_rows=[r for r in tr if label(r)!='direct']; round_labels=[label(r) for r in round_rows]
    gs,ge,gm=train(tr,gate_labels); rs,re,rm=train(round_rows,round_labels)
    gx=gs.transform(np.stack([feats[r['id']] for r in te])); gate=ge.inverse_transform(gm.predict(gx)); rx=rs.transform(np.stack([feats[r['id']] for r in te])); rp=re.inverse_transform(rm.predict(rx))
    actions=['direct' if g=='direct' else p for g,p in zip(gate,rp)]
    def metric(act):
        c=[cs(r)[z] for r,z in zip(te,act)]; d=[cs(r)['direct']['correct'] for r in te]; q=[x['correct'] for x in c]
        return {'accuracy':float(np.mean(q)),'avg_token_cost':float(np.mean([x['token_cost'] for x in c])),'reflection_rate':float(np.mean([z!='direct' for z in act])),'fixes':int(sum(x==0 and y==1 for x,y in zip(d,q))),'harms':int(sum(x==1 and y==0 for x,y in zip(d,q)))}
    result={'n_train':len(tr),'n_confirmation':len(te),'seed':a.seed,'encoder_cache':'outputs/math_fixed_features/83599b3edb63803121f0','train_labels':{'gate':{z:gate_labels.count(z) for z in set(gate_labels)},'round':{z:round_labels.count(z) for z in set(round_labels)}},'methods':{'direct':metric(['direct']*len(te)),'fixed_reflect_1':metric(['reflect_1']*len(te)),'fixed_reflect_2':metric(['reflect_2']*len(te)),'policy':metric(actions)},'status':'internal_holdout_not_independent_benchmark'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)); (out/'REPORT.md').write_text('# GSM8K policy internal holdout\n\n复用已有候选和冻结特征，无大模型调用。该结果是策略内部留出验证，不是独立 benchmark 确认。\n\n'+json.dumps(result['methods'],indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
