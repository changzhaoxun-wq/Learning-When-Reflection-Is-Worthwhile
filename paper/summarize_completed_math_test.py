"""Reconcile saved test outcomes and produce manuscript tables without model calls."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'outputs/math_full_test_failure_accounted_v1'
OUT = PROJECT / 'paper/results_audit_20260918'
DRAFT = PROJECT / 'paper/HIERARCHICAL_REFLECTION_DRAFT_ZH_EN.md'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def label(g):
    return ('CE' if g['recipe']=='weighted_ce_control' else 'CE+Rank') + '/' + (
        'Gate-Round' if g['branch']=='original_round' else 'Gate-R1')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def run(update=False):
    OUT.mkdir(exist_ok=True)
    summary_path = ROOT / 'evaluation/summary.json'
    s = json.loads(summary_path.read_text())
    audit = json.loads((ROOT / 'accounting_audit.json').read_text())
    assert s['accounting_audit_sha256']==sha(ROOT / 'accounting_audit.json')
    assert audit['trace_sha256']==sha(ROOT / 'policy_test_accounted.jsonl')
    rows = read_rows(ROOT / 'policy_test_accounted.jsonl')
    ids = [r['id'] for r in rows]
    assert len(ids)==len(set(ids))==1319
    outcomes = {r['id']: {c['action']:c for c in [*r['candidate_results'],r['independent']]} for r in rows}
    decisions = defaultdict(list)
    for d in read_rows(ROOT / 'evaluation/decisions.jsonl'):
        decisions[d['policy']].append(d)
    assert set(decisions)=={p['name'] for p in s['policies']}
    decomposition, groups = [], defaultdict(list)
    for p in s['policies']:
        ds=decisions[p['name']]
        assert [d['id'] for d in ds]==ids
        counts = dict(gate_missed_repair=0, round_missed_repair=0, introduced_error=0,
                      execution_failure=0, correct=0, oracle_correct=0)
        totals = []
        for d in ds:
            cs=outcomes[d['id']]; c=cs[d['action']]; base=cs['direct']['correct']
            oracle=max(cs[a]['correct'] for a in ('direct','reflect_1','reflect_2'))
            for name, field in [('final_correct','correct'),('prompt_tokens','prompt_tokens'),
                                ('completion_tokens','completion_tokens'),('total_tokens','token_cost')]:
                assert d[name]==c[field]
            assert d['execution_status']==c['execution_status']
            counts['correct']+=c['correct']; counts['oracle_correct']+=oracle
            counts['execution_failure']+=c['execution_status']!='success'
            if oracle and not c['correct']:
                if base:
                    counts['introduced_error']+=1
                elif d['action']=='direct':
                    counts['gate_missed_repair']+=1
                else:
                    counts['round_missed_repair']+=1
            totals.append(c['token_cost'])
        assert counts['oracle_correct']-counts['correct']==sum(counts[k] for k in (
            'gate_missed_repair','round_missed_repair','introduced_error'))
        assert counts['correct']==p['metrics']['correct']
        assert abs(np.mean(totals)-p['metrics']['avg_token_cost'])<1e-9
        assert counts['execution_failure']==p['metrics']['execution_failure_count']
        row={k:p[k] for k in ('name','recipe','branch','seed','epoch','validation_call_budget','direct_threshold')}
        row.update(counts); decomposition.append(row)
        groups[(p['recipe'],p['branch'],p['validation_call_budget'])].append(row)
    grouped=[]
    for g in s['groups']:
        members=groups[(g['recipe'],g['branch'],g['validation_call_budget'])]
        assert sorted(m['seed'] for m in members)==[7,21,42]
        grouped.append({**{k:g[k] for k in ('recipe','branch','validation_call_budget')},
            **{k:float(np.mean([m[k] for m in members])) for k in
               ('gate_missed_repair','round_missed_repair','introduced_error','execution_failure')}})
    transitions={}
    for a,b in [('direct','reflect_1'),('reflect_1','reflect_2')]:
        transitions[a+'->'+b]={
            'fixes':sum(not c[a]['correct'] and c[b]['correct'] for c in outcomes.values()),
            'harms':sum(c[a]['correct'] and not c[b]['correct'] for c in outcomes.values()),
            'added_mean_tokens':float(np.mean([c[b]['token_cost']-c[a]['token_cost'] for c in outcomes.values()]))}
    derived={'analysis_intent':'posthoc descriptive attribution, not model selection', 'n':len(ids),
             'baseline_wrong':len(ids)-s['baselines']['direct']['correct'],
             'per_policy':decomposition,'groups':grouped,'transitions':transitions,
             'all_36_policy_metrics_reconciled':True,'source_summary_sha256':sha(summary_path)}
    dump(OUT/'attribution.json',derived)
    blocks={}
    blocks['main']=table(['方法','验证预算','Acc. (%)','输入 token','输出 token','总 token','调用率 (%)'],[
        [name,'--',f"{m['accuracy']*100:.2f}",f"{m['avg_prompt_tokens']:.2f}",f"{m['avg_completion_tokens']:.2f}",
         f"{m['avg_token_cost']:.2f}",f"{m['reflection_rate']*100:.2f}" if name!='independent' else '不适用']
        for name,m in s['baselines'].items()]+[
        [label(g),f"{g['validation_call_budget']:.0%}",
         f"{g['metrics']['accuracy']['mean']*100:.2f} ± {g['metrics']['accuracy']['std']*100:.2f}",
         f"{g['metrics']['avg_prompt_tokens']['mean']:.2f}",f"{g['metrics']['avg_completion_tokens']['mean']:.2f}",
         f"{g['metrics']['avg_token_cost']['mean']:.2f} ± {g['metrics']['avg_token_cost']['std']:.2f}",
         f"{g['metrics']['reflection_rate']['mean']*100:.2f}"] for g in s['groups']])
    blocks['transitions']=table(['方法','修复数','引错数','修复率 (%)','引错率 (%)','净变化 (pp)','执行失败数'],[
        [name,m['fixes'],m['harms'],f"{100*m['fix_rate']:.2f}",f"{100*m['harm_rate']:.2f}",
         f"{100*(m['accuracy']-s['baselines']['direct']['accuracy']):+.2f}",m['execution_failure_count']]
        for name,m in s['baselines'].items() if name!='direct'])
    blocks['random']=table(['策略','验证预算','策略 Acc. (%)','均匀随机 Acc. (%)','同轮次随机 Acc. (%)',
        '策略/均匀/同轮次 token'],[[label(g),f"{g['validation_call_budget']:.0%}",f"{g['metrics']['accuracy']['mean']*100:.2f}",
        f"{g['random_expected']['uniform_gate']['accuracy']['mean']*100:.2f}",
        f"{g['random_expected']['round_stratified_gate']['accuracy']['mean']*100:.2f}",
        '/'.join(f'{v:.2f}' for v in (g['metrics']['avg_token_cost']['mean'],
          g['random_expected']['uniform_gate']['avg_token_cost']['mean'],g['random_expected']['round_stratified_gate']['avg_token_cost']['mean']))]
        for g in s['groups']])
    blocks['attribution']=table(['策略','验证预算','Gate 漏修复','反思轮次漏修复','对初始正确题引错','执行失败 (重叠计数)'],[
        [label(g),f"{g['validation_call_budget']:.0%}",*[f'{g[k]:.2f}' for k in
         ('gate_missed_repair','round_missed_repair','introduced_error','execution_failure')]] for g in grouped])
    tests={t['name']:t for t in s['mcnemar_vs_direct']}
    def ci(x):
        return f"{100*x['observed_difference']:+.2f} [{100*x['ci95_lower']:+.2f}, {100*x['ci95_upper']:+.2f}]"
    blocks['statistics']=table(['策略','验证预算','相对 Direct ΔAcc. [95% CI] (pp)',
        '三种子 Holm p 范围','相对强模型 ΔAcc. [95% CI] (pp)'],[[label(g),f"{g['validation_call_budget']:.0%}",
        ci(g['paired_bootstrap']['direct']['accuracy']),
        '–'.join(f'{x:.3g}' for x in (min(tests[m['name']]['holm_p'] for m in groups[(g['recipe'],g['branch'],g['validation_call_budget'])]),
                                     max(tests[m['name']]['holm_p'] for m in groups[(g['recipe'],g['branch'],g['validation_call_budget'])]))),
        ci(g['paired_bootstrap']['independent']['accuracy'])] for g in s['groups']])
    blocks['seeds']=table(['策略','验证预算','seed','epoch','阈值 p_D','Acc. (%)','总 token','调用率 (%)','执行失败'],[
        [label(p),f"{p['validation_call_budget']:.0%}",p['seed'],p['epoch'],f"{p['direct_threshold']:.8f}",
         f"{p['metrics']['accuracy']*100:.2f}",f"{p['metrics']['avg_token_cost']:.2f}",
         f"{p['metrics']['reflection_rate']*100:.2f}",p['metrics']['execution_failure_count']] for p in s['policies']])
    (OUT/'tables.md').write_text('\n\n'.join('## '+k+'\n\n'+v for k,v in blocks.items())+'\n')
    with (OUT/'policy_attribution.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(decomposition[0]));writer.writeheader();writer.writerows(decomposition)
    if update:
        draft=DRAFT.read_text()
        for key,value in blocks.items():
            pattern=rf'<!-- RESULTS:{key}:START -->.*?<!-- RESULTS:{key}:END -->'
            draft,count=re.subn(pattern,lambda _:f'<!-- RESULTS:{key}:START -->\n{value}\n<!-- RESULTS:{key}:END -->',draft,flags=re.S)
            assert count==1,(key,count)
        DRAFT.write_text(draft)
    figure(s)
    sources=[summary_path,ROOT/'evaluation/decisions.jsonl',ROOT/'accounting_audit.json',
             ROOT/'gpu_numerical_review.json',ROOT/'evaluation/encoder_audit.json',Path(__file__),OUT/'attribution.json']
    dump(OUT/'provenance.json',{'files_sha256':{str(p):sha(p) for p in sources},
        'human_verification':'pending','no_model_calls':True,'no_policy_selection_or_tuning':True,
        'all_36_saved_decisions_reconciled':True,'n':len(ids),
        'figure_sha256':{str(OUT/f'quality_cost_test.{ext}'):sha(OUT/f'quality_cost_test.{ext}')
                         for ext in ('png','pdf','svg')},
        'figure_uncertainty':'sample SD across three Gate seeds, not CI',
        'figure_transformations':'all twelve groups, mean accuracy multiplied by 100, no smoothing',
        'publisher_requirements':'not assessed; provisional general manuscript figure'})
    print(json.dumps(derived['transitions'],indent=2)); print('Tables, attribution, and figure exported to',OUT)


def figure(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':8,'axes.titlesize':9,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(8.2,3.7),layout='constrained')
        styles=[('weighted_ce_control','original_round','#0072B2','o','-'),
                ('weighted_ce_pairwise','original_round','#D55E00','s','-'),
                ('weighted_ce_control','fixed_r1','#0072B2','^','--'),
                ('weighted_ce_pairwise','fixed_r1','#D55E00','D','--')]
        for recipe,branch,color,marker,ls in styles:
            gs=sorted([g for g in s['groups'] if g['recipe']==recipe and g['branch']==branch],key=lambda g:g['validation_call_budget'])
            x=[g['metrics']['avg_token_cost']['mean'] for g in gs]
            y=[100*g['metrics']['accuracy']['mean'] for g in gs]
            axes[0].errorbar(x,y,xerr=[g['metrics']['avg_token_cost']['std'] for g in gs],
                yerr=[100*g['metrics']['accuracy']['std'] for g in gs],color=color,marker=marker,
                linestyle=ls,linewidth=1,capsize=2,label=label(gs[0]),markersize=4)
            axes[1].errorbar([100*g['validation_call_budget'] for g in gs],
                [100*g['metrics']['reflection_rate']['mean'] for g in gs],
                yerr=[100*g['metrics']['reflection_rate']['std'] for g in gs],
                color=color,marker=marker,linestyle=ls,linewidth=1,capsize=2,markersize=4)
        for name,text,marker in [('direct','Base D','x'),('independent','Strong D','*'),
                                 ('fixed_reflect_1','Fixed R1','P'),('fixed_reflect_2','Fixed R2','X'),
                                 ('oracle_accuracy_first','Oracle','+')]:
            m=s['baselines'][name]; x=m['avg_token_cost']; y=100*m['accuracy']
            axes[0].scatter([x],[y],color='#333333',marker=marker,s=38)
            axes[0].annotate(text,(x,y),xytext=(5,7) if name=='direct' else (0,7),
                             textcoords='offset points',ha='left' if name=='direct' else 'center',fontsize=7)
        axes[0].set(xlim=(180,1840),ylim=(92.7,97.2),xlabel='Mean input + output tokens per question',
                    ylabel='Answer accuracy (%)',title='(a) Quality and generation cost')
        axes[0].legend(loc='lower right',fontsize=6.8,frameon=False)
        axes[1].plot([0,25],[0,25],color='#555555',linestyle=':',linewidth=1,label='Test rate = validation budget')
        axes[1].set(xlim=(0,25),ylim=(0,42),xticks=[5,10,20],xlabel='Validation call-rate budget (%)',
                    ylabel='Observed test call rate (%)',title='(b) Budget transfer to test')
        axes[1].legend(loc='upper left',fontsize=7,frameon=False)
        for ax in axes:
            ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18)
        for ext in ('png','pdf','svg'):
            fig.savefig(OUT/f'quality_cost_test.{ext}',dpi=300,facecolor='white')
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--update-draft-tables',action='store_true')
    run(parser.parse_args().update_draft_tables)
