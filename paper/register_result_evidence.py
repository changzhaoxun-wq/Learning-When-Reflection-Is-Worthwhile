"""Bind the updated manuscript to local results; do not assert human verification."""
import csv
import hashlib
import json
import re

from summarize_completed_math_test import PROJECT, OUT, DRAFT, ROOT, sha, dump

OLD = PROJECT / 'paper/method_audit_20260917'


def run():
    original_checks=json.loads((OLD/'source_checks.json').read_text())['sources']
    for record in original_checks:
        assert sha(PROJECT/record['path'])==record['sha256'],record['path']
    sources=json.loads((OLD/'source_manifest.json').read_text())['sources']
    checks=list(original_checks)
    new=[('E019',ROOT/'evaluation/summary.json','baselines; groups; policies; paired_bootstrap; mcnemar_vs_direct; encoder_info'),
         ('E020',ROOT/'accounting_audit.json','n; logical_calls; paid_attempts; generation_usage; unresolved_execution_failures'),
         ('E021',ROOT/'gpu_numerical_review.json','overlap_n; strict_element_mismatches; linked action audit; accepted'),
         ('E022',OUT/'attribution.json','all_36_policy_metrics_reconciled; groups; transitions'),
         ('E023',PROJECT/'paper/summarize_completed_math_test.py','decision reconciliation; table generation; figure'),
         ('E024',PROJECT/'src/evaluate_accounted_math_test.py','failure_metrics; evaluate; Holm and paired bootstrap'),
         ('E025',ROOT/'evaluation/decisions.jsonl','all saved per-question actions and measured costs'),
         ('E026',ROOT/'gpu_pipeline_20260918_104901.log','encode gpu 1319/1319 elapsed=21s; failed first strict audit')]
    for eid,path,locator in new:
        sources.append({'evidence_id':eid,'source_type':'software' if path.suffix=='.py' else 'report',
            'title':str(path.relative_to(PROJECT)),'authors':[],'year':None,
            'identifiers':{'doi':'','pmid':'','pmcid':'','isbn':'','url':path.as_uri()},
            'locator':locator,'confidentiality':'restricted','verification':{
                'status':'unverified','source_opened':False,'verified_by':'','verified_on':''}})
        checks.append({'evidence_id':eid,'path':str(path.relative_to(PROJECT)), 'sha256':sha(path),
            'locator':locator,'machine_checked_by':'Codex','human_review_pending':True})
    dump(OUT/'source_manifest.json',{'schema_version':'1.0','sources':sources})
    dump(OUT/'source_checks.json',{'sources':checks,'human_review_pending':True})
    fieldnames=['claim_id','section','claim_kind','claim_text_sha256','evidence_ids','verification_status','uncertainty','analysis_intent']
    claims=[]; section=''
    for line in DRAFT.read_text().splitlines():
        if line.startswith('#'):
            section=line.lstrip('# ')
        match=re.search(r'\[claim:(C\d+)\]\s*\[evidence:([^]]+)\]',line)
        if not match:
            continue
        text=line[:match.start()].strip()
        eid=match.group(2).split(',')
        assert set(eid)<={s['evidence_id'] for s in sources}
        claims.append(dict(zip(fieldnames,[match.group(1),section,'result' if int(match.group(1)[1:])>=24 else 'method',
            hashlib.sha256(text.encode()).hexdigest(),';'.join(eid),'unverified','not_estimated','exploratory'])))
    assert len(claims)==len({c['claim_id'] for c in claims})==51
    with (OUT/'claims.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(claims)
    consistency=json.loads((OLD/'consistency_manifest.json').read_text())
    s=json.loads((ROOT/'evaluation/summary.json').read_text())
    newfacts=[('test_denominator',1319,'questions'),('execution_failures',4,'calls')]
    newfacts += [(k+'_accuracy',v['accuracy'],'proportion') for k,v in s['baselines'].items()]
    newfacts += [(k+'_mean_tokens',v['avg_token_cost'],'tokens/question') for k,v in s['baselines'].items()]
    for i,(concept,value,unit) in enumerate(newfacts,start=len(consistency['numeric_facts'])+1):
        consistency['numeric_facts'].append({'fact_id':f'N{i:03d}','concept':concept,'section':'5. Results',
            'value':value,'unit':unit,'numerator':None,'denominator':None,'sample_size':1319,
            'analysis_set':'failure_inclusive_test','evidence_ids':['E019','E020']})
    dump(OUT/'consistency_manifest.json',consistency)
    text=DRAFT.read_text()
    assert 'Full-test evaluation is ongoing' not in text and '完整测试仍在进行' not in text
    for key in ('main','transitions','random','attribution','statistics','seeds'):
        block=re.search(rf'<!-- RESULTS:{key}:START -->(.*?)<!-- RESULTS:{key}:END -->',text,re.S)
        assert block and '|' in block.group(1)
    dump(OUT/'manuscript_checks.json',{'claims':len(claims),'sources':len(sources),
        'claim_hashes_bound':True,'original_18_sources_unchanged':True,'six_generated_tables_populated':True,
        'manuscript_sha256':sha(DRAFT),'human_verification':'pending','submission_ready':False})
    print('Registered',len(claims),'claims;',len(sources),'sources; human verification remains pending')


if __name__=='__main__':
    run()
