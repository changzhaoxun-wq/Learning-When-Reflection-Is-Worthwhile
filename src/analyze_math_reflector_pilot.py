"""Audit completed math pilot artifacts and compute paired, exploratory results."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .analyze_final_policy import mcnemar
from .build_final_report import _bootstrap_diff
from .math_reflector_pilot import candidate, digest, visible_answer, validate_response
from .prepare_qa_hard_development import file_hash
from .utils import read_jsonl, save_json, write_jsonl


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired(a, b, iterations=10000, seed=20260910):
    require(bool(a) and len(a) == len(b), 'Paired results must have equal nonzero lengths')
    return {'mcnemar': mcnemar(a, b),
            'accuracy': _bootstrap_diff(a, b, 'correct', iterations, seed),
            'token_cost': _bootstrap_diff(a, b, 'token_cost', iterations, seed),
            'reward': _bootstrap_diff(a, b, 'reward', iterations, seed)}


def audit(root):
    protocol = json.loads((root/'protocol.json').read_text())
    saved = json.loads((root/'results.json').read_text())
    require(json.loads((root/'status.json').read_text())['stage'] == 'complete', 'Pilot not complete')
    require(saved['protocol_sha256'] == file_hash(root/'protocol.json'), 'Protocol hash mismatch')
    for name, sha in protocol['code_sha256'].items():
        require(file_hash(Path(__file__).parent/name) == sha, f'Generation/scoring code changed: {name}')
    sample = root.parent/'development.jsonl'
    require(file_hash(sample) == protocol['sample_sha256'], 'Sample changed')
    items = read_jsonl(sample)
    ids = [r['id'] for r in items]
    require(len(ids) == len(set(ids)) and len(ids) > 0, 'Invalid sample IDs')
    roles = protocol['roles']
    stages = {role: read_jsonl(root/f'{role}.jsonl') for role in roles}
    checked = {}

    def check_call(call, role, key, action, expected_prompt):
        path = (root/'calls'/role/key/f'{action}.jsonl').resolve()
        require(Path(call['call_log']).resolve() == path, 'Wrong call log path')
        require(call['call_log_sha256'] == file_hash(path), f'Call hash mismatch: {path}')
        require(call['prompt'] == expected_prompt, 'Candidate prompt mismatch')
        if path in checked:
            require(checked[path]['call'] == call, 'Conflicting copies of shared call')
            return
        attempts = read_jsonl(path)
        require(0 < len(attempts) <= len(protocol['output_steps']), 'Unexpected attempt count')
        for index, attempt in enumerate(attempts):
            validate_response(attempt)
            require(attempt['prompt'] == expected_prompt, 'Attempt prompt changed')
            require(expected_prompt.strip() in attempt['wire_prompt'], 'Task prompt absent from request')
            fp = digest(json.dumps({'role':role, 'model':roles[role]['model'],
                                  'wire':attempt['wire_prompt'], 'output_steps':protocol['output_steps'],
                                  'context':protocol['context']}, sort_keys=True))
            require(fp == attempt['fingerprint'], 'Request fingerprint mismatch')
            require(attempt['max_tokens'] == protocol['output_steps'][index], 'Output budget changed')
            require(attempt['metadata']['finish_reason'] == ('stop' if index == len(attempts)-1 else 'length'),
                    'Unresolved or spurious retry')
            req = attempt['metadata']['request']
            for field in ('temperature','seed'):
                require(req[field] == protocol[field], f'Request {field} changed')
            require(req['effective_max_tokens'] == attempt['max_tokens'], 'Server clipped output budget')
            required = attempt['prompt_tokens_preflight'] + protocol['output_steps'][-1] + protocol['safety']
            require(required == attempt['required_window_with_cap'] <= protocol['context'], 'Invalid input reserve')
        last = attempts[-1]
        wire = last['wire_prompt']
        require(visible_answer(last['output'], wire.rfind('<think>') > wire.rfind('</think>')) == call['answer_text'],
                'Visible response mismatch')
        require(call['output'] == last['output'] and call['finish_reason'] == 'stop', 'Final output mismatch')
        require(call['attempt_count'] == len(attempts) and call['recovered_length_calls'] == len(attempts)-1,
                'Retry count mismatch')
        for field in ('prompt_tokens','completion_tokens','total_tokens'):
            require(call[field] == sum(a[field] for a in attempts), 'Lost retry cost')
        checked[path] = {'call':call, 'attempts':attempts}

    for role, rows in stages.items():
        require([r['id'] for r in rows] == ids, f'Incomplete or reordered {role}')
        for i, (row, item) in enumerate(zip(rows, items)):
            for field in ('question','gold_answer','answer'):
                require(row[field] == item[field], f'Sample field changed: {field}')
            base = stages['base'][i]['initial_call']
            require(row['initial_call'] == base and row['initial_answer'] == base['answer_text'], 'Unshared Base response')
            question, key = item['question'], item['id']
            direct_prompt = protocol['direct_prompt'].format(question=question)
            check_call(base, 'base', key, 'direct', direct_prompt)
            expected_actions = ['direct'] if role == 'base' else ['direct','reflect_1','reflect_2']
            require([c['action'] for c in row['candidate_results']] == expected_actions, 'Candidate actions changed')
            previous = [base]
            for index, c in enumerate(row['candidate_results']):
                if index:
                    require(c['calls'][:-1] == previous, 'Reflection prefix mismatch')
                    prompt = protocol['reflection_template'].format(question=question,
                              current_answer=previous[-1]['answer_text'], round_id=index)
                    check_call(c['calls'][-1], role, key, f'reflect_{index}', prompt)
                else:
                    require(c['calls'] == [base], 'Direct cost contains additional calls')
                require(c == candidate(c['action'], c['calls'], item['gold_answer']), 'Candidate score/cost mismatch')
                previous = c['calls']
            if role == 'qwen14':
                independent = row['independent']
                require(len(independent['calls']) == 1, 'Independent call count changed')
                check_call(independent['calls'][0], role, key, 'independent', direct_prompt)
                require(independent == candidate('independent', independent['calls'], item['gold_answer']),
                        'Independent score/cost mismatch')
    actual_files = {p.resolve() for p in (root/'calls').rglob('*.jsonl')}
    require(actual_files == set(checked), 'Missing or unaccounted call logs')
    paid = [a for c in checked.values() for a in c['attempts']]
    require(sum(a['total_tokens'] for a in paid) == saved['total_generation_tokens'], 'Generation ledger mismatch')
    recovered = [{'call_log':str(path), 'attempts':len(c['attempts']),
                  'budgets':[a['max_tokens'] for a in c['attempts']],
                  'total_tokens':c['call']['total_tokens']} for path,c in checked.items() if len(c['attempts'])>1]
    require(len(paid)-len(checked) == saved['length_attempts_recovered'], 'Summary retry count mismatch')
    return items, stages, saved, {
        'n':len(items), 'unique_calls':len(checked), 'paid_attempts':len(paid),
        'total_generation_tokens':sum(a['total_tokens'] for a in paid),
        'unresolved_truncation':0, 'recovered_calls':recovered,
        'all_candidate_costs_scores_and_prefixes_verified':True,
        'source_hashes':{p.name:file_hash(p) for p in [sample,root/'protocol.json',root/'results.json',
                       *[root/f'{role}.jsonl' for role in roles]]}}


def methods_from(stages, items):
    methods={'direct':[r['candidate_results'][0] for r in stages['base']]}
    for role in ('deepseek','qwen14'):
        for index in (1,2):
            methods[f'{role}_reflect_{index}']=[r['candidate_results'][index] for r in stages[role]]
        methods[f'{role}_oracle']=[max(r['candidate_results'],key=lambda c:(c['correct'],-c['token_cost'])) for r in stages[role]]
    methods['qwen14_independent_standalone']=[r['independent'] for r in stages['qwen14']]
    methods['base_then_qwen14_independent']=[candidate('independent',[stages['base'][i]['initial_call'],r['independent']['calls'][0]],items[i]['gold_answer']) for i,r in enumerate(stages['qwen14'])]
    return methods


def run(root, out):
    items, stages, saved, checks = audit(root)
    methods = methods_from(stages, items)
    comparisons = [(m,'direct') for m in methods if 'oracle' not in m and m != 'direct']
    comparisons += [('qwen14_reflect_1','deepseek_reflect_1'),('qwen14_reflect_2','deepseek_reflect_2'),
                    ('qwen14_reflect_2','qwen14_reflect_1'),
                    ('qwen14_reflect_1','base_then_qwen14_independent'),
                    ('qwen14_reflect_2','base_then_qwen14_independent')]
    stats = [{'method':a,'baseline':b,**paired(methods[a],methods[b])} for a,b in comparisons]
    errors=[]
    for i,item in enumerate(items):
        base = methods['direct'][i]
        failures = {m:c[i] for m,c in methods.items() if 'oracle' not in m}
        if not base['correct'] or any(not c['correct'] for c in failures.values()):
            errors.append({'id':item['id'],'question':item['question'],'reference_solution':item['answer'],
                           'gold_answer':item['gold_answer'],'difficulty':item.get('difficulty','unknown'),
                           'methods':{m:{k:c[k] for k in ('pred','correct','final_answer','token_cost','final_marker_present')} for m,c in failures.items()}})
    format_stats={m:{'missing_marker':sum(not c['final_marker_present'] for c in cs),
                     'unextractable':sum(c['pred'] is None for c in cs),
                     'missing_marker_and_wrong':sum(not c['final_marker_present'] and not c['correct'] for c in cs)}
                  for m,cs in methods.items()}
    distributions={role:dict(Counter(c['action'] for c in methods[f'{role}_oracle'])) for role in ('deepseek','qwen14')}
    round_changes={role:mcnemar(methods[f'{role}_reflect_2'],methods[f'{role}_reflect_1']) for role in ('deepseek','qwen14')}
    groups={}
    for difficulty in sorted({i.get('difficulty','unknown') for i in items}):
        indices=[j for j,i in enumerate(items) if i.get('difficulty','unknown')==difficulty]
        group={}
        for m,cs in methods.items():
            chosen=[cs[j] for j in indices]
            base=[methods['direct'][j] for j in indices]
            transition=mcnemar(chosen,base)
            group[m]={'n':len(indices),'accuracy':sum(c['correct'] for c in chosen)/len(indices),
                      'avg_token_cost':sum(c['token_cost'] for c in chosen)/len(indices),
                      'fixes':transition['learned_correct_baseline_wrong'],'harms':transition['learned_wrong_baseline_correct']}
        groups[difficulty]=group
    result={'audit':checks,'comparisons':stats,'format_diagnostics':format_stats,
            'oracle_label_distribution':distributions,'round2_vs_round1':round_changes,
            'by_difficulty':groups,'bootstrap_iterations':10000,'bootstrap_seed':20260910,
            'comparison_status':'exploratory, unadjusted for multiple comparisons',
            'scoring':'original benchmark gold unchanged; candidate errors need manual semantic review',
            'no_llm_calls':True}
    out.mkdir(parents=True,exist_ok=False)
    save_json(result,out/'analysis.json')
    write_jsonl(errors,out/'error_cases.jsonl')
    lines=['# 数学反思开发对照：完整审计与配对分析','',
           '固定训练侧 200 题；不是测试集，不是已训练策略结果。原始标准答案及评分不变。','',
           '## 完整性','',
           f"三阶段各 {checks['n']} 题，共 {checks['unique_calls']} 个独立调用、{checks['paid_attempts']} 次返回尝试。",
           f"未解决截断为 0；生成总开销 {checks['total_generation_tokens']} token，包含输入、输出和已记录重试。",
           '所有共享 Base、两轮前缀、提示、请求预算、usage、文件哈希和候选评分均通过检查。','',
           '## 方法结果','',
           '| 方法 | 准确率 | 输入 | 输出 | 总 token | 奖励 | 修复/引错 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in saved['methods']:
        lines.append(f"| {row['method']} | {row['accuracy']:.4f} | {row['avg_prompt_tokens']:.2f} | {row['avg_completion_tokens']:.2f} | {row['avg_token_cost']:.2f} | {row['avg_reward']:.4f} | {row['fixes']}/{row['harms']} |")
    lines += ['', 'Oracle 按正确性优先、同正确性选较低成本，仅为回溯上界，不可部署。奖励 lambda=0.0001。',
              '独立解题同时报告只调用14B，以及已经调用Base后再调用14B的两种成本，不能混用。','',
              '## 配对统计','',
              '差值为方法减基线。按问题进行10000次配对Bootstrap；McNemar为双侧精确检验。',
              '以下为探索性多重比较，未校正；区间包含0不构成等效或非劣证明。','',
              '| 方法 - 基线 | 准确率差 [95% CI] | token差 [95% CI] | 修复/引错 | McNemar p |',
              '|---|---:|---:|---:|---:|']
    for row in stats:
        a,t,m=row['accuracy'],row['token_cost'],row['mcnemar']
        lines.append(f"| {row['method']} - {row['baseline']} | {a['observed_difference']:.4f} [{a['ci95_lower']:.4f}, {a['ci95_upper']:.4f}] | {t['observed_difference']:.1f} [{t['ci95_lower']:.1f}, {t['ci95_upper']:.1f}] | {m['learned_correct_baseline_wrong']}/{m['learned_wrong_baseline_correct']} | {m['exact_two_sided_p']:.6f} |")
    lines += ['', '## 分类监督空间','', 'Oracle 动作分布（只描述当前200题，不估计全训练集分布）：','']
    lines += [f'- {role}: {labels}' for role,labels in distributions.items()]
    lines += ['', '## 解读与下一步','',
              '- 更强反思器提供了更多可修复候选，但不能据此断言响应条件反思优于独立重新作答。',
              '- 按当前样本点估计，14B独立作答比固定14B反思更准确且更便宜；配对区间用于限定结论强度。',
              '- 策略潜在价值应在较低成本下接近候选上界，而不是仅宣称胜过较弱Base；必须保留强模型独立基线。',
              '- 200题反思正例很少，不用这些题既训练又验证选择器，也不立即进行全量策略训练。',
              '- 先核对 error_cases.jsonl 的歧义/参考解问题，保持本表官方gold不变；任何修订另建有依据的敏感性分析。',
              '- 下一阶段只在训练侧预先锁定扩样和共同验证清单；不再用测试Oracle调门限，不恢复QA。','',
              '格式诊断、难度分组、第二轮正负转移与完整哈希详见 analysis.json。缺少Final answer标记不自动等于错误。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'audit':checks,'oracle_labels':distributions,'output':str(out)},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/math_reflector_pilot_v1/run_v1'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.root.resolve(),(args.output or args.root/'analysis_v1').resolve())


if __name__=='__main__':
    main()
