"""Publish audited completed results, preserving validation-only selection."""
from pathlib import Path
import json, hashlib, sys, subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rl_v5'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def score(s):
    return f"{s['wins']} 胜 / {s['draws']} 平 / {s['losses']} 负"

def paired(rows, controls):
    key = lambda r: (r['map'], r['seed'], r['side'])
    base = {key(r): r['reward'] for r in controls}
    assert len(base) == len(controls) == len(rows)
    delta = [r['reward'] - base[key(r)] for r in rows]
    return '/'.join(str(sum(test(d) for d in delta)) for test in
                    (lambda x: x > 0, lambda x: x < 0, lambda x: x == 0))

def main():
    from model import load as load_model
    from training import base_hash
    ref_dir = ROOT / 'rl_v5/runs/structured-split'
    run = ROOT / 'rl_v5/runs/refined-teachers'
    ref, result = load(ref_dir / 'results.json'), load(run / 'results.json')
    subprocess.run([sys.executable, str(ROOT/'rl_v5/diagnostics/analyze_branches.py'), str(ref_dir)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(ROOT/'rl_v5/diagnostics/preference_progress.py')], check=True, stdout=subprocess.DEVNULL)
    analysis = load(run / 'analysis.json')
    assert analysis['complete'] and analysis['completed_games'] == result['total_games']
    for directory in (ref_dir, run):
        subprocess.run([sys.executable, str(ROOT/'rl_v5/diagnostics/classify_splits.py'), str(directory)], check=True, stdout=subprocess.DEVNULL)
        a = load(directory / 'analysis.json')
        assert all('final_longest' in row for row in a['records'].values()), 'Missing decoded replay'
        assert not sum(row['errors'] + row['lineage_errors'] for row in a['records'].values())
    guard_limits = {}
    for path in (run/'episodes').glob('*.json'):
        if path.name.endswith(('.telemetry.json', '.replay-metrics.json')):
            continue
        meta = load(path)
        if meta.get('job', {}).get('parent_min'):
            guard_limits[path.stem] = meta['job']['parent_min']
    guarded_splits_checked = 0
    for rows in load(run/'split-outcomes.json')['records'].values():
        for row in rows:
            if row['episode'] in guard_limits:
                assert row['parent_length_before'] - row['child_size'] >= guard_limits[row['episode']]
                guarded_splits_checked += 1
    selection = result['selection']
    assert selection == load(run / 'selection.json')
    sensitivity = {}
    for architecture in ('spatial', 'gru'):
        checkpoint = Path(result['validation'][architecture]['refined']['weights'])
        files = sorted((run/'episodes').glob(f'{architecture}-refine-rollout-2-*.npz'))
        assert len(files) == 4
        subprocess.run([sys.executable, str(ROOT/'rl_v5/diagnostics/probe_checkpoint.py'), str(checkpoint), *map(str,files)],
            check=True, stdout=subprocess.DEVNULL)
        sensitivity[architecture] = load(checkpoint.with_suffix('.sensitivity.json'))
    frozen = {}
    for name, choice in selection['selected'].items():
        assert sha(choice['weights']) == selection['checkpoint_sha256'][name]
        frozen[name] = base_hash(load_model(choice['weights']))
        assert frozen[name] == ref['frozen_base_sha256'][name]
    name = selection['preferred_architecture']
    policy = dict(architecture=name, **selection['selected'][name],
        checkpoint_sha256=selection['checkpoint_sha256'][name], frozen_base_sha256=frozen[name],
        selection_basis='Final eight validation matches; independent tests never select the model.',
        validation=selection['final_validation_scores'][name], test=result['tests'][name]['score'],
        use='Local native teacher evaluation only; official sandbox deployment is unverified.',
        source_manifest=str(run / 'manifest.json'))
    (ROOT / 'rl_v5/best-policy.json').write_text(json.dumps(policy, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT/'rl_v5/diagnostics/evaluate_best.py'), '--describe'],check=True,stdout=subprocess.DEVNULL)
    choice = selection['selected'][name]
    lines = ['# v5 优化总结', '',
        f"最终按验证集选用 **{name}**，分裂控制 `{choice['controller']}`（候选 `{choice['key']}`，{'随机执行' if choice.get('stochastic') else '确定性执行'}，父龙最小保留约束：{choice.get('parent_min', '无额外约束')}），喂养 `{choice['feeding']}`。验证 {score(policy['validation'])}；新种子独立测试 {score(policy['test'])}。", '',
        '权重、分裂控制、执行方式和喂养规则必须一起使用，完整配置见 `best-policy.json`。`evaluate-v5-best.cmd --describe` 可校验并查看配置；不加该参数会执行本地评估。这是本地教师，尚未验证官方沙箱运行限制。', '',
        '## 完成的优化', '',
        '- 增加尾部局部图、回合/剩余时间、团队与资源信息、候选分裂后果、已知路径和输入置信度。原移动主干、移动头及 GRU 冻结。',
        '- 分裂采用“是否分裂＋分裂尺寸”的分层决策；用终局胜负训练，辅以子龙/父龙生存与增长预测。',
        '- 两个架构各跑两个训练种子；检测到晚期样本过多、贪心策略几乎不变后，比较更强更新与阶段均衡，并限制整体和各阶段的 KL。',
        '- 分开比较训练后的确定性执行、随机执行，以及未训练的随机分裂先验，区分学习收益与随机扩张收益。',
        '- 根据参考验证中的过早 2＋2 分裂记录，单独比较父龙至少保留长度 3 的执行约束；同时评估未训练先验配合同一约束。不改变网络输入、训练或规则保护分支。',
        '- 单独比较动态回程喂养，以及局部 KING 在获知更大接收者时转为供给者的规则。喂养筛选胜者再用另外 4 局确认。',
        '- 保留原规则、原权重和原始方案1；所有比赛串行、每局新进程。未蒸馏、未提交。', '',
        '## 更新方法对照', '',
        '| 方法 | 开发集（4局） |', '|---|---|']
    for key, v in result['pilots'].items():
        lines.append(f"| {'阶段均衡＋更强更新' if key == 'phase' else '仅更强更新'} | {score(v['score'])} |")
    lines += ['', f"选定方法：`{result['method']}`，随后各架构再收集两批新的 on-policy 对局继续训练。开发集选择采用 arena、schooltime，种子 9151，双方位置。", '',
        '## 验证集选择', '',
        '| 架构 | 参考版最佳分裂（8局） | 追加训练分裂（8局） | 最终完整策略（8局） | 采用分裂 / 喂养 |',
        '|---|---|---|---|---|']
    for n, v in result['validation'].items():
        c = selection['selected'][n]
        lines.append(f"| {n} | {score(v['reference']['score'])} | {score(v['refined']['score'])} | {score(result['final_validation'][n]['score'])} | {c['key']} / {c['feeding']} |")
    lines += ['', '上表“追加训练分裂”采用确定性执行；随机执行的额外对照如下（同 8 局，原喂养）：', '',
        '| 架构 | 未训练随机先验 | 追加训练后随机执行 |', '|---|---|---|']
    for n, variants in result['execution'].items():
        lines.append(f"| {n} | {score(variants['initial-stochastic']['score'])} | {score(variants['refined-stochastic']['score'])} |")
    lines += ['', '父龙最小长度单因素对照（同 8 局，原喂养，确定性执行）：', '',
        '| 架构 | 未训练先验＋父龙≥3 | 追加训练＋父龙≥3 |', '|---|---|---|']
    for n, variants in result['execution'].items():
        lines.append(f"| {n} | {score(variants['initial-parent3']['score'])} | {score(variants['refined-parent3']['score'])} |")
    lines += ['', '该约束在观察参考验证的失败类型后加入，并在任何独立测试开始前固定。GRU 种子 9201 在参考验证中 99 次战略分裂有 85 次为长度 4 拆成 2＋2；子龙 20 回合内死亡为 13/99，积极规则为 3/51。这是相关线索，不证明分裂尺寸导致死亡。此处只测试父龙长度 3 这一阈值，未搜索其他保护条件。']
    lines += ['', '验证地图为 dilemma、queen_of_spades，种子 9411、9417，双方位置。分裂、喂养及推荐架构都在新独立测试之前锁定；同分优先保留对照。', '',
        '## 新种子独立测试', '',
        '| 架构 | 最终策略（8局） | 参考版完整策略（同8局） | 改善 / 退步 / 相同 |',
        '|---|---|---|---|']
    for n, v in result['tests'].items():
        control = result['test_controls'][n]
        lines.append(f"| {n} | {score(v['score'])} | {score(control['score'])} | {paired(v['rows'], control['rows'])} |")
    prior_tests={n:v for n,v in result['test_prior_controls'].items() if not v['reused']}
    if prior_tests:
        lines += ['', '学习归因对照：下表两侧使用相同喂养规则、执行方式、父龙保护约束及测试条件，仅更换分裂权重；随机执行时也保持相同随机数设置。', '',
            '| 架构 | 训练后策略 | 匹配的未训练先验 | 改善 / 退步 / 相同 |', '|---|---|---|---|']
        for n, prior in prior_tests.items():
            v=result['tests'][n]
            lines.append(f"| {n} | {score(v['score'])} | {score(prior['score'])} | {paired(v['rows'],prior['rows'])} |")
    lines += ['', '所有对手均为原始方案1。新测试种子为 9811、9817，地图 Colosseum、devil；种子更新，地图几何与参考版测试相同。上述结果不代表对其他对手或排行榜的胜率。', '',
        '## 结论与限制', '']
    for n, c in selection['selected'].items():
        if c['key'].startswith('initial-'):conclusion='采用未训练的分裂先验及其执行配置，不能把收益归因于分裂学习'
        elif c['controller']=='network':conclusion='最终采用训练后的神经网络分裂决策'
        else:conclusion='最终仍采用规则分裂，追加学习未能在选择指标上胜过对照'
        lines.append(f"- {n}：{conclusion}；移动能力来自保留的旧模型。")
    lines += [f"- 参考实验 {ref['total_games']} 局，反馈优化 {result['total_games']} 局；两个阶段的完整回放均已核对官方终局结果，执行/谱系检查报错计数为 0。", 
        '- 原移动参数哈希保持不变。网络参数约 267 万，本轮训练的分裂与辅助分支共 258,790 参数。',
        f'- 父龙保护对照共核对 {guarded_splits_checked} 次实际战略分裂，父龙剩余长度均满足对应约束；规则保护分支单独保留。',
        '- 样本量仍小，地图上的双方对局并非完全独立；验证集反复用于多个候选选择，有过拟合风险。',
        '- 喂养回收统计只追踪直接回收，不包含分裂继承或再次转移，不能把直接回收量当作全部贡献。', '',
        '## 分裂复盘线索', '',
        '| 最终测试策略 | 战略分裂数 | 子龙20回合内死亡 / 已观察窗口 | 存活完整20回合但无正增长 |',
        '|---|---:|---:|---:|']
    outcomes = load(run/'split-outcomes.json')['groups']
    for n in ('spatial', 'gru'):
        d = outcomes.get(f'test-{n}', {})
        lines.append(f"| {n} | {d.get('splits',0)} | {d.get('child_died20',0)} / {d.get('child_observed20',0)} | {d.get('child_complete_window_no_positive_growth',0)} |")
    lines += ['', '追加训练候选的输入敏感性（抽样训练状态，固定规则先验和掩码）：', '',
        '| 候选架构 | 样本数 | 打乱尾图后动作变化 | 打乱旧主干表示后动作变化 |', '|---|---:|---:|---:|']
    for n, s in sensitivity.items():
        lines.append(f"| {n} | {s['states']} | {s['probes']['tail']['action_changes']} | {s['probes']['base_h']['action_changes']} |")
    lines += ['', '打乱后特征组合可能不自然；该探针只检查网络是否使用输入，不证明因果重要性或泛化能力。此处是追加训练候选，即使最终选择保留规则也照实列出。']
    causal_data=load(ref_dir/'counterfactual-summary.json');causal=causal_data['summary']
    lines += ['', '这些是复查指标，不证明死亡由分裂造成。官方死亡码 A 表示无有效动作，也可能是主动牺牲；H 表示头对头失败。详细实例和原始死亡码见 `split-outcomes.json`，不把主动供给或必要牺牲一律记作错误。', '',
        f"单动作分支干预共 {causal['interventions']} 组，其中 {causal['terminal_outcome_changed']} 组改变终局胜负；{causal['complete_20_round_pairs']} 组具有完整的 20 回合配对观察，其中 {causal['team_total20_changed']} 组改变团队总长度。中期收益变化不直接作为终局偏好标签。", '']
    changed=[r for r in causal_data['rows'] if r['baseline']['reward']!=r['alternative']['reward']]
    label={-1:'负',0:'平',1:'胜'}
    action=lambda size:f'分裂出长度 {size} 的子龙' if size else '继续移动'
    for row in changed[:2]:
        lines.append(f"- 具体案例：{row['architecture']} / {row['map']} / {row['side']}，第 {row['round']} 回合，原长 {row['parent_length']}、角色 {row['role']}；把“{action(row['baseline_child_size'])}”改成“{action(row['alternative_child_size'])}”，固定后续规则下由{label[row['baseline']['reward']]}变{label[row['alternative']['reward']]}。")
    progression = load(ref_dir/'preference-progression.json')['rows']
    if progression:
        lines += ['', '有效偏好局面的“时机”和“尺寸”分别检查。以下直接检查网络原始输出，不施加父龙长度执行约束；均为已用于监督的开发状态，不能当作泛化成绩，也不能将变化单独归因于偏好损失。', '',
            '| 局面 | 最终候选 | 分裂概率 | 证据支持的子龙长度 | 贪心子龙长度（0为继续） | 完整动作匹配 |',
            '|---|---|---:|---:|---:|---|']
        for row in progression:
            final_checks = [c for c in row['checks'] if c['checkpoint'].endswith(('-ppo-4', '-refine-2-fine'))]
            assert len(final_checks) == 3
            for c in final_checks:
                lines.append(f"| {row['architecture']} / {row['map']} / {row['side']} | {c['checkpoint']} | {c['split_probability']:.1%} | {row['preferred_child_length']} | {c['greedy_child_length']} | {'是' if c['exact_action_matches'] else '否'} |")
        lines += ['', '分裂概率升高不等于选中了已验证的分裂尺寸；其他尺寸未经该单动作干预验证，不能推断其胜负。详细概率轨迹见 `runs/structured-split/preference-progression.json`。']
    lines += ['', '## 后续建议', '',
        '先针对独立测试中的死亡、无效分裂、资源未集中等失败类型做回放分析，再扩大训练地图、种子和规则对手池。仅当分裂学习在这些对照中稳定获益后，再考虑解冻移动或蒸馏。', '',
        '详细文件：`架构说明.md`、`实验报告.md`、`runs/refined-teachers/results.json`、`runs/refined-teachers/analysis.json`。', '']
    (ROOT / 'rl_v5/优化总结.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(policy, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
