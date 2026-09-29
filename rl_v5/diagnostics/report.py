"""Write a factual report from a completed run; add interpretation only after review."""
from pathlib import Path
import json,math
ROOT=Path(__file__).resolve().parents[2];run=ROOT/'rl_v5/runs/structured-split'
r=json.loads((run/'results.json').read_text());a=json.loads((run/'analysis.json').read_text())
assert a['complete'] and a['completed_games']==r['total_games']
def score(s):return f"{s['wins']} 胜 / {s['draws']} 平 / {s['losses']} 负"
def pct(x):return f'{100*x:.1f}%'
def paired(rows,base):
    b={(x['map'],x['seed'],x['side']):x['reward'] for x in base}
    d=[x['reward']-b[(x['map'],x['seed'],x['side'])] for x in rows];win=sum(x>0 for x in d);loss=sum(x<0 for x in d);n=win+loss
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(win,loss)+1))/2**n) if n else 1.
    return win,loss,len(d)-n,p
lines=['# v5 实验报告','', '本报告由完成的实验记录生成；所有对手均为原始方案1。教师使用本机原生 Python 推理，方案1使用官方沙箱。没有进行蒸馏或提交。','',
'## 实验规模与架构','',f"完成 {r['total_games']} 场正式对局。训练共 {sum(len(v) for v in r['rollouts'].values())} 场，另有对照、分支干预、验证、喂养比较和独立测试。全部对局累计耗时 {sum(x['seconds'] for x in a['records'].values())/60:.1f} 分钟。",'',
'Spatial 与 GRU 各约 267 万参数，本轮各训练 258,790 参数。新增尾部 29×7×7 地图、32 维阶段/团队上下文和每候选 32 维后果表示；候选共享编码后，同时驱动分裂时机与尺寸选择。原移动主干、移动头和 GRU 全部冻结。详见 `架构说明.md`。','',
'PPO 以终局团队胜/平/负为回报，按对局均衡；辅以 20 回合存活和正向净增长预测。不给龙数、分裂或自爆次数奖励。固定前缀的单动作干预仅在终局胜负改变时形成偏好监督。','',
'## 验证与选择','', '| 架构 | 分裂控制 | 验证（8局） |','|---|---|---|']
for n,cs in r['validation'].items():
    for k,v in cs.items():lines.append(f"| {n} | {k} | {score(v['score'])} |")
lines+=['','验证地图：dilemma、queen_of_spades；种子 9411、9417，A/B 两侧。先按验证得分选择分裂控制，平局偏向原规则；再用 4 局配对验证决定是否采用动态喂养。','', '| 架构 | 原喂养（4局） | 动态喂养（4局） | 选定 |','|---|---|---|---|']
for n,v in r['feeding'].items():lines.append(f"| {n} | {score(v['original']['score'])} | {score(v['dynamic']['score'])} | {r['selection']['selected'][n]['feeding']} |")
lines+=['','## 独立测试','', '全部选择冻结后，才运行 Colosseum、devil 两张未用于本轮训练/选择的地图，种子 9511、9517，A/B 两侧。测试结果不再用于调整本轮选择。','',
'| 架构 | 选定方案 | 原分裂/原喂养控制 | 配对改善/退步/相同 |','|---|---|---|---|']
for n,v in r['tests'].items():
    b=r['test_controls'][n];up,down,tie,p=paired(v['rows'],b['rows'])
    lines.append(f"| {n} | {score(v['score'])} | {score(b['score'])} | {up}/{down}/{tie} |")
lines+=['','样本仍少，且同图不同方位并非完全独立；这些结果只支持当前地图、种子和方案1对手上的判断，不能代表排行榜胜率。','', '## 学习诊断','',
'| 模型 | 最后更新决策数 | 贪心偏离规则 | 仅时机偏离 | 尾图置零动作变化 | 分支偏好对数 |','|---|---:|---:|---:|---:|---:|']
for n in ('spatial','gru'):
    for seed in (9201,9203):
        m=r['training'][f'{n}-s{seed}-ppo-4'];tail=m['input_probe']['tail']
        lines.append(f"| {n}-{seed} | {m['decisions']} | {pct(m['greedy_rule_disagreement_rate'])} | {pct(m['greedy_mode_disagreement_rate'])} | {tail['action_changes']}/{m['probe_states']} | {m['preference_pairs']} |")
full=[]
for n in ('spatial','gru'):
    for seed in (9201,9203):
        path=run/f'{n}-s{seed}-final-full.json'
        if not path.exists():
            from compare_policies import compare
            files=sorted((run/'episodes').glob(f'{n}-s{seed}-rollout-*.npz'))
            assert len(files)==16
            comparison=compare(run/f'{n}-initial-{seed}.pt',run/f'{n}-s{seed}-ppo-4.pt',files)
            path.write_text(json.dumps(comparison,indent=2),encoding='utf-8')
        data=json.loads(path.read_text())
        assert data['new_sha256']==r['checkpoint_sha256'][f'{n}-ppo-{seed}']
        assert data['counts']['states']==sum(r['training'][f'{n}-s{seed}-ppo-{u}']['decisions'] for u in range(1,5))
        full.append((f'{n}-{seed}',data['counts']))
if full:
    lines+=['','额外的全量已保存训练状态检查（不代表未见状态）：','','| 模型 | 状态数 | 与原规则不同的确定性动作 |','|---|---:|---:|']
    for name,c in full:lines.append(f"| {name} | {c['states']} | {c['rule_disagreements']} |")
lines+=['','输入探针来自训练状态，不是额外的泛化胜率测试。上下文/候选置零也会删除固定规则先验，不能把其变化都解释为学会利用新输入；尾图置零没有这个直接先验问题。','',
'每次更新均通过动作概率与价值一致性核对，以及冻结主干哈希检查。此前固定前缀同动作重放的完整输入/输出/事件哈希一致；正式不同动作干预也要求前缀完全相同。','',
'## 回放与资源集中','', '| 独立测试 | 平均己方最长龙 | 平均己方总长度 | 指定供给新增珍珠 | 己方收回 | 被终局最长龙直接收回 |','|---|---:|---:|---:|---:|---:|']
for n in ('spatial','gru'):
    g=a['groups'][f'test-{n}'];t=g['totals'];m=g['means']
    lines.append(f"| {n} | {m.get('final_longest',0):.1f} | {m.get('final_total',0):.1f} | {t.get('feed_deposited',0)} | {t.get('feed_recovered_ally',0)} | {t.get('feed_direct_final_longest',0)} |")
lines+=['','资源归属来自官方回放：仅把死亡事件中在死者身体格上新出现的珍珠归给该龙，再跟踪实际吃到它的龙。不追踪随后分裂继承或再次死亡转移的资源。因此“终局最长龙直接收回”是直接回收量，不能当作全部间接贡献。所有回放重建终局的龙数、最长龙和总长度均与官方结果核对。','',
'## 可复现文件','', '- `runs/structured-split/selection.json`：选中的权重、分裂控制模式与喂养规则，三者应一起使用。','- `runs/structured-split/results.json`：完整训练与评估结果。','- `runs/structured-split/analysis.json`：阶段曲线、子龙生存、回收与错误统计。','- `runs/structured-split/manifest.json`、`sources/`：来源哈希与源码快照。','- `runs/structured-split/episodes/`：逐局数据、遥测和官方回放。','',
'尚未做教师沙箱算力适配；本轮输出是本地训练与评估用教师。']
(ROOT/'rl_v5/实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(ROOT/'rl_v5/实验报告.md')
