# Battlecode v5 教师模型

此分支保存 v5 教师的源码、配置、实验报告和**选中的最佳 checkpoint**，以及加载模型所需的最小规则、先验和地图依赖。未上传其他历史 checkpoint、大型训练数据、回放、日志或 Python 环境。

## 最佳模型

- 权重：[`rl_v5/runs/structured-split/gru-initial-9201.pt`](rl_v5/runs/structured-split/gru-initial-9201.pt)
- SHA256：`2dc768e2e756855af04ab0837861780a2c97a240c499eabda1c8cc10eaecffbb`
- 约 267 万参数，GRU hidden 为 256。
- 实际选中组合：**神经网络移动 + 积极分裂规则 + 原喂养规则 + 紧急规则**。
- `initial` 指 v5 分裂训练的初始点；移动网络继承了此前训练成果，不是随机初始化。checkpoint 中的分裂网络不等于最佳组合实际采用的分裂规则。
- 历史验证为 6 胜 2 负，独立测试为 7 胜 1 负，对手均为原始方案1；不代表排行榜胜率。

## 安装与检查

建议 Python 3.11 或 3.12。在仓库根目录执行：

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python check_teacher.py
```

检查脚本会校验导出文件哈希、加载权重和运行一次合成观测前向，验证四方向概率及 GRU 输出。它不运行比赛或启动训练。若你主动修改源码，原始导出哈希检查失败是预期行为。

在自己的脚本中加载移动策略：

```python
from pathlib import Path
import sys
import numpy as np

root = Path.cwd()  # 当前目录为仓库根目录
sys.path.insert(0, str(root / 'rl_v5'))
from model import Policy

policy = Policy(root / 'rl_v5/runs/structured-split/gru-initial-9201.pt')
hidden = np.zeros(256, np.float32)  # 每条新生龙独立初始化
# obs 必须使用 base_observation.Observer 和 base_actions 构造。
# action, hidden, logp = policy.movement(obs, hidden)
# 概率为 np.exp(logp)，顺序 N/E/S/W。
```

每条龙独立保存记忆。规则接管的回合也应更新 GRU；完整规则集成可参考 `rl_v5/runner.py` 中的 `Dragon`。`Policy.act()` 是分裂网络接口，不能代替最佳组合的积极分裂规则。

## 文件导航

| 内容 | 路径 |
|---|---|
| 教师结构、权重加载 | `rl_v5/model.py`、`base_model.py` |
| 输入特征与记忆 | `rl_v5/base_observation.py`、`features.py` |
| 训练与采样源码 | `rl_v5/training.py`、`collect.py`、`runner.py`、`experiment.py` |
| 后续优化源码 | `rl_v5/refinements/` |
| 超参数与历史最佳选择 | `rl_v5/config.json`、`best-policy.json` |
| 实验说明 | [实验报告](rl_v5/实验报告.md)、[架构说明](rl_v5/架构说明.md)、[优化总结](rl_v5/优化总结.md) |
| 部分历史结果与来源清单 | `rl_v5/runs/structured-split/`、`rl_v5/runs/refined-teachers/` |
| 规则依赖、原始对手 | `方案2-v2/`、`方案1/` |
| 冻结方向先验 | `rl_v1/runs/v1-first/deployment-best.npz` |
| 导出文件完整性 | `export-manifest.json` |

## 训练与复现范围

这是**源码和最佳模型的精简存档**，不是原 1.86 GB 实验目录的完整备份。

- v5 的 `training.py` 主要训练分裂分支，冻结已有移动主干；若要训练移动网络，需要另写训练入口或取得前代移动训练代码。
- 历史 `best-policy.json`、结果和 manifest 保留原始绝对路径及哈希以便溯源；它们不是可在任意机器直接运行的配置。上面的检查/加载示例使用当前仓库相对路径。
- `experiment.py` 从头重放历史实验需要额外的 v4 checkpoint、前代源码/权重；历史测试和分析脚本还可能需要未上传的数据或官方 viewer。不能直接运行它们期待重现完整实验。
- 已选最佳 checkpoint 可以直接加载；进一步训练可以从它出发重新采集数据，无需找回所有旧轨迹。原始 v5 的分裂训练与最终采用的规则分裂须分开评估。
- `check_teacher.py` 的前向检查只证明导出模型可加载，不代表新的比赛验证。

官方工具、helper 与地图来源：<https://github.com/unswcpmsoc/battlecode>，其 MIT 许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本分支未为自有策略源码另行声明开源许可。
