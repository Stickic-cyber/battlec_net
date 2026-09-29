"""Static report figures, made only from completed, frozen experimental results."""
from pathlib import Path
import json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
run=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'rl_v5/runs/structured-split'
r=json.loads((run/'results.json').read_text());a=json.loads((run/'analysis.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors=['#64748b','#d97706','#0284c7','#059669']
for col,name in enumerate(('spatial','gru')):
    val=r['validation'][name];labels=list(val);wins=[v['score']['points'] for v in val.values()]
    axes[0,col].bar(labels,wins,color=colors);axes[0,col].set(ylim=(0,8),ylabel='Validation points / 8',title=name.upper()+' split selection')
    for i,v in enumerate(wins):axes[0,col].text(i,v+.12,str(v),ha='center')
    for key,color in zip(labels,colors):
        g=a['groups'][f'val-{name}-{key}'];curve=g['curve_mean'];xs=[int(x) for x in curve]
        axes[1,col].plot(xs,[curve[str(x)]['longestDragon'] for x in xs],marker='.',label=key,color=color)
    axes[1,col].set(xlabel='Round',ylabel='Mean longest dragon',title='Matched validation trajectories');axes[1,col].legend(fontsize=8)
fig.savefig(run/'validation-summary.png',dpi=160);plt.close(fig)
print(run/'validation-summary.png')
