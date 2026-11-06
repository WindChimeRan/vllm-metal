"""Render the archived measurements; does not run benchmarks."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data=json.loads((ROOT/'benchmark.json').read_text())
lookup={(r['arm'],r['concurrency']):r for r in data['aggregate']}
methods=['target','eagle3','dflash','mlx-dflash']
labels=['MPS\nTarget only','MPS\nEAGLE3 K=3','MPS\nDFlash K=7','MLX\nDFlash K=7']
plt.rcParams.update({'font.family':'sans-serif','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(13,6))
for ax,concurrency in zip(axes,(1,8)):
    for runtime,shift,color,label in [('stable',-.19,'#9ca3af','Torch 2.13'),('nightly',.19,'#327ca4','Default nightly (#1117)')]:
        rows=[lookup[(f'{runtime}-{m}',concurrency)] for m in methods]
        assert all(r['runs']==2 for r in rows)
        rates=np.array([r['mean_tok_s'] for r in rows])
        errors=np.array([[r['mean_tok_s']-r['min_tok_s'] for r in rows],[r['max_tok_s']-r['mean_tok_s'] for r in rows]])
        bars=ax.bar(np.arange(4)+shift,rates,width=.35,color=color,label=label,yerr=errors,capsize=3)
        for bar,value,row in zip(bars,rates,rows):
            ax.annotate(f'{value:.1f}',(bar.get_x()+bar.get_width()/2,row['max_tok_s']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xticks(np.arange(4),labels)
    ax.set_title(f'Concurrency {concurrency}')
    ax.set_ylabel('Output tokens / second')
    ax.set_ylim(0,ax.get_ylim()[1]*1.15)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False,fontsize=9)
fig.suptitle('Qwen3-8B BF16: DFlash serving on MPS and MLX\nPre-rebase measurements · 2026-10-29 · vLLM 0.32',fontsize=14)
fig.text(.5,.015,'36 fixed Speculator Bench prompts × 128 output tokens · Two runs per setting; reversed configuration order\nBars: mean; whiskers: range · Same z-lab DFlash draft · Nightly: 2.16.0.dev20261013 · M5 Pro, 20 GPU cores, 64 GiB',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,.91))
fig.savefig(ROOT/'comparison.png',dpi=180,facecolor='white')
fig.savefig(ROOT/'comparison.pdf',facecolor='white')
