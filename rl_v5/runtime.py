"""Move plus strategic split PPO; existing versions are immutable inputs."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BOT=ROOT/'方案2-v2'
os.environ.setdefault('PROCESSOR_ARCHITECTURE','AMD64')
os.environ['XDG_CACHE_HOME']=str(ROOT/'.cache')
os.environ['UNSWBC_NO_UPDATE']='1'
os.environ['UNSWBC_NO_VSCODE']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(BOT))

def config():return json.loads((ROOT/'rl_v5/config.json').read_text(encoding='utf-8'))
def hashes():
    ps=list((ROOT/'rl_v5').glob('*.py'))+list(BOT.glob('*.py'))+[ROOT/'rl_v5/config.json',ROOT/'方案1/main.py',ROOT/'方案1/helper.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ps}
