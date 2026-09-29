"""Expose the locally installed official viewer's parser without mounting its UI."""
from pathlib import Path
import zipfile,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
source=ROOT/'tools/replay-viewer.vsix'
with zipfile.ZipFile(source) as z:s=z.read('extension/dist/webview/webview.js').decode()
# Pinned symbol boundary: fail loudly if the viewer is upgraded/incompatible.
assert s.startswith('(function(){') and s.count('var Ff=z(')==1
end='''globalThis.decodeReplay=function(bytes){let r=Mf(bytes);return {map:r.map,result:{teamA:Af(r.result.teamA),teamB:Af(r.result.teamB)},events:[...r.events].map(e=>Tf(e,r.formatVersion))}};
globalThis.replayState=function(text){let map=Ra.fromMapText(text);return {map:new Va(map),bodies:new Xa(map.initialDragons),pearl:Da.Pearl}};})();'''
(HERE/'replay-library.cjs').write_text(s[:s.index('var Ff=z(')]+end,encoding='utf-8')
(HERE/'decoder-provenance.json').write_text(json.dumps({'vsix_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'webview_sha256':hashlib.sha256(s.encode()).hexdigest(),'method':'Official Mf/Tf and Ra/Va/Xa decoder/state classes; UI mounting excluded. Reconstructed final teams are checked against replay result.'},indent=2))
