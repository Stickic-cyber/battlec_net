// Offline analysis only; no future replay state is supplied to the playing policy.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
require('./replay-library.cjs');
const analysisHash=crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex');
const key=p=>`${p.x},${p.y}`;
function analyze(file,telemetry){
 const r=decodeReplay(fs.readFileSync(file)),state=replayState(r.map);
 const ledger=new Map(),feedSet=new Set(),actions=new Map(),births=[],deaths=[],consumptions=[],curves=[];
 for(const [id,history] of Object.entries(telemetry.histories))for(const h of history)if(h.feed)feedSet.add(`${id}:${h.round}`);
 let round=-1,turn=null,pendingDeaths=[],pending=[];let ambiguities=0,ambiguousSources=0;
 const standing=team=>{const ds=[...state.bodies.dragons.values()].filter(d=>d.team===team);return {dragonCount:ds.length,longestDragon:Math.max(0,...ds.map(d=>d.body.length)),totalLength:ds.reduce((s,d)=>s+d.body.length,0)}};
 const snapshot=()=>{if(round>=0)curves.push({round,A:standing('A'),B:standing('B')})};
 for(const e of r.events){
   if(e.type==='roundStart'){snapshot();round=e.round;turn=null;pendingDeaths=[];ambiguities+=pending.length;pending=[]}
   if(e.type==='turnStart'){turn=e.id;pendingDeaths=[];ambiguities+=pending.length;pending=[]}
   if(e.type==='dragonAction'){actions.set(e.id,e.action);pendingDeaths=[]}
   if(e.type==='dragonDeath'){
     const d=state.bodies.getById(e.id);
     const death={id:e.id,round,team:d?.team,length:d?.body.length??0,feed:feedSet.has(`${e.id}:${round}`),action:actions.get(e.id)?.kind,
       deposited:0,recoveredSameTeam:0,recoveredOtherTeam:0,recoveredByFinalLongest:0};
     deaths.push(death);if(d)pendingDeaths.push({death,body:new Set(d.body.map(key))});
   }
   if(e.type==='tileChange'){
     const k=key(e.tile),was=state.map.tileAt(e.tile.x,e.tile.y)===state.pearl;
     if(e.hasPearl){
       const possible=pendingDeaths.filter(d=>d.body.has(k));
       if(!was&&possible.length===1){ledger.set(k,possible[0].death);possible[0].death.deposited++}
       else if(!was){ledger.delete(k);if(possible.length>1)ambiguousSources++}
       // A repeated 'true' does not create another physical pearl or reset its source.
     }else{
       if(was)pending.push({tile:k,source:ledger.get(k),round,turn});
       ledger.delete(k);
     }
   }
   if(e.type==='dragonUpdate'){
     const d=state.bodies.getById(e.id),matches=pending.filter(p=>p.tile===key(e.head)&&p.turn===e.id);
     for(const p of matches){
       const x={round,id:e.id,team:d?.team,sourceId:p.source?.id,sourceTeam:p.source?.team,feed:p.source?.feed??false};consumptions.push(x);
       if(p.source&&d){if(p.source.team===d.team)p.source.recoveredSameTeam++;else p.source.recoveredOtherTeam++}
     }
     pending=pending.filter(p=>!matches.includes(p));pendingDeaths=[];
   }
   if(e.type==='dragonSplit'){
     births.push({round,parent:e.parentId,child:e.childId,team:e.team,size:e.childBody.length,parentLength:e.parentBody.length,
       childHead:e.childBody[0],parentHead:e.parentBody[0],parentTail:e.parentBody[e.parentBody.length-1]});pendingDeaths=[];
   }
   state.map.applyEvent(e);state.bodies.applyEvent(e);
 }
 snapshot();ambiguities+=pending.length;
 const final={A:standing('A'),B:standing('B')};
 for(const team of ['A','B'])for(const k of ['dragonCount','longestDragon','totalLength'])if(final[team][k]!==r.result['team'+team][k])throw Error(`Replay reconstruction mismatch ${file} ${team} ${k}`);
 const longest={};for(const team of ['A','B'])longest[team]=new Set([...state.bodies.dragons.values()].filter(d=>d.team===team&&d.body.length===final[team].longestDragon).map(d=>d.id));
 const byId=new Map(deaths.map(d=>[d.id,d]));
 for(const c of consumptions)if(c.sourceId!==undefined&&c.team===c.sourceTeam&&longest[c.team].has(c.id))byId.get(c.sourceId).recoveredByFinalLongest++;
 const survivors=[...state.bodies.dragons.values()].map(d=>({id:d.id,team:d.team,length:d.body.length}));
 return {analysis_sha256:analysisHash,replay_sha256:crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),final,curves,births,deaths,survivors,
   consumption:{total:consumptions.length,fromDeath:consumptions.filter(c=>c.sourceId!==undefined).length,fromDesignatedFeed:consumptions.filter(c=>c.feed).length,unmatchedRemovals:ambiguities,ambiguousSources},
   attribution:'Only newly appearing pearls in a dying dragon body during its death event are attributed; later matched head entry records the eater. Direct collection by a final longest survivor is counted; inherited/re-donated resources are not traced.'};
}
if(require.main===module){
 const folder=process.argv[2];let files=fs.statSync(folder).isDirectory()?fs.readdirSync(folder).filter(x=>x.endsWith('.replay')).map(x=>path.join(folder,x)):[folder];
 for(const f of files){const out=f.replace(/\.replay$/,'.replay-metrics.json');if(fs.existsSync(out)&&JSON.parse(fs.readFileSync(out)).analysis_sha256===analysisHash)continue;const tp=f.replace(/\.replay$/,'.telemetry.json');if(!fs.existsSync(tp))continue;
   fs.writeFileSync(out,JSON.stringify(analyze(f,JSON.parse(fs.readFileSync(tp)))));console.log(path.basename(f));}
}
module.exports={analyze};
