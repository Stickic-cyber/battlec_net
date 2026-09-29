"""Causal per-dragon spatial, entity and event memory. No engine/private state."""
from runtime import ROOT
from collections import deque
import numpy as np

VERSION=4
GRID_NAMES=('known','visible','observation_age','pearl_last_seen','nominal_pearl_eta','spawn_history',
    'self_body','self_body_order','estimated_release','estimated_tail','ally_body_visible','enemy_body_visible',
    'ally_head_visible','enemy_head_visible','part_direction_x','part_direction_y',
    'kelp_N','kelp_E','kelp_S','kelp_W','portal_N','portal_E','portal_S','portal_W',
    'visits','since_visit','previous_target')
COARSE_NAMES=('known_fraction','visible_fraction','mean_known_age','pearl_history_density','spawn_history_density',
    'self_body_density','ally_visible_density','enemy_visible_density','portal_density','visit_density')
ENTITY_TYPES=('null','ally','enemy','king','feed','portal','target','resource')
ENTITY_FIELDS=('dx','dy','length_or_lower_bound','length_reported','age','visible','dir_x','dir_y',
    'id_order','id_known','target_dx','target_dy','target_known','message_type','report_source',
    'exit_dx','exit_dy','exit_known','distance','visible_segments','motion_dx','motion_dy','motion_gap','motion_known')
ENTITY_DIM=len(ENTITY_TYPES)+len(ENTITY_FIELDS)
ENTITY_LIMIT=40
BASE_GLOBAL=('round','length','units','width','height','head_x','head_y','body_known_fraction','map_known_fraction',
    'echo_kelp','echo_ally','echo_ally_head','echo_enemy','echo_enemy_head','unparsed_sonar',
    'previous_length_delta','previous_dx','previous_dy','delta_rounds','age_since_birth','target_age',
    'target_dx','target_dy','target_known','enemy_report_count','ally_report_count','feed_count',
    'repeat_visits','visible_pearl_count','visible_enemy_heads','visible_ally_heads','known_king_age',
    'known_king_length','known_king_present','recent_position_repetitions')
GLOBAL_NAMES=BASE_GLOBAL+tuple('role_'+x for x in ('FARMER','KING','HUNTER','FEEDER'))+tuple('heading_'+x for x in 'NESW')+tuple('id_mod4_'+str(i) for i in range(4))+tuple(f'action_lag_{t}_{a}' for t in range(1,9) for a in ('N','E','S','W','split','none'))
OBS_SHAPES={'grid':(len(GRID_NAMES),15,15),'local':(len(GRID_NAMES),7,7),'coarse':(len(COARSE_NAMES),8,8),
    'entities':(ENTITY_LIMIT,ENTITY_DIM),'entity_mask':(ENTITY_LIMIT,),
    'global':(len(GLOBAL_NAMES),),'legacy':(4,54),'mask':(4,),'action_mask':(9,),'split_context':(16,),'split_candidates':(5,12)}

def wrapped_delta(a,b,size):return ((a-b+size//2)%size)-size//2

class Observer:
    def __init__(self):
        self.base=None
        self.previous_position=None
        self.previous_length=None
        self.previous_round=None
        self.born=None
        self.target=None
        self.target_since=0
        self.actions=deque(maxlen=8)
        self.positions=deque(maxlen=16)
        self.enemy_tracks={}

    def _base_observe(self,r,decision):
        rnd=r.game.get_round_num();ct=r.ct;pos=ct.get_position();hx,hy=pos.x,pos.y
        width,height=r.map_width,r.map_height
        if self.base is None:
            self.base=np.zeros((len(GRID_NAMES),height,width),np.float32)
            self.seen=np.full((height,width),-1,np.int32)
            self.pearl_eta=np.full((height,width),-1,np.int16)
            self.visits=np.zeros((height,width),np.int16)
            self.visited_at=np.full((height,width),-1,np.int32)
            self.born=rnd
            yy,xx=np.indices((height,width))
            self.bin_id=((yy*8//height)*8+(xx*8//width)).ravel()
            self.bin_count=np.bincount(self.bin_id,minlength=64).clip(1)
        g=self.base
        g[1].fill(0);g[6:16].fill(0);g[26].fill(0)
        visible={};heads=[];counts={}
        for tile in ct.get_tiles():
            p=tile.get_position();x,y=p.x,p.y;wm=r.world_map[(x,y)]
            g[0,y,x]=g[1,y,x]=1
            self.seen[y,x]=rnd;self.pearl_eta[y,x]=wm['pearl_time']
            g[3,y,x]=wm['has_pearl'];g[5,y,x]=wm['spawns']
            for dindex,d in enumerate(r._DIRS):
                g[16+dindex,y,x]=d in wm['kelp'];g[20+dindex,y,x]=d in wm['portals']
            part=tile.get_dragon()
            if part is not None:
                visible[(x,y)]=part
                key=(part.get_team(),part.get_id());counts[key]=counts.get(key,0)+1
                dx,dy=part.get_dir().get_offset();g[14,y,x]=dx;g[15,y,x]=dy
                if part.get_team()==r.my_team and part.get_id()!=ct.get_id():
                    g[10,y,x]=1;g[12,y,x]=part.is_head()
                elif part.get_team()!=r.my_team:g[11,y,x]=1;g[13,y,x]=part.is_head()
                if part.is_head() and part.get_id()!=ct.get_id():heads.append((x,y,part))
        known=self.seen>=0
        age=np.maximum(0,rnd-self.seen)
        g[2]=np.where(known,np.minimum(age,100)/100,0)
        g[4]=np.where(known&(self.pearl_eta>=0),np.clip(self.pearl_eta-age,0,100)/100,0)
        for i,(x,y) in enumerate(r.own_body):
            g[6,y,x]=1;g[7,y,x]=i/max(1,len(r.own_body)-1)
            g[8,y,x]=min(len(r.own_body)-i,128)/128;g[9,y,x]=i==len(r.own_body)-1
        self.visits[hy,hx]=min(32760,int(self.visits[hy,hx])+1);self.visited_at[hy,hx]=rnd
        g[24]=np.minimum(self.visits,8)/8
        g[25]=np.where(self.visited_at>=0,np.minimum(rnd-self.visited_at,100)/100,1)
        if r.last_target!=self.target:self.target=r.last_target;self.target_since=rnd
        if self.target is not None:g[26,self.target[1],self.target[0]]=1
        ys=(hy+np.arange(-7,8))%height;xs=(hx+np.arange(-7,8))%width
        crop=g[:,ys[:,None],xs[None,:]]
        coarse_channels=[g[0],g[1],g[2],g[3],g[5],g[6],g[10],g[11],g[20:24].max(0),g[24]]
        coarse=np.stack([np.bincount(self.bin_id,weights=a.ravel(),minlength=64)/self.bin_count for a in coarse_channels]).reshape(10,8,8)
        # Entity records are observations/reports, not omniscient current facts.
        def rel(x,y):return (wrapped_delta(x,hx,width)/max(1,width/2),wrapped_delta(y,hy,height)/max(1,height/2))
        def entity(kind,x=hx,y=hy,**kw):
            row=np.zeros(ENTITY_DIM,np.float32);row[ENTITY_TYPES.index(kind)]=1
            dx,dy=rel(x,y);values={'dx':dx,'dy':dy,'distance':(abs(dx)+abs(dy))/2,**kw}
            for k,v in values.items():row[len(ENTITY_TYPES)+ENTITY_FIELDS.index(k)]=v
            return row
        head_by_id={(p.get_team(),p.get_id()):(x,y,p) for x,y,p in heads}
        allies=[]
        ids=set(r.ally_states)|{p.get_id() for x,y,p in heads if p.get_team()==r.my_team}
        for aid in ids:
            if aid==ct.get_id():continue
            st=r.ally_states.get(aid);vis=head_by_id.get((r.my_team,aid))
            if vis:x,y,part=vis;dx,dy=part.get_dir().get_offset()
            elif st:x,y=st['x'],st['y'];dx=dy=0
            else:continue
            kw=dict(visible=float(vis is not None),dir_x=dx,dir_y=dy,id_order=np.sign(aid-ct.get_id()),id_known=1,
                visible_segments=counts.get((r.my_team,aid),0)/128,length_or_lower_bound=counts.get((r.my_team,aid),0)/128)
            if st:
                tx,ty=rel(st['tx'],st['ty'])
                kw.update(length_or_lower_bound=min(st['len'],256)/128,length_reported=1,age=min(rnd-st['round'],100)/100,
                    target_dx=tx,target_dy=ty,target_known=1,message_type=st['itype']/3,report_source=1)
            allies.append(entity('ally',x,y,**kw))
        allies.sort(key=lambda a:abs(a[8])+abs(a[9]))
        enemies=[];seen_enemy_ids=set()
        for x,y,p in heads:
            if p.get_team()==r.my_team:continue
            eid=p.get_id();seen_enemy_ids.add(eid);old=self.enemy_tracks.get(eid)
            dx,dy=p.get_dir().get_offset()
            kw=dict(visible=1,dir_x=dx,dir_y=dy,id_order=np.sign(eid-ct.get_id()),id_known=1,
                length_or_lower_bound=counts[(p.get_team(),eid)]/128,visible_segments=counts[(p.get_team(),eid)]/128)
            if old and rnd>old['round']:
                kw.update(motion_dx=wrapped_delta(x,old['x'],width)/max(1,width/2),motion_dy=wrapped_delta(y,old['y'],height)/max(1,height/2),motion_gap=min(rnd-old['round'],100)/100,motion_known=1)
            enemies.append(entity('enemy',x,y,**kw))
            self.enemy_tracks[eid]={'x':x,'y':y,'round':rnd,'dx':dx,'dy':dy}
        for eid,st in list(self.enemy_tracks.items()):
            if rnd-st['round']>8:del self.enemy_tracks[eid];continue
            if eid not in seen_enemy_ids:enemies.append(entity('enemy',st['x'],st['y'],age=(rnd-st['round'])/100,dir_x=st['dx'],dir_y=st['dy'],id_order=np.sign(eid-ct.get_id()),id_known=1))
        for st in r.enemy_reports:enemies.append(entity('enemy',st['x'],st['y'],age=(rnd-st['round'])/100,report_source=1))
        enemies.sort(key=lambda a:a[8+ENTITY_FIELDS.index('age')]+(abs(a[8])+abs(a[9]))*.01)
        rows=[entity('null')]+allies[:8]+enemies[:8]
        king=r.known_king_info
        if king['x']>=0:rows.append(entity('king',king['x'],king['y'],length_or_lower_bound=min(king['len'],256)/128,
            length_reported=1,age=min(100,rnd-king['round'])/100,id_order=np.sign(king['id']-ct.get_id()),id_known=king['id']>=0,report_source=1))
        for st in sorted(r.feed_beacons,key=lambda z:rnd-z['round'])[:4]:
            rows.append(entity('feed',st['x'],st['y'],age=(rnd-st['round'])/100,id_known=1,id_order=np.sign(st['id']-ct.get_id()),report_source=1))
        portals=[]
        for (x,y),wm in r.world_map.items():
            for d in wm['portals']:
                dest=r.portal_dest_ready.get((x,y,d));dx,dy=d.get_offset();kw=dict(dir_x=dx,dir_y=dy,age=min(rnd-wm['round'],100)/100)
                if dest:ex,ey=rel(*dest);kw.update(exit_dx=ex,exit_dy=ey,exit_known=1)
                portals.append(entity('portal',x,y,**kw))
        portals.sort(key=lambda a:abs(a[8])+abs(a[9]));rows+=portals[:8]
        if decision is not None:
            for i,move in sorted(decision.candidates.items()):
                if move.get('search_score',-1000)>-999:
                    x,y=move.get('search_target',move['dest']);wm=r.world_map.get((x,y))
                    rows.append(entity('target',x,y,age=min(rnd-wm['round'],100)/100 if wm else 1,
                        visible=float(ct.get_tile(r.Position(x,y)) is not None),message_type=i/3,target_known=wm is not None))
        resources=sorted(((abs(rel(x,y)[0])+abs(rel(x,y)[1]),x,y,wm) for (x,y),wm in r.world_map.items() if wm['spawns']))[:6]
        for _,x,y,wm in resources:rows.append(entity('resource',x,y,age=min(rnd-wm['round'],100)/100,visible=float(ct.get_tile(r.Position(x,y)) is not None)))
        rows=rows[:ENTITY_LIMIT];entities=np.zeros((ENTITY_LIMIT,ENTITY_DIM),np.float32);entities[:len(rows)]=rows
        entity_mask=np.arange(ENTITY_LIMIT)<len(rows)
        enemy_heads=[(x,y) for x,y,p in heads if p.get_team()!=r.my_team]
        role=decision.role if decision is not None else r.determine_role(enemy_heads,visible)
        px,py=self.previous_position or (hx,hy)
        dx,dy=rel(px,py)
        tx,ty=rel(*self.target) if self.target is not None else (0,0)
        echo=ct.get_sonar_echoes()
        basic=[rnd/499,min(ct.get_length(),256)/128,ct.get_unit_count()/64,width/64,height/64,hx/max(1,width-1),hy/max(1,height-1),
            min(1,len(r.own_body)/max(1,ct.get_length())),float(known.mean()),*[min(v,64)/64 for v in echo],float(r.enemy_pinged_me),
            (ct.get_length()-(self.previous_length if self.previous_length is not None else ct.get_length()))/32,-dx,-dy,
            (rnd-self.previous_round)/100 if self.previous_round is not None else 0,(rnd-self.born)/499,min(rnd-self.target_since,100)/100,
            tx,ty,float(self.target is not None),min(len(r.enemy_reports),32)/32,min(len(r.ally_states),64)/64,min(len(r.feed_beacons),16)/16,
            min(int(self.visits[hy,hx])-1,8)/8,sum(t.has_pearl() for t in ct.get_tiles())/49,len(enemy_heads)/16,
            sum(p.get_team()==r.my_team for x,y,p in heads)/16,min(100,max(0,rnd-king['round']))/100,min(king['len'],256)/128,float(king['x']>=0),
            sum(p==(hx,hy) for p in self.positions)/16]
        basic += [float(role==name) for name in ('FARMER','KING','HUNTER','FEEDER')]
        basic += [float(ct.get_dir()==d) for d in r._DIRS]+[float(ct.get_id()%4==i) for i in range(4)]
        for lag in range(1,9):basic += [float(len(self.actions)>=lag and self.actions[-lag]==i) for i in range(6)]
        assert len(basic)==len(GLOBAL_NAMES)
        self.previous_position=(hx,hy);self.previous_length=ct.get_length();self.previous_round=rnd;self.positions.append((hx,hy))
        obs={'grid':crop,'local':crop[:,4:11,4:11],'coarse':coarse,'entities':entities,'entity_mask':entity_mask,
            'global':np.asarray(basic),'legacy':np.asarray(decision.features) if decision is not None else np.zeros((4,54)),
            'mask':np.asarray(decision.mask) if decision is not None else np.ones(4,bool)}
        # Identical float16 quantization at collection/inference/training boundaries.
        return {k:np.asarray(v,dtype=bool if k in ('mask','entity_mask') else np.float32 if k=='legacy' else np.float16).copy() for k,v in obs.items()}

    def observe(self,r,choice):
        obs=self._base_observe(r,choice.movement if choice is not None else None)
        am=np.zeros(9,bool)
        if choice is None:
            am[0]=True;ctx=np.zeros(16,np.float16);candidates=np.zeros((5,12),np.float16)
        else:
            if choice.move_active:am[:4]=choice.movement.mask
            else:am[0]=True
            am[4:]=choice.size_mask;ctx=choice.context.astype(np.float16);candidates=choice.candidates.astype(np.float16)
        obs.update(action_mask=am,split_context=ctx,split_candidates=candidates)
        return obs

    def executed(self,response):
        action=5
        for line in response.decode().splitlines():
            if line.startswith('MOVE '):
                char=line.split()[1][-1]
                if char in 'NESW':action='NESW'.index(char)
            elif line.startswith('SPLIT '):action=4
        self.actions.append(action)
