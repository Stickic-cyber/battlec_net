"""Split-only causal features. Never mutate the frozen movement observation."""
from collections import deque
import numpy as np

TAIL_CHANNELS=29
CONTEXT_NAMES=('round','remaining','length','units_fraction','free_slots','map_width','map_height',
 'body_fraction','king_known','king_age','king_length','king_gap','head_king_eta','head_route_known',
 'ally_reports','ally_mean_length','ally_max_length','nearby_allies','enemy_heads','enemy_sonar',
 'rule_split','role_farmer','role_king','role_hunter','role_feeder','head_space','head_exits',
 'tail_known','tail_visible','tail_space','tail_exits','tail_route_known')
CANDIDATE_NAMES=('split','child_length','parent_length','child_fraction','rule_choice','minimum_child',
 'body_fraction','birth_known','birth_dx','birth_dy','birth_exits','birth_space','birth_pearls','birth_spawns',
 'unknown_boundary','head_space','head_exits','parent_space_ratio','king_eta','route_known','delivery_slack',
 'nearest_ally','nearest_enemy','cut_known','cut_dx','cut_dy','nearby_allies','birth_age',
 'heading_x','heading_y','local_known_fraction','free_slots')
SHAPES={'tail':(29,7,7),'context':(32,),'candidates':(6,32),'mask':(6,),'base_h':(256,)}


def topology(r,start,blocked=None,limit=128,target=None):
    """Known-only graph; unknown portals/walls never become invented connections."""
    if start is None:return {'area':0,'exits':0,'pearls':0,'spawns':0,'unknown':0,'eta':-1}
    blocked=blocked or set();seen={start};q=deque([(start,0)]);result=dict(area=0,exits=0,pearls=0,spawns=0,unknown=0,eta=-1)
    while q and result['area']<limit:
        (x,y),distance=q.popleft();wm=r.world_map.get((x,y))
        if wm is None:result['unknown']+=1;continue
        result['area']+=1;result['pearls']+=int(wm['has_pearl']);result['spawns']+=int(wm['spawns'])
        if target==(x,y):result['eta']=distance;break
        for d in r._DIRS:
            if d in wm['kelp']:continue
            if d in wm['portals']:
                dest=r.portal_dest_ready.get((x,y,d))
                if dest is None:result['unknown']+=1;continue
            else:
                dx,dy=d.get_offset();dest=((x+dx)%r.map_width,(y+dy)%r.map_height)
            if dest in blocked and dest!=start:continue
            if dest not in r.world_map:result['unknown']+=1;continue
            if distance==0:result['exits']+=1
            if dest not in seen:seen.add(dest);q.append((dest,distance+1))
    return result


def observe(r,observer,choice,base_h):
    ct=r.ct;pos=ct.get_position();head=(pos.x,pos.y);rnd=r.game.get_round_num();remaining=500-rnd
    n=ct.get_length();w,h=r.map_width,r.map_height;units=ct.get_unit_count();limit=r.game.get_unit_limit()
    body=list(r.own_body);fraction=min(1,len(body)/max(1,n));complete=len(body)==n
    tail=body[-1] if body else head;tail_known=bool(body)
    def rel(p):return (((p[0]-head[0]+w//2)%w-w//2)/max(1,w/2),((p[1]-head[1]+h//2)%h-h//2)/max(1,h/2))
    def dist(p,q):return min(abs(p[0]-q[0]),w-abs(p[0]-q[0]))+min(abs(p[1]-q[1]),h-abs(p[1]-q[1]))
    occupied=set(body);visible_self=set();enemies=[];allies=[]
    for tile in ct.get_tiles():
        p=tile.get_position();part=tile.get_dragon()
        if part is None:continue
        occupied.add((p.x,p.y))
        if part.get_team()==r.my_team:
            if part.get_id()==ct.get_id():visible_self.add((p.x,p.y))
            elif part.is_head():allies.append((p.x,p.y))
        elif part.is_head():enemies.append((p.x,p.y))
    reports=list(r.ally_states.values());allies=list(set(allies+[(s['x'],s['y']) for s in reports]))
    lengths=[s['len'] for s in reports];king=r.known_king_info
    king_known=king['x']>=0 and rnd-king['round']<=25
    king_pos=(king['x'],king['y']) if king_known else None
    hd=topology(r,head,occupied);td=topology(r,tail,occupied)
    head_eta=topology(r,head,limit=512,target=king_pos)['eta'] if king_known else -1
    tail_eta=topology(r,tail,limit=512,target=king_pos)['eta'] if king_known else -1
    # New corrected occupancy lives only in this copy; the original 27-channel map stays untouched.
    ys=(tail[1]+np.arange(-3,4))%h;xs=(tail[0]+np.arange(-3,4))%w
    crop=np.zeros((29,7,7),np.float32);crop[:27]=observer.base[:,ys[:,None],xs[None,:]]
    body_set=set(body)
    for iy,y in enumerate(ys):
        for ix,x in enumerate(xs):
            if (int(x),int(y)) in visible_self:
                crop[27,iy,ix]=1;crop[28,iy,ix]=float((int(x),int(y)) not in body_set);crop[6,iy,ix]=1
    role=choice.role;echo=ct.get_sonar_echoes();head_near=sum(dist(head,a)<=6 for a in allies)
    context=[rnd/500,remaining/500,n/128,units/max(1,limit),(limit-units)/max(1,limit),w/64,h/64,
      fraction,float(king_known),min(100,max(0,rnd-king['round']))/100,min(king['len'],256)/128,
      np.tanh((king['len']-n)/32) if king_known else 0,max(0,head_eta)/64,float(head_eta>=0),
      len(reports)/32,np.mean(lengths)/128 if lengths else 0,max(lengths,default=0)/128,head_near/8,
      len(enemies)/16,echo.enemy_head/64,float(choice.recommended>0),
      *[float(role==k) for k in ('FARMER','KING','HUNTER','FEEDER')],hd['area']/128,hd['exits']/4,
      float(tail_known),float(observer.base[1,tail[1],tail[0]]),td['area']/128,td['exits']/4,float(tail_eta>=0)]
    mask=np.asarray([True]+choice.size_mask,bool);rows=np.zeros((6,32),np.float32)
    for i in range(6):
        if not mask[i]:continue
        size=choice.sizes[i-1] if i else 0;birth=tail if i else head;space=td if i else hd;eta=tail_eta if i else head_eta
        # Route ETA is a known static-topology estimate. Collection/wait buffer is explicitly heuristic.
        eta_est=eta if eta>=0 else dist(birth,king_pos) if king_known else (w+h)//2
        slack=remaining-eta_est-8-(size+1)//2
        cut_known=bool(i and complete and 0<=n-size-1<len(body));cut=body[n-size-1] if cut_known else head
        bx,by=rel(birth);cx,cy=rel(cut);dx=dy=0
        if i and len(body)>=2:
            px,py=body[-2];rx=(birth[0]-px+w//2)%w-w//2;ry=(birth[1]-py+h//2)%h-h//2
            if abs(rx)+abs(ry)==1:dx,dy=rx,ry
        else:dx,dy=ct.get_dir().get_offset()
        rows[i]=[float(i>0),size/128,(n-size)/128,size/max(1,n),float((not i and not choice.recommended) or (i and size==choice.recommended)),float(size==2),
          fraction,float(tail_known if i else True),bx,by,space['exits']/4,space['area']/128,space['pearls']/32,space['spawns']/32,
          min(space['unknown'],128)/128,hd['area']/128,hd['exits']/4,min(4,hd['area']/max(1,n-size))/4,
          min(128,eta_est)/64,float(eta>=0),np.clip(slack/500,-1,1),min((dist(birth,p) for p in allies),default=32)/32,
          min((dist(birth,p) for p in enemies),default=16)/16,float(cut_known),cx,cy,sum(dist(birth,a)<=6 for a in allies)/8,
          observer.base[2,birth[1],birth[0]],dx,dy,float(crop[0].mean()) if i else float(observer.base[0,head[1],head[0]]),
          (limit-units)/max(1,limit)]
    assert len(context)==32
    return {'tail':crop.astype(np.float16),'context':np.asarray(context,np.float16),'candidates':rows.astype(np.float16),
        'mask':mask,'base_h':np.asarray(base_h,np.float32)}


def aggressive_choice(features,choice,r):
    c=features['context'];rows=features['candidates'];rnd=float(c[0])*500
    target=min(24,max(6,int(float(c[5])*64*float(c[6])*64)//45))
    if rnd>=280 or c[3]>=.75 or r.ct.get_unit_count()>=target:return choice.teacher-3 if choice.teacher>=4 else 0
    candidates=[]
    for i in range(1,6):
        row=rows[i]
        if not features['mask'][i] or row[7]<.5 or row[10]<.25 or row[11]*128<8 or row[20]<.15:continue
        if row[2]*128<3 or row[17]<.25:continue
        candidates.append((abs(float(row[1])*128-3),i))
    return min(candidates)[1] if candidates else (choice.teacher-3 if choice.teacher>=4 else 0)
