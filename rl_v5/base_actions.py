"""Only strategic splitting is newly controlled. Keep rule fallback ordering."""
from dataclasses import dataclass
from hybrid import Decision,features,commit as move_commit
import numpy as np

SPLIT_COUNT=5
SPLIT_FEATURES=12
CONTEXT_DIM=16

@dataclass
class Choice:
    movement: object
    sizes: list
    size_mask: list
    recommended: int
    role: str
    enemies: list
    context: object
    candidates: object
    @property
    def move_active(self):return self.movement is not None and sum(self.movement.mask)>1
    @property
    def active(self):return self.move_active or any(self.size_mask)
    @property
    def teacher(self):
        if self.recommended and self.recommended in self.sizes:return 4+self.sizes.index(self.recommended)
        return self.movement.teacher if self.move_active else 0

def candidates(r,recommended,role,enemies,moves,protected=False):
    n=r.ct.get_length();minimum=r.Constants.MIN_SIZE
    proposed=[recommended if recommended else n//2,minimum,max(minimum,n//4),n//2,n-minimum]
    sizes=[];mask=[];seen=set()
    for size in proposed:
        allowed=not protected and size not in seen and r.ct.can_split(size)
        sizes.append(size if allowed else 0);mask.append(allowed)
        if allowed:seen.add(size)
    pos=r.ct.get_position();w,h=r.map_width,r.map_height
    def rel(x,y):return (((x-pos.x+w//2)%w-w//2)/max(1,w/2),((y-pos.y+h//2)%h-h//2)/max(1,h/2))
    body=r.own_body;complete=len(body)==n
    rows=np.zeros((SPLIT_COUNT,SPLIT_FEATURES),np.float32)
    for i,(size,allowed) in enumerate(zip(sizes,mask)):
        if not allowed:continue
        # Body-derived positions are estimates, indicated separately. No engine state.
        tail_known=bool(body);tx,ty=rel(*body[-1]) if tail_known else (0,0)
        cut_known=complete and 0<=n-size-1<len(body)
        cx,cy=rel(*body[n-size-1]) if cut_known else (0,0)
        rows[i]=[size/128,(n-size)/128,size/max(1,n),float(size==recommended),
            tx,ty,float(tail_known),cx,cy,float(cut_known),len(body)/max(1,n),float(size==minimum)]
    limit=r.game.get_unit_limit();echo=r.ct.get_sonar_echoes()
    ctx=[float(any(mask)),float(recommended>0),recommended/max(1,n),n/128,r.ct.get_unit_count()/max(1,limit),
        (limit-r.ct.get_unit_count())/max(1,limit),r.game.get_round_num()/499,
        *[float(role==x) for x in ('FARMER','KING','HUNTER','FEEDER')],
        min(1,len(body)/max(1,n)),len(enemies)/16,echo.enemy_head/64,len(moves)/4,
        max((m['area'] for m in moves),default=0)/48]
    assert len(ctx)==CONTEXT_DIM
    return sizes,mask,np.asarray(ctx,np.float32),rows

def prepare(r):
    if not hasattr(r,'_clear_graph_cache'):
        from optimizations import install_graph_cache
        r._clear_graph_cache=install_graph_cache(r)
    r._clear_graph_cache();r.turn_start_clock=r.time.perf_counter()
    rnd=r.game.get_round_num();pos=r.ct.get_position()
    visible,enemies,enemy_any,release,own=r.update_perception(rnd);r.process_sonar(rnd)
    role=r.determine_role(enemies,visible)
    feed,feed_data=r.check_endgame_sacrifice_feed(rnd,role,visible,enemy_any)
    if feed=='DETONATE_NOW':r.ct.set_indicator_string('RULE_FEED');return None
    moves,viable=r.evaluate_moves(visible,release,own,role)
    split=r.check_split_decision(role,moves,viable,enemies)
    recommended=split if split>=r.Constants.MIN_SIZE and r.ct.can_split(split) else 0
    # Preserve emergency and active feeding branches, including their original split priority.
    protected=not viable or feed is not None
    if protected and recommended:
        r.ct.do_split(recommended);r.broadcast_sonar(rnd,role,enemies,(pos.x,pos.y));return None
    if not moves:r._make_trapped_sacrifice_move(pos,visible);return None
    if not viable:
        chosen=moves[0];r.ct.make_move(chosen['dir']);r.broadcast_sonar(rnd,role,enemies,chosen['dest']);return None
    approach=feed_data if feed=='APPROACH_AND_PING' else None
    direction,target=r.choose_best_move_via_bfs(viable,visible,release,role,rnd,approach)
    if direction is None:
        chosen=viable[0]
        for move in viable:
            if move['dir']==r.ct.get_dir() and move['danger']<=viable[0]['danger']+5:chosen=move;break
        direction,target=chosen['dir'],chosen['dest']
    values,mask,by_dir=features(r,moves,role,visible,enemies,approach)
    teacher=r._DIRS.index(direction)
    if not mask[teacher]:raise RuntimeError('Rule selected masked movement')
    movement=Decision(values,mask,teacher,by_dir,role,enemies,target,feed=='APPROACH_AND_PING' and rnd>=470)
    sizes,sm,ctx,rows=candidates(r,recommended,role,enemies,viable,protected)
    if sum(mask)==1 and not any(sm):move_commit(r,movement,teacher,'FORCED');return None
    return Choice(movement,sizes,sm,recommended,role,enemies,ctx,rows)

def commit(r,choice,action):
    if action>=4:
        i=action-4
        if not 0<=i<SPLIT_COUNT or not choice.size_mask[i] or not r.ct.can_split(choice.sizes[i]):raise RuntimeError('Illegal split')
        size=choice.sizes[i];r.ct.do_split(size)
        pos=r.ct.get_position();r.broadcast_sonar(r.game.get_round_num(),choice.role,choice.enemies,(pos.x,pos.y))
        return 'network_split',size
    a=action if choice.move_active else choice.movement.teacher
    move_commit(r,choice.movement,a,'TEACHER_V4' if choice.move_active else 'FORCED')
    return ('network_move' if choice.move_active else 'rule_move'),0
