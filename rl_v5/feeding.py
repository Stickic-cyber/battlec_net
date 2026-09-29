"""Optional rules-only logistics variant, evaluated separately after split selection."""
from features import topology
from collections import Counter

def install(r,variant):
    if variant!='dynamic':raise ValueError(variant)
    r._dynamic_audit=Counter()
    role_original=r.determine_role;feed_original=r.check_endgame_sacrifice_feed;move_original=r.choose_best_move_via_bfs
    broadcast_original=r.broadcast_sonar;make_move=r.ct.make_move
    def role(enemies,visible):
        value=role_original(enemies,visible);r._dynamic_home=None
        k=r.known_king_info;rnd=r.game.get_round_num();n=r.ct.get_length();p=r.ct.get_position()
        if value!='KING' and rnd>=360 and k['x']>=0 and rnd-k['round']<=8 and k['len']>n:
            target=(k['x'],k['y']);eta=topology(r,(p.x,p.y),limit=512,target=target)['eta']
            if eta>=0 and 500-rnd<=eta+min(40,n+8)+25:
                r._dynamic_home=target;r._dynamic_audit['dynamic_home_turns']+=1;return 'FEEDER'
        return value
    def feed(rnd,role,visible,enemies):
        original=feed_original(rnd,role,visible,enemies)
        if original[0] is not None:return original
        target=getattr(r,'_dynamic_home',None)
        if target is None or enemies or r.ct.get_unit_count()<2:return None,None
        p=r.ct.get_position();here=(p.x,p.y);mylen=r.ct.get_length()
        for pos,part in visible.items():
            if part.get_team()!=r.my_team or part.get_id()==r.ct.get_id() or not part.is_head():continue
            st=r.ally_states.get(part.get_id())
            if not st or rnd-st['round']>2 or st['len']<=mylen:continue
            # Receipt is an existing TARGET packet explicitly aimed at the donor's tile.
            ack=st['itype']==r.ITYPE_TARGET and (st['tx'],st['ty'])==here
            adjacent=any(r.step_tile_no_blind_portal(p.x,p.y,d)==pos for d in r._DIRS)
            if ack and adjacent and 500-rnd>=3:
                r._dynamic_audit['dynamic_ack_detonations']+=1;return 'DETONATE_NOW',part.get_id()
        return 'APPROACH_AND_PING',target
    def move(viable,visible,release,role,rnd,approach=None):
        return move_original(viable,visible,release,role,rnd,approach or getattr(r,'_dynamic_home',None))
    def make(direction):
        p=r.ct.get_position();r._dynamic_next_destination=r.step_tile_no_blind_portal(p.x,p.y,direction)
        return make_move(direction)
    def broadcast(rnd,role,enemies,target,feed_pos=None):
        if role=='FEEDER' and getattr(r,'_dynamic_home',None) is not None and r._dynamic_next_destination is not None:feed_pos=r._dynamic_next_destination
        if role=='KING' and any(target==(f['x'],f['y']) and rnd-f['round']<=2 for f in r.feed_beacons) and not enemies:
            r._dynamic_audit['dynamic_king_ack_messages']+=1
            p=r.ct.get_position();packet=r.encode_sonar_packet(r.ct.get_id(),p.x,p.y,r.ct.get_length(),*target,r.ITYPE_TARGET,rnd)
            for d in r._DIRS:r.ct.send_sonar(d,packet)
        else:broadcast_original(rnd,role,enemies,target,feed_pos)
    r.determine_role=role;r.check_endgame_sacrifice_feed=feed;r.choose_best_move_via_bfs=move
    r.broadcast_sonar=broadcast;r.ct.make_move=make
