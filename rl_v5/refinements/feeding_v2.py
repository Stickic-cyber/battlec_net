"""Delivery-window role correction for a local KING that knows a larger recipient.

The original early-game KING role also includes any dragon of length >=14.
That is not proof it is the team's largest dragon. Preserve original behavior
except when a fresh, larger known recipient and a known delivery route exist.
"""
from feeding import install as install_base
from features import topology

def install(r,variant):
    if variant=='dynamic':return install_base(r,variant)
    if variant!='dynamic_v2':raise ValueError(variant)
    original=r.determine_role
    def role(enemies,visible):
        value=original(enemies,visible);k=r.known_king_info;rnd=r.game.get_round_num();n=r.ct.get_length()
        if value=='KING' and rnd>=360 and k['x']>=0 and rnd-k['round']<=8 and k['len']>n:
            p=r.ct.get_position();eta=topology(r,(p.x,p.y),limit=512,target=(k['x'],k['y']))['eta']
            if eta>=0 and 500-rnd<=eta+min(40,n+8)+25:
                r._dynamic_audit['local_king_reassigned']+=1
                return 'FEEDER'
        return value
    r.determine_role=role
    install_base(r,'dynamic')
