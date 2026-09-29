from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'rl_v5'))
import runtime
from helper import Direction
from types import SimpleNamespace as NS
from feeding_v2 import install

class TestDeliveryRole(unittest.TestCase):
    def rules(self,rnd=446,length=28,age=0):
        world={(x,y):dict(kelp=set(),portals={},has_pearl=False,spawns=False) for x in range(4) for y in range(4)}
        return NS(determine_role=lambda *_:'KING',check_endgame_sacrifice_feed=lambda *_:(None,None),
            choose_best_move_via_bfs=lambda *_:None,broadcast_sonar=lambda *_:None,
            ct=NS(get_length=lambda:length,get_position=lambda:NS(x=0,y=0),make_move=lambda *_:None),
            game=NS(get_round_num=lambda:rnd),known_king_info=dict(x=1,y=1,len=30,round=rnd-age,id=7),
            world_map=world,map_width=4,map_height=4,portal_dest_ready={},_DIRS=Direction.get_direction_list())
    def test_larger_recipient_in_delivery_window(self):
        r=self.rules();install(r,'dynamic_v2')
        self.assertEqual(r.determine_role([],{}),'FEEDER');self.assertEqual(r._dynamic_home,(1,1))
        self.assertEqual(r._dynamic_audit['local_king_reassigned'],1)
    def test_current_dynamic_does_not_reassign_local_king(self):
        r=self.rules();install(r,'dynamic');self.assertEqual(r.determine_role([],{}),'KING');self.assertIsNone(r._dynamic_home)
    def test_stale_larger_self_early_or_unreachable_stay_king(self):
        for r in (self.rules(age=9),self.rules(length=31),self.rules(rnd=300),self.rules(rnd=400)):
            install(r,'dynamic_v2');self.assertEqual(r.determine_role([],{}),'KING')
        r=self.rules();r.world_map[(0,0)]['kelp']=set(r._DIRS);install(r,'dynamic_v2');self.assertEqual(r.determine_role([],{}),'KING')

if __name__=='__main__':unittest.main(verbosity=2)
