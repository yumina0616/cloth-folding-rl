"""Offline checks for the updated dataset loader. Run with pytest from project root."""
import json
import numpy as np
import pytest

from scripts.learning.demonstration_dataset import load_episode, split_episodes


def make_episode(tmp_path, name="ep", overlap=False):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "context.json").write_text(json.dumps({"grasp_mode":"physical", "pairs":[[0,1],[2,3]],
        "pick_xy":"measured TCP at manual close", "place_xy":"measured TCP immediately before opening"}))
    (folder / "metadata.json").write_text(json.dumps({"success":True}))
    corners = np.array([[.3,-.1,.01],[.3,.1,.01],[.6,-.1,.01],[.6,.1,.01]],dtype=np.float32)
    records = [{"observation":{"corners":corners.tolist()},"action":{"pick":[.3,-.1],"place":[.3,.1]},
                "low_level_start":0,"low_level_end":3},
               {"observation":{"corners":corners.tolist()},"action":{"pick":[.6,-.1],"place":[.6,.1]},
                "low_level_start":2 if overlap else 3,"low_level_end":6}]
    (folder / "high_level.jsonl").write_text("\n".join(map(json.dumps,records)))
    phases=np.array(["carry","carry","physical_release","carry","carry","physical_release"])
    actions=np.tile(np.array([.2,.1,.3,1.,0.,0.,0.,-1.],dtype=np.float32),(6,1))
    actions[:,0]=np.arange(6)/10+.2
    actions[2,7]=actions[5,7]=1
    np.savez(folder / "low_level.npz",time_s=np.arange(6,dtype=float),phase=phases,action=actions,
             corners_w=np.tile(corners,(6,1,1)),tcp_w=np.tile(np.array([.3,.1,.2]),(6,1)),
             grasping=np.array([True,True,True,True,True,True]))
    return folder


def test_shapes_and_time_alignment(tmp_path):
    ep=load_episode(make_episode(tmp_path))
    assert ep.observations.shape == (2,30)
    assert ep.actions.shape == (2,2)
    assert ep.low_observations.shape == (4,28)
    assert ep.low_actions.shape == (4,8)
    # sample 0: state from t=0, target from t=1
    assert ep.low_observations[0,15] == pytest.approx(.2)
    assert ep.low_actions[0,0] == pytest.approx(.3)
    # last release is retained
    assert ep.low_actions[-1,7] == 1


def test_reject_overlapping_segments(tmp_path):
    with pytest.raises(ValueError,match="overlapping"):
        load_episode(make_episode(tmp_path,overlap=True))


def test_episode_split(tmp_path):
    episodes=[load_episode(make_episode(tmp_path, f"ep{i}")) for i in range(3)]
    train,val=split_episodes(episodes,seed=5)
    assert {e.path for e in train}.isdisjoint({e.path for e in val})
