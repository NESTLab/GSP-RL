# utility/ Package

## Overview
1 file, 1 class. ZMQ binary message parsing for CoppeliaSim robotics simulation.

## zmq_utility.py -- ZMQ_Utility
- Parses binary messages from a CoppeliaSim (formerly V-REP) robotics simulator via ZMQ sockets
- Uses Python `struct` module to unpack C++ float/int binary data
- Not used by the core RL library -- only by external robotics integration code

### Message Types
| Method | Fields | Format | Returns |
|--------|--------|--------|---------|
| `get_params(msg)` | num_robots, num_obstacles, num_obs, num_actions, num_stats, alphabet_size, use_gate, distance_to_goal_normalization_factor | 8f | sets self.params dict |
| `parse_status(msg)` | exp_done, episode_done, reached_goal | 3B (unsigned char) | tuple of bools |
| `parse_obs(msg)` | 31 fields per robot (dist2goal, angle2goal, wheel speeds, proximity sensors) | 31f per robot | list of numpy arrays |
| `parse_rewards(msg)` | reward per robot | 1f per robot | list of numpy arrays |
| `parse_failures(msg)` | failure flag per robot | 1I per robot | list of numpy arrays |
| `parse_stats(msg)` | magnitude, angle, deltaX, deltaY per robot | 4f per robot | list of numpy arrays |
| `parse_robot_stats(msg)` | x,y,z pos + x,y,z deg per robot | 6f per robot | list of numpy arrays |
| `parse_obj_stats(msg)` | x,y,z pos + x,y,z deg + cyl_angle2goal | 7f | numpy array |
| `parse_obstacle_stats(msg)` | x,y per obstacle (dynamic count) | Nf | numpy array |
| `parse_gate_stats(msg)` | wall positions and lengths | 4f | numpy array |
| `serialize_actions(actions)` | lwheel, rwheel, failure per robot | 3f per robot | bytearray |

### Usage Pattern
```python
zmq_util = ZMQ_Utility()
zmq_util.get_params(param_msg)        # Call first to set robot/obstacle counts
zmq_util.set_obstacles_fields()        # Then set dynamic obstacle fields
obs = zmq_util.parse_obs(obs_msg)      # Then parse per-step messages
rewards = zmq_util.parse_rewards(reward_msg)
action_msg = zmq_util.serialize_actions(actions)
```
