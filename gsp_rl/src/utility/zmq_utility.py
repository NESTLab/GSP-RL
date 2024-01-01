from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np

class ZMQ_Utility:
    def __init__(self):
        self.PARAMS_FIELDS = ['num_robots','num_obstacles', 'num_obs','num_actions', 'num_stats', 'alphabet_size',
                               'use_gate', 'distance_to_goal_normalization_factor']
        self.PARAMS_FMT = '8f'
        self.EXPERIMENT_FIELDS = ['exp_done', 'episode_done', 'reached_goal']
        self.EXPERIMENT_FMT = '3B'
        self.OBS_FIELDS = ['robot_dist2goal', 'robot_angle2goal', 'robot_lwheel',
                           'robot_rwheel', 'cyl_dist2robot', 'cyl_angle2robot', 'cyl_dist2goal',
                           'ProxVal_0',  'ProxVal_1',  'ProxVal_2',  'ProxVal_3',
                           'ProxVal_4',  'ProxVal_5',  'ProxVal_6',  'ProxVal_7',
                           'ProxVal_8',  'ProxVal_9',  'ProxVal_10', 'ProxVal_11',
                           'ProxVal_12', 'ProxVal_13', 'ProxVal_14', 'ProxVal_15',
                           'ProxVal_16', 'ProxVal_17', 'ProxVal_18', 'ProxVal_19',
                           'ProxVal_20', 'ProxVal_21', 'ProxVal_22', 'ProxVal_23']
        self.OBS_FMT = '31f'
        self.FAILURE_FIELDS = ['failure']
        self.FAILURE_FMT = '1I'
        self.REWARDS_FIELDS = ['reward']
        self.REWARDS_FMT = '1f'
        self.STATS_FIELDS = ['magnitude', 'angle', 'deltaX', 'deltaY']
        self.STATS_FMT = '4f'
        self.ROBOT_STATS_FIELDS = ['x_pos', 'y_pos', 'z_pos', 'x_deg', 'y_deg', 'z_deg']
        self.ROBOT_STATS_FMT = '6f'
        self.OBJ_STATS_FIELDS = ['x_pos', 'y_pos', 'z_pos', 'x_deg', 'y_deg', 'z_deg', 'cyl_angle2goal']
        self.OBJ_STATS_FMT = '7f'
        self.GATE_STATS_FIELDS = ['neg_wall_1_x', 'neg_wall_1_length_y',
                                  'pos_wall_2_x', 'pos_wall_2_length_y']
        self.GATE_STATS_FMT = '4f'
        # Need to account for differing numbers of obstacles
        self.OBSTACLE_STATS_FIELDS = []


        self.ACTIONS_FIELDS = ['lwheel', 'rwheel', 'failure']
        self.ACTIONS_FMT = '3f'

        # Byte size of float in C++
        self.FLOAT_SIZE = 4
        # Byte size of int in C++
        self.INT_SIZE = 4


    def get_params(self, msg):
        self.params = self.parse_msg(msg, 'params', self.PARAMS_FIELDS, self.PARAMS_FMT)
        self.params['num_robots'] = int(self.params['num_robots'])
        self.params['num_obstacles'] = int(self.params['num_obstacles'])
        self.params['num_obs'] = int(self.params['num_obs'])
        self.params['num_actions'] = int(self.params['num_actions'])
        self.params['num_stats'] = int(self.params['num_stats'])
        self.params['alphabet_size'] = int(self.params['alphabet_size'])
        self.params['use_gate'] = int(self.params['use_gate'])

    def set_obstacles_fields(self):
        for i in range(self.params['num_obstacles']):
            self.OBSTACLE_STATS_FIELDS.append('obs_'+str(i)+'_x')
            self.OBSTACLE_STATS_FIELDS.append('obs_'+str(i)+'_y')
        self.OBSTACLE_STATS_FMT = str(self.params['num_obstacles']*2)+'f'

    #
    # Parse the fields from a message
    # Returns a dictionary
    #
    def parse_msg(self, msg, msgtype, fields, fmt):
        # Make an empty named tuple with the given fields
        Tx = namedtuple(msgtype, fields)
        # Fill the tuple with the contents of the message
        x = Tx._make(unpack(fmt, msg))
        # Return the tuple as a dictionary
        return x._asdict()

    #
    # Parse the experiment status
    # Returns a tuple (exp_done, episode_done, reached_goal)
    #
    def parse_status(self, msg):
        data = self.parse_msg(msg, 'status', self.EXPERIMENT_FIELDS, self.EXPERIMENT_FMT)
        exp_done = (data['exp_done'] == 1)
        episode_done = (data['episode_done'] == 1)
        reached_goal = (data['reached_goal'] == 1)
        return exp_done, episode_done, reached_goal

    #
    # Parse the observations
    # Returns a list of numpy arrays, one per robot
    #
    def parse_obs(self, msg):
        obs = []
        # For each robot
        for r in range(0, self.params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * self.params['num_obs'] * self.FLOAT_SIZE:(r+1) * self.params['num_obs'] * self.FLOAT_SIZE]
            # Parse the bytes into a dictionary
            data = self.parse_msg(m, 'obs', self.OBS_FIELDS, self.OBS_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.float32, count=len(data))
            # Append it to the observations
            obs.append(nparr)
        return obs

    def parse_failures(self, msg):
        failures = []
        for r in range(0, self.params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * self.INT_SIZE:(r+1)*self.INT_SIZE]
            # Parse the bytes into a dictionary
            data = self.parse_msg(m, 'failure', self.FAILURE_FIELDS, self.FAILURE_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.intc, count = len(data))
            # Append it to the rewards
            failures.append(nparr)
        return failures

    def parse_rewards(self, msg):
        rewards = []
        for r in range(0, self.params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * self.FLOAT_SIZE:(r+1)*self.FLOAT_SIZE]
            # Parse the bytes into a dictionary
            data = self.parse_msg(m, 'reward', self.REWARDS_FIELDS, self.REWARDS_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.float32, count = len(data))
            # Append it to the rewards
            rewards.append(nparr)
        return rewards

    def parse_stats(self, msg):
        stats = []
        for r in range(0, self.params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * self.params['num_stats'] * self.FLOAT_SIZE:(r+1) * self.params['num_stats'] * self.FLOAT_SIZE]
            # Parse the bytes into a dictionary
            data = self.parse_msg(m, 'stats', self.STATS_FIELDS, self.STATS_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.float32, count = len(data))
            # Append it to the stats array
            stats.append(nparr)
        return stats
    
    def parse_robot_stats(self, msg):
        robot_stats = []
        for r in range(0, self.params['num_robots']):
            m = msg[r *len(self.ROBOT_STATS_FIELDS)* self.FLOAT_SIZE:(r+1) *len(self.ROBOT_STATS_FIELDS)* self.FLOAT_SIZE] 
            # Parse the bytes into a dictionary
            data = self.parse_msg(m, 'robot_stats', self.ROBOT_STATS_FIELDS, self.ROBOT_STATS_FMT)
            # Make a numpy array
            robot_stats.append(np.fromiter(data.values(), dtype=np.float32, count = len(data)))
        return robot_stats

    def parse_obj_stats(self, msg):
        # Parse the bytes into a dictionary
        data = self.parse_msg(msg, 'obj_stats', self.OBJ_STATS_FIELDS, self.OBJ_STATS_FMT)
        # Make a numpy array
        obj_stats = np.fromiter(data.values(), dtype=np.float32, count = len(data))
        return obj_stats

    def parse_obstacle_stats(self, msg):
        # Parse the bytes into a dictionary
        data = self.parse_msg(msg, 'obstacle_stats', self.OBSTACLE_STATS_FIELDS, self.OBSTACLE_STATS_FMT)
        # Make a numpy array
        obj_stats = np.fromiter(data.values(), dtype=np.float32, count = len(data))
        return obj_stats

    def parse_gate_stats(self, msg):
        # Parse the bytes into a dictionary
        data = self.parse_msg(msg, 'gate_stats', self.GATE_STATS_FIELDS, self.GATE_STATS_FMT)
        # Make a numpy array
        obj_stats = np.fromiter(data.values(), dtype=np.float32, count = len(data))
        return obj_stats


    def serialize_actions(self, actions):
        packer = Struct(self.ACTIONS_FMT)
        msg = bytearray(self.FLOAT_SIZE * self.params['num_actions'] * self.params['num_robots'])
        # For each robot
        for r in range(0, self.params['num_robots']):
            offset = self.FLOAT_SIZE * self.params['num_actions'] * r
            packer.pack_into(msg, offset, *(actions[r]))
        return msg