from urllib.parse import uses_relative
from .agent import Agent
from gsp_rl.src.utility import ZMQ_Utility as zmq_utility
#from python_code.comms_viz import viz

import argparse
from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np
import math
import copy
import zmq
import csv
import os
import time
import torch as T
import matplotlib.pyplot as plt

Utility = zmq_utility.ZMQ_Utility()

# get path to containing folder so this works where ever it is used
containing_folder = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
parser.add_argument("--learning_scheme",default = 'None')
parser.add_argument("--test", default = False, action = "store_true")
parser.add_argument("--model_path")
parser.add_argument("--trained_num_robots")                                          # if we are testing a model trained on a different number of robots. This should be set to the training number of robots so that the network is built properly.
parser.add_argument("--no_print", default = False, action = "store_true")
parser.add_argument("--port", default = "55555")
parser.add_argument("--independent_learning", default = False, action = "store_true")
parser.add_argument("--global_knowledge", default = False, action = "store_true")   # append knowledge of other agents to the observation space
parser.add_argument("--intention", default=False, action = "store_true")
parser.add_argument("--recurrent", default= False, action= 'store_true')
parser.add_argument("--attention", default= False, action= 'store_true')
parser.add_argument("--gnn", default=False, action = 'store_true')
parser.add_argument("--neighbors", default=False, action = 'store_true')
parser.add_argument("--recurrent-rl", default=False, action = 'store_true')
parser.add_argument("--attention-rl", default=False, action="store_true")
parser.add_argument("--meta_param_size", default=0, type=int)
parser.add_argument("--share_prox_values", default=False, action = 'store_true')    # Robots will share their averaged prox values with eachother

args = parser.parse_args()

recording_path = os.path.join(containing_folder, args.recording_path)
if args.model_path is not None:
    model_file_path = os.path.join(containing_folder, args.model_path)
learning_scheme = args.learning_scheme
port = args.port
test_mode = args.test
train_mode = not test_mode
#
# Initialize zmq
#
# Create context
context = zmq.Context()
# create socket
socket = context.socket(zmq.REP)
# wait for connections on specified port, defaults to 55555
socket.bind("tcp://*:" + port)
print("Server Started")
# Get Parameters
Utility.get_params(socket.recv())
if not args.no_print:
    print("PARAMETERS:")
    print("  num_robots ----", Utility.params['num_robots'])
    print("  num_obstacles -", Utility.params['num_obstacles'])
    print("  num_obs -------", Utility.params['num_obs'])
    print("  alphabet_size -", Utility.params['alphabet_size'])
    print("  num_actions ---", Utility.params['num_actions'])
    print("  num_stats -----", Utility.params['num_stats'])

Utility.set_obstacles_fields()
# Path to save data
data_file_path = recording_path + '/Data/'

if args.share_prox_values:
    num_obs = Utility.params['num_obs'] +Utility.params['num_robots']   #need to account for num_robots extra observations
elif args.global_knowledge:
    num_obs = Utility.params['num_obs']+(Utility.params['num_robots']-1)*4  #need to account for the x and y positions and the x and y velocitis for each robot
else:
    num_obs = Utility.params['num_obs']

edge_index = np.array([
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
], dtype=np.int64)

agent_args = {'n_agents':Utility.params['num_robots'],
              'n_obs':num_obs, 
              'n_actions': Utility.params['num_actions']-1,
              'learning_scheme': args.learning_scheme,
              'options_per_action':3,
              'n_chars':Utility.params['alphabet_size'],
              'meta_param_size':1, 
              'use_intention':args.intention, 
              'use_recurrent':args.recurrent,
              'attention':args.attention,
              'gnn': args.gnn,
              'intention_neighbors':args.neighbors,
              'intention_look_back':2,
              'edge_index': edge_index,
              'prox_filter_angle':60}


if args.independent_learning:
    models = [Agent.Agent(id=i, **agent_args) for i in range(Utility.params['num_robots'])]
    if test_mode:
        [models[i].load_model(model_file_path) for i in range(Utility.params['num_robots'])]
else:
    if args.trained_num_robots is not None:
        agent_args['n_agents'] = int(args.trained_num_robots)
        model = Agent.Agent(id = 0, **agent_args)
    else:
        model = Agent.Agent(id = 0, **agent_args)
    if test_mode:
        model.load_model(model_file_path)


# Send acknowledgment
socket.send(b"ok")

#######################################################################
#                           MAIN LOOP
#######################################################################
exp_done = False
ep_counter = 0
exp_rewards = []
exp_mean_rewards = []
high_score = -np.inf
mean_axis = []
experiment_start_time = time.time()
Testing_Failures = 0
Testing_Successes = 0
var_grad = 0
gate = 0
gate_stats = 0
obstacles = 0
obstacle_stats = 0

while not exp_done:
    #receive initial observations
    msgs = socket.recv_multipart()
    exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
    data_file_name = 'Data_Episode_'+str(ep_counter)+'.csv'
    with open(data_file_path+data_file_name, 'a+') as output:
        writer = csv.writer(output, delimiter = ',')
        writer.writerow(['reward', 'epsilon', 'termination', 'loss', 'force magnitude', 'force angle', 'average force vector', 'cyl_x_pos', 'cyl_y_pos', 'cyl_angle', 'gate_stats', 'obstacle_stats', 'intention_reward', 'intention_heading', 'run_time', 'robots_x_pos', 'robots_y_pos', 'robot_angle', 'robot_failures', 'env_observations', 'agent_actions'])

        if not exp_done:
            time_steps = 0
            
            object_positions = []
            agent_prox_flags = []
            last_object_heading = None

            next_heading_intention = np.zeros(Utility.params['num_robots'])
            old_heading_intention = np.zeros(Utility.params['num_robots'])
            episode_intention_rewards = np.zeros(Utility.params['num_robots'])

            # Receive initial observations from the environment
            env_observations = Utility.parse_obs(msgs[1])
            failures = Utility.parse_failures(msgs[2])
            rewards = Utility.parse_rewards(msgs[3])
            stats = Utility.parse_stats(msgs[4])
            robot_stats = Utility.parse_robot_stats(msgs[5])
            obj_stats = Utility.parse_obj_stats(msgs[6])

            object_positions.append([obj_stats[0], obj_stats[1]])
            old_cyl_ang = obj_stats[5]

            #print('[ROBOT 0 DIST]: ', env_observations[0][4])

            if Utility.params['num_obstacles'] > 0:
                obstacle_stats = Utility.parse_obstacle_stats(msgs[7])
            elif Utility.params['use_gate'] == 1:
                gate_stats = Utility.parse_gate_stats(msgs[7])

            # Store the object stats in agent for learning later
            if args.independent_learning:
                for i in range(Utility.params['num_robots']):
                    [models[i].store_object_stats(obj_stats, time_steps>2) for i in range(Utility.params['num_robots'])]
                    [models[i].reset_intention_sequence()]
            else:
                model.store_object_stats(obj_stats, time_steps>2)
                model.reset_intention_sequence()

            agent_states = []
            force_mags = []
            force_angs = []
            if args.independent_learning:
                running_reward = []
            else:
                running_reward = 0
            
            for i in range(Utility.params['num_robots']):
                if failures[i][0]:
                    agent_prox_flags.append(0)
                else:
                    prox_values = env_observations[i][7:]
                    # Add logic to filter prox values that are observing the object
                    prox_values, filtered_indeces = model.filter_prox_values(prox_values, env_observations[i][5])
                    for j in range(len(filtered_indeces)):
                        env_observations[i][7+filtered_indeces[j]] = 0.0
                    prox_value = np.sum(prox_values)
                    agent_prox_flags.append(prox_value/float(len(filtered_indeces)))
            
            #Define Global Knowledge: [positions, velocities]
            global_knowledge=np.zeros((Utility.params['num_robots'])*4)
            for i in range(Utility.params['num_robots']):
                global_knowledge[i*4] = robot_stats[i][0]           #x position
                global_knowledge[i*4+1] = robot_stats[i][1]         #y position
                global_knowledge[i*4+2] = stats[i][2]               #velocity X
                global_knowledge[i*4+3] = stats[i][3]               #velocity Y

            for i in range(Utility.params['num_robots']):
                g_knowledge = np.zeros((Utility.params['num_robots']-1)*4)
                counter = 0
                for j in range(Utility.params['num_robots']):
                    if i != j:
                        g_knowledge[counter*4] = global_knowledge[j*4]
                        g_knowledge[counter*4+1] = global_knowledge[j*4+1]
                        g_knowledge[counter*4+2] = global_knowledge[j*4+2]
                        g_knowledge[counter*4+3] = global_knowledge[j*4+3]
                        counter+=1
                if args.independent_learning:
                    running_reward.append(0)
                    if args.intention:
                        if args.global_knowledge:
                            agent_state = models[i].make_agent_state(env_observations[i], heading_intention = next_heading_intention[i], global_knowledge=g_knowledge) 
                        else:
                            agent_state = models[i].make_agent_state(env_observations[i], heading_intention = next_heading_intention[i])
                    else:
                        if args.global_knowledge:
                            agent_state = models[i].make_agent_state(env_observations[i], global_knowledge = g_knowledge)
                        else:
                            agent_state = env_observations[i]
                        
                else:
                    if args.intention:
                        if args.global_knowledge:
                            agent_state = model.make_agent_state(env_observations[i], heading_intention=next_heading_intention[i], global_knowledge=g_knowledge)
                        else:
                            agent_state = model.make_agent_state(env_observations[i], heading_intention=next_heading_intention[i])
                    else: 
                        if args.share_prox_values:
                            agent_state = np.concatenate((env_observations[i], agent_prox_flags))
                        else:
                            if args.global_knowledge:
                                agent_state = model.make_agent_state(env_observations[i], global_knowledge=g_knowledge)
                            else:
                                agent_state = env_observations[i]
                agent_states.append(agent_state)
                force_mags.append(stats[i][0])
                force_angs.append(stats[i][1])

            # reward is the same across all agents. If it were per agent then this would need to move into the loop above
            if args.independent_learning:
                for i in range(Utility.params['num_robots']):
                    running_reward[i]+= rewards[i]
            else:
                running_reward += rewards[0]
            # failures should all be false because we havent started the episode yet
            failure = failures[0]

            #
            # Start the Episode Loop
            #

            episode_start_time = time.time()
            while not episode_done:
                if not exp_done:
                    reward = []
                    actions = []
                    actions_to_take = []
                    time_steps += 1
                    robot_failures = []

                    for i in range(Utility.params['num_robots']):
                        # Choose an action
                        if args.independent_learning:
                            action, action_num = models[i].choose_agent_action(agent_states[i], failures[i], test_mode)
                        else:
                            action, action_num = model.choose_agent_action(agent_states[i], failures[i], test_mode)
                        actions_to_take.append(action)
                        actions.append(action_num)

                    old_failures = failures[:]
                    # Take Step
                    socket.send(Utility.serialize_actions(actions_to_take))
                    msgs = socket.recv_multipart()

                    exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
                    env_observations = Utility.parse_obs(msgs[1])
                    failures = Utility.parse_failures(msgs[2])
                    #print('[DEBUG] Received Failures ', failures)
                    rewards = Utility.parse_rewards(msgs[3])
                    stats = Utility.parse_stats(msgs[4])
                    robot_stats = Utility.parse_robot_stats(msgs[5])
                    obj_stats = Utility.parse_obj_stats(msgs[6])
                    robot_x_pos = []
                    robot_y_pos = []
                    robot_angle = []
                    for i in range(Utility.params['num_robots']):
                        robot_x_pos.append(robot_stats[i][0])
                        robot_y_pos.append(robot_stats[i][1])
                        robot_angle.append(robot_stats[i][5])
                    if Utility.params['num_obstacles'] > 0:
                        obstacle_stats = Utility.parse_obstacle_stats(msgs[7])
                    elif Utility.params['use_gate'] == 1:
                        gate_stats = Utility.parse_gate_stats(msgs[7])

                    old_object_positions = copy.deepcopy(object_positions)
                    object_positions.append([obj_stats[0], obj_stats[1]])

                    intention_reward = []
                    label = 0
                    ############################## INTENTION REWARD ##############################################
                    if args.intention:
                        # shift to get between 0 and 2 Pi
                        old_cyl_ang += math.pi
                        new_cyl_ang = obj_stats[5] + math.pi
                        # check edge wrap at 0
                        if old_cyl_ang < math.pi and new_cyl_ang > math.pi:
                            diff = old_cyl_ang - new_cyl_ang + 2*math.pi
                        # check edge wrap at 2 pi
                        elif old_cyl_ang < math.pi and new_cyl_ang < math.pi:
                            diff = new_cyl_ang - old_cyl_ang + 2*math.pi
                        # otherwise we are not wrapping
                        else:
                            diff = old_cyl_ang - new_cyl_ang
                        label=diff
                        x1 = math.cos(diff)
                        y1 = math.sin(diff)                        
                        for i in range(Utility.params['num_robots']):
                                x2 = math.cos(next_heading_intention[i] * math.pi)
                                y2 = math.sin(next_heading_intention[i] * math.pi)

                                error = np.dot([x1, y1], [x2, y2])
                                intention_reward.append(-1 + error)
                                episode_intention_rewards[i] += intention_reward[i]

                    else:
                        intention_reward = [0 for i in range(Utility.params['num_robots'])]
                    
                    old_cyl_ang = obj_stats[5]

                    # store object stats for learning later
                    if args.independent_learning:
                        for i in range(Utility.params['num_robots']):
                            models[i].store_object_stats(obj_stats, time_steps>2)
                    else:
                        model.store_object_stats(obj_stats, time_steps>2)

                    # Store Transitions and Learn
                    old_agent_prox_flags = copy.deepcopy(agent_prox_flags)
                    neighbors_old_heading_intention = copy.deepcopy(old_heading_intention)
                    old_heading_intention = copy.deepcopy(next_heading_intention)

                    new_agent_states = []
                    force_mags = []
                    force_angs = []
                    r = []
                    agent_prox_flags = []
                    next_object_heading = np.zeros(Utility.params['num_robots'])
                    
                    for i in range(Utility.params['num_robots']):
                        robot_failures.append(failures[i][0])
                        if failures[i][0]:
                            agent_prox_flags.append(0)
                        else:
                            prox_values = env_observations[i][7:]
                            prox_values, filtered_indeces = model.filter_prox_values(prox_values, env_observations[i][5])
                            for j in range(len(filtered_indeces)):
                                env_observations[i][7+filtered_indeces[j]] = 0.0
                            prox_value = np.sum(prox_values)              
                            agent_prox_flags.append(prox_value/float(len(filtered_indeces)))

                    if args.intention:
                        if args.attention:
                            if args.independent_learning:
                                for i in range(Utility.params['num_robots']):
                                    # Check the syntax on object positions to make sure we are giving an X and Y
                                    next_object_heading[i] = models[i].choose_object_intention(agent_prox_flags, edge_index, test_mode)
                                    next_heading_intention[i] = next_object_heading[i]
                            else:
                                if model.intention_neighbors:
                                    agent_intention_states = model.build_intention_states(agent_prox_flags, old_heading_intention)
                                    ctde_intention = model.choose_object_intention(agent_intention_states, edge_index, test_mode)
                                else:
                                    ctde_intention = model.choose_object_intention(agent_prox_flags, edge_index, test_mode)
                                for i in range(Utility.params['num_robots']):
                                    next_heading_intention[i] = ctde_intention[i]

                        else:
                            if args.independent_learning:
                                for i in range(Utility.params['num_robots']):
                                    next_object_heading[i] = models[i].choose_object_intention(agent_prox_flags, edge_index, test_mode)
                                    next_heading_intention[i] = next_object_heading[i]
                            else:
                                if model.intention_neighbors:
                                    agent_intention_states = model.build_intention_states(agent_prox_flags, old_heading_intention)
                                    ctde_intention = model.choose_object_intention(agent_intention_states, edge_index, test_mode)
                                else:
                                    ctde_intention = model.choose_object_intention(agent_prox_flags, edge_index, test_mode)
                                for i in range(Utility.params['num_robots']):
                                    next_heading_intention[i] = ctde_intention[i]
                        #store transitions of intentions
                        if args.attention:
                            if args.independent_learning:
                                    for i in range(Utility.params['num_robots']):
                                        models[i].store_intention_transition(agent_prox_flags, label, 0, 0, 0)
                            else:
                                if args.neighbors:
                                    states = model.build_intention_states(old_agent_prox_flags, neighbors_old_heading_intention)
                                    new_states = model.build_intention_states(agent_prox_flags, old_heading_intention)
                                    for i in range(Utility.params['num_robots']):
                                        state = states[i]
                                        action = old_heading_intention[i]
                                        reward = intention_reward[i]
                                        new_state = new_states[i]
                                        model.store_intention_transition(state, action, reward, new_state, 0)
                                else:
                                    model.store_intention_transition(agent_prox_flags, label, 0, 0, 0)

                        if args.neighbors:
                            states = model.build_intention_states(old_agent_prox_flags, neighbors_old_heading_intention)
                            new_states = model.build_intention_states(agent_prox_flags, old_heading_intention)
                            for i in range(Utility.params['num_robots']):
                                state = states[i]
                                action = old_heading_intention[i]
                                reward = intention_reward[i]
                                new_state = new_states[i]
                                model.store_intention_transition(state, action, reward, new_state, 0)
                        else:
                            for i in range(Utility.params['num_robots']):
                                if args.independent_learning:
                                    state = np.array(old_agent_prox_flags)
                                    action = old_heading_intention[i]
                                    reward = intention_reward[i]
                                    new_state = np.array(agent_prox_flags)
                                    models[i].store_intention_transition(state, action, reward, new_state, 0)
                                else:
                                    state = np.array(old_agent_prox_flags)
                                    action = old_heading_intention[i]
                                    reward = intention_reward[i]
                                    new_state = np.array(agent_prox_flags)
                                model.store_intention_transition(state, action, reward, new_state, 0)


                    #Define Global Knowledge: [positions, velocities]
                    global_knowledge=np.zeros((Utility.params['num_robots'])*4)
                    for i in range(Utility.params['num_robots']):
                        global_knowledge[i*4] = robot_stats[i][0]           #x position
                        global_knowledge[i*4+1] = robot_stats[i][1]         #y position
                        global_knowledge[i*4+2] = stats[i][2]               #velocity X
                        global_knowledge[i*4+3] = stats[i][3]               #velocity Y


                    for i in range(Utility.params['num_robots']):
                        g_knowledge = np.zeros((Utility.params['num_robots']-1)*4)
                        counter = 0
                        for j in range(Utility.params['num_robots']):
                            if i != j:
                                g_knowledge[counter*4] = global_knowledge[j*4]
                                g_knowledge[counter*4+1] = global_knowledge[j*4+1]
                                g_knowledge[counter*4+2] = global_knowledge[j*4+2]
                                g_knowledge[counter*4+3] = global_knowledge[j*4+3]
                                counter+=1
                        #print('[DEBUG] ROBOT', i)
                        #reward = rewards[i]
                        #print('[DEBUG] OBS:', env_observations[i])
                        prox_values = env_observations[i][7:]
                        #print('[DEBUG] Prox Values', prox_values)
                        prox_value = np.sum(prox_values)
                        #print('[DEBUG] Prox Value Reward:', (-0.1)*prox_value)
                        #reward += (-1)*prox_value
                        rewards[i] += (-1)*prox_value
                        force_mags.append(stats[i][0])
                        force_angs.append(stats[i][1])

                        if args.independent_learning:
                            if args.intention:
                                if args.global_knowledge:
                                    new_agent_state = models[i].make_agent_state(env_observations[i], heading_intention = next_heading_intention[i], global_knowledge=g_knowledge) 
                                else:
                                    new_agent_state = models[i].make_agent_state(env_observations[i], heading_intention = next_heading_intention[i])
                            else:
                                if args.global_knowledge:
                                    new_agent_state = models[i].make_agent_state(env_observations[i], global_knowledge = g_knowledge)
                                else:
                                    new_agent_state = env_observations[i]
                                
                        else:
                            if args.intention:
                                if args.global_knowledge:
                                    new_agent_state = model.make_agent_state(env_observations[i], heading_intention=next_heading_intention[i], global_knowledge=g_knowledge)
                                else:
                                    new_agent_state = model.make_agent_state(env_observations[i], heading_intention=next_heading_intention[i])
                            else: 
                                if args.share_prox_values:
                                    new_agent_state = np.concatenate((env_observations[i], agent_prox_flags))
                                else:
                                    if args.global_knowledge:
                                        new_agent_state = model.make_agent_state(env_observations[i], global_knowledge=g_knowledge)
                                    else:
                                        new_agent_state = env_observations[i]

                        new_agent_states.append(new_agent_state)
                        if time_steps > 2:
                            if train_mode:
                                if learning_scheme != 'None':
                                    if not old_failures[i] and not failures[i]:
                                        if not episode_done:
                                            if args.independent_learning:
                                                models[i].store_agent_transition(agent_states[i],
                                                                    (actions[i], actions_to_take[i]),
                                                                    rewards[i],
                                                                    new_agent_states[i],
                                                                    episode_done)
                                            else:
                                                model.store_agent_transition(agent_states[i],
                                                                    (actions[i], actions_to_take[i]),
                                                                    rewards[i],
                                                                    new_agent_states[i],
                                                                    episode_done)
                                                    
                        r.append(rewards[i][0])
                    #print('[DEBUG] Rewards:', r)
                    #print('[DEBUG] Reward Average:', np.average(r))
                    #for i in range(Utility.params['num_robots']):
                    #    print('[DEBUG] robot %i memory:'%i, message_memory[i])

                    if train_mode and args.learning_scheme != 'None':
                        if args.independent_learning:
                            for i in range(Utility.params['num_robots']):
                                loss = models[i].learn()
                        else:
                            loss = model.learn(edge_index)
                    else:
                        loss = 0

                    if args.independent_learning:
                        for i in range(Utility.params['num_robots']):
                            running_reward[i] += r[i]
                    else:
                        running_reward += np.average(r)
                    # Store New Observations
                    agent_states = new_agent_states
                    actions = []

                    # Calculate average force vector
                    average_force_mag = None
                    average_force_ang = None
                    for i in range(Utility.params['num_robots']):
                        if average_force_mag is None:
                            average_force_mag = force_mags[i]
                            average_force_ang = force_angs[i]
                        else:
                            angle = abs(average_force_ang - force_angs[i])
                            #average_force_mag = math.sqrt(average_force_mag**2 + force_mags[i]**2 + 2*(average_force_mag)*(force_mags[i])*math.cos(math.radians(angle)))
                            #average_force_ang = math.asin(force_mags[i]*math.sin(math.radians(180 - angle)) / average_force_mag)
                            average_force_mag = 0
                            average_force_ang = 0

                    if type(gate_stats) != int:
                        gate = []
                        for i in range(len(gate_stats)):
                            gate.append(gate_stats[i])
                    if type(obstacle_stats) != int:
                        obstacles = []
                        for i in range(len(obstacle_stats)):
                            obstacles.append(obstacle_stats[i])
                    if args.independent_learning:
                        tmp_epsilon = models[0].epsilon
                    else:
                        tmp_epsilon = model.epsilon

                    writer.writerow([r, tmp_epsilon, reached_goal, loss, force_mags, force_angs, 
                                    [average_force_mag, math.degrees(average_force_ang)], obj_stats[0], obj_stats[1],
                                    obj_stats[5], gate, obstacles, intention_reward, next_heading_intention[0], 
                                    time.time() - episode_start_time, robot_x_pos, robot_y_pos, robot_angle, 
                                    robot_failures, env_observations, actions_to_take])

                    if episode_done:
                        run_time = time.time() - episode_start_time
                        if not args.no_print:
                            print('[RUN TIME] %.2f' % run_time)
                        if args.independent_learning:
                            exp_rewards.append(np.average(running_reward))
                        else:
                            exp_rewards.append(running_reward)
                        if not reached_goal:
                            if not args.no_print:
                                print("Episode", ep_counter ,"timed out")
                            if test_mode:
                                Testing_Failures += 1
                        else:
                            if not args.no_print:
                                print("Episode", ep_counter ,"reached goal")
                            if test_mode:
                                Testing_Successes += 1
                        if not args.no_print:
                            for i in range(Utility.params['num_robots']):
                                if args.independent_learning:
                                    print('Agent', i, 'reward %.1f' % running_reward[i],
                                          'epsilon:%.2f' % models[i].epsilon,
                                          'steps:', models[i].learn_step_counter)
                                else:
                                    print('Agent', i, 'reward %.1f' % running_reward,
                                          'epsilon:%.2f' % model.epsilon,
                                          'steps:', model.networks['learn_step_counter'])
                                    print('Intention rewards %.2f' % episode_intention_rewards[i])

                        if ep_counter % 10 == 0:
                            exp_mean_rewards.append(np.mean(exp_rewards))
                            exp_rewards = []
                            file_name = 'Episode_'+str(ep_counter)
                            path = recording_path + "/Models/" +file_name
                            if train_mode:
                                if args.independent_learning:
                                    for i in range(Utility.params['num_robots']):
                                        models[i].save_model(path)
                                else:
                                    model.save_model(path)
                            if not args.no_print:
                                print('reward last 10 eps:%.2f'%exp_mean_rewards[-1],'\n')
                        ep_counter += 1
                        if args.independent_learning:
                            for i in range(Utility.params['num_robots']):
                                models[i].reset_obj_stats()
                        else:
                            model.reset_obj_stats()
                        #if not args.no_print:
                        #    print("[INFO] Max Object Statistics: ", model.max_obj_stats)
                        #    print("[INFO] Min Object Statistics: ", model.min_obj_stats)

                        # Send acknowledgment
                        socket.send(b"ok")
print("[RUN TIME] Experiment: %.2f" % (time.time() - experiment_start_time))
if test_mode:
    print('Experiment:', args.recording_path)
    print("[Statistics] Success Percentage", (Testing_Successes/(Testing_Successes+Testing_Failures)))
    print("[Statistics] Failure Percentage", (Testing_Failures/(Testing_Successes+Testing_Failures)))
print("Closing Server")
#socket.unbind("tcp://:" + port)
#socket.close()
print("Experiment Done\n")