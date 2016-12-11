import numpy as np
from random import randrange
import pdb
from value_iteration import calculate_successor_states
from value_iteration import determine_if_crashed_or_finished
from copy import deepcopy

def sarsa(grid_map, map_height, map_width, start_locs, learning_rate, discount_factor, epsilon, crash_type, track_name, q_value_map_init=None):
   # Initialize q values for all state-action pairs randomly and
   # determine possible states the racecar can have
   possible_actions = [(acc_x,acc_y) for acc_x in xrange(-1,2) for acc_y in xrange(-1,2)]
   possible_velocities = [(velo_x,velo_y) for velo_x in xrange(-5,6) for velo_y in xrange(-5,6)]
   q_value_map = dict()
   possible_states = list()
   for row in xrange(map_height):
       q_value_map.setdefault(row, {})
       for col in xrange(map_width):
           q_value_map[row].setdefault(col, {})
           for velocity in possible_velocities:
               q_value_map[row][col].setdefault(velocity[0], {})
               q_value_map[row][col][velocity[0]].setdefault(velocity[1], {})
               for action in possible_actions:
                   q_value_map[row][col][velocity[0]][velocity[1]].setdefault(action[0], {})
                   q_value_map[row][col][velocity[0]][velocity[1]][action[0]].setdefault(action[1], 0)
                   if grid_map[row,col] != '#':
                       possible_states.append(((row,col),velocity))
   if q_value_map_init != None:
       q_value_map = q_value_map_init

   states_to_visit = get_states_to_visit(track_name, possible_velocities)
   # Loop over episodes, one episode per state?
   epsilon_update = float(epsilon) / len(states_to_visit)
   episode_count = 1
   for state_to_visit in states_to_visit:
       print "Episode %d of %d" % (episode_count, len(states_to_visit))

       # Choose a starting position
       state = state_to_visit
       #print_map(grid_map, state)

       # Calculate q values of states resulting from taking each action
       reachable_state_values = \
           calculate_reachable_state_values(
               grid_map, q_value_map, state, possible_actions, crash_type, start_locs)
       # Choose an action using epsilon-greedy method
       # First update epsilon
       if epsilon - epsilon_update > 0:
           epsilon -= epsilon_update
       else:
           epsilon = 0
       action_idx = 0
       if np.random.choice(['greedy','random'], [1.0-epsilon, epsilon]) == 'greedy':
           # Choose the best action
           action = reachable_state_values[action_idx][0]
       else:
           # Choose a random action
           action_idx = np.random.choice(xrange(len(reachable_state_values)))
           action = reachable_state_values[action_idx][0]

       # While not at finish line
       itr = 0
       while grid_map[state[0][0],state[0][1]] != 'F':
           # Take action a, observe reward, calculate successor state s'
           successor_state = reachable_state_values[action_idx][1]

           # Choose next action (a') using epsilon-greedy method from Q
           reachable_state_values = \
               calculate_reachable_state_values(
                   grid_map, q_value_map, successor_state, possible_actions, crash_type, start_locs)
           # Choose an action using epsilon-greedy method
           successor_action_idx = 0
           if np.random.choice(['greedy','random'], [1.0-epsilon, epsilon]) == 'greedy':
               # Choose the best action
               successor_action = reachable_state_values[0][0]
           else:
               # Choose a random action
               successor_action_idx = np.random.choice(xrange(len(reachable_state_values)))
               successor_action = reachable_state_values[successor_action_idx][0]

           # Cache q values because accessing them is ugly
           q_value = q_value_map[state[0][0]][state[0][1]] \
                                 [state[1][0]][state[1][1]] \
                                 [action[0]][action[1]]
           successor_q_value = q_value_map[successor_state[0][0]][successor_state[0][1]] \
                                       [successor_state[1][0]][successor_state[1][1]] \
                                       [successor_action[0]][successor_action[1]]

           # Update q value (s = current state):
           # Q_t+1(s,a) = Q(s,a) + learning_rate * (reward + discount_factor * Q(s',a')-Q(s,a))
           q_value_map[state[0][0]][state[0][1]][state[1][0]][state[1][1]][action[0]][action[1]] = \
               q_value + learning_rate * \
                   (get_reward(grid_map, successor_state) + discount_factor * successor_q_value - q_value)
               
           # s = s', a = a'
           state = successor_state
           action_idx = successor_action_idx
           action = successor_action
           #print_map(grid_map, state)
           itr += 1
           if itr == 500:
               break;
       episode_count += 1
   return q_value_map

'''This method applies each action in possible_actions to curr_state
   and stores the resulting state's q value in reachable_state_values.'''
def calculate_reachable_state_values( \
    grid_map, q_value_map, curr_state, possible_actions, crash_type, start_locs):
    reachable_state_values = list()
    for action in possible_actions:
        next_row_pos, next_col_pos, next_row_vel, next_col_vel = \
            calculate_new_state(grid_map, curr_state, action, crash_type, start_locs)
        reachable_state_values.append(
            (action, 
            ((next_row_pos,next_col_pos), (next_row_vel,next_col_vel)), 
            q_value_map[next_row_pos][next_col_pos][next_row_vel][next_col_vel][action[0]][action[1]]))
    reachable_state_values = sorted(reachable_state_values, key=lambda tup: tup[2])
    reachable_state_values.reverse()
    return reachable_state_values

def calculate_new_state(grid_map, old_state, action, crash_type, start_locs):
    vel_row = action[0] + old_state[1][0]
    # Make sure velocity stays in between -5 and 5 inclusive
    if vel_row > 5:
        vel_row = 5
    if vel_row < -5:
        vel_row = -5
    vel_col = action[1] + old_state[1][1]
    # Make sure velocity stays in between -5 and 5 inclusive
    if vel_col > 5:
        vel_col = 5
    if vel_col < -5:
        vel_col = -5
    pos_row = vel_row + old_state[0][0]
    pos_col = vel_col + old_state[0][1]
    # Check to make sure it didn't cross any walls or we finished the race
    crashed_or_finished = \
        determine_if_crashed_or_finished(grid_map, old_state[0], (vel_row,vel_col))
    crashed = crashed_or_finished[0]
    finished = crashed_or_finished[1]
    finished_coors = crashed_or_finished[2]
    if finished:
        pos_row = finished_coors[0]
        pos_col = finished_coors[1]
    elif crashed:
        vel_row = 0; vel_col = 0;
        if crash_type == 'soft':
            pos_row = old_state[0][0]
            pos_col = old_state[0][1]
        else:
            new_pos_loc = np.choice(start_locs)
            pos_row = new_pos_loc[0]
            pos_col = new_pos_loc[1]

    return pos_row,pos_col,vel_row,vel_col

def get_reward(grid_map, state):
    if grid_map[state[0][0],state[0][1]] == 'F':
        return 0.0
    else:
        return -1.0

def print_map(grid_map, state):
    # Print racetrack map
    display_map = deepcopy(grid_map)
    display_map[state[0][0],state[0][1]] = 'R'
    for row in xrange(display_map.shape[0]):
        print ' '.join(map(str, display_map[row]))
    print "Velocity = %d, %d" % (state[1][0], state[1][0])

def get_states_to_visit(track_name, possible_velocities):
    states_to_visit = list()
    if track_name == 'L-track.txt':
        for _ in xrange(0, 30):
            '''for row_idx in xrange(2, 10):
                for col_idx in xrange(32, 36):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in xrange(6, 10):
                for col_idx in reversed(xrange(26, 32)):
                    for velocity in possible_velocities:
                            states_to_visit.append(((row_idx,col_idx), velocity))'''
            '''for row_idx in xrange(6, 10):
                for col_idx in reversed(xrange(20, 26)):
                    for velocity in possible_velocities:
                            states_to_visit.append(((row_idx,col_idx), velocity))'''
            '''for row_idx in xrange(6, 10):
                for col_idx in reversed(xrange(12, 20)):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))'''
            for row_idx in xrange(6, 10):
                for col_idx in reversed(xrange(6, 12)):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            '''for row_idx in xrange(6, 10):
                for col_idx in reversed(xrange(1, 6)):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))'''
    elif track_name == 'O-track.txt':
        for _ in xrange(0, 10):
            for row_idx in xrange(13, 20):
                for col_idx in xrange(1, 5):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [20]:
                for col_idx in [2,3,4,5]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [21]:
                for col_idx in [3,4,5,6]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [22]:
                for col_idx in xrange(3,22):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [23]:
                for col_idx in xrange(4,21):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [21]:
                for col_idx in [18,19,20,21]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [20]:
                for col_idx in [19,20,21,22]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [19,18,17,16,15,14,13,12,11,10,9,8,7,6,5]:
                for col_idx in [20,21,22,23]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [4]:
                for col_idx in [19,20,21,22]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [3]:
                for col_idx in [18,19,20,21]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [2]:
                for col_idx in [21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [1]:
                for col_idx in [20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [3]:
                for col_idx in [3,4,5,6]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [4]:
                for col_idx in [2,3,4,5]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [5,6,7,8,9,10]:
                for col_idx in [2,3,4,5]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
    else:
        for _ in xrange(0, 10):
            for row_idx in [25,26]:
                for col_idx in [24,25,26,27,28]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [23,22,21,20,19,18,17]:
                for col_idx in [23,24,25,26,27]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [16]:
                for col_idx in [21,22,23,24,25,26,27]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [15]:
                for col_idx in [19,20,21,22,23]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [14]:
                for col_idx in [16,17,18,19,20]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [13]:
                for col_idx in [14,15,16,17,18]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [12]:
                for col_idx in [13,14,15,16,17]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [11]:
                for col_idx in [12,13,14,15,16]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [10]:
                for col_idx in [10,11,12,13,14]:
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [9]:
                for col_idx in xrange(12,16):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [8]:
                for col_idx in xrange(14,18):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [7]:
                for col_idx in xrange(15,19):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [6]:
                for col_idx in xrange(17,22):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [5]:
                for col_idx in xrange(19,24):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [4]:
                for col_idx in xrange(18,25):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [3]:
                for col_idx in xrange(3,27):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [2]:
                for col_idx in xrange(5,26):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [1]:
                for col_idx in xrange(9,22):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in [4]:
                for col_idx in xrange(2,8):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in xrange(5,12):
                for col_idx in xrange(2,7):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in xrange(12,16):
                for col_idx in xrange(1,5):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in xrange(16,21):
                for col_idx in xrange(2,7):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))
            for row_idx in xrange(21,27):
                for col_idx in xrange(1,6):
                    for velocity in possible_velocities:
                        states_to_visit.append(((row_idx,col_idx), velocity))

    return states_to_visit


