from copy import deepcopy
import numpy as np
import pdb

'''This method performs value iteration on the racecar control problem with
   the inputed parameters.'''
def value_iteration(grid_map, map_height, map_width, 
                    start_locs, goal_locs, discount_factor, epsilon, crash_type):
    # Initialize the value and policy map
    # Both data structures will use a multi-level nested map structure that is indexed
    # in the following order: row in the grid_map, column in the grid_map, velocity in row 
    # direction, velocity in the column direction
    possible_velocities = [(velo_x,velo_y) for velo_x in xrange(-5,6) for velo_y in xrange(-5,6)]
    policy_map = dict()
    value_function_map = dict()
    for row in xrange(map_height):
        policy_map[row] = dict()
        value_function_map[row] = dict()
        for col in xrange(map_width):
            policy_map[row][col] = dict()
            value_function_map[row][col] = dict()
            for velocity in possible_velocities:
                policy_map[row][col][velocity[0]] = dict()
                policy_map[row][col][velocity[0]][velocity[1]] = list()
                value_function_map[row][col][velocity[0]] = dict()

    # Calculate all possible states
    # which will be stored in a list of tuples, each tuple containing a tuple for position and velocity
    # i.e. some_state = ( (row_position, column_position), (row_velocity, column_velocity) )
    possible_states = list()
    for row in xrange(map_height):
        for col in xrange(map_width):
            for velocity in possible_velocities:
                if grid_map[row,col] != '#':
                    possible_states.append(((row,col),velocity))
                    policy_map[row][col][velocity[0]][velocity[1]] = \
                        [(acc_x,acc_y) for acc_x in xrange(-1,2) for acc_y in xrange(-1,2)]
                    value_function_map[row][col][velocity[0]][velocity[1]] = 0.0
                else:
                    # Initialize non-racetrack cells differently
                    value_function_map[row][col][velocity[0]][velocity[1]] = -1000

    # Main value iteration loop
    # Continue until the successive largest differences in V drop 
    # below the Bellman error magnitude (epsilon)
    max_value_increase = epsilon
    while max_value_increase >= epsilon:
        max_value_increase = float("-inf")
        # old_value_map = value function map from t-1
        old_value_map = deepcopy(value_function_map)

        # For each possible state, action pair
        for state in possible_states:
            row = state[0][0]; col = state[0][1]; vel_row = state[1][0]; vel_col = state[1][1];
            q_values= list()
            for action in [(acc_x,acc_y) for acc_x in xrange(-1,2) for acc_y in xrange(-1,2)]:
                # q_value(state,action) = 
                #     Reward(state, action) + 
                #     discount_factor *
                #     sum over successor states [Transition(s,a,s') * 
                #     value_estimate_t-1(successor_state)]
                # For the racetrack problem, there are two successor states:
                #     1. The acceleration update is successful (probability = 80%)
                #     2. The acceleration is rejected (probability = 20%)
                successor_states = \
                    calculate_successor_states(grid_map, state, action, crash_type, start_locs)
                successor_states_sum = 0.8 * \
                    old_value_map[successor_states[0][0]] \
                                 [successor_states[0][1]] \
                                 [successor_states[0][2]] \
                                 [successor_states[0][3]]
                successor_states_sum += 0.2 * \
                    old_value_map[successor_states[1][0]] \
                                 [successor_states[1][1]] \
                                 [successor_states[1][2]] \
                                 [successor_states[1][3]]
                # Reward = 0 if we reach the finish line, -1 otherwise
                if (successor_states[0][0],successor_states[0][1]) in goal_locs:
                    reward = 0
                else:
                    reward = -1
                q_values.append((action, reward + discount_factor * successor_states_sum))
            # Update the policy map by choosing the action with the best q value
            q_values = sorted(q_values, key=lambda tup: tup[1])
            q_values.reverse()
            q_values = np.array(q_values)
            # There may be multiple actions with the same best q_value so grab them all
            best_q_values = np.where(q_values[:,1] == q_values[0][1])[0]
            policy_map[row][col][vel_row][vel_col] = q_values[best_q_values][:,0]
            # Calculate update difference, this is the value that terminates the 
            # loop
            value_difference = abs(q_values[0][1] - old_value_map[row][col][vel_row][vel_col])
            if value_difference > max_value_increase:
                max_value_increase = value_difference
            value_function_map[row][col][vel_row][vel_col] = q_values[0][1]
    return policy_map, value_function_map
 
'''This method performs the kinematic updates given an old state and an action to
   apply. A new position and velocity is calculated both for when the given action
   was accepted by the model and when it is ignored. In addition, checks are made
   to ensure velocity constraints are adhered to and for crashes.'''
def calculate_successor_states(grid_map, old_state, action, crash_type, start_locs):
    # Need to calculate two successor states
    #     1. Action is accepted
    #     2. Action is ignored
    # In either case we need to check if we hit a wall, if so use crash_type ('soft','hard')
    # to figure out what the new state is
    # Start with action being accepted
    vel_row_accepted = action[0] + old_state[1][0]
    # Make sure velocity stays in between -5 and 5 inclusive
    if vel_row_accepted > 5:
        vel_row_accepted = 5
    if vel_row_accepted < -5:
        vel_row_accepted = -5
    vel_col_accepted = action[1] + old_state[1][1]
    # Make sure velocity stays in between -5 and 5 inclusive
    if vel_col_accepted > 5:
        vel_col_accepted = 5
    if vel_col_accepted < -5:
        vel_col_accepted = -5
    pos_row_accepted = vel_row_accepted + old_state[0][0]
    pos_col_accepted = vel_col_accepted + old_state[0][1]
    # Check to make sure it didn't cross any walls
    if determine_if_crashed(grid_map, old_state[0], (vel_row_accepted,vel_col_accepted)):
        vel_row_accepted = 0; vel_col_accepted = 0;
        if crash_type == 'soft':
            pos_row_accepted = old_state[0][0]
            pos_col_accepted = old_state[0][1]
        else:
            new_pos_loc = np.choice(start_locs)
            pos_row_accepted = new_pos_loc[0]
            pos_col_accepted = new_pos_loc[1]

    # Now let's calculate the case when the action is ignored
    vel_row_ignored = old_state[1][0]
    vel_col_ignored = old_state[1][1]
    pos_row_ignored = vel_row_ignored + old_state[0][0]
    pos_col_ignored = vel_col_ignored + old_state[0][1]
    # Check to make sure it didn't cross any walls
    if determine_if_crashed(grid_map, old_state[0], (vel_row_ignored,vel_col_ignored)):
        vel_row_ignored = 0; vel_col_ignored = 0;
        if crash_type == 'soft':
            pos_row_ignored = old_state[0][0]
            pos_col_ignored = old_state[0][1]
        else:
            new_pos_loc = np.choice(start_locs)
            pos_row_ignored = new_pos_loc[0]
            pos_col_ignored = new_pos_loc[1]

    # Return an array of size 2 (one for each case) of 
    # 4-tuples (row pos, col pos, row velocity, col velocity)
    return [(pos_row_accepted,pos_col_accepted,vel_row_accepted,vel_col_accepted),
        (pos_row_ignored,pos_col_ignored,vel_row_ignored,vel_col_ignored)]

def determine_if_crashed(grid_map, start_position, velocity):
    # To determine if the racer crashed into the wall, we'll 
    # move the racer first by velocity_row then by velocity_col starting at the
    # start_position, one step at a time. 
    # Next we'll move the racer first by velocity_col then by
    # velocity_row starting at the start_position, one step at a time. 
    # If at least one wall is encountered in each walk, we consider it a crash.

    # First walk: velocity_row then velocity_col
    first_walk_crash = False
    velocity_row = velocity[0]; velocity_col = velocity[1]
    current_position_row = start_position[0]; current_position_col = start_position[1]
    # velocity_row
    while velocity_row != 0:
        # Update velocity differently if it is negative
        if velocity_row > 0:
            current_position_row += 1
            velocity_row -= 1
        else:
            current_position_row -= 1
            velocity_row += 1
        if grid_map[current_position_row, current_position_col] == '#':
            first_walk_crash = True
            break
    # velocity_col
    while velocity_col != 0:
        # Update velocity differently if it is negative
        if velocity_col > 0:
            current_position_col += 1
            velocity_col -= 1
        else:
            current_position_col -= 1
            velocity_col += 1
        if grid_map[current_position_row, current_position_col] == '#':
            first_walk_crash = True
            break

    # Second walk: velocity_col then velocity_row
    second_walk_crash = False
    if first_walk_crash:
        velocity_row = velocity[0]; velocity_col = velocity[1]
        current_position_row = start_position[0]; current_position_col = start_position[1]
        # velocity_col
        while velocity_col != 0:
            # Update velocity differently if it is negative
            if velocity_col > 0:
                current_position_col += 1
                velocity_col -= 1
            else:
                current_position_col -= 1
                velocity_col += 1
            if grid_map[current_position_row, current_position_col] == '#':
                second_walk_crash = True
                break
        # velocity_row
        while velocity_row != 0:
            # Update velocity differently if it is negative
            if velocity_row > 0:
                current_position_row += 1
                velocity_row -= 1
            else:
                current_position_row -= 1
                velocity_row += 1
            if grid_map[current_position_row, current_position_col] == '#':
                second_walk_crash = True
                break

    return first_walk_crash and second_walk_crash
