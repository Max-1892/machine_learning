import numpy as np
import pdb
import sys
from sarsa import *
import pickle

# Grab the racetrack name from the command line args
racetrack_file = sys.argv[1]
crash_type = sys.argv[2]
q_value_map_file = sys.argv[3]

# Read in racetrack file
racetrack_tokens = list()
with open(racetrack_file) as FileObj:
    for lines in FileObj:
        racetrack_tokens.append(lines.replace("\n", ""))

# Extract racetrack height and width
racetrack_height = int(racetrack_tokens[0].split(',')[0])
racetrack_width = int(racetrack_tokens[0].split(',')[1])

# Translate lines of racetrack into a 2-d array
expand_track = list()
for row in racetrack_tokens[1:]:
    expand_row = list()
    for element in row:
        expand_row.append(element)
    expand_track.append(expand_row)
grid_map = np.array(expand_track).astype(object)

# Extract start locations
start_locs = list()
indices = np.where(grid_map == 'S')
for row,col in zip(indices[0], indices[1]):
    start_locs.append((row,col))

# Extract goal locations
goal_locs = list()
indices = np.where(grid_map == 'F')
for row,col in zip(indices[0], indices[1]):
    goal_locs.append((row,col))

# Load q value map
afile = open(q_value_map_file, 'rb')
q_value_map = pickle.load(afile)
afile.close()

possible_actions = [(acc_x,acc_y) for acc_x in xrange(-1,2) for acc_y in xrange(-1,2)]
np.random.seed(314)
possible_row_positions, possible_col_positions = np.where(grid_map == '.')
num_of_start_positions = len(possible_row_positions)
successes = 0
print num_of_start_positions
#for start_loc in start_locs:
for start_row,start_col in zip(possible_row_positions,possible_col_positions):
    # Pick a random start location
    #start_loc = start_locs[np.random.choice(xrange(len(start_locs)))]
    start_loc = (start_row,start_col)
    curr_row_pos = start_loc[0]
    curr_col_pos = start_loc[1]
    # Initialize velocity
    curr_row_vel = 0
    curr_col_vel = 0
    state = ((curr_row_pos,curr_col_pos),(curr_row_vel,curr_col_vel))
    iterations = 0
    # while we haven't reached the finish line, loop
    '''display_map = deepcopy(grid_map)
    display_map[curr_row_pos,curr_col_pos] = 'R'
    for row in xrange(display_map.shape[0]):
        print ' '.join(map(str, display_map[row]))
    print "Velocity = %d, %d" % (curr_row_vel, curr_col_vel)'''
    while grid_map[curr_row_pos, curr_col_pos] != 'F' and iterations < 100:
        #print "Iteration = %d" % iterations
        # Calculate q values of states resulting from taking each action
        reachable_state_values = \
            calculate_reachable_state_values(
                grid_map, q_value_map, state, possible_actions, crash_type, start_locs)
        # Choose an action using epsilon-greedy method
        accl = reachable_state_values[0][0]
        old_row_pos = curr_row_pos
        old_col_pos = curr_col_pos
        # Determine if the acceleration update will be applied or ignored
        if np.random.choice(['accept','ignore'], p=[0.8, 0.2]) == 'accept':
            # Apply acceleration
            curr_row_vel = accl[0] + curr_row_vel
            # Make sure velocity stays in between -5 and 5 inclusive
            if curr_row_vel > 5:
                curr_row_vel = 5
            if curr_row_vel < -5:
                curr_row_vel = -5
            curr_col_vel = accl[1] + curr_col_vel
            # Make sure velocity stays in between -5 and 5 inclusive
            if curr_col_vel> 5:
                curr_col_vel = 5
            if curr_col_vel < -5:
                curr_col_vel = -5
            # Update position
            curr_row_pos = curr_row_vel + curr_row_pos
            curr_col_pos = curr_col_vel + curr_col_pos
        else:
            # Acceleration was ignored, update position with old velocity
            curr_row_pos = curr_row_vel + curr_row_pos
            curr_col_pos = curr_col_vel + curr_col_pos
    
        # Determine if crash happened
        crashed_or_finished = \
            determine_if_crashed_or_finished(grid_map, (old_row_pos,old_col_pos), (curr_row_vel,curr_col_vel))
        crashed = crashed_or_finished[0]
        finished = crashed_or_finished[1]
        finished_coors = crashed_or_finished[2]
        if finished:
            curr_row_pos = finished_coors[0]
            curr_col_pos = finished_coors[1]
        elif crashed:
            #print "CRASH!"
            curr_row_vel = 0; curr_col_vel = 0;
            # Are we running with a soft or hard crash?
            if crash_type == 'soft':
                # For soft crashes, put the racecar back to its previous location
                curr_row_pos = old_row_pos
                curr_col_pos = old_col_pos
            else:
                # For hard crashes, put the racecar back at a random start location
                new_pos_loc = start_locs[np.random.choice(xrange(len(start_locs)))]
                curr_row_pos = new_pos_loc[0]
                curr_col_pos = new_pos_loc[1]
        # Print racetrack map
        '''display_map = deepcopy(grid_map)
        display_map[curr_row_pos,curr_col_pos] = 'R'
        for row in xrange(display_map.shape[0]):
            print ' '.join(map(str, display_map[row]))
        print "Velocity = %d, %d" % (curr_row_vel, curr_col_vel)'''
        iterations += 1
    if grid_map[curr_row_pos, curr_col_pos] == 'F':
        successes += 1
    '''display_map = deepcopy(grid_map)
    display_map[curr_row_pos,curr_col_pos] = 'R'
    for row in xrange(display_map.shape[0]):
        print ' '.join(map(str, display_map[row]))
    print "Velocity = %d, %d" % (curr_row_vel, curr_col_vel)'''
    #print "The racecar completed the race in %d moves." % iterations
print "Percent of successes = %f" % (float(successes) / num_of_start_positions)
