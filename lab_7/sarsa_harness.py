import numpy as np
import pdb
import sys
from sarsa import sarsa
import pickle

# Grab the racetrack name from the command line args
racetrack_file = sys.argv[1]
learning_rate = float(sys.argv[2])
discount_factor = float(sys.argv[3])
epsilon = float(sys.argv[4])
crash_type = sys.argv[5]

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

# Run SARSA
print "q_value_map_racetrack_%s_learning_rate_%f_discount_factor_%f_epsilon_%f_crash_type_%s.pkl" % (racetrack_file[:-4], learning_rate, discount_factor, epsilon, crash_type)
q_value_map = sarsa(grid_map, racetrack_height, racetrack_width, start_locs, learning_rate, discount_factor, epsilon, crash_type, racetrack_file)
filename = "q_value_map_racetrack_%s_learning_rate_%f_discount_factor_%f_epsilon_%f_crash_type_%s.pkl" % (racetrack_file, learning_rate, discount_factor, epsilon, crash_type)
afile = open(filename, 'wb')
pickle.dump(q_value_map, afile)
afile.close()
