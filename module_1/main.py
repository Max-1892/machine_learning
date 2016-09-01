from winnow_model import *
model = WinnowModel(2, 0.5, 3, 1)
f = open('test', 'r')
for line in f:
    line_split = line.split(',')
    model.learn(map(int, line_split[:-1]), int(line_split[-1]))
print model.output_model()
