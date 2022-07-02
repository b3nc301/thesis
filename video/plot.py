import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


import csv

with open('res.txt') as csvfile:
    data = [(int(x), int(y)) for x, y in csv.reader(csvfile, delimiter= ' ')]



verts = [
   (245, 210),  # right, bottom
   (80, 210),  # left, bottom
   (80, 55),  # left, top
   (210, 55),  # right, top
   #(0., 0.), # ignored
]

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    #Path.CLOSEPOLY,
]
path = Path(verts, codes)
patch = patches.PathPatch(path, lw=2, facecolor='none')
fig, ax = plt.subplots()

ax.add_patch(patch)
ax.scatter(*zip(*data))

#ax.scatter(data[0][0],data[0][1], c='green')



plt.scatter(80, 210, c='red')
plt.scatter(80, 55, c='red')
plt.scatter(210, 55, c='red')
plt.scatter(245, 210, c='red')
#
plt.show()