import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


import csv

with open('p1.txt') as csvfile:
    data1 = [(int(x), int(y)) for x, y in csv.reader(csvfile, delimiter= ' ')]
with open('p2.txt') as csvfile:
    data2 = [(int(x), int(y)) for x, y in csv.reader(csvfile, delimiter= ' ')]
with open('p3.txt') as csvfile:
    data3 = [(int(x), int(y)) for x, y in csv.reader(csvfile, delimiter= ' ')]

choose= "p1"
max = 0
avg = 0
avgn = 0
min = 100
x=80
errdist=10
errdistcount=0
for d in data1:
    if abs(d[0]-x) > max:
        max=abs(d[0]-x)
    if abs(d[0]-x) < min:
        min=abs(d[0]-x)
    avg += abs(d[0]-x)
    avgn += 1
    if abs(d[0]-x) < errdist:
        errdistcount+=1
y=55
for d in data2:
    if abs(d[1]-y) > max:
        max=abs(d[1]-y)
    if abs(d[1]-y) < min:
        min=abs(d[0]-y)
    avg += abs(d[1]-y)
    avgn += 1
    if abs(d[1]-y) < errdist:
        errdistcount+=1
y=210
for d in data3:
    if abs(d[1]-y) > max:
        max=abs(d[1]-y)
    if abs(d[1]-y) < min:
        min=abs(d[1]-y)
    avg += abs(d[1]-y)
    avgn += 1

    if abs(d[1]-y) < errdist:
        errdistcount+=1
print("Maximum")
print(max)
print("Átlag")
print(avg/avgn)
print("Minimum")
print(min)