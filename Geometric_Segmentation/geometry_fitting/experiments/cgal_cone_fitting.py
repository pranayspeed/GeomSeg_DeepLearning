from __future__ import print_function
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Shape_detection import *

import os

import numpy as np


datafile = 'cone_points.txt'

points_np = np.loadtxt(datafile)
points=Point_set_3()
for pts in points_np:
    points.insert(Point_3(pts[0],pts[1],pts[2]))
#points = Point_set_3(datafile)
print(points.size(), "points read")
for p in points.points():
    print(p)
    break


print("Detecting cylinders with efficient RANSAC")
cone_map = points.add_int_map("cone_index")
cones = efficient_RANSAC(points,
                             cone_map,
                             planes=False,
                             spheres=True)
print(len(cones), "cone(s) detected, first 10 cones are:")
for s in range(min(len(cones), 10)):
    print(" *", s, ":", cones[s])

print(
    "Detecting everything possible with efficient RANSAC (custom parameters)")
shape_map = points.add_int_map("shape_index")
shapes = efficient_RANSAC(points,
                          shape_map,
                          min_points=5,
                          epsilon=1.0,
                          cluster_epsilon=1.2,
                          normal_threshold=0.85,
                          planes=True,
                          cylinders=True,
                          spheres=True,
                          cones=True,
                          tori=True)
print(len(shapes), "shapes(s) detected, first 10 shapes are:")
for s in range(min(len(shapes), 10)):
    print(" *", s, ":", shapes[s])

# Counting types of shapes
nb_cones = 0
nb_cylinders = 0
nb_planes = 0
nb_spheres = 0
nb_tori = 0
for s in shapes:
    _type = s.split()[1]
    if _type == "cone":
        nb_cones += 1
    if _type == "cylinder":
        nb_cylinders += 1
    if _type == "plane":
        nb_planes += 1
    if _type == "sphere":
        nb_spheres += 1
    if _type == "torus":
        nb_tori += 1
print("Number of shapes by type:")
print(" *", nb_cones, "cone(s)")
print(" *", nb_cylinders, "cylinder(s)")
print(" *", nb_planes, "plane(s)")
print(" *", nb_spheres, "sphere(s)")
print(" *", nb_tori, "torus/i")

print("Recovering inliers of first shape")
inliers_of_first_shape = Point_set_3()
for idx in points.indices():
    if shape_map.get(idx) == 0:
        inliers_of_first_shape.insert(points.point(idx))
print(inliers_of_first_shape.size(), "inliers(s) recovered")