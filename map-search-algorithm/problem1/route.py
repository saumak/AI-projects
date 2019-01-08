#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:31:29 2017
@author: PulkitMaloo
"""
#==============================================================================
# route.py : Solve the routing problem
#
# Abstraction:
#    Initial State: start_city
#    Goal State: end_city
#    Successor: returns all the cities that are connected to the current city
#    State space: all the cities
#    Cost Function: cost_function
#    Heuristic: Great circle distance
#
# (1) Uniform Cost Search seems to work best for finding a route between two cities.
#       It is computationally faster for finding the optimal route than astar
#       and bfs, dfs doesn't gurantees to find the optimal route
#
# (2) For example:
#     1. start_city = San_Jose,_California
#        end_city = Miami,_Florida
#        Time taken by    dfs = 9.339 ms
#                         bfs = 11.122 ms
#                         uniform = 41.419 ms
#                         astar = 64.850 ms
#     2. start_city = Bloomington,_Indiana
#        end_city = Seattle,_Washington
#        Time taken by    dfs = 5.203 ms
#                         bfs = 11.052 ms
#                         uniform = 55.244 ms
#                         astar = 100.631 ms
#
#  The above experiments clearly illustrate that Depth First Search is the
#  fastest in terms of the amount of computation time required
#  Depth first search is around 10-20 times faster than astar on an average case, but
#  the factor totally depends on the start_city and end_city and so its hard to generalize
#  However, the dfs could run really slow at times when it will have to explore all the cities
#
#
# (3) Uniform cost search requires the least memory on an average case.
#       Since the astar is performing search algorithm #2 so it revisits the cities and so would the size of the fringe increases
#       Uniform keeps track of the visited cities so its fringe size is lesser than astar
#       bfs, dfs fringe size would totally depend on the start_city and end_city
#       On an average case through several experiments, uniform cost requires the least memory by 2-5 times than astar
#
#
# (4) The heuristic function calculates the great circle distance between the
# current city and end_city given it has the latitude and longitude of both the cities
# If the latitude, longitude is missing of the current city then it returns
# the heuristic of nearest city(which has a latitude, longitude) - cost to the nearest city
# If the nearest city is the goal state then dist will become negative so it will simply return 0
# If none of the adjacent cities to the current city have latitude or longitude then heuristic is 0
# I've also taken the assumption that end_city will not be a Junction(which are missing lat lon) and in that case the heuristic will return 0
# If the cost_function is distance the heuristic returns the distance calculated
# If the cost_function is time the heuristic returns distance/max_speed
# If the cost_function is segments the heuristic returns distance/max_distance
#
# The heuristic which calculates the great circle distance should ideally be consistent
# However, Due to the anomalies in the dataset(missing lat lon values, anomalous distance values and many more)
# the heuristic in this case is not even guaranteed to be admissable all the time, but it is the closest to admissability we could get
# We cannot say about consistent because of so many anomalies in the dataset, also in this case if the heuristic returns 0 then it won't be consistent
#
# The heuristic can be improved by maybe multiplying it with some heuristic factor to make it guaranteed to be admissable
# but by doing some experiments, the factor was coming out to be around 0.05 to make it guranteed to be admissable
# this factor is way small and would not add much value to our using the heuristic
#==============================================================================


from __future__ import division
import sys
import heapq

radius_of_earth = 3959

# Assumption 1
# Speed limit of highways, which have missing speed limit
s = {'NS_106' : 62, 'NB_11': 50, 'NB_8': 50, 'NS_103': 50}
# speed = 45 for missing or 0 values
# since majority of the road segments seem to have speed 45
speed_limit = 45
max_speed = 65
max_distance = 923
heuristic_factor = 1
# Assumption 2
# When speed=0 and distance=0 there is no route possible
# Because the highway name is ferry, there's no road there
# =============================================================
#   The format of data is:
# data =
#   {
#   city:
#       {
#           "latitude": value(float) / None
#           "longitude": value(float) / None
#           "visited": False(bool)
#           "parent": city(str)
#           "cost": value(int)
#           "to_city":
#               {
#                   city: {"distance": value(int), "speed": value(int), "time":value(int), segments:1(int) "highway": value(str)},
#                   city: {..},
#                   ...
#               }
#       }
#   }
# =============================================================
def reading_files():
    data = dict()
    f_city = open('city-gps.txt', 'r')
    f_road = open('road-segments.txt', 'r')
    for city in f_city:
        city = city.split()
        data[city[0]] = {'latitude': float(city[1]), 'longitude': float(city[2]), 'visited': False, 'parent': None, 'cost': 0}
    for seg in f_road:
        seg = seg.split()
        # One city has a route to itself
        if seg[0] == seg[1]:
            continue
        # Updating missing speed values by a constant speed_limit
        # or by looking at their highway speeds
        if len(seg) != 5:
            if seg[-1] in s:
                seg = seg[:3] + [s[seg[-1]]] + seg[3:]
            else:
                seg = seg[:3] + [speed_limit] + seg[3:]
        # Ferry cases where distance = 0 and speed = 0
        if int(seg[2])==0:
            continue
        # Updating speed = 0
        if int(seg[3]) == 0:
            seg[3] = speed_limit
        # Some road segment are not in city-gps, they don't have lat or lon
        if seg[0] not in data:
            data[seg[0]] = {'latitude': None, 'longitude': None, 'visited': False, 'parent': None, 'cost': 0}
        # from_city to to_city
        if 'to_city' not in data[seg[0]]:
            data[seg[0]]['to_city'] = dict()
        data[seg[0]]['to_city'][seg[1]] = {'distance':int(seg[2]), 'speed':int(seg[3]), 'time':int(seg[2])/int(seg[3]), 'segments':1, 'highway':seg[4]}
        # to_city to from_city
        if seg[1] not in data:
            data[seg[1]] = {'latitude': None, 'longitude': None, 'visited': False,  'parent': None, 'cost': 0}
        if 'to_city' not in data[seg[1]]:
            data[seg[1]]['to_city'] = dict()
        data[seg[1]]['to_city'][seg[0]] = {'distance':int(seg[2]), 'speed':int(seg[3]), 'time':int(seg[2])/int(seg[3]), 'segments':1, 'highway':seg[4]}
    f_city.close()
    f_road.close()
    return data

# Returns the nearest city(which has lat, lon) and its distance from the current city
def dist_nearest_city(city):
    nearest_cities = successors(city)
    d = []
    for c in nearest_cities:
        if data[c]['latitude'] == None:
            continue
        heapq.heappush(d, (distance(city, c), c))
    return heapq.heappop(d)

def lat_lon(city):
    return data[city]['latitude'], data[city]['longitude']

def great_circle_distance(from_city, to_city):
    from math import radians, sin, cos, acos
    slat, slon = map(radians, lat_lon(from_city))
    elat, elon = map(radians, lat_lon(to_city))
    dist = radius_of_earth*acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
    return dist

def heuristic(city):
    dist = 0
    try:
        if data[city]['latitude']:
            dist = great_circle_distance(city, end_city)
        else:
#            If city's latitude longitude is missing
            d, nearest_city = dist_nearest_city(city)
            dist = heuristic(nearest_city) - d
            dist = dist if dist > 0 else 0
    except:
        return 0
    dist = dist*heuristic_factor
    if cost_function == 'distance':
        return dist
    elif cost_function == 'time':
        return dist/max_speed
    else:
        return dist/max_distance

def distance(from_city, to_city):
    return data[from_city]['to_city'][to_city]['distance']

def speed(from_city, to_city):
    return data[from_city]['to_city'][to_city]['speed']

def time(from_city, to_city):
    return data[from_city]['to_city'][to_city]['time']

def cost(from_city, to_city):
        return data[from_city]['to_city'][to_city][cost_function]

def distance_of_path(path):
    return sum([distance(path[i], path[i+1]) for i in range(len(path)-1)])

def time_of_path(path):
    return sum([time(path[i], path[i+1]) for i in range(len(path)-1)])

def segments_of_path(path):
    return len(path)-1

def cost_of_path(path):
    return eval(cost_function+"_of_path(path)")

def path(city):
    current = city
    current_path = [city]
    while current != start_city:
        current_path.append(data[current]['parent'])
        current = data[current]['parent']
    return distance_of_path(current_path), time_of_path(current_path), current_path[::-1]

def successors(city):
    return data[city]['to_city'].keys()

def solve1(start_city, end_city):
    if start_city == end_city:
        return path(end_city)
    # For switching between bfs dfs, use pop(0) for BFS, pop() for DFS
    i = {'bfs': 0, 'dfs': -1}[routing_algorithm]
    # fringe is a list of cities which can explored further
    fringe = [start_city]
    data[start_city]['visited'] = True
    # While there are still cities to be explored
    while fringe:
        # Retreive the city from the fringe to be expanded
        curr_city = fringe.pop(i)
        # For all cities that we can go from the current city
        for next_city in data[curr_city]['to_city']:
            # And if the next_city is already visited, discard
            if data[next_city]['visited']:
                continue
            data[next_city]['parent'] = curr_city
            # Check if it's our goal state then return the path
            if next_city == end_city:
                return path(next_city)
            # Mark this city visited
            data[next_city]['visited'] = True
            # Add the new city to our fringe
            fringe.append(next_city)
    # No route found
    return False

#===  SA #2   =================================================
# 1. If GOAL?(initial-state) then return initial-state
# 2. INSERT(initial-node, FRINGE)
# 3. Repeat:
# 4. 	If empty(FRINGE) then return failure
# 5.		s  REMOVE(FRINGE)
# 6.		If GOAL?(s) then return s and/or path
# 7.    For every state s’ in SUCC(s):
# 8.		    INSERT(s’, FRINGE)
#==============================================================
def solve2(start_city, end_city):
    fringe = [[heuristic(start_city), start_city]]
    while fringe:
        curr_city = heapq.heappop(fringe)[1]
        g_curr_cost = data[curr_city]['cost']
        if curr_city == end_city:
            return path(curr_city)
        for next_city in data[curr_city]['to_city']:
            g_next_city = g_curr_cost + cost(curr_city, next_city)
            h_next_city = heuristic(next_city)
            f_next_city = g_next_city + h_next_city
            if data[next_city]['parent']:
                if g_next_city > data[next_city]['cost']:
                    continue
                else:
                    try:
                        fringe.remove([data[next_city]['cost'] + h_next_city, next_city])
                        heapq.heapify(fringe)
                    except:
                        pass
            data[next_city]['parent'] = curr_city
            data[next_city]['cost'] = g_next_city
            heapq.heappush(fringe, [f_next_city, next_city])
    return False

#==== SA #3   =================================================
# 1. If GOAL?(initial-state) then return initial-state
# 2. INSERT(initial-node, FRINGE)
# 3. Repeat:
# 4.   If empty(FRINGE) then return failure
# 5.   s  REMOVE(FRINGE)
# 6.   INSERT(s, CLOSED)
# 7.   If GOAL?(s) then return s and/or path
# 8.   For every state s’ in SUCC(s):
# 9.       If s’ in CLOSED, discard s’
# 10.     If s’ in FRINGE with larger s’, remove from FRINGE
# 11.	    If s’ not in FRINGE, INSERT(s’, FRINGE)
#==============================================================
def solve3(start_city, end_city):
    fringe = [[0, start_city]]
    while fringe:
        curr_city = heapq.heappop(fringe)[1]
        curr_cost = data[curr_city]['cost']
        data[curr_city]['visited'] = True
        if curr_city == end_city:
            return path(curr_city)
        for next_city in data[curr_city]['to_city']:
            if data[next_city]['visited']:
                continue
            next_cost = curr_cost + cost(curr_city, next_city)
            if data[next_city]['parent']:
                if next_cost > data[next_city]['cost']:
                    continue
                else:
                    fringe.remove([data[next_city]['cost'], next_city])
                    heapq.heapify(fringe)
            data[next_city]['parent'] = curr_city
            data[next_city]['cost'] = next_cost
            heapq.heappush(fringe, [next_cost, next_city])
    return False

def solve4(start_city, end_city):
    fringe = [[-heuristic(start_city), start_city]]
    while fringe:
        curr_city = heapq.heappop(fringe)[1]
        g_curr_cost = data[curr_city]['cost']
        if curr_city == end_city:
            return path(curr_city)
        for next_city in data[curr_city]['to_city']:
            g_next_city = g_curr_cost - cost(curr_city, next_city)
            try:
                h_next_city = -heuristic(next_city)
            except:
                h_next_city = 0
            f_next_city = g_next_city + h_next_city
            if data[next_city]['parent']:
                if g_next_city < data[next_city]['cost']:
                    continue
                else:
                    try:
                        fringe.remove([data[next_city]['cost'] + h_next_city, next_city])
                        heapq.heapify(fringe)
                    except:
                        pass
            data[next_city]['parent'] = curr_city
            data[next_city]['cost'] = g_next_city
            heapq.heappush(fringe, [f_next_city, next_city])
    return False

#start_city = 'Bloomington,_Indiana'
#end_city = 'Seattle,_Washington'
#routing_algorithm = 'uniform'
#cost_function = 'distance'

start_city = sys.argv[1]
end_city = sys.argv[2]
routing_algorithm = sys.argv[3]
cost_function = sys.argv[4]

import timeit
data = reading_files()
s = timeit.default_timer()
try:
    if routing_algorithm in ['bfs', 'dfs']:
        solution = solve1(start_city, end_city)
    elif routing_algorithm == 'uniform':
        solution = solve3(start_city, end_city)
    elif routing_algorithm == 'astar':
        solution = solve2(start_city, end_city)
    elif routing_algorithm == 'longtour':
        solution = solve4(start_city, end_city)
    else:
        print("Implement statetour")
    e = timeit.default_timer()
#    print e-s
    for i in range(len(solution[2])-1):
        c1 = solution[2][i]
        c2 = solution[2][i+1]
        d = data[c1]['to_city'][c2]['distance']
        t = int(data[c1]['to_city'][c2]['time']*60)
        h = data[c1]['to_city'][c2]['highway']
        print "Travel via", h, "from", c1, "to", c2, "for", t, "minutes", "(", d, "miles )"
    print solution[0], solution[1] , ' '.join(solution[2])

except TypeError:
    print("No route found!")


