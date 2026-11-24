import random
import math

floor = [[0,3,2,0], [0,0,0,0], [0,4,6,8], [7,9,0,10]]

def hillClimbing(start):
    curr = start
    r,c = start
    currScore = floor[r][c]
    path = [curr]

    while True:
        neighbors = generate_neighbors(curr)
        bestNeighbor = None
        bestNeighborScore = currScore

        for neighbor in neighbors:
            r,c = neighbor
            score = floor[r][c]
            if score > bestNeighborScore:
                bestNeighbor = neighbor
                bestNeighborScore = score
        if bestNeighbor is None:
            break

        curr = bestNeighbor
        path.append(curr)
        currScore = bestNeighborScore

    return path



# solution is a location (x,y)
def generate_neighbors(solution): # returns a tuple of locations
    res = []

    if solution[0] != 0:
        res.append((solution[0] - 1, solution[1]))
    if solution[0] != 3:
        res.append((solution[0] + 1, solution[1]))
    if solution[1] != 0:
        res.append((solution[0], solution[1] - 1))
    if solution[1] != 3:
        res.append((solution[0], solution[1] + 1))

    return res

print(hillClimbing((0,3)))