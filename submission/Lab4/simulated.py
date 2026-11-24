import random
import math

floor = [[0,3,2,0], [3,6,0,0], [5,0,6,8], [4,9,0,10]]

def simulatedAnnealing(start, temperature, coolingRate):
    currLoc = start
    r,c = start
    currScore = floor[r][c]

    path = [currLoc]
    dirtiest = currLoc
    bestScore = currScore

    step = 1

    print(f"Step = {step} | Path = {path} | Dirtiest = {bestScore}\n")

    while temperature > 0.1:
        newLoc = currLoc

        neighbors = generate_neighbors(currLoc)
        i = random.sample(range(len(neighbors)), 1)
        newLoc = neighbors[i[0]]
        r,c = newLoc

        score = floor[r][c]

        if score > currScore or random.random() < math.exp((score - currScore) / temperature):
            path.append(newLoc)
            currLoc = newLoc
            currScore = score
            if score > bestScore:
                dirtiest = newLoc
                bestScore = score

        temperature *= coolingRate
        step += 1
        print(f"Step = {step} | Path = {path} | Dirtiest = {bestScore}\n")
    
    return dirtiest


# solution is a location (x,y)
def generate_neighbors(loc): # returns a tuple of locations
    res = []

    if loc[0] != 0:
        res.append((loc[0] - 1, loc[1]))
    if loc[0] != 3:
        res.append((loc[0] + 1, loc[1]))
    if loc[1] != 0:
        res.append((loc[0], loc[1] - 1))
    if loc[1] != 3:
        res.append((loc[0], loc[1] + 1))

    return res    

result = simulatedAnnealing((0,3), 1000, 0.8)
print("Dirtiest spot:", result)

