# Search-Based Test Generation - Part 1


```python
import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import inspect
import ast
import astor

import sys

# For presenting as slides
#plt.rcParams['figure.figsize'] = [12, 8]
#plt.rcParams.update({'font.size': 22})
#plt.rcParams['lines.linewidth'] = 3
```

## The Test Data Generation Problem

The problem we will consider in this chapter is the following: Given an entry point to a program (a function), we want to find values for the parameters of this function such that the execution of the function reaches a particular point in the program. In other words, we aim to find a test input to the program that covers a target statement. We will then generalise this problem to finding test inputs to cover _all_ statements in the program.

Assume we are aiming to test the following function under test:


```python
def test_me(x, y):
    if x == 2 * (y + 1):
        return True
    else:
        return False
```

The `test_me` function has two input parameters, `x`and `y`, and it returns `True` or `False` depending on how the parameters relate:


```python
test_me(10, 10)
```




    False




```python
test_me(22, 10)
```




    True



In order to address the test generation problem as a search problem, we need to decide on an encoding, and derive appropriate search operators. It is possible to use bitvectors like we did on previous problems; however, a simpler interpretation of the parameters of a function is a list of the actual parameters. That is, we encode a test input as a list of parameters; we will start by assuming that all parameters are numeric.

The representation for inputs of this function is lists of length two, one element for `x` and one for `y`. As numeric values in Python are unbounded, we need to decide on some finite bounds for these values, e.g.:


```python
MAX = 1000
MIN = -MAX
```

For generating inputs we can now uniformly sample in the range (MIN, MAX). The length of the vector shall be the number of parameters of the function under test. Rather than hard coding such a parameter, we can also make our approach generalise better by using inspection to determine how many parameters the function under test has:


```python
from inspect import signature
sig = signature(test_me)
num_parameters = len(sig.parameters)
num_parameters
```




    2



As usual, we will define the representation implicitly using a function that produces random instances.


```python
def get_random_individual():
    return [random.randint(MIN, MAX) for _ in range(num_parameters)]
```


```python
get_random_individual()
```




    [-426, -89]



We need to define search operators matching this representation. To apply local search, we need to define the neighbourhood. For example, we could define one upper and one lower neighbour for each parameter:

- `x-1, y`
- `x+1, y`
- `x, y+1`
- `x, y-1`


```python
def get_neighbours(individual):
    neighbours = []
    for p in range(len(individual)): 
        if individual[p] > MIN:
            neighbour = individual[:]
            neighbour[p] = individual[p] - 1
            neighbours.append(neighbour)
        if individual[p] < MAX:
            neighbour = individual[:]
            neighbour[p] = individual[p] + 1
            neighbours.append(neighbour)
            
    return neighbours
```


```python
x = get_random_individual()
x
```




    [-391, -973]




```python
get_neighbours(x)
```




    [[-392, -973], [-390, -973], [-391, -974], [-391, -972]]



Before we can apply search, we also need to define a fitness function.  Suppose that we are interested in covering the `True` branch of the if-condition in the `test_me()` function, i.e. `x == 2 * (y + 1)`.


```python
def test_me(x, y):
    if x == 2 * (y + 1):
        return True
    else:
        return False
```

How close is a given input tuple for this function from reaching the target (true) branch of `x == 2 * (y + 1)`?

Let's consider an arbitrary point in the search space, e.g. `(274, 153)`. The if-condition compares the following values:


```python
x = 274
y = 153
x, 2 * (y + 1)
```




    (274, 308)



In order to make the branch true, both values need to be the same. Thus, the more they differ, the further we are away from making the comparison true, and the less they differ, the closer we are from making the comparison true. Thus, we can quantify "how false" the comparison is by calculating the difference between `x` and `2 * (y + 1)`. Thus, we can calculate this distance as `abs(x - 2 * (y + 1))`:


```python
def calculate_distance(x, y):
    return abs(x - 2 * (y + 1))
```


```python
calculate_distance(274, 153)
```




    34



We can use this distance value as our fitness function, since we can nicely measure how close we are to an optimal solution. Note, however, that "better" doesn't mean "bigger" in this case; the smaller the distance the better. This is not a problem, since any algorithm that can maximize a value can also be made to minimize it instead.

For each value in the search space of integer tuples, this distance value defines the elevation in our search landscape. Since our example search space is two-dimensional, the search landscape is three-dimensional and we can plot it to see what it looks like:


```python
x = np.outer(np.linspace(-10, 10, 30), np.ones(30))
y = x.copy().T
z = calculate_distance(x, y)

fig = plt.figure(figsize=(12, 12))
ax  = plt.axes(projection='3d')

ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0);
```


    
![png](6/output_30_0.png)
    


The optimal values, i.e. those that make the if-condition true, have fitness value 0 and can be clearly seen at the bottom of the plot. The further away from the optimal values, the higher elevated the points in the search space.

This distance can serve as our fitness function if we aim to cover the true branch of the program in our example:


```python
def get_fitness(individual):
    x = individual[0]
    y = individual[1]
    return abs(x - 2 * (y + 1))
```

We can now use any local search algorithm we have defined previously, with only one modification: In the prior examples where we applied local search we were always maximising fitness values; now we are minimising, so a hillclimber, for example, should only move to neighbours with _smaller_ fitness values:


```python
max_steps = 10000
fitness_values = []
```

Let's use a steepest ascent hillclimber:


```python
def hillclimbing():
    current = get_random_individual()
    fitness = get_fitness(current)
    best, best_fitness = current[:], fitness
    print(f"Starting at fitness {best_fitness}: {best}")

    step = 0
    while step < max_steps and best_fitness > 0:
        neighbours = [(x, get_fitness(x)) for x in get_neighbours(current)]
        best_neighbour, neighbour_fitness = min(neighbours, key=lambda i: i[1])
        step += len(neighbours)        
        fitness_values.extend([best_fitness] * len(neighbours))
        if neighbour_fitness < fitness:
            current = best_neighbour
            fitness = neighbour_fitness
            if fitness < best_fitness:
                best = current[:]
                best_fitness = fitness
        else:
            # Random restart if no neighbour is better
            current = get_random_individual()
            fitness = get_fitness(current)
            step += 1
            if fitness < best_fitness:
                best = current[:]
                best_fitness = fitness
            fitness_values.append(best_fitness)


    print(f"Solution fitness after {step} fitness evaluations: {best_fitness}: {best}")
    return best
```


```python
max_steps = 10000
fitness_values = []
hillclimbing()
```

    Starting at fitness 2685: [791, -948]
    Solution fitness after 5372 fitness evaluations: 0: [790, 394]





    [790, 394]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10fb7d2a0>]




    
![png](6/output_39_1.png)
    


Since there are no local optima, the hillclimber will easily find the solution, even without restarts. However, this can take a while, in particular if we use a larger input space:


```python
MAX = 100000
MIN = -MAX
```


```python
fitness_values = []
hillclimbing()
plt.plot(fitness_values)
```

    Starting at fitness 172725: [72171, -50278]
    Solution fitness after 10000 fitness evaluations: 167725: [72171, -47778]





    [<matplotlib.lines.Line2D at 0x10fbfc850>]




    
![png](6/output_42_2.png)
    


Unless the randomly chosen initial point is already close to an optimal solution, the hillclimber is going to be hopeless in moving through the search space within a reasonable number of iterations.

## Alternating Variable Method

The search problem represented by the `test_me` function has an easy fitness landscape with no local optima. However, it still takes quite long to reach the optimum, depending on where the random starting point lies in the search space. This is because the neighbourhood for real program inputs can be quite large, depending on the number of parameters, and even just the search space for each parameter individually can already be very large. In our example we restricted `MAX` and `MIN` to a very narrow range, but imagine doing this for 32 bit integers. Both these problems are addressed with an adapted version of our hillclimber known as the _Alternating Variable Method_, which differs from the hillclimber in two ways: 
1. Rather than considering the neighbourhood of all input parameters at once, we apply search to each parameter individually in turn
2. Rather than taking only small steps of size 1, we allow larger jumps in the search space.

Let's first consider the second aspect, larger jumps in the search space. The idea is to apply a _pattern_ search where we first decide on a direction in the search space to move, and then apply increasingly larger steps in that direction as long as the fitness improves. We only consider a single parameter, thus the "direction" simply refers to whether one increases or decreases this value. The function thus takes (1) the individual on which to perform the search, (2) a particular parameter we are considering, (3) the direction of the search, and (4) the starting fitness values. 


```python
def pattern_search(individual, parameter, direction, fitness):
    print(f"  {individual}, direction {direction}, fitness {fitness}")

    individual[parameter] = individual[parameter] + direction
    new_fitness = get_fitness(individual)
    if new_fitness < fitness:
        fitness_values.append(new_fitness)
        return pattern_search(individual, parameter, 2 * direction, new_fitness)
    else:
        # If fitness is not better we overshot. Undo last move, and return
        fitness_values.append(fitness)
        individual[parameter] = individual[parameter] - direction
        return fitness
```

For example, let's assume `y` is a large value (1000), and `x` is considerably smaller. For our example function, the optimal value for `x` would thus be at 2002. Applying the search to `x` we thus need to move in the positive direction (`1`), and the function will do this with increasing steps until it "overshoots".


```python
x = [0, 1000]
f = get_fitness(x)
pattern_search(x, 0, 1, get_fitness(x))
```

      [0, 1000], direction 1, fitness 2002
      [1, 1000], direction 2, fitness 2001
      [3, 1000], direction 4, fitness 1999
      [7, 1000], direction 8, fitness 1995
      [15, 1000], direction 16, fitness 1987
      [31, 1000], direction 32, fitness 1971
      [63, 1000], direction 64, fitness 1939
      [127, 1000], direction 128, fitness 1875
      [255, 1000], direction 256, fitness 1747
      [511, 1000], direction 512, fitness 1491
      [1023, 1000], direction 1024, fitness 979
      [2047, 1000], direction 2048, fitness 45





    45



If `x` is larger than `y` we would need to move in the other direction, and the search does this until it undershoots the target of 2002:


```python
x = [10000, 1000]
f = get_fitness(x)
pattern_search(x, 0, -1, get_fitness(x))
```

      [10000, 1000], direction -1, fitness 7998
      [9999, 1000], direction -2, fitness 7997
      [9997, 1000], direction -4, fitness 7995
      [9993, 1000], direction -8, fitness 7991
      [9985, 1000], direction -16, fitness 7983
      [9969, 1000], direction -32, fitness 7967
      [9937, 1000], direction -64, fitness 7935
      [9873, 1000], direction -128, fitness 7871
      [9745, 1000], direction -256, fitness 7743
      [9489, 1000], direction -512, fitness 7487
      [8977, 1000], direction -1024, fitness 6975
      [7953, 1000], direction -2048, fitness 5951
      [5905, 1000], direction -4096, fitness 3903
      [1809, 1000], direction -8192, fitness 193





    193



The AVM algorithm applies the pattern search as follows:
1. Start with the first parameter
2. Probe the neighbourhood of the parameter to find the direction of the search
3. Apply pattern search in that direction
4. Repeat probing + pattern search until no more improvement can be made
5. Move to the next parameter, and go to step 2

Like a regular hillclimber, the search may get stuck in local optima and needs to use random restarts. The algorithm is stuck if it probed all parameters in sequence and none of the parameters allowed a move that improved fitness.


```python
def probe_and_search(individual, parameter, fitness):
    new_parameters = individual[:]
    value = new_parameters[parameter]
    new_fitness = fitness
    # Try +1
    new_parameters[parameter] = individual[parameter] + 1
    print(f"Trying +1 at fitness {fitness}: {new_parameters}")
    new_fitness = get_fitness(new_parameters)
    if new_fitness < fitness:
        fitness_values.append(new_fitness)
        new_fitness = pattern_search(new_parameters, parameter, 2, new_fitness)
    else:
        # Try -1
        fitness_values.append(fitness)
        new_parameters[parameter] = individual[parameter] - 1
        print(f"Trying -1 at fitness {fitness}: {new_parameters}")
        new_fitness = get_fitness(new_parameters)
        if new_fitness < fitness:
            fitness_values.append(new_fitness)
            new_fitness = pattern_search(new_parameters, parameter, -2, new_fitness)
        else:
            fitness_values.append(fitness)
            new_parameters[parameter] = individual[parameter]
            new_fitness = fitness
            
    return new_parameters, new_fitness
```


```python
def avm():
    current = get_random_individual()
    fitness = get_fitness(current)
    best, best_fitness = current[:], fitness
    fitness_values.append(best_fitness)    
    print(f"Starting at fitness {best_fitness}: {current}")

    changed = True
    while len(fitness_values) < max_steps and best_fitness > 0:
        # Random restart
        if not changed: 
            current = get_random_individual()
            fitness = get_fitness(current)
            fitness_values.append(fitness)
        changed = False
            
        parameter = 0
        while parameter < len(current):
            print(f"Current parameter: {parameter}")
            new_parameters, new_fitness = probe_and_search(current, parameter, fitness)
            if current != new_parameters:
                # Keep on searching
                changed = True
                current = new_parameters
                fitness = new_fitness
                if fitness < best_fitness:
                    best_fitness = fitness
                    best = current[:]
            else:
                parameter += 1

    print(f"Solution fitness {best_fitness}: {best}")
    return best
```


```python
fitness_values = []
avm()
```

    Starting at fitness 190741: [25079, -82832]
    Current parameter: 0
    Trying +1 at fitness 190741: [25080, -82832]
    Trying -1 at fitness 190741: [25078, -82832]
      [25078, -82832], direction -2, fitness 190740
      [25076, -82832], direction -4, fitness 190738
      [25072, -82832], direction -8, fitness 190734
      [25064, -82832], direction -16, fitness 190726
      [25048, -82832], direction -32, fitness 190710
      [25016, -82832], direction -64, fitness 190678
      [24952, -82832], direction -128, fitness 190614
      [24824, -82832], direction -256, fitness 190486
      [24568, -82832], direction -512, fitness 190230
      [24056, -82832], direction -1024, fitness 189718
      [23032, -82832], direction -2048, fitness 188694
      [20984, -82832], direction -4096, fitness 186646
      [16888, -82832], direction -8192, fitness 182550
      [8696, -82832], direction -16384, fitness 174358
      [-7688, -82832], direction -32768, fitness 157974
      [-40456, -82832], direction -65536, fitness 125206
      [-105992, -82832], direction -131072, fitness 59670
    Current parameter: 0
    Trying +1 at fitness 59670: [-105991, -82832]
    Trying -1 at fitness 59670: [-105993, -82832]
      [-105993, -82832], direction -2, fitness 59669
      [-105995, -82832], direction -4, fitness 59667
      [-105999, -82832], direction -8, fitness 59663
      [-106007, -82832], direction -16, fitness 59655
      [-106023, -82832], direction -32, fitness 59639
      [-106055, -82832], direction -64, fitness 59607
      [-106119, -82832], direction -128, fitness 59543
      [-106247, -82832], direction -256, fitness 59415
      [-106503, -82832], direction -512, fitness 59159
      [-107015, -82832], direction -1024, fitness 58647
      [-108039, -82832], direction -2048, fitness 57623
      [-110087, -82832], direction -4096, fitness 55575
      [-114183, -82832], direction -8192, fitness 51479
      [-122375, -82832], direction -16384, fitness 43287
      [-138759, -82832], direction -32768, fitness 26903
      [-171527, -82832], direction -65536, fitness 5865
    Current parameter: 0
    Trying +1 at fitness 5865: [-171526, -82832]
      [-171526, -82832], direction 2, fitness 5864
      [-171524, -82832], direction 4, fitness 5862
      [-171520, -82832], direction 8, fitness 5858
      [-171512, -82832], direction 16, fitness 5850
      [-171496, -82832], direction 32, fitness 5834
      [-171464, -82832], direction 64, fitness 5802
      [-171400, -82832], direction 128, fitness 5738
      [-171272, -82832], direction 256, fitness 5610
      [-171016, -82832], direction 512, fitness 5354
      [-170504, -82832], direction 1024, fitness 4842
      [-169480, -82832], direction 2048, fitness 3818
      [-167432, -82832], direction 4096, fitness 1770
    Current parameter: 0
    Trying +1 at fitness 1770: [-167431, -82832]
      [-167431, -82832], direction 2, fitness 1769
      [-167429, -82832], direction 4, fitness 1767
      [-167425, -82832], direction 8, fitness 1763
      [-167417, -82832], direction 16, fitness 1755
      [-167401, -82832], direction 32, fitness 1739
      [-167369, -82832], direction 64, fitness 1707
      [-167305, -82832], direction 128, fitness 1643
      [-167177, -82832], direction 256, fitness 1515
      [-166921, -82832], direction 512, fitness 1259
      [-166409, -82832], direction 1024, fitness 747
      [-165385, -82832], direction 2048, fitness 277
    Current parameter: 0
    Trying +1 at fitness 277: [-165384, -82832]
    Trying -1 at fitness 277: [-165386, -82832]
      [-165386, -82832], direction -2, fitness 276
      [-165388, -82832], direction -4, fitness 274
      [-165392, -82832], direction -8, fitness 270
      [-165400, -82832], direction -16, fitness 262
      [-165416, -82832], direction -32, fitness 246
      [-165448, -82832], direction -64, fitness 214
      [-165512, -82832], direction -128, fitness 150
      [-165640, -82832], direction -256, fitness 22
    Current parameter: 0
    Trying +1 at fitness 22: [-165639, -82832]
    Trying -1 at fitness 22: [-165641, -82832]
      [-165641, -82832], direction -2, fitness 21
      [-165643, -82832], direction -4, fitness 19
      [-165647, -82832], direction -8, fitness 15
      [-165655, -82832], direction -16, fitness 7
    Current parameter: 0
    Trying +1 at fitness 7: [-165654, -82832]
    Trying -1 at fitness 7: [-165656, -82832]
      [-165656, -82832], direction -2, fitness 6
      [-165658, -82832], direction -4, fitness 4
      [-165662, -82832], direction -8, fitness 0
    Current parameter: 0
    Trying +1 at fitness 0: [-165661, -82832]
    Trying -1 at fitness 0: [-165663, -82832]
    Current parameter: 1
    Trying +1 at fitness 0: [-165662, -82831]
    Trying -1 at fitness 0: [-165662, -82833]
    Solution fitness 0: [-165662, -82832]





    [-165662, -82832]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10fc60bb0>]




    
![png](6/output_56_1.png)
    


The pattern search even works efficiently if we increase the size of the search space to 64-bit numbers:


```python
MAX = 2**32
MIN = -MAX
```


```python
fitness_values = []
avm()
```

    Starting at fitness 1198636659: [2195987891, 498675615]
    Current parameter: 0
    Trying +1 at fitness 1198636659: [2195987892, 498675615]
    Trying -1 at fitness 1198636659: [2195987890, 498675615]
      [2195987890, 498675615], direction -2, fitness 1198636658
      [2195987888, 498675615], direction -4, fitness 1198636656
      [2195987884, 498675615], direction -8, fitness 1198636652
      [2195987876, 498675615], direction -16, fitness 1198636644
      [2195987860, 498675615], direction -32, fitness 1198636628
      [2195987828, 498675615], direction -64, fitness 1198636596
      [2195987764, 498675615], direction -128, fitness 1198636532
      [2195987636, 498675615], direction -256, fitness 1198636404
      [2195987380, 498675615], direction -512, fitness 1198636148
      [2195986868, 498675615], direction -1024, fitness 1198635636
      [2195985844, 498675615], direction -2048, fitness 1198634612
      [2195983796, 498675615], direction -4096, fitness 1198632564
      [2195979700, 498675615], direction -8192, fitness 1198628468
      [2195971508, 498675615], direction -16384, fitness 1198620276
      [2195955124, 498675615], direction -32768, fitness 1198603892
      [2195922356, 498675615], direction -65536, fitness 1198571124
      [2195856820, 498675615], direction -131072, fitness 1198505588
      [2195725748, 498675615], direction -262144, fitness 1198374516
      [2195463604, 498675615], direction -524288, fitness 1198112372
      [2194939316, 498675615], direction -1048576, fitness 1197588084
      [2193890740, 498675615], direction -2097152, fitness 1196539508
      [2191793588, 498675615], direction -4194304, fitness 1194442356
      [2187599284, 498675615], direction -8388608, fitness 1190248052
      [2179210676, 498675615], direction -16777216, fitness 1181859444
      [2162433460, 498675615], direction -33554432, fitness 1165082228
      [2128879028, 498675615], direction -67108864, fitness 1131527796
      [2061770164, 498675615], direction -134217728, fitness 1064418932
      [1927552436, 498675615], direction -268435456, fitness 930201204
      [1659116980, 498675615], direction -536870912, fitness 661765748
      [1122246068, 498675615], direction -1073741824, fitness 124894836
    Current parameter: 0
    Trying +1 at fitness 124894836: [1122246069, 498675615]
    Trying -1 at fitness 124894836: [1122246067, 498675615]
      [1122246067, 498675615], direction -2, fitness 124894835
      [1122246065, 498675615], direction -4, fitness 124894833
      [1122246061, 498675615], direction -8, fitness 124894829
      [1122246053, 498675615], direction -16, fitness 124894821
      [1122246037, 498675615], direction -32, fitness 124894805
      [1122246005, 498675615], direction -64, fitness 124894773
      [1122245941, 498675615], direction -128, fitness 124894709
      [1122245813, 498675615], direction -256, fitness 124894581
      [1122245557, 498675615], direction -512, fitness 124894325
      [1122245045, 498675615], direction -1024, fitness 124893813
      [1122244021, 498675615], direction -2048, fitness 124892789
      [1122241973, 498675615], direction -4096, fitness 124890741
      [1122237877, 498675615], direction -8192, fitness 124886645
      [1122229685, 498675615], direction -16384, fitness 124878453
      [1122213301, 498675615], direction -32768, fitness 124862069
      [1122180533, 498675615], direction -65536, fitness 124829301
      [1122114997, 498675615], direction -131072, fitness 124763765
      [1121983925, 498675615], direction -262144, fitness 124632693
      [1121721781, 498675615], direction -524288, fitness 124370549
      [1121197493, 498675615], direction -1048576, fitness 123846261
      [1120148917, 498675615], direction -2097152, fitness 122797685
      [1118051765, 498675615], direction -4194304, fitness 120700533
      [1113857461, 498675615], direction -8388608, fitness 116506229
      [1105468853, 498675615], direction -16777216, fitness 108117621
      [1088691637, 498675615], direction -33554432, fitness 91340405
      [1055137205, 498675615], direction -67108864, fitness 57785973
      [988028341, 498675615], direction -134217728, fitness 9322891
    Current parameter: 0
    Trying +1 at fitness 9322891: [988028342, 498675615]
      [988028342, 498675615], direction 2, fitness 9322890
      [988028344, 498675615], direction 4, fitness 9322888
      [988028348, 498675615], direction 8, fitness 9322884
      [988028356, 498675615], direction 16, fitness 9322876
      [988028372, 498675615], direction 32, fitness 9322860
      [988028404, 498675615], direction 64, fitness 9322828
      [988028468, 498675615], direction 128, fitness 9322764
      [988028596, 498675615], direction 256, fitness 9322636
      [988028852, 498675615], direction 512, fitness 9322380
      [988029364, 498675615], direction 1024, fitness 9321868
      [988030388, 498675615], direction 2048, fitness 9320844
      [988032436, 498675615], direction 4096, fitness 9318796
      [988036532, 498675615], direction 8192, fitness 9314700
      [988044724, 498675615], direction 16384, fitness 9306508
      [988061108, 498675615], direction 32768, fitness 9290124
      [988093876, 498675615], direction 65536, fitness 9257356
      [988159412, 498675615], direction 131072, fitness 9191820
      [988290484, 498675615], direction 262144, fitness 9060748
      [988552628, 498675615], direction 524288, fitness 8798604
      [989076916, 498675615], direction 1048576, fitness 8274316
      [990125492, 498675615], direction 2097152, fitness 7225740
      [992222644, 498675615], direction 4194304, fitness 5128588
      [996416948, 498675615], direction 8388608, fitness 934284
    Current parameter: 0
    Trying +1 at fitness 934284: [996416949, 498675615]
      [996416949, 498675615], direction 2, fitness 934283
      [996416951, 498675615], direction 4, fitness 934281
      [996416955, 498675615], direction 8, fitness 934277
      [996416963, 498675615], direction 16, fitness 934269
      [996416979, 498675615], direction 32, fitness 934253
      [996417011, 498675615], direction 64, fitness 934221
      [996417075, 498675615], direction 128, fitness 934157
      [996417203, 498675615], direction 256, fitness 934029
      [996417459, 498675615], direction 512, fitness 933773
      [996417971, 498675615], direction 1024, fitness 933261
      [996418995, 498675615], direction 2048, fitness 932237
      [996421043, 498675615], direction 4096, fitness 930189
      [996425139, 498675615], direction 8192, fitness 926093
      [996433331, 498675615], direction 16384, fitness 917901
      [996449715, 498675615], direction 32768, fitness 901517
      [996482483, 498675615], direction 65536, fitness 868749
      [996548019, 498675615], direction 131072, fitness 803213
      [996679091, 498675615], direction 262144, fitness 672141
      [996941235, 498675615], direction 524288, fitness 409997
      [997465523, 498675615], direction 1048576, fitness 114291
    Current parameter: 0
    Trying +1 at fitness 114291: [997465524, 498675615]
    Trying -1 at fitness 114291: [997465522, 498675615]
      [997465522, 498675615], direction -2, fitness 114290
      [997465520, 498675615], direction -4, fitness 114288
      [997465516, 498675615], direction -8, fitness 114284
      [997465508, 498675615], direction -16, fitness 114276
      [997465492, 498675615], direction -32, fitness 114260
      [997465460, 498675615], direction -64, fitness 114228
      [997465396, 498675615], direction -128, fitness 114164
      [997465268, 498675615], direction -256, fitness 114036
      [997465012, 498675615], direction -512, fitness 113780
      [997464500, 498675615], direction -1024, fitness 113268
      [997463476, 498675615], direction -2048, fitness 112244
      [997461428, 498675615], direction -4096, fitness 110196
      [997457332, 498675615], direction -8192, fitness 106100
      [997449140, 498675615], direction -16384, fitness 97908
      [997432756, 498675615], direction -32768, fitness 81524
      [997399988, 498675615], direction -65536, fitness 48756
      [997334452, 498675615], direction -131072, fitness 16780
    Current parameter: 0
    Trying +1 at fitness 16780: [997334453, 498675615]
      [997334453, 498675615], direction 2, fitness 16779
      [997334455, 498675615], direction 4, fitness 16777
      [997334459, 498675615], direction 8, fitness 16773
      [997334467, 498675615], direction 16, fitness 16765
      [997334483, 498675615], direction 32, fitness 16749
      [997334515, 498675615], direction 64, fitness 16717
      [997334579, 498675615], direction 128, fitness 16653
      [997334707, 498675615], direction 256, fitness 16525
      [997334963, 498675615], direction 512, fitness 16269
      [997335475, 498675615], direction 1024, fitness 15757
      [997336499, 498675615], direction 2048, fitness 14733
      [997338547, 498675615], direction 4096, fitness 12685
      [997342643, 498675615], direction 8192, fitness 8589
      [997350835, 498675615], direction 16384, fitness 397
    Current parameter: 0
    Trying +1 at fitness 397: [997350836, 498675615]
      [997350836, 498675615], direction 2, fitness 396
      [997350838, 498675615], direction 4, fitness 394
      [997350842, 498675615], direction 8, fitness 390
      [997350850, 498675615], direction 16, fitness 382
      [997350866, 498675615], direction 32, fitness 366
      [997350898, 498675615], direction 64, fitness 334
      [997350962, 498675615], direction 128, fitness 270
      [997351090, 498675615], direction 256, fitness 142
      [997351346, 498675615], direction 512, fitness 114
    Current parameter: 0
    Trying +1 at fitness 114: [997351347, 498675615]
    Trying -1 at fitness 114: [997351345, 498675615]
      [997351345, 498675615], direction -2, fitness 113
      [997351343, 498675615], direction -4, fitness 111
      [997351339, 498675615], direction -8, fitness 107
      [997351331, 498675615], direction -16, fitness 99
      [997351315, 498675615], direction -32, fitness 83
      [997351283, 498675615], direction -64, fitness 51
      [997351219, 498675615], direction -128, fitness 13
    Current parameter: 0
    Trying +1 at fitness 13: [997351220, 498675615]
      [997351220, 498675615], direction 2, fitness 12
      [997351222, 498675615], direction 4, fitness 10
      [997351226, 498675615], direction 8, fitness 6
      [997351234, 498675615], direction 16, fitness 2
    Current parameter: 0
    Trying +1 at fitness 2: [997351235, 498675615]
    Trying -1 at fitness 2: [997351233, 498675615]
      [997351233, 498675615], direction -2, fitness 1
    Current parameter: 0
    Trying +1 at fitness 1: [997351234, 498675615]
    Trying -1 at fitness 1: [997351232, 498675615]
      [997351232, 498675615], direction -2, fitness 0
    Current parameter: 0
    Trying +1 at fitness 0: [997351233, 498675615]
    Trying -1 at fitness 0: [997351231, 498675615]
    Current parameter: 1
    Trying +1 at fitness 0: [997351232, 498675616]
    Trying -1 at fitness 0: [997351232, 498675614]
    Solution fitness 0: [997351232, 498675615]





    [997351232, 498675615]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10fcac220>]




    
![png](6/output_60_1.png)
    


## Program Instrumentation

Deriving fitness functions is not quite so easy. Of course we could come up with an equation that captures the relation between the sides of the triangle, but then essentially we would need to reproduce the entire program logic again in a function, which certainly does not help generalising to other programs. For example, consider how the fitness function would look like if the comparison was not made on the input parameters, but on values derived through computation within the function under test. Ideally, what we would want is to be able to pick a point in the program and come up with a fitness function automatically that describes how close we are to reaching this point.

There are two central ideas in order to achieve this:

- First, rather than trying to guess how close a program inputs gets to a target statement, we simply _run_ the program with the input and observe how close it actually gets.

- Second, during the execution we keep track of distance estimates like the one we calculated for the `test_me` function whenever we come across conditional statements.

In order to observe what an execution does, we need to *instrument* the program: We add new code immediately before or after the branching condition to keep track of the values observed and calculate the distance using these values.

Let's first consider what is done here conceptually. We first define a global variable in which we will store the distance, so that we can access it after the execution:


```python
distance = 0
```

Now the instrumented version just has to update the global variable immediately before executing the branching condition:


```python
def test_me_instrumented(x, y):
    global distance
    distance = abs(x - 2 * (y + 1))
    if x == 2 * (y + 1):
        return True
    else:
        return False
```

Let's try this out for a couple of example values:


```python
test_me_instrumented(0, 0)
distance
```




    2




```python
test_me_instrumented(22, 10)
distance
```




    0



Using this instrumented version of `test_me()`, we can define a fitness function which simply calculates the distance for the condition being true:


```python
def get_fitness(individual):
    global distance
    test_me_instrumented(*individual)
    fitness = distance
    return fitness
```

Let's try this on some example inputs:


```python
get_fitness([0, 0])
```




    2



When we have reached the target branch, the distance will be 0:


```python
get_fitness([22, 10])
```




    0



When implementing the instrumentation, we need to consider that the branching condition may have side-effects. For example, suppose that the branching condition were `x == 2 * foo(y)`, where `foo()` is a function that takes an integer as input. Naively instrumenting would lead to the following code:

```
    distance = abs(x - 2 * foo(y))
	if x == 2 * foo(y):
	...
```

Thus, the instrumentation would lead to `foo()` being executed *twice*. Suppose `foo()` changes the state of the system (e.g., by printing something, accessing the file system, changing some state variables, etc.), then clearly invoking `foo()` a second time is a bad idea. One way to overcome this problem is to _transform_ the conditions, rather than _adding_ tracing calls. For example, one can create temporary variables that hold the values necessary for the distance calculation and then use these in the branching condition:

```
	tmp1 = x
	tmp2 = 2 * foo(y)
	distance = compute_distance(tmp1, tmp2)
	if tmp1 == tmp2:
	...
```


```python
def evaluate_equals(op1, op2):
    global distance
    distance = abs(op1 - op2)
    if distance == 0:
        return True
    else:
        return False
```

Now the aim would be to transform the program automatically such that it looks like so:


```python
def test_me_instrumented(x, y):
    tmp1 = x
    tmp2 = 2 * (y + 1)    
    if evaluate_equals(tmp1, tmp2):
        return True
    else:
        return False
```

Replacing comparisons automatically is actually quite easy in Python, using the abstract syntax tree (AST) of the program. In the AST, a comparison will typically be a tree node with an operator attribute and two children for the left-hand and right-hand operators. To replace such comparisons with a call to `calculate_distance()` one simply needs to replace the comparison node in the AST with a function call node, and this is what the `BranchTransformer` class does using a NodeTransformer from Python's `ast` module:


```python
import ast
```


```python
class BranchTransformer(ast.NodeTransformer):

    def visit_FunctionDef(self, node):
        node.name = node.name + "_instrumented"
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if not isinstance(node.ops[0], ast.Eq):
            return node

        return ast.Call(func=ast.Name("evaluate_equals", ast.Load()),
                        args=[node.left,
                              node.comparators[0]],
                        keywords=[],
                        starargs=None,
                        kwargs=None)
```

The `BranchTransformer` parses a target Python program using the built-in parser `ast.parse()`, which returns the AST. Python provides an API to traverse and modify this AST. To replace the comparison with a function call we use an `ast.NodeTransformer`, which uses the visitor pattern where there is one `visit_*` function for each type of node in the AST. As we are interested in replacing comparisons, we override `visit_Compare`, where instead of the original comparison node we return a new node of type `ast.Func`, which is a function call node. The first parameter of this node is the name of the function `calculate_distance`, and the arguments are the two operands that our `calculate_distance` function expects.

You will notice that we also override `visit_FunctionDef`; this is just to change the name of the method by appending `_instrumented`, so that we can continue to use the original function together with the instrumented one.

The following code parses the source code of the `test_me()` function to an AST, then transforms it, and prints it out again (using the `to_source()` function from the `astor` library):


```python
import inspect
import ast
import astor
```


```python
source = inspect.getsource(test_me)
node = ast.parse(source)
BranchTransformer().visit(node)

# Make sure the line numbers are ok before printing
node = ast.fix_missing_locations(node)
print(astor.to_source(node))
```

    def test_me_instrumented(x, y):
        if evaluate_equals(x, 2 * (y + 1)):
            return True
        else:
            return False
    


To calculate a fitness value with the instrumented version, we need to compile the instrumented AST again, which is done using Python's `compile()` function. We then need to make the compiled function accessible, for which we first retrieve the current module from `sys.modules`, and then add the compiled code of the instrumented function to the list of functions of the current module using `exec`. After this, the `cgi_decode_instrumented()` function can be accessed.


```python
import sys
```


```python
def create_instrumented_function(f):
    source = inspect.getsource(f)
    node = ast.parse(source)
    node = BranchTransformer().visit(node)

    # Make sure the line numbers are ok so that it compiles
    node = ast.fix_missing_locations(node)

    # Compile and add the instrumented function to the current module
    current_module = sys.modules[__name__]
    code = compile(node, filename="<ast>", mode="exec")
    exec(code, current_module.__dict__)
```


```python
create_instrumented_function(test_me)
```


```python
test_me_instrumented(0, 0)
```




    False




```python
distance
```




    2




```python
test_me_instrumented(22, 10)
```




    True




```python
distance
```




    0



The estimate for any relational comparison of two values is defined in terms of the _branch distance_. Our `evaluate_equals` function indeed implements the branch distance function for an equality comparison. To generalise this we need similar estimates for other types of relational comparisons. Furthermore, we also have to consider the distance to such conditions evaluating to false, not just to true. Thus, each if-condition actually has two distance estimates, one to estimate how close it is to being true, and one how close it is to being false. If the condition is true, then the true distance is 0; if the condition is false, then the false distance is 0. That is, in a comparison `a == b`, if `a` is smaller than `b`, then the false distance is `0` by definition. 

The following table shows how to calculate the distance for different types of comparisons:

| Condition | Distance True | Distance False |
| ------------- |:-------------:| -----:|
| a == b      | abs(a - b) | 1 |
| a != b      | 1          | abs(a - b) |
| a < b       | b - a + 1  | a - b      |
| a <= b      | b - a      | a - b + 1  |
| a > b       | a - b + 1  | b - a      |


Note that several of the calculations add a constant `1`. The reason for this is quite simple: Suppose we want to have `a < b` evaluate to true, and let `a = 27` and `b = 27`. The condition is not true, but simply taking the difference would give us a result of `0`. To avoid this, we have to add a constant value. It is not important whether this value is `1` -- any positive constant works.

We generalise our `evaluate_equals` function to an `evaluate_condition` function that takes the operator as an additional parameter, and then implements the above table. In contrast to the previous `calculate_equals`, we will now calculate both, the true and the false distance:


```python
def evaluate_condition(op, lhs, rhs):
    distance_true = 0
    distance_false = 0
    if op == "Eq":
        if lhs == rhs:
            distance_false = 1
        else:
            distance_true = abs(lhs - rhs)

    # ... code for other types of conditions

    if distance_true == 0:
        return True
    else:
        return False
```

Let's consider a slightly larger function under test. We will use the well known triangle example, originating in Glenford Meyer's classical Art of Software Testing book 


```python
def triangle(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return 4 # invalid
    
    if a + b <= c or a + c <= b or b + c <= a:
        return 4 # invalid
    
    if a == b and b == c:
        return 1 # equilateral
    
    if a == b or b == c or a == c:
        return 2 # isosceles
    
    return 3 # scalene
```

The function takes as input the length of the three sides of a triangle, and returns a number representing the type of triangle:


```python
triangle(4,4,4)
```




    1



Adapting our representation is easy, we just need to correctly set the number of parameters:


```python
sig = signature(triangle)
num_parameters = len(sig.parameters)
num_parameters
```




    3



For the `triangle` function, however, we have multiple if-conditions; we have to add instrumentation to each of these using `evaluate_condition`. We also need to generalise from our global `distance` variable, since we now have two distance values per branch, and potentially multiple branches. Furthermore, a condition might be executed multiple times within a single execution (e.g., if it is in a loop), so rather than storing all values, we will only keep the _minimum_ value observed for each condition:


```python
distances_true = {}
distances_false = {}
```


```python
def update_maps(condition_num, d_true, d_false):
    global distances_true, distances_false

    if condition_num in distances_true.keys():
        distances_true[condition_num] = min(distances_true[condition_num], d_true)
    else:
        distances_true[condition_num] = d_true

    if condition_num in distances_false.keys():
        distances_false[condition_num] = min(distances_false[condition_num], d_false)
    else:
        distances_false[condition_num] = d_false
```

Now we need to finish implementing the `evaluate_condition` function. We add yet another parameter to denote the ID of the branch we are instrumenting:


```python
def evaluate_condition(num, op, lhs, rhs):
    distance_true = 0
    distance_false = 0

    # Make sure the distance can be calculated on number and character
    # comparisons (needed for cgi_decode later)
    if isinstance(lhs, str):
        lhs = ord(lhs)
    if isinstance(rhs, str):
        rhs = ord(rhs)

    if op == "Eq":
        if lhs == rhs:
            distance_false = 1
        else:
            distance_true = abs(lhs - rhs)

    elif op == "Gt":
        if lhs > rhs:
            distance_false = lhs - rhs
        else:
            distance_true = rhs - lhs + 1
    elif op == "Lt":
        if lhs < rhs:
            distance_false = rhs - lhs
        else:
            distance_true = lhs - rhs + 1
    elif op == "LtE":
        if lhs <= rhs:
            distance_false = rhs - lhs + 1
        else:
            distance_true = lhs - rhs
    # ...
    # handle other comparison operators
    # ...

    elif op == "In":
        minimum = sys.maxsize
        for elem in rhs.keys():
            distance = abs(lhs - ord(elem))
            if distance < minimum:
                minimum = distance

        distance_true = minimum
        if distance_true == 0:
            distance_false = 1
    else:
        assert False

    update_maps(num, normalise(distance_true), normalise(distance_false))

    if distance_true == 0:
        return True
    else:
        return False
```

We need to normalise branch distances since different comparisons will be on different scales, and this would bias the search. We will use the normalisation function defined in the previous chapter:


```python
def normalise(x):
    return x / (1.0 + x)
```

We also need to extend our instrumentation function to take care of all comparisons, and not just equality comparisons:


```python
import ast
class BranchTransformer(ast.NodeTransformer):

    branch_num = 0

    def visit_FunctionDef(self, node):
        node.name = node.name + "_instrumented"
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if node.ops[0] in [ast.Is, ast.IsNot, ast.In, ast.NotIn]:
            return node

        self.branch_num += 1
        return ast.Call(func=ast.Name("evaluate_condition", ast.Load()),
                        args=[ast.Num(self.branch_num - 1),
                              ast.Str(node.ops[0].__class__.__name__),
                              node.left,
                              node.comparators[0]],
                        keywords=[],
                        starargs=None,
                        kwargs=None)
```

We can now take a look at the instrumented version of `triangle`:


```python
source = inspect.getsource(triangle)
node = ast.parse(source)
transformer = BranchTransformer()
transformer.visit(node)

# Make sure the line numbers are ok before printing
node = ast.fix_missing_locations(node)
num_branches = transformer.branch_num

print(astor.to_source(node))
```

    def triangle_instrumented(a, b, c):
        if evaluate_condition(0, 'LtE', a, 0) or evaluate_condition(1, 'LtE', b, 0
            ) or evaluate_condition(2, 'LtE', c, 0):
            return 4
        if evaluate_condition(3, 'LtE', a + b, c) or evaluate_condition(4,
            'LtE', a + c, b) or evaluate_condition(5, 'LtE', b + c, a):
            return 4
        if evaluate_condition(6, 'Eq', a, b) and evaluate_condition(7, 'Eq', b, c):
            return 1
        if evaluate_condition(8, 'Eq', a, b) or evaluate_condition(9, 'Eq', b, c
            ) or evaluate_condition(10, 'Eq', a, c):
            return 2
        return 3
    


To define an executable version of the instrumented triangle function, we can use our `create_instrumented_function` function again:


```python
create_instrumented_function(triangle)
```


```python
triangle_instrumented(4, 4, 4)
```




    1




```python
distances_true
```




    {0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.8, 6: 0.0, 7: 0.0}




```python
distances_false
```




    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.5, 7: 0.5}



The branch distance functions above are defined only for atomic comparisons. However, in the `triangle` program all of the atomic comparisons are part of larger predicates, joined together by `and` and `or` connectors. 

For conjunctions the branch distance is defined such that the distance to make `A and B` true equals the sum of the branch distances for `A` and `B`, as both of the two conditions would need to be true. Similarly, the branch distance to make `A or B` true would be the _minimum_ of the two branch distances of `A` and `B`, as it suffices if one of the two conditions is true to make the entire expression true (and the false distance would be the sum of false distances of the conditions). For a negation `not A`, we can simply switch from the true distance to the false distance, or vice versa. Since predicates can consist of nested conditions, one would need to recursively calculate the branch distance.


Assume we want to find an input that covers the third if-condition, i.e., it produces a triangle where all sides have equal length. Considering the instrumented version of the triangle function we printed above, in order for this if-condition to evaluate to true we require conditions 0, 1, 2, 3, 4, and 5 to evaluate to false, and 6 and 7 to evaluate to true. Thus, the fitness function for this branch would be the sum of false distances for branches 0-5, and true distances for branches 6 and 7.


```python
def get_fitness(x):
    # Reset any distance values from previous executions
    global distances_true, distances_false
    distances_true  = {x: 1.0 for x in range(num_branches)}
    distances_false = {x: 1.0 for x in range(num_branches)}

    # Run the function under test
    triangle_instrumented(*x)

    # Sum up branch distances for our specific target branch
    fitness = 0.0
    for branch in [6, 7]:
        fitness += distances_true[branch]

    for branch in [0, 1, 2, 3, 4, 5]:
        fitness += distances_false[branch]

    return fitness
```


```python
get_fitness([5,5,5])
```




    0.0




```python
get_fitness(get_random_individual())
```




    7.99999999947165




```python
MAX = 10000
MIN = -MAX
fitness_values = []
max_gen = 1000
hillclimbing()
```

    Starting at fitness 7.999338624338624: [-1510, 704, -2624]
    Solution fitness after 10002 fitness evaluations: 5.999595141700405: [1, 704, -2468]





    [1, 704, -2468]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10fd3b1c0>]




    
![png](6/output_129_1.png)
    



```python
fitness_values = []
avm()
```

    Starting at fitness 5.999796458375737: [7749, 9661, -4911]
    Current parameter: 0
    Trying +1 at fitness 5.999796458375737: [7750, 9661, -4911]
    Trying -1 at fitness 5.999796458375737: [7748, 9661, -4911]
    Current parameter: 1
    Trying +1 at fitness 5.999796458375737: [7749, 9662, -4911]
    Trying -1 at fitness 5.999796458375737: [7749, 9660, -4911]
    Current parameter: 2
    Trying +1 at fitness 5.999796458375737: [7749, 9661, -4910]
      [7749, 9661, -4910], direction 2, fitness 5.999796416938111
      [7749, 9661, -4908], direction 4, fitness 5.9997963340122205
      [7749, 9661, -4904], direction 8, fitness 5.999796167957603
      [7749, 9661, -4896], direction 16, fitness 5.999795835034708
      [7749, 9661, -4880], direction 32, fitness 5.999795165915608
      [7749, 9661, -4848], direction 64, fitness 5.99979381443299
      [7749, 9661, -4784], direction 128, fitness 5.999791057250313
      [7749, 9661, -4656], direction 256, fitness 5.999785315586088
      [7749, 9661, -4400], direction 512, fitness 5.999772830531576
      [7749, 9661, -3888], direction 1024, fitness 5.9997429305912595
      [7749, 9661, -2864], direction 2048, fitness 5.999651081646895
      [7749, 9661, -816], direction 4096, fitness 5.99877750611247
      [7749, 9661, 3280], direction 8192, fitness 1.9994772608468374
    Current parameter: 2
    Trying +1 at fitness 1.9994772608468374: [7749, 9661, 3281]
    Trying -1 at fitness 1.9994772608468374: [7749, 9661, 3279]
    Current parameter: 0
    Trying +1 at fitness 1.9994772608468374: [7750, 9661, 3280]
      [7750, 9661, 3280], direction 2, fitness 1.9994769874476988
      [7752, 9661, 3280], direction 4, fitness 1.999476439790576
      [7756, 9661, 3280], direction 8, fitness 1.9994753410283317
      [7764, 9661, 3280], direction 16, fitness 1.999473129610116
      [7780, 9661, 3280], direction 32, fitness 1.9994686503719448
      [7812, 9661, 3280], direction 64, fitness 1.9994594594594595
      [7876, 9661, 3280], direction 128, fitness 1.9994400895856663
      [8004, 9661, 3280], direction 256, fitness 1.999396863691194
      [8260, 9661, 3280], direction 512, fitness 1.9992867332382311
      [8772, 9661, 3280], direction 1024, fitness 1.998876404494382
      [9796, 9661, 3280], direction 2048, fitness 1.9926470588235294
    Current parameter: 0
    Trying +1 at fitness 1.9926470588235294: [9797, 9661, 3280]
    Trying -1 at fitness 1.9926470588235294: [9795, 9661, 3280]
      [9795, 9661, 3280], direction -2, fitness 1.9925925925925925
      [9793, 9661, 3280], direction -4, fitness 1.9924812030075187
      [9789, 9661, 3280], direction -8, fitness 1.9922480620155039
      [9781, 9661, 3280], direction -16, fitness 1.9917355371900827
      [9765, 9661, 3280], direction -32, fitness 1.9904761904761905
      [9733, 9661, 3280], direction -64, fitness 1.9863013698630136
      [9669, 9661, 3280], direction -128, fitness 1.8888888888888888
    Current parameter: 0
    Trying +1 at fitness 1.8888888888888888: [9670, 9661, 3280]
    Trying -1 at fitness 1.8888888888888888: [9668, 9661, 3280]
      [9668, 9661, 3280], direction -2, fitness 1.875
      [9666, 9661, 3280], direction -4, fitness 1.8333333333333335
      [9662, 9661, 3280], direction -8, fitness 1.5
    Current parameter: 0
    Trying +1 at fitness 1.5: [9663, 9661, 3280]
    Trying -1 at fitness 1.5: [9661, 9661, 3280]
      [9661, 9661, 3280], direction -2, fitness 0.9998433093074272
    Current parameter: 0
    Trying +1 at fitness 0.9998433093074272: [9662, 9661, 3280]
    Trying -1 at fitness 0.9998433093074272: [9660, 9661, 3280]
    Current parameter: 1
    Trying +1 at fitness 0.9998433093074272: [9661, 9662, 3280]
    Trying -1 at fitness 0.9998433093074272: [9661, 9660, 3280]
    Current parameter: 2
    Trying +1 at fitness 0.9998433093074272: [9661, 9661, 3281]
      [9661, 9661, 3281], direction 2, fitness 0.9998432847516063
      [9661, 9661, 3283], direction 4, fitness 0.9998432356168678
      [9661, 9661, 3287], direction 8, fitness 0.9998431372549019
      [9661, 9661, 3295], direction 16, fitness 0.999842940160201
      [9661, 9661, 3311], direction 32, fitness 0.999842544481184
      [9661, 9661, 3343], direction 64, fitness 0.9998417471118848
      [9661, 9661, 3407], direction 128, fitness 0.9998401278976818
      [9661, 9661, 3535], direction 256, fitness 0.9998367879875959
      [9661, 9661, 3791], direction 512, fitness 0.9998296712655425
      [9661, 9661, 4303], direction 1024, fitness 0.9998133980220191
      [9661, 9661, 5327], direction 2048, fitness 0.9997693194925029
      [9661, 9661, 7375], direction 4096, fitness 0.9995627459554001
      [9661, 9661, 11471], direction 8192, fitness 0.9994478188845941
    Current parameter: 2
    Trying +1 at fitness 0.9994478188845941: [9661, 9661, 11472]
    Trying -1 at fitness 0.9994478188845941: [9661, 9661, 11470]
      [9661, 9661, 11470], direction -2, fitness 0.9994475138121547
      [9661, 9661, 11468], direction -4, fitness 0.9994469026548672
      [9661, 9661, 11464], direction -8, fitness 0.9994456762749445
      [9661, 9661, 11456], direction -16, fitness 0.9994432071269488
      [9661, 9661, 11440], direction -32, fitness 0.999438202247191
      [9661, 9661, 11408], direction -64, fitness 0.9994279176201373
      [9661, 9661, 11344], direction -128, fitness 0.9994061757719715
      [9661, 9661, 11216], direction -256, fitness 0.9993573264781491
      [9661, 9661, 10960], direction -512, fitness 0.9992307692307693
      [9661, 9661, 10448], direction -1024, fitness 0.998730964467005
      [9661, 9661, 9424], direction -2048, fitness 0.9957983193277311
    Current parameter: 2
    Trying +1 at fitness 0.9957983193277311: [9661, 9661, 9425]
      [9661, 9661, 9425], direction 2, fitness 0.9957805907172996
      [9661, 9661, 9427], direction 4, fitness 0.9957446808510638
      [9661, 9661, 9431], direction 8, fitness 0.9956709956709957
      [9661, 9661, 9439], direction 16, fitness 0.9955156950672646
      [9661, 9661, 9455], direction 32, fitness 0.9951690821256038
      [9661, 9661, 9487], direction 64, fitness 0.9942857142857143
      [9661, 9661, 9551], direction 128, fitness 0.990990990990991
      [9661, 9661, 9679], direction 256, fitness 0.9473684210526315
    Current parameter: 2
    Trying +1 at fitness 0.9473684210526315: [9661, 9661, 9680]
    Trying -1 at fitness 0.9473684210526315: [9661, 9661, 9678]
      [9661, 9661, 9678], direction -2, fitness 0.9444444444444444
      [9661, 9661, 9676], direction -4, fitness 0.9375
      [9661, 9661, 9672], direction -8, fitness 0.9166666666666666
      [9661, 9661, 9664], direction -16, fitness 0.75
    Current parameter: 2
    Trying +1 at fitness 0.75: [9661, 9661, 9665]
    Trying -1 at fitness 0.75: [9661, 9661, 9663]
      [9661, 9661, 9663], direction -2, fitness 0.6666666666666666
      [9661, 9661, 9661], direction -4, fitness 0.0
    Current parameter: 2
    Trying +1 at fitness 0.0: [9661, 9661, 9662]
    Trying -1 at fitness 0.0: [9661, 9661, 9660]
    Solution fitness 0.0: [9661, 9661, 9661]





    [9661, 9661, 9661]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10fd8ba60>]




    
![png](6/output_131_1.png)
    


Besides the local search algorithms, we can also use evolutionary search in order to find solutions to our test generation problem. We therefore need to define the usual search operators:


```python
tournament_size = 3
def tournament_selection(population):
    candidates = random.sample(population, tournament_size)        
    winner = min(candidates, key = lambda x: get_fitness(x))
    return winner
```


```python
elite_size = 2
def elitism_standard(population):
    population.sort(key = lambda k: get_fitness(k))
    return population[:elite_size]
```


```python
def mutate(solution):
    P_mutate = 1/len(solution)
    mutated = solution[:]
    for position in range(len(solution)):
        if random.random() < P_mutate:
            mutated[position] = int(random.gauss(mutated[position], 20))
    return mutated
```


```python
def singlepoint_crossover(parent1, parent2):
    pos = random.randint(0, len(parent1))
    offspring1 = parent1[:pos] + parent2[pos:]
    offspring2 = parent2[:pos] + parent1[pos:]
    return (offspring1, offspring2)
```


```python
population_size = 20
P_xover = 0.7
max_gen = 100
selection = tournament_selection
crossover = singlepoint_crossover
elitism = elitism_standard
MAX = 1000
MIN = -MAX
```


```python
def ga():
    population = [get_random_individual() for _ in range(population_size)]
    best_fitness = sys.maxsize
    for p in population:
        fitness = get_fitness(p)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = p
    print(f"Iteration 0, best fitness: {best_fitness}")

    for iteration in range(max_gen):
        fitness_values.append(best_fitness)
        new_population = elitism(population)
        while len(new_population) < len(population):
            parent1 = selection(population)
            parent2 = selection(population)

            if random.random() < P_xover:
                offspring1, offspring2 = crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2

            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            
            new_population.append(offspring1)
            new_population.append(offspring2)

        population = new_population
        for p in population:
            fitness = get_fitness(p)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = p
        print(f"Iteration {iteration}, best fitness: {best_fitness}, size {len(best_solution)}")

    return best_solution
```


```python
fitness_values = []
ga()
```

    Iteration 0, best fitness: 1.987012987012987
    Iteration 0, best fitness: 1.9830508474576272, size 3
    Iteration 1, best fitness: 1.967741935483871, size 3
    Iteration 2, best fitness: 1.9666666666666668, size 3
    Iteration 3, best fitness: 1.9666666666666668, size 3
    Iteration 4, best fitness: 1.5, size 3
    Iteration 5, best fitness: 1.5, size 3
    Iteration 6, best fitness: 1.5, size 3
    Iteration 7, best fitness: 1.5, size 3
    Iteration 8, best fitness: 1.5, size 3
    Iteration 9, best fitness: 1.5, size 3
    Iteration 10, best fitness: 1.5, size 3
    Iteration 11, best fitness: 1.5, size 3
    Iteration 12, best fitness: 1.5, size 3
    Iteration 13, best fitness: 0.9946236559139785, size 3
    Iteration 14, best fitness: 0.9946236559139785, size 3
    Iteration 15, best fitness: 0.9946236559139785, size 3
    Iteration 16, best fitness: 0.9905660377358491, size 3
    Iteration 17, best fitness: 0.9905660377358491, size 3
    Iteration 18, best fitness: 0.9894736842105263, size 3
    Iteration 19, best fitness: 0.9894736842105263, size 3
    Iteration 20, best fitness: 0.989010989010989, size 3
    Iteration 21, best fitness: 0.989010989010989, size 3
    Iteration 22, best fitness: 0.975, size 3
    Iteration 23, best fitness: 0.975, size 3
    Iteration 24, best fitness: 0.9, size 3
    Iteration 25, best fitness: 0.9, size 3
    Iteration 26, best fitness: 0.9, size 3
    Iteration 27, best fitness: 0.6666666666666666, size 3
    Iteration 28, best fitness: 0.6666666666666666, size 3
    Iteration 29, best fitness: 0.6666666666666666, size 3
    Iteration 30, best fitness: 0.6666666666666666, size 3
    Iteration 31, best fitness: 0.6666666666666666, size 3
    Iteration 32, best fitness: 0.6666666666666666, size 3
    Iteration 33, best fitness: 0.6666666666666666, size 3
    Iteration 34, best fitness: 0.6666666666666666, size 3
    Iteration 35, best fitness: 0.5, size 3
    Iteration 36, best fitness: 0.5, size 3
    Iteration 37, best fitness: 0.5, size 3
    Iteration 38, best fitness: 0.5, size 3
    Iteration 39, best fitness: 0.0, size 3
    Iteration 40, best fitness: 0.0, size 3
    Iteration 41, best fitness: 0.0, size 3
    Iteration 42, best fitness: 0.0, size 3
    Iteration 43, best fitness: 0.0, size 3
    Iteration 44, best fitness: 0.0, size 3
    Iteration 45, best fitness: 0.0, size 3
    Iteration 46, best fitness: 0.0, size 3
    Iteration 47, best fitness: 0.0, size 3
    Iteration 48, best fitness: 0.0, size 3
    Iteration 49, best fitness: 0.0, size 3
    Iteration 50, best fitness: 0.0, size 3
    Iteration 51, best fitness: 0.0, size 3
    Iteration 52, best fitness: 0.0, size 3
    Iteration 53, best fitness: 0.0, size 3
    Iteration 54, best fitness: 0.0, size 3
    Iteration 55, best fitness: 0.0, size 3
    Iteration 56, best fitness: 0.0, size 3
    Iteration 57, best fitness: 0.0, size 3
    Iteration 58, best fitness: 0.0, size 3
    Iteration 59, best fitness: 0.0, size 3
    Iteration 60, best fitness: 0.0, size 3
    Iteration 61, best fitness: 0.0, size 3
    Iteration 62, best fitness: 0.0, size 3
    Iteration 63, best fitness: 0.0, size 3
    Iteration 64, best fitness: 0.0, size 3
    Iteration 65, best fitness: 0.0, size 3
    Iteration 66, best fitness: 0.0, size 3
    Iteration 67, best fitness: 0.0, size 3
    Iteration 68, best fitness: 0.0, size 3
    Iteration 69, best fitness: 0.0, size 3
    Iteration 70, best fitness: 0.0, size 3
    Iteration 71, best fitness: 0.0, size 3
    Iteration 72, best fitness: 0.0, size 3
    Iteration 73, best fitness: 0.0, size 3
    Iteration 74, best fitness: 0.0, size 3
    Iteration 75, best fitness: 0.0, size 3
    Iteration 76, best fitness: 0.0, size 3
    Iteration 77, best fitness: 0.0, size 3
    Iteration 78, best fitness: 0.0, size 3
    Iteration 79, best fitness: 0.0, size 3
    Iteration 80, best fitness: 0.0, size 3
    Iteration 81, best fitness: 0.0, size 3
    Iteration 82, best fitness: 0.0, size 3
    Iteration 83, best fitness: 0.0, size 3
    Iteration 84, best fitness: 0.0, size 3
    Iteration 85, best fitness: 0.0, size 3
    Iteration 86, best fitness: 0.0, size 3
    Iteration 87, best fitness: 0.0, size 3
    Iteration 88, best fitness: 0.0, size 3
    Iteration 89, best fitness: 0.0, size 3
    Iteration 90, best fitness: 0.0, size 3
    Iteration 91, best fitness: 0.0, size 3
    Iteration 92, best fitness: 0.0, size 3
    Iteration 93, best fitness: 0.0, size 3
    Iteration 94, best fitness: 0.0, size 3
    Iteration 95, best fitness: 0.0, size 3
    Iteration 96, best fitness: 0.0, size 3
    Iteration 97, best fitness: 0.0, size 3
    Iteration 98, best fitness: 0.0, size 3
    Iteration 99, best fitness: 0.0, size 3





    [559, 559, 559]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10fde4eb0>]




    
![png](6/output_140_1.png)
    


We set `MAX` to a value as low as 1000, because the optimisation with our small mutational steps may take long to achieve that multiple values are equal, which some of the branches of the triangle program require (such as the one we are optimising for currently).


```python
MAX = 100000
MIN = -MAX
fitness_values = []
ga()
plt.plot(fitness_values)
```

    Iteration 0, best fitness: 1.9999274679045478
    Iteration 0, best fitness: 1.9999274679045478, size 3
    Iteration 1, best fitness: 1.9999272409778812, size 3
    Iteration 2, best fitness: 1.9999272409778812, size 3
    Iteration 3, best fitness: 1.9999271667880554, size 3
    Iteration 4, best fitness: 1.999927076496755, size 3
    Iteration 5, best fitness: 1.9999269059279292, size 3
    Iteration 6, best fitness: 1.9999268471104608, size 3
    Iteration 7, best fitness: 1.9999266539533518, size 3
    Iteration 8, best fitness: 1.9999266001174398, size 3
    Iteration 9, best fitness: 1.9999263568745858, size 3
    Iteration 10, best fitness: 1.9999262917373037, size 3
    Iteration 11, best fitness: 1.9999261829187274, size 3
    Iteration 12, best fitness: 1.9999258600237249, size 3
    Iteration 13, best fitness: 1.9999258545265812, size 3
    Iteration 14, best fitness: 1.9999255730872283, size 3
    Iteration 15, best fitness: 1.9999255620068483, size 3
    Iteration 16, best fitness: 1.9999254399045632, size 3
    Iteration 17, best fitness: 1.9999253118231384, size 3
    Iteration 18, best fitness: 1.999925216871074, size 3
    Iteration 19, best fitness: 1.9999251777029556, size 3
    Iteration 20, best fitness: 1.9999251160700915, size 3
    Iteration 21, best fitness: 1.9999248685199098, size 3
    Iteration 22, best fitness: 1.999924800721913, size 3
    Iteration 23, best fitness: 1.999924670433145, size 3
    Iteration 24, best fitness: 1.999924590905663, size 3
    Iteration 25, best fitness: 1.9999245226054798, size 3
    Iteration 26, best fitness: 1.9999243913503704, size 3
    Iteration 27, best fitness: 1.999924265374129, size 3
    Iteration 28, best fitness: 1.9999241101920013, size 3
    Iteration 29, best fitness: 1.9999239717174788, size 3
    Iteration 30, best fitness: 1.9999237688672054, size 3
    Iteration 31, best fitness: 1.9999237688672054, size 3
    Iteration 32, best fitness: 1.9999236349751812, size 3
    Iteration 33, best fitness: 1.9999234654829328, size 3
    Iteration 34, best fitness: 1.9999231596741969, size 3
    Iteration 35, best fitness: 1.9999230710054618, size 3
    Iteration 36, best fitness: 1.9999229880631497, size 3
    Iteration 37, best fitness: 1.9999226843977116, size 3
    Iteration 38, best fitness: 1.9999226843977116, size 3
    Iteration 39, best fitness: 1.9999223903764065, size 3
    Iteration 40, best fitness: 1.9999223361292326, size 3
    Iteration 41, best fitness: 1.9999221547563444, size 3
    Iteration 42, best fitness: 1.9999221547563444, size 3
    Iteration 43, best fitness: 1.9999221244451366, size 3
    Iteration 44, best fitness: 1.9999219237976265, size 3
    Iteration 45, best fitness: 1.9999219237976265, size 3
    Iteration 46, best fitness: 1.999921630094044, size 3
    Iteration 47, best fitness: 1.999921630094044, size 3
    Iteration 48, best fitness: 1.9999214762465645, size 3
    Iteration 49, best fitness: 1.9999211480838985, size 3
    Iteration 50, best fitness: 1.999920904848533, size 3
    Iteration 51, best fitness: 1.999920904848533, size 3
    Iteration 52, best fitness: 1.9999207669756753, size 3
    Iteration 53, best fitness: 1.99992055926279, size 3
    Iteration 54, best fitness: 1.9999204075135308, size 3
    Iteration 55, best fitness: 1.999920369485587, size 3
    Iteration 56, best fitness: 1.9999201787994891, size 3
    Iteration 57, best fitness: 1.9999200767263428, size 3
    Iteration 58, best fitness: 1.9999199102995355, size 3
    Iteration 59, best fitness: 1.9999197431781701, size 3
    Iteration 60, best fitness: 1.9999197174052665, size 3
    Iteration 61, best fitness: 1.999919523579591, size 3
    Iteration 62, best fitness: 1.9999193808448887, size 3
    Iteration 63, best fitness: 1.9999193483345432, size 3
    Iteration 64, best fitness: 1.9999191200258815, size 3
    Iteration 65, best fitness: 1.9999191200258815, size 3
    Iteration 66, best fitness: 1.999918864097363, size 3
    Iteration 67, best fitness: 1.999918864097363, size 3
    Iteration 68, best fitness: 1.9999184538856722, size 3
    Iteration 69, best fitness: 1.9999184538856722, size 3
    Iteration 70, best fitness: 1.9999184538856722, size 3
    Iteration 71, best fitness: 1.9999183273440053, size 3
    Iteration 72, best fitness: 1.9999182204775923, size 3
    Iteration 73, best fitness: 1.999918200408998, size 3
    Iteration 74, best fitness: 1.9999179588153253, size 3
    Iteration 75, best fitness: 1.999917844232665, size 3
    Iteration 76, best fitness: 1.9999176751461265, size 3
    Iteration 77, best fitness: 1.9999175529722155, size 3
    Iteration 78, best fitness: 1.999917328042328, size 3
    Iteration 79, best fitness: 1.9999169504194003, size 3
    Iteration 80, best fitness: 1.9999169504194003, size 3
    Iteration 81, best fitness: 1.9999169090153717, size 3
    Iteration 82, best fitness: 1.9999167221852099, size 3
    Iteration 83, best fitness: 1.9999166597216433, size 3
    Iteration 84, best fitness: 1.9999164089275265, size 3
    Iteration 85, best fitness: 1.9999161917532686, size 3
    Iteration 86, best fitness: 1.999916022841787, size 3
    Iteration 87, best fitness: 1.9999157681940702, size 3
    Iteration 88, best fitness: 1.9999157681940702, size 3
    Iteration 89, best fitness: 1.999915725602562, size 3
    Iteration 90, best fitness: 1.9999154977184384, size 3
    Iteration 91, best fitness: 1.9999152470548351, size 3
    Iteration 92, best fitness: 1.99991503101368, size 3
    Iteration 93, best fitness: 1.999915002124947, size 3
    Iteration 94, best fitness: 1.9999148066110068, size 3
    Iteration 95, best fitness: 1.9999146101955425, size 3
    Iteration 96, best fitness: 1.9999146101955425, size 3
    Iteration 97, best fitness: 1.9999146101955425, size 3
    Iteration 98, best fitness: 1.9999144494824193, size 3
    Iteration 99, best fitness: 1.9999143322196522, size 3





    [<matplotlib.lines.Line2D at 0x10fe44610>]




    
![png](6/output_142_2.png)
    


Different mutation operators may yield different results: For example, rather than just adding random noise to the individual parameters, we can also probabilistically copy values from other parameters:


```python
def mutate(solution):
    P_mutate = 1/len(solution)
    mutated = solution[:]
    for position in range(len(solution)):
        if random.random() < P_mutate:
            if random.random() < 0.9:
                mutated[position] = int(random.gauss(mutated[position], 20))
            else:
                mutated[position] = random.choice(solution)
    return mutated
```

Let's see the performance of the resulting algorithm:


```python
fitness_values = []
MAX = 100000
MIN = -MAX
ga()
```

    Iteration 0, best fitness: 2.9998454404945907
    Iteration 0, best fitness: 1.9998728705822528, size 3
    Iteration 1, best fitness: 1.9998728705822528, size 3
    Iteration 2, best fitness: 1.999871877001922, size 3
    Iteration 3, best fitness: 1.9998717126363053, size 3
    Iteration 4, best fitness: 1.95, size 3
    Iteration 5, best fitness: 1.75, size 3
    Iteration 6, best fitness: 1.75, size 3
    Iteration 7, best fitness: 1.6666666666666665, size 3
    Iteration 8, best fitness: 1.6666666666666665, size 3
    Iteration 9, best fitness: 1.6666666666666665, size 3
    Iteration 10, best fitness: 0.8571428571428571, size 3
    Iteration 11, best fitness: 0.8571428571428571, size 3
    Iteration 12, best fitness: 0.8, size 3
    Iteration 13, best fitness: 0.8, size 3
    Iteration 14, best fitness: 0.0, size 3
    Iteration 15, best fitness: 0.0, size 3
    Iteration 16, best fitness: 0.0, size 3
    Iteration 17, best fitness: 0.0, size 3
    Iteration 18, best fitness: 0.0, size 3
    Iteration 19, best fitness: 0.0, size 3
    Iteration 20, best fitness: 0.0, size 3
    Iteration 21, best fitness: 0.0, size 3
    Iteration 22, best fitness: 0.0, size 3
    Iteration 23, best fitness: 0.0, size 3
    Iteration 24, best fitness: 0.0, size 3
    Iteration 25, best fitness: 0.0, size 3
    Iteration 26, best fitness: 0.0, size 3
    Iteration 27, best fitness: 0.0, size 3
    Iteration 28, best fitness: 0.0, size 3
    Iteration 29, best fitness: 0.0, size 3
    Iteration 30, best fitness: 0.0, size 3
    Iteration 31, best fitness: 0.0, size 3
    Iteration 32, best fitness: 0.0, size 3
    Iteration 33, best fitness: 0.0, size 3
    Iteration 34, best fitness: 0.0, size 3
    Iteration 35, best fitness: 0.0, size 3
    Iteration 36, best fitness: 0.0, size 3
    Iteration 37, best fitness: 0.0, size 3
    Iteration 38, best fitness: 0.0, size 3
    Iteration 39, best fitness: 0.0, size 3
    Iteration 40, best fitness: 0.0, size 3
    Iteration 41, best fitness: 0.0, size 3
    Iteration 42, best fitness: 0.0, size 3
    Iteration 43, best fitness: 0.0, size 3
    Iteration 44, best fitness: 0.0, size 3
    Iteration 45, best fitness: 0.0, size 3
    Iteration 46, best fitness: 0.0, size 3
    Iteration 47, best fitness: 0.0, size 3
    Iteration 48, best fitness: 0.0, size 3
    Iteration 49, best fitness: 0.0, size 3
    Iteration 50, best fitness: 0.0, size 3
    Iteration 51, best fitness: 0.0, size 3
    Iteration 52, best fitness: 0.0, size 3
    Iteration 53, best fitness: 0.0, size 3
    Iteration 54, best fitness: 0.0, size 3
    Iteration 55, best fitness: 0.0, size 3
    Iteration 56, best fitness: 0.0, size 3
    Iteration 57, best fitness: 0.0, size 3
    Iteration 58, best fitness: 0.0, size 3
    Iteration 59, best fitness: 0.0, size 3
    Iteration 60, best fitness: 0.0, size 3
    Iteration 61, best fitness: 0.0, size 3
    Iteration 62, best fitness: 0.0, size 3
    Iteration 63, best fitness: 0.0, size 3
    Iteration 64, best fitness: 0.0, size 3
    Iteration 65, best fitness: 0.0, size 3
    Iteration 66, best fitness: 0.0, size 3
    Iteration 67, best fitness: 0.0, size 3
    Iteration 68, best fitness: 0.0, size 3
    Iteration 69, best fitness: 0.0, size 3
    Iteration 70, best fitness: 0.0, size 3
    Iteration 71, best fitness: 0.0, size 3
    Iteration 72, best fitness: 0.0, size 3
    Iteration 73, best fitness: 0.0, size 3
    Iteration 74, best fitness: 0.0, size 3
    Iteration 75, best fitness: 0.0, size 3
    Iteration 76, best fitness: 0.0, size 3
    Iteration 77, best fitness: 0.0, size 3
    Iteration 78, best fitness: 0.0, size 3
    Iteration 79, best fitness: 0.0, size 3
    Iteration 80, best fitness: 0.0, size 3
    Iteration 81, best fitness: 0.0, size 3
    Iteration 82, best fitness: 0.0, size 3
    Iteration 83, best fitness: 0.0, size 3
    Iteration 84, best fitness: 0.0, size 3
    Iteration 85, best fitness: 0.0, size 3
    Iteration 86, best fitness: 0.0, size 3
    Iteration 87, best fitness: 0.0, size 3
    Iteration 88, best fitness: 0.0, size 3
    Iteration 89, best fitness: 0.0, size 3
    Iteration 90, best fitness: 0.0, size 3
    Iteration 91, best fitness: 0.0, size 3
    Iteration 92, best fitness: 0.0, size 3
    Iteration 93, best fitness: 0.0, size 3
    Iteration 94, best fitness: 0.0, size 3
    Iteration 95, best fitness: 0.0, size 3
    Iteration 96, best fitness: 0.0, size 3
    Iteration 97, best fitness: 0.0, size 3
    Iteration 98, best fitness: 0.0, size 3
    Iteration 99, best fitness: 0.0, size 3





    [69793, 69793, 69793]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10fe8f6a0>]




    
![png](6/output_147_1.png)
    


In our fitness function, we manually determined which branches need to evaluate which way, and how to sum up the fitness functions. In practice, this can be automated by combining the branch distance metric with the _approach level_, which was introduced (originally named approximation level) in this paper:

Wegener, J., Baresel, A., & Sthamer, H. (2001). Evolutionary test environment for automatic structural testing. Information and software technology, 43(14), 841-854.

The approach level calculates the distances of an execution from a target node in terms of graph distances on the control dependence graph. However, we will not cover the approach level in this chapter.

## Whole Test Suite Optimisation

Besides the question of how the best fitness function for a coverage goal looks like, there are some related questions: How much time should we spend on optimising for each coverage goal? It is possible that some coverage goals are infeasible (e.g., dead code, or infeasible branches), so any time spent on these is wasted, while it may be missing for other goals that are feasible but would need more time. Test cases typically cover multiple goals at the same time; even if a test is optimised for one specific line or branch, it may coincidentally cover others along the execution. Thus, the order in which we select coverage goals for optimisation may influence the overall result, and the number of tests we require. In principle, one way to address these issues would be by casting test generation as a multi-objective optimisation problem, and aiming to produce tests for _all_ coverage goals at the same time. However, there is an issue with this: Multi-objective algorithms like the ones we considered in the previous chapter typically work well on 2-3 objectives, but code will generally have many more coverage objectives, rendering classical multi-objective algorithms infeasible (Pareto-dominance happens rarely with higher numbers of objectives). We will therefore now consider some alternatives.

The first alternative we consider is to switch our representation: Rather than optimising individual test cases for individual coverage objectives, we optimise entire test _suites_ to cover _all_ coverage objectives at the same time. Our encoding thus should describe multiple tests. But how many? This is very much problem specific. Thus, rather than hard coding the number of tests, we will only define an upper bound, and let the search decide what is the necessary number of tests.


```python
num_tests = 30
```


```python
def get_random_individual():
    num = random.randint(1, num_tests)
    return [[random.randint(MIN, MAX) for _ in range(num_parameters)] for _ in range(num)]
```

When applying mutation, we need to be able to modify individual tests as before. To keep things challenging, we will not use our optimised mutation that copies parameters, but aim to achieve the entire optimisation just using small steps:


```python
def mutate_test(solution):
    P_mutate = 1/len(solution)
    mutated = solution[:]
    for position in range(len(solution)):
        if random.random() < P_mutate:
            mutated[position] = min(MAX, max(MIN, int(random.gauss(mutated[position], MAX*0.01))))
            
    return mutated
```

However, modifying tests is only one of the things we can do when mutating our actual individuals, which consist of multiple tests. Besides modifying existing tests, we could also delete or add tests, for example like this.


```python
def mutate_set(solution):
    P_mutate = 1/len(solution)
    mutated = []
    for position in range(len(solution)):
        if random.random() >= P_mutate:
            mutated.append(solution[position][:])
            
    if not mutated:
        mutated = solution[:]
    for position in range(len(mutated)):
        if random.random() < P_mutate:
            mutated[position] = mutate_test(mutated[position])
 
    ALPHA = 1/3
    count = 1
    while random.random() < ALPHA ** count:
        count += 1
        mutated.append([random.randint(MIN, MAX) for _ in range(num_parameters)])
    
    return mutated
```

With a certain probability, each of the tests can be removed from a test suite; similarly, each remaining test may be mutated like we mutated tests previously. Finally, with a probability `ALPHA` we insert a new test; if we do so, we insert another one with probability `ALPHA`$^2$, and so on.

When crossing over two individuals, they might have different length, which makes choosing a crossover point difficult. For example, we might pick a crossover point that is longer than one of the parent chromosomes, and then what do we do? A simple solution would be to pick two different crossover points.


```python
def variable_crossover(parent1, parent2):
    pos1 = random.randint(1, len(parent1))
    pos2 = random.randint(1, len(parent2))
    offspring1 = parent1[:pos1] + parent2[pos2:]
    offspring2 = parent2[:pos2] + parent1[pos1:]
    return (offspring1, offspring2)
```

To see this works, we need to define the fitness function. Since we want to cover _everything_ we simply need to make sure that every single branch is covered at least once in a test suite. A branch is covered if its minimum branch distance is 0; thus, if everything is covered, then the sum of minimal branch distances should be 0.

There is one special case: If an if-statement is executed only once, then optimising the true/false distance may lead to a suboptimal, oscillising evolution. We therefore also count how often each if-condition was executed. If it was only executed once, then the fitness value for that branch needs to be higher than if it was executed twice. For this, we extend our `update_maps` function to also keep track of the execution count:


```python
condition_count = {}
def update_maps(condition_num, d_true, d_false):
    global distances_true, distances_false, condition_count

    if condition_num in condition_count.keys():
        condition_count[condition_num] = condition_count[condition_num] + 1
    else:
        condition_count[condition_num] = 1
        
    if condition_num in distances_true.keys():
        distances_true[condition_num] = min(
            distances_true[condition_num], d_true)
    else:
        distances_true[condition_num] = d_true

    if condition_num in distances_false.keys():
        distances_false[condition_num] = min(
            distances_false[condition_num], d_false)
    else:
        distances_false[condition_num] = d_false
```

The actual fitness function now is the sum of minimal distances after all tests have been executed. If an if-condition was not executed at all, then the true distance and the false distance will be 1, resulting in a sum of 2 for the if-condition. If the condition was covered only once, we set the fitness to exactly 1. If the condition was executed more than once, then at least either the true or false distance has to be 0, such that in sum, true and false distances will be less than 0.


```python
def get_fitness(x):
    # Reset any distance values from previous executions
    global distances_true, distances_false, condition_count
    distances_true =  {x: 1.0 for x in range(num_branches)}
    distances_false = {x: 1.0 for x in range(num_branches)}
    condition_count = {x:   0 for x in range(num_branches)}

    # Run the function under test
    for test in x:
        triangle_instrumented(*test)

    # Sum up branch distances
    fitness = 0.0
    for branch in range(num_branches):
        if condition_count[branch] == 1:
            fitness += 1
        else:
            fitness += distances_true[branch]
            fitness += distances_false[branch]

    return fitness
```

Before we run some experiments on this, let's make a small addition to our genetic algorithm: Since the size of individuals is variable it will be interesting to observe how this evolves. We'll capture the average population size in a separate list. Since the costs of evaluating fitness are no longer constant per individual but depend on the number of tests executed, we will also change our stopping criterion to the number of executed tests.


```python
from statistics import mean

length_values = []
max_executions = 10000

def ga():
    population = [get_random_individual() for _ in range(population_size)]
    best_fitness = sys.maxsize
    tests_executed = 0
    for p in population:
        fitness = get_fitness(p)
        tests_executed += len(p)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = p
    while tests_executed < max_executions:
        fitness_values.append(best_fitness)
        length_values.append(mean([len(x) for x in population]))
        new_population = elitism(population)
        while len(new_population) < len(population):
            parent1 = selection(population)
            parent2 = selection(population)

            if random.random() < P_xover:
                offspring1, offspring2 = crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2

            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            
            new_population.append(offspring1)
            new_population.append(offspring2)
            tests_executed += len(offspring1) + len(offspring2)

        population = new_population
        for p in population:
            fitness = get_fitness(p)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = p
    print(f"Best fitness: {best_fitness}, size {len(best_solution)}")

    return best_solution
```

Since we now have all the operators we need in place, let's run a first experiment aiming to achieve 100% coverage on the triangle example.


```python
max_executions = 1000000
MAX = 1000
MIN = -MAX
crossover = variable_crossover
selection = tournament_selection
elitism   = elitism_standard 
mutate    = mutate_set
tournament_size = 4
population_size = 50
fitness_values = []
length_values = []
ga()
```

    Best fitness: 0.9791666666666666, size 1203





    [[201, -482, 72],
     [-943, 51, -207],
     [951, -679, -760],
     [-630, -773, 621],
     [393, 27, -12],
     [-933, -427, -210],
     [590, -975, 73],
     [-836, -839, -27],
     [-237, 617, -764],
     [-139, -112, 933],
     [-221, 617, -764],
     [-621, -759, -960],
     [-943, 997, -887],
     [231, 411, 148],
     [199, 69, 888],
     [-547, 176, -58],
     [967, -29, 709],
     [-406, 356, 19],
     [-237, 617, -764],
     [-621, -759, -960],
     [-243, 31, -747],
     [-848, 414, 458],
     [576, 269, 130],
     [581, 894, 584],
     [-406, 356, 19],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [576, 270, 145],
     [581, 894, 584],
     [-161, -991, -276],
     [107, 644, 644],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [967, -29, 709],
     [-406, 356, 19],
     [-451, 662, 254],
     [-139, -130, 933],
     [-237, 617, -764],
     [-621, -759, -960],
     [-243, 31, -747],
     [-114, 984, -15],
     [107, 642, 644],
     [752, 304, 422],
     [517, -939, -950],
     [-848, 414, 458],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -29, 709],
     [-406, 356, 19],
     [-237, 617, -764],
     [588, -73, -741],
     [720, 697, 186],
     [-943, 997, -887],
     [581, 883, 582],
     [231, 411, 148],
     [199, 69, 888],
     [-547, 176, -58],
     [967, -29, 709],
     [-406, 356, 19],
     [-237, 617, -764],
     [-621, -759, -960],
     [-243, 31, -747],
     [-848, 414, 458],
     [576, 278, 130],
     [581, 894, 584],
     [842, -554, 247],
     [228, -418, 556],
     [731, 244, -388],
     [199, 69, 888],
     [588, -76, -739],
     [692, 704, 186],
     [731, 244, -388],
     [-55, -27, 285],
     [199, 69, 888],
     [588, -76, -739],
     [694, 704, 177],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 939],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [581, 894, 584],
     [912, 461, -345],
     [-161, -991, -276],
     [86, -257, 370],
     [393, 27, -12],
     [-933, -427, -210],
     [-548, -122, -531],
     [590, -975, 73],
     [-836, -839, -27],
     [152, -220, 940],
     [882, -563, -932],
     [-976, -717, -793],
     [842, -554, 247],
     [731, 244, -388],
     [-55, -27, 285],
     [588, -73, -741],
     [720, 697, 186],
     [750, 196, 1000],
     [281, -724, -363],
     [612, 775, -391],
     [-406, 356, 19],
     [331, 523, -462],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [207, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -29, 709],
     [-406, 356, 19],
     [-237, 617, -764],
     [588, -73, -741],
     [720, 697, 186],
     [-943, 997, -887],
     [576, 270, 145],
     [581, 898, 581],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -39, 709],
     [-406, 356, 19],
     [-451, 662, 254],
     [-237, 617, -764],
     [-240, 21, -747],
     [731, 244, -388],
     [-55, -27, 285],
     [86, -257, 370],
     [393, 27, -12],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [-406, 356, 19],
     [331, 523, -462],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -29, 709],
     [-406, 356, 19],
     [-237, 617, -764],
     [231, 411, 148],
     [199, 69, 888],
     [-547, 176, -58],
     [967, -29, 709],
     [-406, 356, 19],
     [-237, 617, -764],
     [-621, -759, -960],
     [-243, 31, -747],
     [-848, 414, 458],
     [576, 269, 130],
     [581, 894, 584],
     [842, -554, 247],
     [228, -418, 556],
     [731, 244, -388],
     [-55, -27, 285],
     [199, 69, 888],
     [588, -76, -739],
     [692, 704, 186],
     [731, 244, -388],
     [-55, -27, 285],
     [199, 69, 888],
     [685, 683, 181],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [-47, 533, 807],
     [-139, -119, 933],
     [-237, 617, -764],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [576, 270, 145],
     [581, 894, 584],
     [-161, -991, -276],
     [86, -257, 370],
     [393, 27, -12],
     [-933, -427, -210],
     [-548, -122, -531],
     [590, -975, 73],
     [-836, -839, -27],
     [107, 644, 644],
     [745, 304, 422],
     [-672, 632, -870],
     [688, 181, 297],
     [-715, -529, -227],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -39, 709],
     [-406, 356, 19],
     [-943, 997, -887],
     [576, 270, 145],
     [581, 894, 584],
     [231, 426, 148],
     [199, 55, 890],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -39, 709],
     [-406, 356, 19],
     [-451, 662, 254],
     [-240, 21, -747],
     [731, 244, -388],
     [-55, -27, 285],
     [86, -257, 370],
     [393, 27, -12],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [-406, 356, 19],
     [331, 523, -462],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [612, 775, -391],
     [-55, -27, 285],
     [199, 69, 888],
     [588, -76, -739],
     [685, 683, 181],
     [746, 196, 1000],
     [281, -724, -363],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [612, 775, -391],
     [967, -29, 709],
     [-406, 356, 19],
     [-451, 662, 254],
     [-139, -125, 933],
     [-237, 617, -764],
     [-621, -759, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [745, 304, 422],
     [517, -939, -950],
     [-848, 414, 458],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [725, 697, 183],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -29, 709],
     [-406, 356, 19],
     [-237, 617, -764],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [-943, 997, -887],
     [576, 270, 145],
     [581, 894, 584],
     [231, 426, 148],
     [199, 55, 890],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -39, 709],
     [-406, 356, 19],
     [-451, 662, 254],
     [-237, 617, -764],
     [-240, 21, -747],
     [731, 244, -388],
     [-55, -27, 285],
     [86, -257, 370],
     [393, 27, -12],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [-406, 356, 19],
     [331, 523, -462],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [612, 775, -391],
     [-55, -27, 285],
     [199, 69, 888],
     [588, -76, -739],
     [685, 683, 181],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 933],
     [-237, 617, -764],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [581, 894, 584],
     [912, 461, -345],
     [-161, -991, -276],
     [86, -257, 370],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 758, -391],
     [967, -39, 709],
     [-237, 617, -764],
     [-240, 21, -747],
     [731, 244, -388],
     [-55, -27, 285],
     [86, -257, 370],
     [393, 27, -12],
     [746, 196, 1000],
     [281, -725, -363],
     [-547, 176, -58],
     [-43, -29, 285],
     [199, 69, 888],
     [588, -76, -739],
     [694, 704, 177],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [177, 479, 771],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 688],
     [-47, 533, 807],
     [-139, -119, 933],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [692, 704, 186],
     [731, 244, -388],
     [-55, -27, 285],
     [199, 69, 873],
     [588, -76, -739],
     [694, 704, 177],
     [746, 196, 1000],
     [281, -728, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [847, -554, 247],
     [228, -418, 556],
     [731, 244, -388],
     [199, 69, 888],
     [588, -76, -739],
     [692, 704, 186],
     [-55, -27, 285],
     [199, 69, 888],
     [588, -76, -739],
     [685, 683, 181],
     [746, 196, 1000],
     [281, -724, -363],
     [612, 775, -391],
     [469, 965, 367],
     [-16, -22, -710],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 933],
     [-237, 617, -764],
     [-621, -766, -960],
     [-243, 31, -747],
     [177, 479, 771],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 688],
     [-47, 533, 807],
     [-139, -119, 933],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-47, 533, 807],
     [-139, -119, 933],
     [-237, 617, -764],
     [-243, 31, -747],
     [177, 479, 771],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 688],
     [-47, 533, 807],
     [-139, -119, 933],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-715, -529, -227],
     [124, 124, 77],
     [-430, -643, 805],
     [-209, -180, -544],
     [199, 69, 888],
     [588, -76, -739],
     [694, 704, 177],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 939],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [581, 894, 584],
     [912, 469, -348],
     [-161, -991, -276],
     [86, -257, 370],
     [393, 27, -12],
     [-548, -122, -531],
     [590, -975, 73],
     [-836, -839, -27],
     [152, -220, 940],
     [882, -563, -932],
     [-976, -717, -793],
     [842, -554, 247],
     [731, 244, -388],
     [-55, -27, 285],
     [588, -73, -741],
     [720, 697, 186],
     [750, 196, 1000],
     [281, -724, -363],
     [581, 894, 584],
     [231, 426, 148],
     [199, 55, 890],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -39, 709],
     [-406, 356, 19],
     [-451, 662, 254],
     [-240, 21, -747],
     [731, 244, -388],
     [-55, -27, 285],
     [86, -257, 370],
     [393, 27, -12],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [-406, 356, 19],
     [331, 523, -462],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [612, 775, -391],
     [-55, -27, 285],
     [199, 69, 888],
     [588, -76, -739],
     [685, 683, 181],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 933],
     [-237, 617, -764],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [581, 894, 584],
     [912, 461, -345],
     [-161, -991, -276],
     [86, -257, 370],
     [393, 27, -12],
     [-933, -427, -210],
     [-548, -122, -531],
     [590, -975, 73],
     [-836, -839, -27],
     [107, 644, 644],
     [-672, 632, -870],
     [-868, 104, 224],
     [688, 181, 297],
     [-715, -529, -227],
     [124, 124, 57],
     [-430, -643, 805],
     [-209, -180, -544],
     [199, 69, 888],
     [588, -76, -739],
     [694, 704, 177],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 939],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 997, -21],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [581, 894, 584],
     [912, 469, -348],
     [-161, -991, -276],
     [86, -257, 370],
     [393, 27, -12],
     [-548, -122, -531],
     [590, -975, 73],
     [-836, -839, -27],
     [152, -220, 940],
     [882, -563, -932],
     [-976, -717, -793],
     [842, -554, 247],
     [731, 244, -388],
     [-55, -27, 285],
     [588, -73, -741],
     [720, 697, 186],
     [750, 196, 1000],
     [281, -724, -363],
     [612, 775, -382],
     [-406, 356, 19],
     [331, 523, -462],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -29, 709],
     [-406, 356, 19],
     [-237, 617, -764],
     [588, -73, -741],
     [720, 697, 186],
     [-943, 997, -887],
     [576, 270, 145],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 758, -391],
     [967, -39, 709],
     [-237, 617, -764],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -29, 709],
     [-406, 356, 19],
     [-237, 617, -764],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [-943, 997, -887],
     [576, 270, 145],
     [581, 894, 584],
     [231, 426, 148],
     [199, 55, 890],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -39, 709],
     [-406, 356, 19],
     [-451, 662, 254],
     [-237, 617, -764],
     [-240, 21, -747],
     [731, 244, -388],
     [-55, -27, 285],
     [86, -257, 370],
     [393, 27, -12],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [-406, 356, 19],
     [331, 523, -462],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [612, 775, -391],
     [-55, -27, 285],
     [199, 69, 888],
     [588, -76, -739],
     [685, 683, 181],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 933],
     [-237, 617, -764],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [581, 894, 584],
     [912, 461, -345],
     [-161, -991, -276],
     [86, -257, 370],
     [393, 27, -12],
     [-933, -427, -210],
     [-548, -122, -531],
     [590, -975, 73],
     [-836, -839, -27],
     [107, 644, 644],
     [745, 304, 422],
     [-672, 632, -870],
     [-868, 104, 224],
     [688, 181, 297],
     [-715, -529, -227],
     [124, 124, 77],
     [-430, -643, 805],
     [-209, -180, -544],
     [199, 69, 888],
     [588, -76, -739],
     [694, 704, 177],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 939],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [581, 894, 584],
     [912, 469, -348],
     [-161, -991, -276],
     [86, -257, 370],
     [393, 27, -12],
     [-548, -122, -531],
     [590, -975, 73],
     [-836, -839, -27],
     [152, -220, 940],
     [882, -563, -932],
     [-976, -717, -793],
     [842, -554, 247],
     [731, 244, -388],
     [-55, -27, 285],
     [588, -73, -741],
     [720, 697, 186],
     [750, 196, 1000],
     [281, -724, -363],
     [581, 894, 584],
     [231, 426, 148],
     [199, 55, 890],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -39, 709],
     [-406, 356, 19],
     [-451, 662, 254],
     [-240, 21, -747],
     [731, 244, -388],
     [-55, -27, 285],
     [86, -257, 370],
     [393, 27, -12],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [-406, 356, 19],
     [331, 523, -462],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [612, 775, -391],
     [-55, -27, 285],
     [199, 69, 888],
     [588, -76, -739],
     [685, 683, 181],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [-406, 356, 19],
     [331, 523, -462],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [612, 775, -391],
     [-55, -27, 285],
     [199, 69, 888],
     [588, -76, -739],
     [685, 683, 181],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 933],
     [-237, 617, -764],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [581, 894, 584],
     [912, 461, -345],
     [-161, -991, -276],
     [86, -257, 370],
     [393, 27, -12],
     [-933, -427, -210],
     [-548, -122, -531],
     [590, -975, 73],
     [-836, -839, -27],
     [107, 644, 644],
     [745, 304, 422],
     [-672, 632, -870],
     [-868, 104, 224],
     [688, 181, 297],
     [-715, -529, -227],
     [124, 124, 77],
     [-430, -643, 805],
     [-209, -180, -544],
     [199, 69, 888],
     [588, -76, -739],
     [694, 704, 177],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 939],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [581, 894, 584],
     [912, 469, -348],
     [-161, -991, -276],
     [86, -257, 370],
     [393, 27, -12],
     [-548, -122, -531],
     [590, -975, 73],
     [-836, -839, -27],
     [152, -220, 940],
     [882, -563, -932],
     [-976, -717, -793],
     [842, -554, 247],
     [731, 244, -388],
     [-55, -27, 285],
     [588, -73, -741],
     [720, 697, 186],
     [750, 196, 1000],
     [281, -724, -363],
     [581, 894, 584],
     [231, 426, 148],
     [199, 55, 890],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [-547, 176, -58],
     [612, 775, -391],
     [967, -39, 709],
     [-406, 356, 19],
     [-451, 662, 254],
     [-240, 21, -747],
     [731, 244, -388],
     [-55, -27, 285],
     [86, -257, 370],
     [393, 27, -12],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [-406, 356, 19],
     [331, 523, -462],
     [580, 216, 805],
     [132, 540, -220],
     [509, 818, 753],
     [587, 269, 130],
     [581, 894, 584],
     [231, 426, 148],
     [199, 69, 888],
     [588, -73, -741],
     [720, 697, 186],
     [746, 196, 1000],
     [612, 775, -391],
     [-55, -27, 285],
     [199, 69, 888],
     [588, -76, -739],
     [685, 683, 181],
     [746, 196, 1000],
     [281, -724, -363],
     [-547, 176, -58],
     [612, 775, -391],
     [933, 554, -28],
     [469, 965, 367],
     [-3, -30, -710],
     [483, -211, 686],
     [-47, 533, 807],
     [-139, -119, 933],
     [-237, 617, -764],
     [-621, -766, -960],
     [-243, 31, -747],
     [-114, 995, -15],
     [107, 642, 644],
     [-683, -297, 363],
     [-687, -684, 912],
     [177, 479, 771],
     [576, 270, 145],
     [581, 894, 584],
     [912, 461, -345],
     [-161, -991, -276],
     [86, -257, 370],
     [393, 27, -12],
     [-933, -427, -210],
     [-548, -122, -531],
     [590, -975, 73],
     [-836, -839, -27],
     [107, 644, 644],
     [745, 304, 422],
     [-672, 632, -870],
     [-868, 104, 224],
     [688, 181, 297],
     ...]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10ff63250>]




    
![png](6/output_170_1.png)
    


The plot shows iterations of the genetic algorithm on the x-axis. Very likely, the result likely isn't great. But why? Let's look at the average population length.


```python
plt.plot(length_values)
```




    [<matplotlib.lines.Line2D at 0x10ff59f90>]




    
![png](6/output_172_1.png)
    


What you can see here is a phenomenon called _bloat_: Individuals grow in size through mutation and crossover, and the search has no incentive to reduce their size (adding a test can never decrease coverage; removing a test can). As a result the individuals just keep growing, and quickly eat up all the available resources for the search. How to deal with this problem will be covered in the next chapter.
