def successor(route: List[int]) -> List[List[int]]:
    """ YOUR CODE HERE """
    length = len(route)
    result = []
    tempList = []
    for i in range(1, length * length):
        tempList = copy.deepcopy(route)
        x = 0
        y = 0
        while x == y:
            x = random.randint(0, length - 1)
            y = random.randint(0, length - 1)
        tempList[x], tempList[y] = tempList[y], tempList[x]
        result.append(tempList)
    return result
    """ YOUR CODE END HERE """

def evaluation_func(
    cities: int,
    distances: List[Tuple[int]],
    route: List[int]
) -> float:
    """ YOUR CODE HERE """
    distMatrix = [[0 for _ in range(cities)] for _ in range(cities)]
    for distance in distances:
        dist = distance[2]
        distMatrix[distance[0]][distance[1]] = dist
        distMatrix[distance[1]][distance[0]] = dist
    
    current = 0
    visited = [False] * cities
    
    for i in range(cities):        
        x = (i + 1) % cities
        currCity = route[i]
        nextCity = route[x]

        if (visited[currCity]):
            return float('inf')
        visited[currCity] = True
       
        current += distMatrix[currCity][nextCity]
    return -current
    """ YOUR CODE END HERE """

def hill_climbing(
    cities: int,
    distances: List[Tuple[int]],
    successor: Callable,
    evaluation_func: Callable
) -> List[int]:
    route = random.sample(list(range(cities)), cities)
    curr_hn = evaluation_func(cities, distances, route)
    while True:
        new_routes = successor(route)
        best_new_route = max(new_routes, key=lambda x: evaluation_func(cities, distances, x))
        h_n = evaluation_func(cities, distances, best_new_route)
        if h_n <= curr_hn:
            return route

        curr_hn = h_n
        route = best_new_route

### Task 1.6: Improved hill-climbing
def hill_climbing_improved(
    cities: int,
    distances: List[Tuple[int]],
    successor: Callable,
    evaluation_func: Callable,
    hill_climbing: Callable
) -> List[int]:
    """ YOUR CODE HERE """
    count = 5;
    bestPerformance = float("-inf")
    bestRoute = None
    while count > 0:
        route = hill_climbing(cities, distances, successor, evaluation_func)
        currPerformance = evaluation_func(cities, distances, route)
        count -= 1

        if (currPerformance > bestPerformance):
            count = 5
            bestPerformance = currPerformance
            bestRoute = route

    return bestRoute
    """ YOUR CODE END HERE """
