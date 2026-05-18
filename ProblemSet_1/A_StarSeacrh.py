def heuristic_func(problem: cube.Cube, state: cube.State) -> float:
    h_n = 0.0
    goals = problem.goal
    dimension = goals.shape[0]
    """ YOUR CODE HERE """
    for i in range(dimension*dimension):
        if goals.layout[i] != state.layout[i]:
            h_n += 1
    """ YOUR CODE END HERE """
    return h_n/dimension

### Task 2.2 - Implement A* search 
def expand(problem, node): # Generate children nodes to be explored
    for act in problem.actions(node.state):
        state = problem.result(node.state, act)
        current_cost = problem.path_cost(node.cost, node.state, act, state)
        child = Node(parent=node, 
                    act=act, 
                    state=state, 
                    cost=current_cost)
        yield child

def astar_search(problem: cube.Cube, heuristic_func: Callable):
    fail = True
    solution = []
    count = 0

    reached = set()
    frontier = PriorityQueue()
    initial = problem.initial
    curr = Node(parent=None, 
                act=None, 
                state=initial,
                cost=0)
    """ YOUR CODE HERE """
    frontier.push(heuristic_func(problem, curr.state), curr)

    while frontier:
        curr = frontier.pop()

        if curr.state in reached:
            continue

        reached.add(curr.state)

        if (problem.goal_test(curr.state)):
            fail = False
            break;

        for neighbours in expand(problem, curr):
            frontier.push(heuristic_func(problem, neighbours.state) + neighbours.cost   , neighbours)
            count+=1
    """ YOUR CODE END HERE """
    print(f"informed: {count}")
    if not fail:
        while curr.parent:
            solution.append(curr.act)
            curr = curr.parent
    solution.reverse()

    if fail:
        return False
    return solution

