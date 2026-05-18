def uninformed_search(problem: cube.Cube):
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
    frontier.push(0, curr)

    while frontier:
        curr = frontier.pop()

        if curr.state in reached:
            continue

        reached.add(curr.state)

        if (problem.goal_test(curr.state)):
            fail = False
            break;

        for neighbours in expand(problem, curr):
            frontier.push(neighbours.cost   , neighbours)
            count += 1
    """ YOUR CODE END HERE """
    print(f"uninformed: {count}")
    if not fail:
        while curr.parent:
            solution.append(curr.act)
            curr = curr.parent
    solution.reverse()

    if fail:
        return False
    return solution