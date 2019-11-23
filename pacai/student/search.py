"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    q = Stack()
    v = []
    order = []
    order.append([])
    start = problem.startingState()
    flag = False
    if len(start) == 2:  # problem 4 or 7
        flag = True
    q.push(start)
    while not q.isEmpty():
        point = q.pop()
        c = order.pop(0)
        if flag:
            visit = point
        else:
            visit = (point[0], point[2])
        if visit not in v:
            if problem.isGoal(point):
                return c
            v.append(visit)
            for item in problem.successorStates(point):
                child = item[0]
                dir = item[1]
                if flag:
                    vis = child
                    item = child
                else:
                    vis = (item[0], item[2])
                if vis not in v:
                    q.push(item)
                    new_path = c.copy()
                    new_path.append(dir)
                    order.append(new_path)

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    q = Queue()
    v = []
    order = []
    order.append([])
    start = problem.startingState()
    flag = False
    if len(start) == 2:  # problem 4 or 7
        flag = True
    q.push(start)
    while not q.isEmpty():
        point = q.pop()
        c = order.pop(0)
        if flag:
            visit = point
        else:
            visit = (point[0], point[2])
        if visit not in v:
            if problem.isGoal(point):
                return c
            v.append(visit)
            for item in problem.successorStates(point):
                child = item[0]
                dir = item[1]
                if flag:
                    vis = child
                    item = child
                else:
                    vis = (item[0], item[2])
                if vis not in v:
                    q.push(item)
                    new_path = c.copy()
                    new_path.append(dir)
                    order.append(new_path)


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    # *** Your Code Here ***
    q = PriorityQueue()
    v = []
    n = problem.startingState()
    flag = False
    if len(n) == 2:  # problem 4 or 7
        flag = True
    actions = []
    q.push([n, actions], 0)

    while True:
        if q.isEmpty():
            return None
        (n, alist) = q.pop()
        actions = alist
        if problem.isGoal(n):
            return actions
        if not flag:
            visit = (n[0], n[2])
        else:
            visit = n
        if visit not in v:
            v.append(visit)
            for item in problem.successorStates(n):
                child = item[0]
                dir = item[1]
                actions.append(dir)
                if flag:
                    item = child
                    vis = child
                else:
                    vis = (item[0], item[2])
                cost = problem.actionsCost(actions)
                if vis not in v:
                    q.push([item, actions.copy()], cost)
                actions.pop(len(actions) - 1)

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    q = PriorityQueue()
    v = []
    n = problem.startingState()
    flag = False
    if len(n) == 2:  # problem 4 or 7
        flag = True
    actions = []
    q.push([n, actions], 0)

    while True:
        if q.isEmpty():
            return None
        (n, alist) = q.pop()
        actions = alist
        if problem.isGoal(n):
            return actions
        if not flag:
            visit = (n[0], n[2])
        else:
            visit = n
        if visit not in v:
            for item in problem.successorStates(n):
                child = item[0]
                dir = item[1]
                actions.append(dir)
                if flag:
                    item = (child[0], child[1])
                    vis = item
                else:
                    vis = (item[0], item[2])
                cost = problem.actionsCost(actions) + heuristic(item, problem)
                if vis not in v:
                    q.push([item, actions.copy()], cost)
                actions.pop(len(actions) - 1)
            if flag:
                visited = (n[0], n[1])
            else:
                visited = (n[0], n[2])
            v.append(visited)
