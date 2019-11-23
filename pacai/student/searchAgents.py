"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
from pacai.core.directions import Directions
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core import distance
from pacai.student.search import uniformCostSearch

DEFAULT_COST_FUNCTION = lambda x: 1

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))
        self.costFn = DEFAULT_COST_FUNCTION

    def startingState(self):
        cornerState = [False, False, False, False]
        direction = ""
        return (self.startingPosition, direction, cornerState)

    def isGoal(self, state):
        for i in range(len(self.corners)):
            if state[0] == self.corners[i]:
                state[2][i] = True
        for item in state[2]:
            if not item:
                return False

        return True

    def successorStates(self, state):
        successors = []

        for action in Directions.CARDINAL:
            x, y = state[0]
            corners = state[2]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            if (not self.walls[nextx][nexty]):
                successors.append(((nextx, nexty), action, corners.copy()))

        self._numExpanded += 1
        if (state[0] not in self._visitedLocations):
            self._visitedLocations.add(state[0])
            self._visitHistory.append(state[0])
        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # *** Your Code Here ***
    currentPos = state[0]

    corners = problem.corners
    maxlen = distance.euclidean(corners[0], corners[3])
    midx = 0
    midy = 0
    for c in corners:
        midx += c[0]
        midy += c[1]
    middle = (midx / 4, midy / 4)
    cornerlist = state[2]
    i = 0
    for item in cornerlist:
        if item:
            i += 1
    if i == 0:  # if 0 corners discovered, return distance from middle
        r = maxlen - distance.euclidean(middle, currentPos)
    elif i == 1:  # if 1 corner found, return distance from that corner
        (midx, midy) = middle
        a = 0
        b = -1
        for item in corners:
            if item:
                b = a
            a += 1
        (cx, cy) = corners[b]
        x = (midx + cx) / 2
        y = (midy + cy) / 2
        r = maxlen - distance.euclidean((x, y), currentPos)
    elif i == 2:
        a = -1
        b = -1
        c = 0
        for item in corners:
            if item and a != -1:
                b = c
            elif item:
                a = c
            c += 1

        if set([a, b]) == set([0, 3]) or set([a, b]) == set([1, 2]):
            # returns distance from midpoint of corners if corners are next to eachother
            (xo, yo) = corners[a]
            (xt, yt) = corners[b]
            xz = (xo + xt) / 2
            yz = (yo + yt) / 2
            r = maxlen - distance.euclidean((xz, yz), currentPos)
        else:  # returns distance from a line going through diagonal corners
            denom = distance.euclidean(corners[a], corners[b])
            (xo, yo) = corners[a]
            (xt, yt) = corners[b]
            (xz, yz) = currentPos
            numer = ((yt - yo) * xz) - ((xt - xo) * yz) + (xt * yo) + (yt * xo)
            numer = abs(numer)
            r = maxlen - (numer / denom)
    elif i == 3:
        a = 0
        b = -1
        for item in cornerlist:
            if not item:
                b = a
            a += 1
        r = distance.euclidean(corners[b], currentPos)
    return (r * 3)

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """
    position, foodGrid = state

    # *** Your Code Here ***
    i = 0
    max = 0
    d = 0
    for x in foodGrid:
        j = 0
        for y in x:
            if y:
                d = distance.manhattan((i, j), position)
                if d > max:
                    max = d
            j += 1
        i += 1
    return max

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        # *** Your Code Here ***
        startpos = gameState.getPacmanPosition()
        min = float('inf')
        minpos = (0, 0)
        a = 0
        for i in gameState.getFood():
            b = 0
            for j in i:
                if j:
                    d = distance.euclidean((a, b), startpos)
                    if d < min:
                        min = d
                        minpos = (a, b)
                b += 1
            a += 1
        problem = AnyFoodSearchProblem(gameState, start = startpos, goal = minpos)
        actions = uniformCostSearch(problem)
        (x, y) = minpos
        gameState._food[x][y] = False
        return actions

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None, goal = None):
        super().__init__(gameState, goal = goal, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()

    def isGoal(self, state):
        (x, y) = state
        if self.food[x][y]:
            self._visitedLocations.add(state)
            self._visitHistory.append(state)
            return True
        return False

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
