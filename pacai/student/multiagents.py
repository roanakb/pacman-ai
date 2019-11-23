import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        x = oldFood.getWidth()
        y = oldFood.getHeight()
        maxlen = distance.manhattan((x, y), (0, 0))
        halflen = maxlen
        minfooddist = maxlen
        baddist = maxlen
        badflag = True

        for i in range(x):
            for j in range(y):
                m = oldFood[i][j]
                if m:
                    d = distance.manhattan((i, j), newPosition)
                    if d < minfooddist:
                        minfooddist = d

        for i in range(len(newScaredTimes)):
            ghost = newGhostStates[i]
            d = distance.manhattan(newPosition, ghost._position)
            if newScaredTimes[i] == 0:
                badflag = False
                if d < baddist:
                    baddist = d

        if baddist == 0:
            return -100
        if badflag:
            baddist = 1
        elif baddist > halflen:
            baddist = 1
        ev = (-1 / baddist) + (successorGameState.getScore() / 2)
        if minfooddist == 0:
            ev += 10
        else:
            ev += (2 / minfooddist)
        if badflag:
            ev += 1
        elif baddist > halflen:
            ev += 1
        return ev

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        succs = state.getLegalActions(0)
        depth = 0
        v = -float('inf')
        bestaction = ''
        for s in succs:
            if s != 'Stop':
                j = state.generateSuccessor(0, s)
                m = self.min_value(j, depth, 1)
                w = v
                v = max(v, m)
                if v > w:
                    bestaction = s
        return bestaction

    def min_value(self, state, depth, agentind):
        actions = state.getLegalActions(agentind)
        if depth == self.getTreeDepth() or len(actions) == 0:
            j = self.getEvaluationFunction()(state)
            return j
        v = float('inf')
        numagents = len(state._agentStates)
        if agentind < (numagents - 1):
            for action in actions:
                if action != 'Stop':
                    newstate = state.generateSuccessor(agentind, action)
                    values = self.min_value(newstate, depth, agentind + 1)
                    v = min(v, values)
        else:
            for action in actions:
                if action != 'Stop':
                    newstate = state.generateSuccessor(agentind, action)
                    values = self.max_value(newstate, depth + 1)
                    v = min(v, values)
        return v

    def max_value(self, state, depth):
        actions = state.getLegalActions(0)
        if depth == self.getTreeDepth() or len(actions) == 0:
            return self.getEvaluationFunction()(state)
        v = -(float('inf'))
        for action in actions:
            if action != 'Stop':
                newstate = state.generateSuccessor(0, action)
                values = self.min_value(newstate, depth + 1, 1)
                v = max(v, values)
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        succs = state.getLegalActions(0)
        depth = 0
        v = -float('inf')
        pinf = float('inf')
        ninf = -float('inf')
        bestaction = ''
        for s in succs:
            if s != 'Stop':
                j = state.generateSuccessor(0, s)
                m = self.max_value(j, depth, ninf, pinf)
                w = v
                v = max(v, m)
                if v > w:
                    bestaction = s
        return bestaction

    def min_value(self, state, depth, agentind, alpha, beta):
        actions = state.getLegalActions(agentind)
        if depth == self.getTreeDepth() or len(actions) == 0:
            j = self.getEvaluationFunction()(state)
            return j
        v = float('inf')
        numagents = len(state._agentStates)
        if agentind < (numagents - 1):
            for action in actions:
                if action != 'Stop':
                    newstate = state.generateSuccessor(agentind, action)
                    values = self.min_value(newstate, depth, agentind + 1, alpha, beta)
                    v = min(v, values)
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
        else:
            for action in actions:
                if action != 'Stop':
                    newstate = state.generateSuccessor(agentind, action)
                    values = self.max_value(newstate, depth + 1, alpha, beta)
                    v = min(v, values)
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
        return v

    def max_value(self, state, depth, alpha, beta):
        actions = state.getLegalActions(0)
        if depth == self.getTreeDepth() or len(actions) == 0:
            return self.getEvaluationFunction()(state)
        v = -(float('inf'))
        for action in actions:
            if action != 'Stop':
                newstate = state.generateSuccessor(0, action)
                values = self.min_value(newstate, depth + 1, 1, alpha, beta)
                v = max(v, values)
                if v >= beta:
                    return v
                alpha = max(alpha, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        succs = state.getLegalActions(0)
        depth = 0
        v = -float('inf')
        bestaction = ''
        for s in succs:
            if s != 'Stop':
                j = state.generateSuccessor(0, s)
                m = self.min_value(j, depth, 1)
                w = v
                v = max(v, m)
                if v > w:
                    bestaction = s
        return bestaction

    def min_value(self, state, depth, agentind):
        actions = state.getLegalActions(agentind)
        if depth == self.getTreeDepth() or len(actions) == 0:
            j = self.getEvaluationFunction()(state)
            return j
        numagents = len(state._agentStates)
        e = 0
        if agentind < (numagents - 1):
            for action in actions:
                if action != 'Stop':
                    newstate = state.generateSuccessor(agentind, action)
                    values = self.min_value(newstate, depth, agentind + 1)
                    e += values
        else:
            for action in actions:
                if action != 'Stop':
                    newstate = state.generateSuccessor(agentind, action)
                    values = self.max_value(newstate, depth + 1)
                    e += values
        return e / len(actions)

    def max_value(self, state, depth):
        actions = state.getLegalActions(0)
        if depth == self.getTreeDepth() or len(actions) == 0:
            return self.getEvaluationFunction()(state)
        v = -(float('inf'))
        for action in actions:
            if action != 'Stop':
                newstate = state.generateSuccessor(0, action)
                values = self.min_value(newstate, depth + 1, 1)
                v = max(v, values)
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    newPosition = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

    # *** Your Code Here ***
    x = oldFood.getWidth()
    y = oldFood.getHeight()
    maxlen = distance.manhattan((x, y), (0, 0))
    halflen = maxlen
    minfooddist = maxlen
    scareddist = maxlen
    baddist = maxlen
    scaredflag = True
    badflag = True

    for i in range(x):
        for j in range(y):
            m = oldFood[i][j]
            if m:
                d = distance.manhattan(newPosition, (i, j))
                if d < minfooddist:
                    minfooddist = d

    for i in range(len(newScaredTimes)):
        ghost = newGhostStates[i]
        d = distance.manhattan(newPosition, ghost._position)
        if newScaredTimes[i] > 0:
            scaredflag = False
            if d < scareddist:
                scareddist = d
        else:
            badflag = False
            if d < baddist:
                baddist = d

    if baddist == 0:
        return -100
    if scaredflag:
        scareddist = 1
    if badflag:
        baddist = 1
    elif baddist > halflen:
        baddist = 1
    ev = (1 / scareddist) + (-1 / baddist) + (currentGameState.getScore() / 2)
    if minfooddist == 0:
        ev += 10
    else:
        ev += (2 / minfooddist)
    if scaredflag:
        ev -= 1
    if badflag:
        ev += 1
    elif baddist > halflen:
        ev += 1
    return ev

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
