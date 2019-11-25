from pacai.core import distance
from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = OffensiveAgent(firstIndex)
    secondAgent = reflection.qualifiedImport(second)

    return [
        firstAgent,
        secondAgent(secondIndex)
    ]



class OffensiveAgent(CaptureAgent):

    def __init__(self, index, evalFn = 'pacai.core.eval.score', depth = 2):
        super().__init__(index)
        self._evaluationFunction = reflection.qualifiedImport(evalFn)
        self._treeDepth = int(depth)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

    def chooseAction(self, gameState):
        action = self.getAct(gameState)
        print(action)
        return action

    def getEvaluationFunction(self):
        return self.betterEvaluationFunction

    def getTreeDepth(self):
        return self._treeDepth

    def getAct(self, state):
        succs = state.getLegalActions(self.index)
        depth = 0
        v = -float('inf')
        pinf = float('inf')
        ninf = -float('inf')
        bestaction = ''
        for s in succs:
            if s != 'Stop':
                j = state.generateSuccessor(self.index, s)
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
        actions = state.getLegalActions(self.index)
        if depth == self.getTreeDepth() or len(actions) == 0:
            return self.getEvaluationFunction()(state)
        v = -(float('inf'))
        for action in actions:
            if action != 'Stop':
                newstate = state.generateSuccessor(self.index, action)
                values = self.min_value(newstate, depth + 1, 1, alpha, beta)
                v = max(v, values)
                if v >= beta:
                    return v
                alpha = max(alpha, v)
        return v

    def betterEvaluationFunction(self, currentGameState):
        """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

        DESCRIPTION: <write something here so we know what you did>
        """

        # Compute distance to the nearest food.
        foodList = self.getFood(currentGameState).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = currentGameState.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        if minDistance == 0:
            return 10
        return 1 / minDistance