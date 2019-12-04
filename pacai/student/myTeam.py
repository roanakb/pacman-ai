from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core import distance
from pacai.core.directions import Directions
from pacai.util import reflection, counter
from pacai.util import util
from pacai.agents.capture.capture import CaptureAgent
from pacai.bin.capture import AgentRules

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
    secondAgent = DefensiveReflexAgent(secondIndex)

    return [
        firstAgent,
        secondAgent
    ]

class OffensiveAgent(ReflexCaptureAgent):

    def __init__(self, index, evalFn = 'pacai.core.eval.score'):
        super().__init__(index)
        self._evaluationFunction = evalFn

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """
        self._treeDepth = 3
        super().registerInitialState(gameState)

    def getTreeDepth(self):
        return self._treeDepth

    def getAction(self, gameState):
        self.observationHistory.append(gameState)
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if (myPos != util.nearestPoint(myPos)):
            # We're halfway from one position to the next.
            return gameState.getLegalActions(self.index)[0]
        else:
            actions = gameState.getLegalActions(self.index)
            depth = 0
            max = -float('inf')
            pinf = float('inf')
            ninf = -float('inf')
            bestaction = ''
            for action in actions:
                if action != 'Stop' :
                    value = self.max_value(gameState, depth, ninf, pinf, action)
                    if value > max:
                        max = value
                        bestaction = action
            return bestaction

    # Min value calculates the optimal successor states for enemy agents
    def min_value(self, state, depth, agentind, alpha, beta, prevAction):
        successorState = state.generateSuccessor(agentind, prevAction)
        if depth == self.getTreeDepth():
            j = self.evaluate(state, prevAction)
            return j
        v = float('inf')
        opps = []
        if state.isOnBlueTeam(self.index):
            opps = state.getBlueTeamIndices()
        else:
            opps = state.getRedTeamIndices()
        if agentind == opps[0]:
            actions = successorState.getLegalActions(opps[1])
            for action in actions:
                if action != 'Stop':
                    values = self.min_value(successorState, depth, opps[1], alpha, beta, action)
                    v = min(v, values)
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
        else:
            actions = successorState.getLegalActions(self.index)
            for action in actions:
                if action != 'Stop':
                    values = self.max_value(successorState, depth + 1, alpha, beta, action)
                    v = min(v, values)
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
        return v

    # Max value calculates maximum evaluation value for Offensive Agent
    # of successor states given by enemies through min_value
    def max_value(self, state, depth, alpha, beta, prevAction):
        opps = []
        if state.isOnBlueTeam(self.index):
            opps = state.getBlueTeamIndices()
        else:
            opps = state.getRedTeamIndices()
        successorState = state.generateSuccessor(self.index, prevAction)
        actions = successorState.getLegalActions(opps[0])
        if depth == self.getTreeDepth() or len(actions) == 0:
            return self.evaluate(state, prevAction)
        v = -(float('inf'))
        for action in actions:
            if action != 'Stop':
                values = self.min_value(successorState, depth + 1, opps[0], alpha, beta, action)
                v = max(v, values)
                if v >= beta:
                    return v
                alpha = max(alpha, v)
        return v

    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()
        myAgent = successor.getAgentState(self.index)
        myPos = myAgent.getPosition()
        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            if minDistance == 0:
                features['distanceToFood'] = 10
            else:
                features['distanceToFood'] = 1 / minDistance

        opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [a for a in opponents if not a.isPacman() and a.getPosition() is not None]
        if len(defenders) > 0:
            minOppDist = min([self.getMazeDistance(myPos, opp.getPosition()) for opp in defenders])
            if minOppDist == 0:
                features['distFromDefender'] = 10
            else:
                features['distFromDefender'] = 1 / minOppDist
        else:
            features['distFromDefender'] = -1

        if not myAgent.isPacman():
            features['distFromDefender'] = 1

        capsuleList = self.getCapsules(successor)
        if len(capsuleList) > 0:
            minCapDist = min([self.getMazeDistance(myPos, food) for food in capsuleList])
            if minCapDist == 0:
                features['capsuleDist'] = 10
            else:
                features['capsuleDist'] = 1 / minCapDist
        else:
            features['capsuleDist'] = 10

        if (action == Directions.STOP):
            features['stop'] = 1
        else:
            features['stop'] = 0

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': 1,
            'distFromDefender': -2,
            'capsuleDist': 1,
            'stop': -100,
            'reverse': 20
        }

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        features['successorScore'] = 0
        features['distanceToFood'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
            features['onDefense'] = 1
            features['distFromDefender'] = 0
        else:
            features['onDefense'] = 0
            myPos = successor.getAgentState(self.index).getPosition()
            foodList = self.getFood(successor).asList()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
            features['successorScore'] = self.getScore(successor)
            features['invaderDistance'] = 0
            opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            defenders = [a for a in opponents if not a.isPacman() and a.getPosition() is not None]
            if len(defenders) > 0:
                minOppDist = min([self.getMazeDistance(myPos, opp.getPosition()) for opp in defenders])
                if minOppDist == 0:
                    features['distFromDefender'] = 10
                else:
                    features['distFromDefender'] = 1 / minOppDist
            else:
                features['distFromDefender'] = -1

            if not myState.isPacman():
                features['distFromDefender'] = 1

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2,
            'distFromDefender': -2,
            'distanceToFood': -1,
            'successorScore': 102
        }
