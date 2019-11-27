from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.util import counter
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
from pacai.util import counter

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
        secondAgent,
    ]

class OffensiveAgent(ReflexCaptureAgent):
    """
      A reflex agent that seeks food.
      This agent will give you an idea of what an offensive agent might look like,
      but it is by no means the best or only way to build an offensive agent.
      """

    def __init__(self, index, **kwargs):
        super().__init__(index)

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
                features['distanceToFood'] = 2
            else:
                features['distanceToFood'] = 1 / minDistance

        features['numFood'] = -len(foodList)
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


        return features


    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': 8,
            'numFood': 1,
            'distFromDefender': -3,
            'capsuleDist': 10
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

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

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
            'reverse': -2
        }
