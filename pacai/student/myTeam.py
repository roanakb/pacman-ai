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

    def __init__(self, index):
        super().__init__(index)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.
        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """
        self._treeDepth = 4
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
                if action != 'Stop':
                    value = self.max_value(
                        gameState.generateSuccessor(self.index, action), depth, ninf, pinf)
                    if value > max:
                        max = value
                        bestaction = action
            return bestaction

    # Min value calculates the optimal successor states for enemy agents
    def min_value(self, state, depth, agentind, alpha, beta):
        if depth == self.getTreeDepth():
            return self.evaluate(state, None)
        v = float('inf')
        opps = []
        if state.isOnBlueTeam(self.index):
            opps = state.getRedTeamIndices()
        else:
            opps = state.getBlueTeamIndices()
        if agentind == opps[0]:
            actions = state.getLegalActions(opps[0])
            if len(actions) == 0:
                return self.evaluate(state, None)
            for action in actions:
                if action != 'Stop':
                    values = self.min_value(
                        state.generateSuccessor(opps[0], action), depth, opps[1], alpha, beta)
                    v = min(v, values)
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
        else:
            actions = state.getLegalActions(opps[1])
            if len(actions) == 0:
                return self.evaluate(state, None)
            for action in actions:
                if action != 'Stop':
                    values = self.max_value(
                        state.generateSuccessor(opps[1], action), depth + 1, alpha, beta)
                    v = min(v, values)
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
        return v

    # Max value calculates maximum evaluation value for Offensive Agent
    # of successor states given by enemies through min_value
    def max_value(self, state, depth, alpha, beta):
        opps = []
        if state.isOnBlueTeam(self.index):
            opps = state.getRedTeamIndices()
        else:
            opps = state.getBlueTeamIndices()
        actions = state.getLegalActions(self.index)
        if depth == self.getTreeDepth() or len(actions) == 0:
            return self.evaluate(state, None)
        v = -(float('inf'))
        for action in actions:
            if action != 'Stop':
                values = self.min_value(state.generateSuccessor(self.index, action),
                                        depth + 1, opps[0], alpha, beta)
                v = max(v, values)
                if v >= beta:
                    return v
                alpha = max(alpha, v)
        return v

    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = gameState
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
        defenders = [opp for opp in opponents
                     if not opp.isPacman() and opp.getPosition() is not None]

        if len(defenders) > 0 and myAgent.isPacman():
            opps = [opp for opp in defenders if opp._scaredTimer == 0]
            scaredys = [opp for opp in defenders if opp._scaredTimer > 0]
            if len(opps) > 0:
                oppD = min([self.getMazeDistance(myPos, opp.getPosition()) for opp in opps])
                if oppD == 0:
                    features['distFromDefender'] = 100
                elif oppD < 5:
                    features['distFromDefender'] = 0
                else:
                    features['distFromDefender'] = 1 / oppD
            else:
                features['distFromDefender'] = 0
            if len(scaredys) > 0:
                scareD = min([self.getMazeDistance(myPos, opp.getPosition()) for opp in scaredys])
                if scareD == 0:
                    features['distFromScared'] = 100
                else:
                    features['distFromScared'] = 1 / scareD
            else:
                features['distFromScared'] = 0
        else:
            if myAgent.isPacman():
                features['distFromDefender'] = -10
            else:
                features['distFromDefender'] = 0.7
            features['distFromScared'] = 0

        capsuleList = self.getCapsules(successor)
        if len(capsuleList) > 0:
            minCapDist = min([self.getMazeDistance(myPos, food) for food in capsuleList])
            if minCapDist == 0:
                features['capsuleDist'] = 1000
            else:
                features['capsuleDist'] = 1 / minCapDist
        else:
            features['capsuleDist'] = 10

        invaders = [a for a in opponents if a.isPacman() and a.getPosition() is not None]
        if (len(invaders) > 0):
            minInvDist = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])
            if minInvDist == 0:
                features['invaderDistance'] = 100
            else:
                features['invaderDistance'] = 1 / minInvDist
        else:
            features['invaderDistance'] = 10

        team = []
        if successor.isOnBlueTeam(self.index):
            team = successor.getBlueTeamIndices()
        else:
            team = successor.getRedTeamIndices()
        teammate = -1
        for num in team:
            if num != self.index:
                teammate = num
        otherAgent = successor.getAgentState(teammate)
        otherPos = otherAgent.getPosition()
        teamDist = self.getMazeDistance(myPos, otherPos)
        if teamDist == 0:
            features['teammateDist'] = 10
        else:
            features['teammateDist'] = 1 / teamDist

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': 2,
            'distFromDefender': -3,
            'capsuleDist': 4,
            'invaderDistance': 1,
            'distFromScared': 1,
            'teammateDist': -0.5
        }

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index):
        super().__init__(index)

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
                if action != 'Stop':
                    value = self.max_value(gameState.generateSuccessor(self.index, action),
                                           depth, ninf, pinf)
                    if value > max:
                        max = value
                        bestaction = action
            return bestaction

    # Min value calculates the optimal successor states for enemy agents
    def min_value(self, state, depth, agentind, alpha, beta):
        if depth == self.getTreeDepth():
            return self.evaluate(state, None)
        v = float('inf')
        opps = []
        if state.isOnBlueTeam(self.index):
            opps = state.getRedTeamIndices()
        else:
            opps = state.getBlueTeamIndices()
        if agentind == opps[0]:
            actions = state.getLegalActions(opps[0])
            if len(actions) == 0:
                return self.evaluate(state, None)
            for action in actions:
                if action != 'Stop':
                    values = self.min_value(
                        state.generateSuccessor(opps[0], action), depth, opps[1], alpha, beta)
                    v = min(v, values)
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
        else:
            actions = state.getLegalActions(opps[1])
            if len(actions) == 0:
                return self.evaluate(state, None)
            for action in actions:
                if action != 'Stop':
                    values = self.max_value(
                        state.generateSuccessor(opps[1], action), depth + 1, alpha, beta)
                    v = min(v, values)
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
        return v

    # Max value calculates maximum evaluation value for Offensive Agent
    # of successor states given by enemies through min_value
    def max_value(self, state, depth, alpha, beta):
        opps = []
        if state.isOnBlueTeam(self.index):
            opps = state.getRedTeamIndices()
        else:
            opps = state.getBlueTeamIndices()
        actions = state.getLegalActions(self.index)
        if depth == self.getTreeDepth() or len(actions) == 0:
            return self.evaluate(state, None)
        v = -(float('inf'))
        for action in actions:
            if action != 'Stop':
                values = self.min_value(
                    state.generateSuccessor(self.index, action), depth + 1, opps[0], alpha, beta)
                v = max(v, values)
                if v >= beta:
                    return v
                alpha = max(alpha, v)
        return v

    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = gameState

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

            foodDef = self.getFoodYouAreDefending(successor).asList()
           
            if len(foodDef) > 0:
                for enemy in invaders:
                    for Dfood in foodDef:
                        minFoodDist = (self.getMazeDistance(enemy.getPosition(), Dfood))

            if minFoodDist < 10:
                features['onDefense'] = 1
                features['RunBack'] = 1
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
                # scaredys = [opp for opp in defenders if opp._scaredTimer > 0]
                # if len(scaredys) > 0:
                #     scareD = min([self.getMazeDistance(myPos, opp.getPosition()) for opp in scaredys])
                #     if scareD == 0:
                #         features['distFromScared'] = 100
                #     else:
                #         features['distFromScared'] = 1 / scareD
                # else:
                #     features['distFromScared'] = 0
            else:
                features['distFromDefender'] = -1
            
        team = []
        if successor.isOnBlueTeam(self.index):
            team = successor.getBlueTeamIndices()
        else:
            team = successor.getRedTeamIndices()
        teammate = -1
        for num in team:
            if num != self.index:
                teammate = num
        otherAgent = successor.getAgentState(teammate)
        otherPos = otherAgent.getPosition()
        teamDist = self.getMazeDistance(myPos, otherPos)
        if teamDist == 0:
            features['teammateDist'] = 10
        else:
            features['teammateDist'] = 1 / teamDist

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -100,
            'distanceToFood': -1,
            'successorScore': 1000,
            'teammateDist': -0.2,
            # 'distFromScared': 1,
            'distFromDefender': -15,
            'RunBack': 100
        }

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights