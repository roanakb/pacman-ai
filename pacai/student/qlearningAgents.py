import random

from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection, probability, counter

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: I created a qValue counter in init
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.qVals = counter.Counter()

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        return self.qVals[(state, action)]

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0
        max = -float('inf')
        for action in actions:
            val = self.getQValue(state, action)
            if val > max:
                max = val
        return max

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
        max = -float('inf')
        maxPair = ()
        for action in actions:
            val = self.getQValue(state, action)
            if val > max:
                max = val
                maxPair = (state, action)
        return maxPair[1]

    def getAction(self, state):
        ep = self.epsilon
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
        if probability.flipCoin(ep):
            return random.choice(actions)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        actions = self.getLegalActions(nextState)
        value = self.getQValue(state, action)
        max = -float('inf')
        flag = False
        for act in actions:
            val = self.getQValue(nextState, act)
            if val > max:
                flag = True
                max = val
        sample = reward
        if flag:
            sample = reward + (self.getDiscountRate() * max)
        alpha = self.getAlpha()
        upd = value + (alpha * (sample - value))
        self.qVals[(state, action)] = upd

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)
        self.weight = 0
        self.weights = counter.Counter()
        # You might want to initialize weights here.

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(self, state, action)
        total = 0
        for feature in features:
            total += features[feature] * self.weights[feature]
        return total

    def update(self, state, action, nextState, reward):
        features = self.featExtractor.getFeatures(self, state, action)
        alpha = self.getAlpha()
        for feature in features:
            actions = self.getLegalActions(nextState)
            value = self.getQValue(state, action)
            max = -float('inf')
            flag = False
            for act in actions:
                val = self.getQValue(nextState, act)
                if val > max:
                    flag = True
                    max = val
            sample = reward
            if flag:
                sample = reward + (self.getDiscountRate() * max)
            sample -= value
            self.weights[feature] += alpha * sample * features[feature]

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            return
