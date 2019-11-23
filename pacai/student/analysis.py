"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    With 0 noise, our agent will never worry about making an unintended move,
    and not be scared of falling off the bridge.
    This allows the agent to cross the bridge with full confidence.
    """

    answerDiscount = 0.9
    answerNoise = 0

    return answerDiscount, answerNoise

def question3a():
    """
    By decreasing noise, we can make the agent choose riskier routes
    as the risk is decreased. We then make the agent
    take the smaller reward in less moves by increasing the discount rate,
    as this increases the penalty for movement.
    """

    answerDiscount = 0.3
    answerNoise = 0.01
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    By significantly increasing the discount rate
    from the default, we are able to force the agent to choose a
    smaller reward in less moves because there is a harsher
    penalty for making moves.
    """

    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    By simply reducing the noise from the default,
    we can reduce the agent's risk of falling off the cliff
    so that the agent chooses to take that route.
    """

    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    This default value avoids the cliff and prefers the distant exit.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    By significantly increasing noise, we can make the agent essentially do
    nothing since his policies have such a small effect.
    """

    answerDiscount = 0.9
    answerNoise = 0.9999
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    There was no possible combination of (epsilon, learning rate) that allowed
    the learning rate to be learned in 50 iterations
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
