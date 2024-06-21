# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

"""
Observations:
1. There are two exits: the further exit is two steps beyond the closer one. The valuation of the distant exit
   must compensate for the extra steps, typically calculated as `second_exit * discount^2 + living_reward * (1 + discount)`.
   This should be less than the reward for the closer exit for the closer exit to be preferred.
2. Noise should be minimized when taking a risky path to avoid accidental moves into dangerous areas (cliffs).
3. A high living reward can encourage the agent to avoid exits in favor of continuing the episode, leading to a preference
   for safer, longer paths or indefinite continuation.
"""

def question2a():
    """
    Prefer the close exit with a risky path.
    """
    answerDiscount = 0.5  # A low discount factor to prioritize immediate rewards.
    answerNoise = 0       # No noise to ensure predictability in movements.
    answerLivingReward = -1  # A negative living reward to encourage quick termination at the close exit.
    return answerDiscount, answerNoise, answerLivingReward

def question2b():
    """
    Prefer the close exit with a safe path.
    """
    answerDiscount = 0.3  # Emphasize immediate rewards, avoiding distant exits.
    answerNoise = 0.1     # Some noise, but not too high, maintaining a balance.
    answerLivingReward = 0  # Neutral living reward, not discouraging or encouraging continuation.
    return answerDiscount, answerNoise, answerLivingReward

def question2c():
    """
    Prefer the distant exit with a risky path.
    """
    answerDiscount = 1    # High discount to value delayed gratification equally with immediate gains.
    answerNoise = 0       # No noise to avoid accidental movement into the cliff.
    answerLivingReward = -0.1  # Slight negative to encourage reaching the exit over prolonging the episode.
    return answerDiscount, answerNoise, answerLivingReward

def question2d():
    """
    Prefer the distant exit with a safe path.
    """
    answerDiscount = 0.7  # Higher discount to value the distant, more rewarding exit.
    answerNoise = 0.2     # Moderate noise to reduce the likelihood of taking risky paths.
    answerLivingReward = 0  # Neutral living reward.
    return answerDiscount, answerNoise, answerLivingReward

def question2e():
    """
    Avoid both exits and prefer to continue indefinitely.
    """
    answerDiscount = 0   # A zero discount factor makes future rewards meaningless.
    answerNoise = 0      # No noise for predictable behavior.
    answerLivingReward = 1  # High positive living reward to incentivize continuation.
    return answerDiscount, answerNoise, answerLivingReward

    

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
