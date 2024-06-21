# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        for _ in range(self.iterations):
            
            #initialization
            Vk=util.Counter()
            
            # Iterate over all states in the MDP
            for state in self.mdp.getStates():
                Q_list=[self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
                if Q_list:
                    Vk[state]=max(Q_list)
            
            # Update the agent's value estimates with the newly computed values for this iteration       
            self.values= Vk
                 

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
    
        def get_probabilistic_utility(trans_state,probability):
            """
            Calculate the utility for a specific transition state and probability. This utility is the expected reward for moving to the transition state from the current state via the specified action

            :param trans_state: The state that may be reached from the current state and action.
            :param probability: The probability of transitioning to this state when taking the action.
            :return: The weighted utility of transitioning to this state.
            """
            return probability*(self.mdp.getReward(state, action, trans_state)+self.discount*self.getValue(trans_state))
        
        # Compute the Q-value by summing the probabilistic utilities of all possible transitions
        
        trasitions_and_Probs=self.mdp.getTransitionStatesAndProbs(state, action)
        return sum([get_probabilistic_utility(state,proba) for state,proba in trasitions_and_Probs])

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Initialization
        running_Q=float("-inf")
        best_action=None
        
        # Retrieve the list of possible actions from the current state
        action_list=self.mdp.getPossibleActions(state)
        
        
        for action in action_list:
            current_Q=self.getQValue(state,action)
            if current_Q>running_Q:
                # Update best_action and associated Q to this action
                best_action=action
                running_Q=current_Q
                
        return best_action
        
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
