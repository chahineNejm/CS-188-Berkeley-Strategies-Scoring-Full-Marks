# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        legalMoves = successorGameState.getLegalActions()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Helper functions
        def manhattan_distance(x, y):
            return abs(x[0] - y[0]) + abs(x[1] - y[1])

        def distance_to_ghosts(pos, ghost_list):
            """
            Calculate the minimum Manhattan distance from the current position to any ghost.
            """
            return min(manhattan_distance(pos, ghost.getPosition()) for ghost in ghost_list)

        def closest_food(food, pos):
            """
            Calculate the Manhattan distance to the closest food pellet from the current position.
            """
            food_distances = [manhattan_distance(pos, f) for f in food.asList()]
            return min(food_distances) if food_distances else 0

        # Calculate the evaluation score
        ghost_dist = distance_to_ghosts(newPos, newGhostStates)
        food_dist = closest_food(newFood, newPos)
        food_left = len(newFood.asList())
        

        # Penalize being too close to ghosts
        if ghost_dist == 1:
            return -float("inf")

        # Reward for eating all the food
        if food_left == 0:
            return float("inf")

        # Calculate score considering distance to ghosts, closest food, and remaining food
        score = successorGameState.getScore()
        score += ghost_dist * 1  # Encourage staying away from ghosts
        score -= food_dist * 2     # Encourage moving towards food
        score -= food_left * 50    # Penalize based on the remaining food with abnormaly large values So that if Pacman left some food on the other end, he is still enticed to go get it.

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        pacman_index=0
        
        def assessing_minimax(agentIndex, depth, gameState):
        # Check for terminal state (win/lose)
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
        
        # Get legal actions for the current agent
            legalMoves = gameState.getLegalActions(agentIndex)
            if not legalMoves:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:  # Pacman (maximizer)
                bestScore = max(assessing_minimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in legalMoves)
                return bestScore
            else:  # Ghosts (minimizers)
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                bestScore = float("inf")
                for action in legalMoves:
                    score = assessing_minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action))
                    bestScore = min(bestScore, score)
            return bestScore

        # Initial call for Pacman's turn
        legalMoves = gameState.getLegalActions(0)
        scores = [assessing_minimax(pacman_index+1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        
        if bestIndices:
            chosenIndex = random.choice(bestIndices)

            return legalMoves[chosenIndex]
        else:
            return Directions.STOP 

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacman_index = 0


        def assessing_alphabeta(gameState, depth, agentIndex, alpha=-float("inf"), beta=float("inf")):
            """
            Args:
                depth (int): number of consecutive steps each agent takes , including th choice of not moving
                agentIndex (int): 0 for pacman and >=1 for ghosts
                alpha (float): running maximum for pacman nodes
                beta (float): running minimum for ghost nodes
            """
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            
            if agentIndex == 0:  # Pacman's turn (maximizer)
                max_value = float('-inf')
                best_action = None
                for action in gameState.getLegalActions(agentIndex):
                    new_state = gameState.generateSuccessor(agentIndex, action)
                    value, _ = assessing_alphabeta(new_state, depth, 1, alpha, beta)  # Call with the next agent (first ghost)
                    if value > max_value:
                        max_value = value
                        best_action = action
                    alpha = max(alpha, value)
                    if alpha > beta:  # Beta cutoff
                        break
                return max_value, best_action
            else:  # Ghosts turn (minimizers)
                min_value = float('inf')
                best_action = None
                next_agent = (agentIndex + 1) % gameState.getNumAgents()
                next_depth = depth + 1 if next_agent == 0 else depth
                for action in gameState.getLegalActions(agentIndex):
                    new_state = gameState.generateSuccessor(agentIndex, action)
                    value, _ = assessing_alphabeta(new_state, next_depth, next_agent, alpha, beta)
                    if value < min_value:
                        min_value = value
                        best_action = action
                    beta = min(beta, value)
                    if beta < alpha:  # Alpha cutoff
                        break
                return min_value, best_action

        # Initial call for Pacman's turn
        _, best_move = assessing_alphabeta(gameState, pacman_index, 0)
        return best_move
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacman_index=0
        
        def assessing_expectimax(agentIndex, depth, gameState):
        # Check for terminal state (win/lose)
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
        
        # Get legal actions for the current agent
            legalMoves = gameState.getLegalActions(agentIndex)
            if not legalMoves:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:  # Pacman (maximizer)
                bestScore = max(assessing_expectimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in legalMoves)
                return bestScore
            else:  # Ghosts (minimizers)
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                Expected_Score = 0
                for action in legalMoves:
                    score = assessing_expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action))
                    Expected_Score = Expected_Score+score
                return Expected_Score/len(legalMoves)

        # Initial call for Pacman's turn
        legalMoves = gameState.getLegalActions(0)
        scores = [assessing_expectimax(pacman_index+1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        if bestIndices:
            chosenIndex = random.choice(bestIndices)

            return legalMoves[chosenIndex]
        else:
            return Directions.STOP 

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
 
    def manhattan_distance(x, y):
        return abs(x[0] - y[0]) + abs(x[1] - y[1])

    def distance_to_ghosts(pos, ghost_list,ScaredTimes=None):
        """
            Calculate the minimum Manhattan distance from the current position to any ghost.
         """
        return min(manhattan_distance(pos, ghost.getPosition()) for ghost in ghost_list)

    def closest_food(food, pos):
        """
            Calculate the Manhattan distance to the closest food pellet from the current position.
        """
        food_distances = [manhattan_distance(pos, f) for f in food.asList()]
        return min(food_distances) if food_distances else 0

        # Calculate the evaluation score
    ghost_dist = distance_to_ghosts(Pos, GhostStates)
    food_dist = closest_food(Food, Pos)
    food_left = len(Food.asList())
        

        # Penalize being too close to ghosts
    if ghost_dist == 0:
        return -float("inf")

        # Reward for eating all the food
    if food_left == 0:
        return float("inf")

        # Calculate score considering distance to ghosts, closest food, and remaining food
    score = currentGameState.getScore()
    score += ghost_dist * 1  # Encourage staying away from ghosts
    score -= food_dist * 2     # Encourage moving towards food
    score -= food_left * 50    # Penalize based on the remaining food with abnormaly large values So that if Pacman left some food on the other end, he is still enticed to go get it.
    score += sum(ScaredTimes)
    return score


# Abbreviation
better = betterEvaluationFunction
