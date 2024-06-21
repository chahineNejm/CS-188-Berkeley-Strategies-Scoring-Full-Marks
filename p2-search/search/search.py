# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    start=problem.getStartState()
    
    if problem.isGoalState(start):
        return []# Return an empty path if the start is already the goal
    
    # Initialize the frontier with the start state and the visited set
    visited=set()
    frontier=util.Stack()
    frontier.push(start)
    parent={start:(None,None)}
    
        
    def get_path(parent, vertex):
        """Function to construct the path from start to the goal

        Args:
            parent (dict): contains parent and direction
            vertex (tuple): the point from which to start computing (arrival point)

        Returns:
            list:correct path
        """
        suite = []
        while vertex is not None:
            pvtx, direction = parent[vertex]
            if pvtx is None:
                break
            suite.insert(0, direction)  
            vertex = pvtx
        return suite
    
     # Explore the frontier until it is empty
    while not frontier.isEmpty():
        vertex = frontier.pop() 
        if vertex not in visited:
            visited.add(vertex)
            
            if problem.isGoalState(vertex):
                
                return get_path(parent,vertex)
            # Expand the current vertex to its neighbors
            for neighbor, direction ,_ in problem.getSuccessors(vertex):
                
                if neighbor not in visited:
                    frontier.push(neighbor)
                    parent[neighbor]=(vertex,direction)
                   

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    
    if problem.isGoalState(start):
        return []# Return an empty path if the start is already the goal
    
    # Initialize the frontier with the start state and the visited set
    visited = set()
    frontier = util.Queue()
    frontier.push(start)
    visited.add(start)  
    parent= {start: (None, None)}

    def get_path(parent, vertex):
        """Function to construct the path from start to the goal

        Args:
            parent (dict): contains parent and direction
            vertex (tuple): the point from which to start computing (arrival point)

        Returns:
            list:correct path
        """
        suite = []
        while vertex is not None:
            pvtx, direction = parent[vertex]
            if pvtx is None:
                break
            suite.insert(0, direction)  
            vertex = pvtx
        return suite
     # Explore the frontier until it is empty
    while not frontier.isEmpty():
        vertex = frontier.pop()
        if problem.isGoalState(vertex):
            return get_path(parent, vertex)
        # Expand the current vertex to its neighbors
        for neighbor, direction ,_ in problem.getSuccessors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)  
                frontier.push(neighbor)
                parent[neighbor] = (vertex, direction)



def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    start = problem.getStartState()
    
    if problem.isGoalState(start):
        return []# Return an empty path if the start is already the goal
    
    # Initialize the frontier with the start state and the visited set
    visited = set()
    frontier = util.PriorityQueue()
    frontier.push(start,0)
    parent = {start: (None, None,0)}
   
    def get_path(parent, vertex):
        """Function to construct the path from start to the goal

        Args:
            parent (dict): contains parent and direction
            vertex (tuple): the point from which to start computing (arrival point)

        Returns:
            list:correct path
        """
        suite = []
        while vertex is not None:
            pvtx, direction , _ = parent[vertex]
            if pvtx is None:
                break
            suite.insert(0, direction)  
            vertex = pvtx
        return suite
     # Explore the frontier until it is empty
    while not frontier.isEmpty():
        
        vertex = frontier.pop()
        current_prio=parent[vertex][2]
        
        if vertex not in visited:
            visited.add(vertex)
            
            if problem.isGoalState(vertex):
                return get_path(parent,vertex)
            # Expand the current vertex to its neighbors
            for neighbor, direction ,priority_nghbr in problem.getSuccessors(vertex):
            
                if neighbor not in visited:
                    total_prio=current_prio+priority_nghbr
                    
                    frontier.update(neighbor,total_prio)
                    if neighbor not in parent or total_prio<parent[neighbor][2]:
                        parent[neighbor]=(vertex,direction,total_prio)
                
    


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    start = problem.getStartState()
    
    if problem.isGoalState(start):
        return []
    
    # Initialize the visited set, frontier priority queue, and parent map including expected travel
    visited = set()
    frontier = util.PriorityQueue()
    frontier.push(start,heuristic(start,problem))
    parent = {start: (None, None,0)}
   
    def get_path(parent, vertex):
        
        """
        Function to construct the path from start to the goal

        Args:
            parent (dict): contains parent and direction
            vertex (tuple): the point from which to start computing (arrival point)

        Returns:
            list:correct path
        """
        
        suite = []
        while vertex is not None:
            pvtx, direction , _ = parent[vertex]
            if pvtx is None:
                break
            suite.insert(0, direction)  
            vertex = pvtx
        return suite
    
    while not frontier.isEmpty():
        # Get the vertex with the lowest heuristic cost
        vertex = frontier.pop()
        current_run=parent[vertex][2]
        
        if vertex not in visited:
            visited.add(vertex)
            
            if problem.isGoalState(vertex):
                return get_path(parent,vertex)
            # Explore each neighbor of the current vertex
            for neighbor, direction ,step_cost in problem.getSuccessors(vertex):
            
                if neighbor not in visited:
                    total_run=current_run+step_cost
                    total_estimate=total_run+heuristic(neighbor,problem)
                    frontier.update(neighbor,total_estimate)
                    # Update the parent if a cheaper path to neighbor is found
                    if neighbor not in parent or total_run<parent[neighbor][2]:
                        parent[neighbor]=(vertex,direction,total_run)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
