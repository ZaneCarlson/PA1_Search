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
from Code.util import manhattanDistance


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
    from util import Stack
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    """check if the start state is the goal state"""
    startState = problem.getStartState()
    if problem.isGoalState(startState):
        print("The Start State is the end state - somehow")
        return [] # is done if somehow the start state is also the end state

    stack = Stack() # Empty LIFO queue
    stack.push(startState)

    visited = {startState}

    # To keep track of what has been explored
    parent = {
        startState: (None, None) # (startState is the key : --> (previous state, action to reach state))
    } #

    # The Main DFS logic loop: Keep expanding the last state in stack until you reach the goal.
    while not stack.isEmpty():
        currentState = stack.pop() # (set of nodes we've discovered but haven't expanded yet) S.pop() --> expand a state and remove it from the Que

        if problem.isGoalState(currentState): # if this state is the goal
            actions = [] # list of actions taken to get to goal state
            current = currentState
            while parent[current][0] is not None:
                prev, act = parent[current] # unpack ( previous state, action taken)
                actions.append(act)
                current = prev
            actions.reverse()
            return actions


        # Successor puzzle, the next action i.e. 'down' 'left' ect, and cost step (always 1) is returned from problem.getSuccessorss(currentState)
        for (Successor, action, stepCost) in problem.getSuccessors(currentState):
            if Successor not in visited:
                visited.add(Successor)
                parent[Successor] = (currentState, action)
                stack.push(Successor)
    return []

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue



    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    """check if the start state is the goal state"""
    startState = problem.getStartState()
    if problem.isGoalState(startState):
        print("The Start State is the end state - somehow")
        return [] # is done if somehow the start state is also the end state

    Que = Queue() # Empty FIFO queue
    Que.push(startState)

    visited = {startState}

    # To keep track of what has been explored
    parent = {
        startState: (None, None) # (startState is the key : --> (previous state, action to reach state))
    } #

    # The Main BFS logic loop: Keep expanding until no states remain in the Queue
    while not Que.isEmpty():
        currentState = Que.pop() # (set of nodes we've discovered but haven't expanded yet) Q.pop() --> expand a state and remove it from the Que

        if problem.isGoalState(currentState): # if this state is the goal
            actions = [] # list of actions taken to get to goal state
            current = currentState
            while parent[current][0] is not None:
                prev, act = parent[current] # unpack ( previous state, action taken)
                actions.append(act)
                current = prev
            actions.reverse()
            return actions


        # Successor puzzle, the next action i.e. 'down' 'left' ect, and cost step (always 1) is returned from problem.getSuccessorss(currentState)
        for (Successor, action, stepCost) in problem.getSuccessors(currentState):
            if Successor not in visited:
                visited.add(Successor)
                parent[Successor] = (currentState, action)
                Que.push(Successor)
    return []

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    """check if the start state is the goal state"""
    startState = problem.getStartState()
    if problem.isGoalState(startState):
        print("The Start State is the end state - somehow")
        return []  # is done if somehow the start state is also the end state

    priorityQueue = PriorityQueue()  # Empty Pripority queue
    priorityQueue.push(startState, 1)

    visited = {startState}

    # To keep track of what has been explored
    parent = {
        startState: (None, None, None)  # (startState is the key : --> (previous state, action to reach state))
    }  #

    # The Main BFS logic loop: Keep expanding until no states remain in the Queue
    while not priorityQueue.isEmpty():
        currentState = priorityQueue.pop()  # (set of nodes we've discovered but haven't expanded yet) Q.pop() --> expand a state and remove it from the Que

        if problem.isGoalState(currentState):  # if this state is the goal
            actions = []  # list of actions taken to get to goal state
            current = currentState
            while parent[current][0] is not None:
                prev, act, stepCst = parent[current]  # unpack ( previous state, action taken, stepCost)
                actions.append(act)
                current = prev
            actions.reverse()
            return actions

        # Successor puzzle, the next action i.e. 'down' 'left' ect, and cost step (always 1) is returned from problem.getSuccessorss(currentState)
        for (Successor, action, stepCost) in problem.getSuccessors(currentState):
            if Successor not in visited:
                visited.add(Successor)
                parent[Successor] = (currentState, action, stepCost)
                priorityQueue.push(Successor, stepCost)
    return []






    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    """check if the start state is the goal state"""
    startState = problem.getStartState()
    if problem.isGoalState(startState):
        print("The Start State is the end state - somehow")
        return []  # is done if somehow the start state is also the end state

    priorityQueue = PriocfrityQueue()  # Empty Pripority queue
    priorityQueue.push(startState, 1)

    visited = {startState}

    # To keep track of what has been explored
    parent = {
        startState: (None, None, None)  # (startState is the key : --> (previous state, action to reach state))
    }  #

    # The Main BFS logic loop: Keep expanding until no states remain in the Queue
    while not priorityQueue.isEmpty():
        currentState = priorityQueue.pop()  # (set of nodes we've discovered but haven't expanded yet) Q.pop() --> expand a state and remove it from the Que

        if problem.isGoalState(currentState):  # if this state is the goal
            actions = []  # list of actions taken to get to goal state
            current = currentState
            while parent[current][0] is not None:
                prev, act, stepCst = parent[current]  # unpack ( previous state, action taken, stepCost)
                actions.append(act)
                current = prev
            actions.reverse()
            return actions

        # Successor puzzle, the next action i.e. 'down' 'left' ect, and cost step (always 1) is returned from problem.getSuccessorss(currentState)
        for (Successor, action, stepCost) in problem.getSuccessors(currentState):
            if Successor not in visited:
                visited.add(Successor)
                parent[Successor] = (currentState, action, stepCost)
                priorityQueue.push(Successor, stepCost)
    return []



    #Helper function
    def getHeuristic(state, problem=None):
        goal = [[1,2,3,],[8,0,4],[7,6,5]]
        goal_position = {goal[row][column]: (row, column) for row in range (3) for column in range (3)}

        total = 0
        for row in range(3):
            for column in range (3):
                tile = state.cells[row][column]
                if tile != 0:
                    total = total + manhattanDistance((row, column), goal_position[tile])



    problem.isGoalState()

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
