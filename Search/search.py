# Arshiya Ansari aa9yk
# Surbhi Singh ss4bz 


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


# Sources: https://www.hackerearth.com/practice/algorithms/graphs/depth-first-search/tutorial/
# http://www.mathcs.emory.edu/~cheung/Courses/171/Syllabus/11-Graph/dfs.html
# https://www.programiz.com/dsa/graph-dfs
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
        """

#    print("Start:", problem.getStartState())
#    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
#    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """ Start: (34, 16), Start's successors: [((34, 15), 'South', 1), ((33, 16), 'West', 1)]"""
    
    "*** YOUR CODE HERE ***"
    
    initState = problem.getStartState() # initialize the problem start state 
#    print(initState)
    visited = [] # establish my visited nodes
    fringe = util.Stack() # establish my stack for DFS
    fringe.push((initState, [], 0)) # push the initial state, list to hold directions (return type like tinyMaze), & cost
#    print(fringe)
    
    while not fringe.isEmpty(): # while there are values in the fringe
        curr, path, cost = fringe.pop() # pop off the current node, the current list of actions, and cost 
#        print(curr, actions, cost)
        if curr not in visited: # if the current node is not visited 
            visited.append(curr) # append the current node to visited 
            if problem.isGoalState(curr): # when the initital state becomes the goal state
                return path  # return the list of actions
            else:
                successors = problem.getSuccessors(curr) # get all the successors of the current node
                for nxt, action, cst in successors: # for next node, action, and cost in successors 
                    copy = path.copy()
                    copy.append(action)
                    curr_path = copy
                    new_cost = cost + cst  # add new extra cost to cost 
                    fringe.push((nxt, curr_path, new_cost)) # push this onto the fringe 
                
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Same code as above, but with a different fringe 
    initState = problem.getStartState() # initialize the problem start state 
#    print(initState)
    visited = [] # establish my visited nodes
    fringe = util.Queue() # establish my queue for BFS
    fringe.push((initState, [], 0)) # push the initial state, list to hold directions (return type like tinyMaze), & cost
#    print(fringe)
    
    while not fringe.isEmpty(): # while there are values in the fringe
        curr, path, cost = fringe.pop() # pop off the current node, the current list of actions, and cost 
#        print(curr, actions, cost)
        if curr not in visited: # if the current node is not visited 
            visited.append(curr) # append the current node to visited 
            if problem.isGoalState(curr): # when the initital state becomes the goal state
                return path  # return the list of actions
            else:
                successors = problem.getSuccessors(curr) # get all the successors of the current node
                for nxt, action, cst in successors: # for next node, action, and cost in successors 
                    copy = path.copy()
                    copy.append(action)
                    curr_path = copy
                    new_cost = cost + cst  # add new extra cost to cost 
                    fringe.push((nxt, curr_path, new_cost)) # push this onto the fringe 

# Sources: https://algorithmicthoughts.wordpress.com/2012/12/15/artificial-intelligence-uniform-cost-searchucs/
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    initState = problem.getStartState() # initialize the problem start state 
#    print(initState)
    visited = [] # establish my visited nodes
    fringe = util.PriorityQueue() # establish priority queue for UCS
    fringe.push((initState, [], 0), 0) # push initState, list of direction, cost AND priority 

    while not fringe.isEmpty(): # while there are values in the fringe
        curr, path, cost = fringe.pop()  # pop off the current node, the current list of actions, and cost 
 #        print(curr, actions, cost)
        if curr not in visited: # if the current node is not visited 
            visited.append(curr) # append the current node to visited 
            if problem.isGoalState(curr): # when the initital state becomes the goal state
                return path  # return the list of actions
            else:
                successors = problem.getSuccessors(curr) # get all the successors of the current node
                for nxt, action, cst in successors: # for next node, action, and cost in successors 
                    copy = path.copy()
                    copy.append(action)
                    curr_path = copy
                    new_cost = cost + cst # add new extra cost to cost 
                    fringe.push((nxt, curr_path, new_cost), new_cost) # push onto the fringe, but push new priority value which is cost 
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
# Sources: http://theory.stanford.edu/~amitp/GameProgramming/ImplementationNotes.html
# https://www.growingwiththeweb.com/2012/06/a-pathfinding-algorithm.html
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    initState = problem.getStartState() # initialize the problem start state 
#    print(initState)
    visited = [] # establish my visited nodes
    fringe = util.PriorityQueue() # establish priority queue for UCS
    fringe.push((initState, [], 0), 0) # push initState, list of direction, cost AND priority 

    while not fringe.isEmpty(): # while there are values in the fringe
        curr, path, cost = fringe.pop()  # pop off the current node, the current list of actions, and cost 
 #        print(curr, actions, cost)
        if curr not in visited: # if the current node is not visited 
            visited.append(curr) # append the current node to visited 
            if problem.isGoalState(curr): # when the initital state becomes the goal state
                return path  # return the list of actions
            else:
                successors = problem.getSuccessors(curr) # get all the successors of the current node
                for nxt, action, cst in successors: # for next node, action, and cost in successors 
                    copy = path.copy()
                    copy.append(action)
                    curr_path = copy
                    new_cost = cost + cst # add new extra cost to cost 
                    # print(heur)                 # heuristic is function defined as nullHeuristic
                    heur = heuristic(nxt, problem) + new_cost # calculate new heuristic value 
                    fringe.push((nxt, curr_path, new_cost), heur) # push onto the fringe, and push the heuristic as the new priority principle 


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
