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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        # if the next move wins the game, return a very large score
        score = 0
        if successorGameState.isWin():
            return 9999999999999
        ghostPos = currentGameState.getGhostPosition(1)
        distToGhost = util.manhattanDistance(ghostPos, newPos) + 1
        # print("DIST TO GHOST " + str(distToGhost))
        closestfood = 1000000
        foodlist = newFood.asList()
        for foodpos in foodlist:
            thisdist = util.manhattanDistance(foodpos, newPos)
            if (thisdist < closestfood):
                closestfood = thisdist
        # print("DIST TO CLOSESET FOOD " + str(closestfood))
        # if the pacman gets stuck, reduce the score
        if action == Directions.STOP:
            score -= 50
        # if the number of food pellets reduces in successor game state, increase the score
        if (successorGameState.getNumFood() <currentGameState.getNumFood()):
            score += 50
        # # if pacman's new position is at a capsule, increase score
        # for state in currentGameState.getCapsules():
        #     if(successorGameState.getPacmanPosition() == state):
        #         score += 150
        # the further the food is, the worse the score
        return score + successorGameState.getScore() - closestfood + distToGhost

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        def max_agent(gameState, depth):
            #initialize score to small number and action to none
            score = -999999999
            action = None
            #when you reach a terminal state, return score and no action
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (gameState.getScore(), None)
            #calculate score for each possible action by recursively calling min_agent
            for currAction in gameState.getLegalActions(0) :
                currgameState = gameState.generateSuccessor(0, currAction)
            #recursively call the min_agent to get the pacmans next action and score
                min_score = min_agent(currgameState, depth,1)[0]
            #finds the largest score for each possible action
                if (min_score > score):
                    score, action = min_score, currAction
            return (score, action)

        def min_agent(gameState, depth, ghost_num):
            #initialize score to large number and action to none
            score = 999999999
            action = None
            #when you reach a terminal state, return score and no action
            if gameState.isWin() or gameState.isLose():
                return (gameState.getScore(), None)
            #calculate score for each possible action by recursively calling min_agent
            for currAction in gameState.getLegalActions(ghost_num):
                currgameState = gameState.generateSuccessor(ghost_num, currAction)
            # if you are on the last ghost, it is actually pacman, so call max_agent
                if (ghost_num == gameState.getNumAgents() - 1):
                    next_score = max_agent(currgameState, depth + 1)[0]
                else:
                    next_score = min_agent(currgameState,depth, ghost_num+1)[0]
                # finds the smallest score for each possible action
                if (next_score < score):
                    score, action = next_score, currAction
            return (score, action)
        #returns minimax action form current gamestate
        return max_agent(gameState, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # prune a node when alpha becomes greater than or equal to beta 
        "*** YOUR CODE HERE ***"
        # initialize alpha and beta values to worst possible cases
        alpha = -(float("inf"))
        beta = float("inf")

        # alpha is the best choice so far for max agent (highest possible value)
        def max_agent(gameState, depth, alpha, beta):
            # initialize score to small number and action to none
            score = -999999999
            action = None
            #when you reach a terminal state, return score and no action
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (gameState.getScore(), None)
            #calculate score for each possible action by recursively calling min_agent
            for currAction in gameState.getLegalActions(0) :
                currgameState = gameState.generateSuccessor(0, currAction)
            #recursively call the min_agent to get the pacmans next action and score, 1 represents the ghost number
                min_score = min_agent(currgameState, depth, 1, alpha, beta)[0]
                #finds the largest score for each possible action
                if (min_score > score):
                    score, action = min_score, currAction
                # if our beta value is less than score, don't prune node
                if (beta < score):
                    return (score, action)
                # get new max alpha value 
                alpha = max(alpha, score)
            return (score, action)

        def min_agent(gameState, depth, ghost_num, alpha, beta):
            # initialize score to small number and action to none
            score = 999999999
            action = None
            #when you reach a terminal state, return score and no action
            if gameState.isWin() or gameState.isLose():
                return (gameState.getScore(), None)
            #calculate score for each possible action by recursively calling min_agent
            for currAction in gameState.getLegalActions(ghost_num):
                currgameState = gameState.generateSuccessor(ghost_num, currAction)
                # if you are on the last ghost, it is actually pacman, so call max_agent
                if (ghost_num == gameState.getNumAgents() - 1):
                    # get the next score of the max agent 
                    next_score = max_agent(currgameState, depth + 1, alpha, beta)[0]
                else:
                    # get the next score of the min agent 
                    next_score = min_agent(currgameState, depth, ghost_num + 1, alpha, beta)[0]
                # finds the smallest score for each possible action
                if (score > next_score):
                    score, action = next_score, currAction
                # if our alpha value is greater than score, don't prune
                if (alpha > score):
                    return (score, action)
                # get the new min beta value 
                beta = min(beta, score)
            return (score, action)
            # returns agent 
        return max_agent(gameState, 0, alpha, beta)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth):
            #initialize score to small number and action to none
            score = -999999999
            action = None
            #when you reach a terminal state, return score and no action
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (gameState.getScore(), None)
            #calculate score for each possible action by recursively calling min_agent
            for currAction in gameState.getLegalActions(0) :
                currgameState = gameState.generateSuccessor(0, currAction)
            #recursively call the min_agent to get the pacmans next action and score
                min_score = exp_value(currgameState, depth,1)[0]
            #finds the largest score for each possible action
                if (min_score > score):
                    score, action = min_score, currAction
            return (score, action)

        def exp_value(gameState, depth, ghost_num):
            #initialize score to large number and action to none
            score = 0
            action = None
            #when you reach a terminal state, return score and no action
            if gameState.isWin() or gameState.isLose():
                return (gameState.getScore(), None)
            #calculate score for each possible action by recursively calling min_agent
            for currAction in gameState.getLegalActions(ghost_num):
                currgameState = gameState.generateSuccessor(ghost_num, currAction)
            # if you are on the last ghost, it is actually pacman, so call max_agent
                if (ghost_num == gameState.getNumAgents() - 1):
                    next_score = max_value(currgameState, depth + 1)[0]
                else:
                    next_score = exp_value(currgameState,depth, ghost_num+1)[0]
                # finds the average score for each possible action
                actions = gameState.getLegalActions(ghost_num)
                avg_val = next_score/len(actions)
                score += avg_val 
            return (score, action)
        #returns minimax action form current gamestate
        return max_value(gameState, 0)[1]


        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    "*** YOUR CODE HERE ***"    
    score = 0
    #if game is over, return highest score
    if successorGameState.isWin():
        return 9999999999999
    for i in range(len(newGhostStates)):
        ghostPos = currentGameState.getGhostPosition(i+1)
        if newScaredTimes[i] > 0:
            # if the ghost is not scared, decrease the score
            score -=  (2/util.manhattanDistance(ghostPos, newPos)+ 1) 
        else:
            # if the ghost is scared, increase the score
            score +=  (2/util.manhattanDistance(ghostPos, newPos)+1) 
    # distToGhost = util.manhattanDistance(ghostPos, newPos) + 1
    closestfood = 99999999999
    foodlist = newFood.asList()
    for foodpos in foodlist:
        thisdist = util.manhattanDistance(foodpos, newPos)
        if (thisdist < closestfood):
            closestfood = thisdist
    if (successorGameState.getNumFood() <currentGameState.getNumFood()):
        score += 50
    return score + successorGameState.getScore() + (1/closestfood) #-  (1/distToGhost)

# Abbreviation
better = betterEvaluationFunction
