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


from re import M
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
        newPos = successorGameState.getPacmanPosition()#Pos of pac man
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        FoodList = newFood.asList()
        FoodList.sort(key = lambda pos: manhattanDistance(pos, newPos))
        GhostList = []
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                GhostList.append(ghost.getPosition())
        GhostList.sort(key = lambda pos: manhattanDistance(pos, newPos))
        if len(FoodList) > 0:
            foodeval = manhattanDistance(newPos, FoodList[0])
        else:
            foodeval = 0
        if len(GhostList) > 0:
            ghosteval = manhattanDistance(newPos, GhostList[0])
        else:
            ghosteval = 0
        evaluation = successorGameState.getScore() - 10/(ghosteval+1) -  foodeval/5
        return evaluation

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
        
        depth = self.depth # depth of minmax
        score, action = self.MinMax(gameState, 0, depth)
        return action #It should be an action
    def MinMax(self,gameState, agentIndex, depth):
        if gameState.isWin():
            return  self.evaluationFunction(gameState), Directions.STOP
        if gameState.isLose():
            return  self.evaluationFunction(gameState), Directions.STOP
        if depth == 0:
            return  self.evaluationFunction(gameState), Directions.STOP
        if agentIndex == 0:
            return self.Max(gameState, agentIndex, depth)
        else:
            return self.Min(gameState, agentIndex, depth)
        util.raiseNotDefined()

    def Max(self, gameState, agentIndex, depth):
        if agentIndex == gameState.getNumAgents() -1:
            next_agent = 0
            next_depth = depth-1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        actions = gameState.getLegalActions(agentIndex)
        max_score = -1e8
        max_action = Directions.STOP
        for act in actions:
            next_state = gameState.generateSuccessor(agentIndex, act)
            next_score, anotheract = self.MinMax(next_state, next_agent, next_depth)
            if next_score > max_score:
                max_score = next_score
                max_action = act
        return max_score, max_action

        util.raiseNotDefined()

    def Min(self, gameState, agentIndex, depth):
        if agentIndex == gameState.getNumAgents() -1:
            next_agent = 0
            next_depth = depth-1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        actions = gameState.getLegalActions(agentIndex)
        min_score = 1e8
        min_action = Directions.STOP
        for act in actions:
            next_state = gameState.generateSuccessor(agentIndex, act)
            next_score, anotheract = self.MinMax(next_state, next_agent, next_depth)
            if next_score < min_score:
                min_score = next_score
                min_action = act
        return min_score, min_action
        


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth # depth of minmax
        score, action = self.MinMax(gameState, 0, depth, -1e8, 1e8)
        return action #It should be an action
    def MinMax(self,gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin():
            return  self.evaluationFunction(gameState), Directions.STOP
        if gameState.isLose():
            return  self.evaluationFunction(gameState), Directions.STOP
        if depth == 0:
            return  self.evaluationFunction(gameState), Directions.STOP
        if agentIndex == 0:
            return self.Max(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.Min(gameState, agentIndex, depth, alpha, beta)
        util.raiseNotDefined()

    def Max(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex == gameState.getNumAgents() -1:
            next_agent = 0
            next_depth = depth-1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        actions = gameState.getLegalActions(agentIndex)
        max_score = -1e8
        max_action = Directions.STOP
        for act in actions:
            next_state = gameState.generateSuccessor(agentIndex, act)
            next_score, anotheract = self.MinMax(next_state, next_agent, next_depth, alpha, beta)
            if next_score > max_score:
                max_score = next_score
                max_action = act
            if next_score > beta:
                return next_score, act
            alpha = max(alpha, max_score)
        return max_score, max_action

        util.raiseNotDefined()

    def Min(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex == gameState.getNumAgents() -1:
            next_agent = 0
            next_depth = depth-1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        actions = gameState.getLegalActions(agentIndex)
        min_score = 1e8
        min_action = Directions.STOP
        for act in actions:
            next_state = gameState.generateSuccessor(agentIndex, act)
            next_score, anotheract = self.MinMax(next_state, next_agent, next_depth, alpha, beta)
            if next_score < min_score:
                min_score = next_score
                min_action = act
            if next_score < alpha:
                return next_score, act
            beta = min(beta, min_score)
        return min_score, min_action

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
        depth = self.depth
        score, action = self.expMax(gameState, 0 ,depth)
        return action

    def expMax(self, gameState, agentIndex, depth):
        max_score = -1e8
        max_act = Directions.STOP
        if gameState.isWin():
            return  self.evaluationFunction(gameState), Directions.STOP
        if gameState.isLose():
            return  self.evaluationFunction(gameState), Directions.STOP
        if depth == 0:
            return  self.evaluationFunction(gameState), Directions.STOP
        if agentIndex == 0:
            actions = gameState.getLegalActions(0)
            next_agent = agentIndex + 1
            next_depth = depth
            for act in actions:
                next_state = gameState.generateSuccessor(agentIndex, act)
                next_score, next_action = self.expMax(next_state, next_agent, next_depth)
                if next_score > max_score:
                    max_score = next_score
                    max_act = act
            return max_score, max_act
        
        if agentIndex != 0:
            exp_score = 0.0
            actions = gameState.getLegalActions(agentIndex)
            if agentIndex == gameState.getNumAgents() -1:
                next_agent = 0
                next_depth = depth-1
            else:
                next_agent = agentIndex + 1
                next_depth = depth
            for act in actions:
                next_state = gameState.generateSuccessor(agentIndex, act)
                next_score, next_action = self.expMax(next_state, next_agent, next_depth)
                exp_score += next_score
            exp_score/=len(actions)
            return exp_score, Directions.STOP

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    '''
    The main concern is how to take scared ghost into account.

    Here, I add "PreyList" to store all the scared ghosts, whose scared time is larger than its manhattanDistance with 
    pacman, which means they can be reached. Then, like q1, I set the evalutation function as the manhattanDistance
    of the nearest scared ghost. The sign is negative, which means pacman should hunt the pray.

    Also, I found that, using the previous parameters, the pacman is too timid. Som I tuned these parameters.
    
    
    '''
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()#Pos of pac man
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    FoodList = newFood.asList()
    FoodList.sort(key = lambda pos: manhattanDistance(pos, newPos))
    GhostList = []
    PreyList = []
    for ghost in newGhostStates:
        if ghost.scaredTimer == 0:
            GhostList.append(ghost.getPosition())
        else:
            if ghost.scaredTimer >= manhattanDistance(newPos, ghost.getPosition()):
                PreyList.append(ghost.getPosition())
    PreyList.sort(key = lambda pos: manhattanDistance(pos, newPos))
    GhostList.sort(key = lambda pos: manhattanDistance(pos, newPos))
    if len(PreyList) > 0:
        preyeval = manhattanDistance(newPos, PreyList[0])
    else:
        preyeval = 0
    if len(FoodList) > 0:
        foodeval = manhattanDistance(newPos, FoodList[0])
    else:
        foodeval = 0
    if len(GhostList) > 0:
        ghosteval = manhattanDistance(newPos, GhostList[0])
    else:
        ghosteval = 0
    evaluation = currentGameState.getScore() - 3/(ghosteval+1) -  foodeval/3 - 10*preyeval
    return evaluation
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
