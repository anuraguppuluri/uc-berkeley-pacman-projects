# Anurag Uppuluri
# 110352456
# 3/16/2019
# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent
from pprint import pprint

#sys.setrecursionlimit(10000)

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.

      Note: A capable reflex agent will have to consider both food locations and ghost locations to perform well.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        #pprint(vars(gameState))
        #print(gameState.__dict__)

        return legalMoves[chosenIndex]

    def evaluationFunction1(self, currentGameState, action):
        """
        Note: As features, try the reciprocal of important values (such as distance to food) rather than just the values themselves.

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
        res = successorGameState.getScore()
        #res = 0 #pacman dies if I don't initialize res with the actual game score

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        fList = newFood.asList() #just the coor of the remaining food
        #mList = [len(search.bfs(PositionSearchProblem(successorGameState, start=newPos, goal=food, warn=False, visualize=False)))  for food in fgL]
        fDistList = [manhattanDistance(newPos, food) for food in fList]
        #in this case bfs (30.7 sec) is faster that both ucs (57.7 sec) and astar (57.3 sec); path cost = 60, search nodes expanded = 4137 and score = 570 in all three cases 

        if fDistList:
            minFDist = min(fDistList)
            if minFDist: res += 10/minFDist
            #if maxFDist: res += maxFDist #pacman dies if I don't take the reciprocal

        newGhostStates = successorGameState.getGhostStates()
        gList = [newGhostState.getPosition() for newGhostState in newGhostStates]
        gDistList = [manhattanDistance(newPos, ghost) for ghost in gList]

        if gDistList:
            minGDist = min(gDistList)
            if minGDist: res -= 10/minGDist
            #if minGDist: res -= minGDist #pacman dies if I don't take the reciprocal

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if newScaredTimes:
            minSTime = min(newScaredTimes)
            if minSTime: res += 10/minSTime 

        "*** YOUR CODE HERE ***"
        #print res
        return res

    def evaluationFunction(self, currentGameState, action):
        """
        Note: As features, try the reciprocal of important values (such as distance to food) rather than just the values themselves.

        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # initializing the final result returned by the evaluation function
        res = successorGameState.getScore()
        # pacman dies super fast if we don't initialize res with the successor game state score
        #res = 0 # no can do baby
        #res = currentGameState.getScore() # not even this

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        fList = newFood.asList() # just the coordinates of the remaining food pellets
        fDistList = [manhattanDistance(newPos, food) for food in fList] # the number of moves to each pellet

        if fDistList: # if the list is not empty
            minFDist = min(fDistList) # distance to the nearest food pellet
            if minFDist: res += 10/minFDist # if the minimum distance is not 0
            # pacman dies super fast if we don't take the reciprocal of important values like food distance, ghost distance and scared times
            #if maxFDist: res += maxFDist # nada

        newGhostStates = successorGameState.getGhostStates()
        gList = [newGhostState.getPosition() for newGhostState in newGhostStates] # just the coordinates of all the ghosts
        gDistList = [manhattanDistance(newPos, ghost) for ghost in gList] # the number of moves to each ghost

        if gDistList:
            minGDist = min(gDistList) # distance to the nearest ghost

        newScaredTimes = [newghostState.scaredTimer for newghostState in newGhostStates]

        if newScaredTimes:
            #minSTime = min(newScaredTimes)
            # pacman scores more if we take the reciprocal of the max scared time of any ghost and subtract it from res as opposed to taking the min and adding it to res
            maxSTime = max(newScaredTimes)

        numPowPel = len(currentGameState.getCapsules())

        # case 1: if there is any scared ghost at all
        if (maxSTime > 0):
            #if minGDist: res += 10/minGDist + minSTime - numPowPel
            if minGDist: res += 10/minGDist - 10/maxSTime - numPowPel
            # pacman dies fast if we take the reciprocal of unimportant values like the number of power pellets

        # case 2: there is no scared ghost
        else:
            if minGDist: res += -10/minGDist + numPowPel

        return res

    def evaluationFunctionDetailed(self, currentGameState, action):
        """
        Note: As features, try the reciprocal of important values (such as distance to food) rather than just the values themselves.

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
        res = successorGameState.getScore()
        #res = 0 #pacman dies if I don't initialize res with the actual game score

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        fList = newFood.asList() #just the coor of the remaining food
        #mList = [len(search.bfs(PositionSearchProblem(successorGameState, start=newPos, goal=food, warn=False, visualize=False)))  for food in fgL]
        fDistList = [manhattanDistance(newPos, food) for food in fList]
        #in this case bfs (30.7 sec) is faster that both ucs (57.7 sec) and astar (57.3 sec); path cost = 60, search nodes expanded = 4137 and score = 570 in all three cases 

        if fDistList: #if the list is not empty
            minFDist = min(fDistList)
            if minFDist: res += 10/minFDist #if the minimum distance is not 0
            #if maxFDist: res += maxFDist #pacman dies if I don't take the reciprocal

        newGhostStates = successorGameState.getGhostStates()
        gList = [newGhostState.getPosition() for newGhostState in newGhostStates]
        gDistList = [manhattanDistance(newPos, ghost) for ghost in gList]

        if gDistList:
            minGDist = min(gDistList)
            #if minGDist: res -= 10/minGDist
            #if minGDist: res -= minGDist #pacman dies if I don't take the reciprocal

        newScaredTimes = [newghostState.scaredTimer for newghostState in newGhostStates]

        if newScaredTimes:
            #minSTime = min(newScaredTimes)
            maxSTime = max(newScaredTimes)
            #if minSTime: res += 10/minSTime 

        numPowPel = len(currentGameState.getCapsules())

        if (maxSTime > 0):
            #if minGDist: res += 10/minGDist + minSTime - numPowPel
            if minGDist: res += 10/minGDist - 10/maxSTime - numPowPel


        else:
            if minGDist: res += -10/minGDist + numPowPel

        "*** YOUR CODE HERE ***"
        #print res
        return res

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

      Your minimax agent should work with any number of ghosts. Your minimax tree will have multiple min layers (one for each ghost) for every max layer. 

      Your code should also expand the game tree to an arbitrary depth. Score the leaves of your minimax tree with the supplied self.evaluationFunction, which defaults to scoreEvaluationFunction. MinimaxAgent extends MultiAgentSearchAgent, which gives access to self.depth and self.evaluationFunction.

      A single search ply is considered to be one Pacman move and all the ghosts' responses, so depth 2 search will involve Pacman and each ghost moving two times.

      The autograder will be very picky about how many times you call GameState.generateSuccessor. If you call it any more or less than necessary, the autograder will complain.

      Now we're evaluating *states* rather than actions, as we were for the reflex agent. Look-ahead agents evaluate future states whereas reflex agents evaluate actions from the current state.

      Pacman is always agent 0, and the agents move in order of increasing agent index.

      All states in minimax should be GameStates, either passed in to getAction or generated via GameState.generateSuccessor. In this project, you will not be abstracting to simplified states.
    """
    '''
    But the gamestate passed in, does it correspond to the pacman agent? should we return an action for the pacman agent to take? 
    affirmative it belongs to the pacman agent - it's his turn now, please make a choice
    minimax_value will be for the start state i.e., the gameState passed in
    you are interested in the value of the next step; so you really wanna know the value of each successor state; so then you'll want to decide which of those actions is associated with the best output state
    right now you are calculating the minimax value of the root but that won't help you decide which action to take; you really want the minimax value of each of the subsequent states
    minimax_value will look a little bit like max; it will loop through each action and it will keep track of the max_value and the action, act, that was associated with that max value and then it might have to return an action
    so the in order to get the action you'll have to generate the minimax value for each of the successor states from the game state
    you can call min first on each successor state in minimax_value
    so the only difference is here rather than call max the first time we would want to have a loop that looks like this that would call min each time but rather than taking just the max of all of them in this loop, we'd want to save the max, save the value and the action for the max of those states; so here you're just using the max function but you'll actually want to do an if or else one way you could do it is you might want to have the min_value and the action as a tuple and then you can take the max of the two as long as the value is first and the action is next
    wherever you currently are is the root of the state
    max_value has to call min_value because it's alternating 
    call max_value if the next agent number is 0 i.e., the index of the successor is 0
    illustrates how the tree structure is organized for the multiple ghosts

    '''

    def maxValueMinimax(self, gameState, depth, agentNum):

        if gameState.isWin() or gameState.isLose() or depth >= self.depth:
            return self.evaluationFunction(gameState)

        maxVal = -sys.maxint - 1
        legalActionList = gameState.getLegalActions(agentNum)
        for act in legalActionList:
            succ = gameState.generateSuccessor(agentNum, act)
            maxVal = max(maxVal, self.minValueMinimax(succ, depth, 1))
        return maxVal


    def minValueMinimax(self, gameState, depth, agentNum):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        minVal = sys.maxint
        legalActionList = gameState.getLegalActions(agentNum)
        for act in legalActionList:
            succ = gameState.generateSuccessor(agentNum, act)
            if (agentNum == gameState.getNumAgents() - 1): #if agentNum == numGhosts then next up is pacman
                minVal = min(minVal, self.maxValueMinimax(succ, depth+1, 0))
            else:
                minVal = min(minVal, self.minValueMinimax(succ, depth, agentNum+1))
        return minVal


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
        """
        "*** YOUR CODE HERE ***"

        succValueList = []
        legalActionList = gameState.getLegalActions(self.index)
        #self.index has been initialized to 0, which means we are concerned only with returning the max action for pacman
        for act in legalActionList:
            succ = gameState.generateSuccessor(self.index, act)
            #after pacman, enter the first ghost
            succValueList.append((self.minValueMinimax(succ, 0, 1), act))

        return max(succValueList, key=lambda item:item[0])[1]

        #util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """


    def maxValueAlphaBeta(self, gameState, depth, agentNum, alpha, beta):

        if gameState.isWin() or gameState.isLose() or depth >= self.depth:
            return self.evaluationFunction(gameState)

        maxVal = -sys.maxint - 1
        legalActionList = gameState.getLegalActions(agentNum)
        for act in legalActionList:
            succ = gameState.generateSuccessor(agentNum, act)
            maxVal = max(maxVal, self.minValueAlphaBeta(succ, depth, 1, alpha, beta))
            #if maxVal >= beta: return maxVal
            '''
            >= results in the following error (the same with <= for minValueAlphaBeta()): -
            *** FAIL: test_cases\q3\6-tied-root.test
            ***     Incorrect generated nodes for depth=3
            ***         Student generated nodes: A B max min1 min2
            ***         Correct generated nodes: A B C max min1 min2
            ***     Tree:
            ***         max
            ***        /   \
            ***     min1    min2
            ***      |      /  \
            ***      A      B   C
            ***     10     10   0
            '''
            if maxVal > beta: return maxVal
            alpha = max(alpha, maxVal)
        return maxVal


    def minValueAlphaBeta(self, gameState, depth, agentNum, alpha, beta):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        minVal = sys.maxint
        legalActionList = gameState.getLegalActions(agentNum)
        for act in legalActionList:
            succ = gameState.generateSuccessor(agentNum, act)
            if (agentNum == gameState.getNumAgents() - 1): #if agentNum == numGhosts then next up is pacman
                minVal = min(minVal, self.maxValueAlphaBeta(succ, depth+1, 0, alpha, beta))
                #if minVal <= alpha: return minVal
                if minVal < alpha: return minVal
                beta = min(beta, minVal)
            else:
                minVal = min(minVal, self.minValueAlphaBeta(succ, depth, agentNum+1, alpha, beta))
                if minVal < alpha: return minVal 
                beta = min(beta, minVal)
        return minVal


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        succValueList = []
        legalActionList = gameState.getLegalActions(self.index)
        #self.index has been initialized to 0, which means we are concerned only with returning the max action for pacman
        alpha = -sys.maxint -1
        beta = sys.maxint
        for act in legalActionList:
            succ = gameState.generateSuccessor(self.index, act)
            #after pacman, enter the first ghost
            succVal = self.minValueAlphaBeta(succ, 0, 1, alpha, beta)
            succValueList.append((succVal, act))
            if succVal > beta: return act
            alpha = max(alpha, succVal)
        return max(succValueList, key=lambda item:item[0])[1]

        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """


    def maxValueExpectimax(self, gameState, depth, agentNum):

        if gameState.isWin() or gameState.isLose() or depth >= self.depth:
            # even depth == self.depth emulates the exact same kind of gameplay when I run autograder.py -q -q4
            return self.evaluationFunction(gameState)

        maxVal = -sys.maxint - 1
        legalActionList = gameState.getLegalActions(agentNum)
        for act in legalActionList:
            succ = gameState.generateSuccessor(agentNum, act)
            maxVal = max(maxVal, self.expectedValue(succ, depth, 1))
        return maxVal


    def expectedValue(self, gameState, depth, agentNum):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        #minVal = sys.maxint
        legalActionList = gameState.getLegalActions(agentNum)
        totalExpVal = 0
        for act in legalActionList:
            succ = gameState.generateSuccessor(agentNum, act)
            if (agentNum == gameState.getNumAgents() - 1): #if agentNum == numGhosts then next up is pacman
                #expVal = min(minVal, self.maxValueExpectimax(succ, depth+1, 0))
                expVal = self.maxValueExpectimax(succ, depth+1, 0)
            else:
                #expVal = min(minVal, self.expectedValue(succ, depth, agentNum+1))
                expVal = self.expectedValue(succ, depth, agentNum+1)
            totalExpVal += expVal
        if not legalActionList:
            return 0
        return float(totalExpVal)/float(len(legalActionList))


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        succValueList = []
        legalActionList = gameState.getLegalActions(self.index)
        #self.index has been initialized to 0, which means we are concerned only with returning the max action for pacman
        for act in legalActionList:
            succ = gameState.generateSuccessor(self.index, act)
            #after pacman, enter the first ghost
            succValueList.append((self.expectedValue(succ, 0, 1), act))

        return max(succValueList, key=lambda item:item[0])[1]

        #util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      Apparently, what we (pacman) see is what we get. So let's minimize (optimize) the performance metric like distance to food and scared ghosts and maximize (optimize) the performance metric like distance from normal ghosts. And so in this implementation, the lesser the distance to a food pellet or a scared ghost is, that we have subtracted from the score of the current state, the more we have optimized that performance metric and also the score of pacman. And so pacman will try to get to that nearest objective first. The case is just the opposite for the negative performance metrics like the distance from active ghosts.

      Hints and Observations
      As for your reflex agent evaluation function, you may want to use the reciprocal of important values (such as distance to food) rather than the values themselves.
      One way you might want to write your evaluation function is to use a linear combination of features. That is, compute values for features about the state that you think are important, and then combine those features by multiplying them by different values and adding the results together. You might decide what to multiply each feature by based on how important you think it is.
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    res = currentGameState.getScore()
    #res = 0 #pacman dies if I don't initialize res with the actual game score

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    fList = newFood.asList() #just the coor of the remaining food
    #mList = [len(search.bfs(PositionSearchProblem(successorGameState, start=newPos, goal=food, warn=False, visualize=False)))  for food in fgL]
    fDistList = [manhattanDistance(newPos, food) for food in fList]
    #in this case bfs (30.7 sec) is faster that both ucs (57.7 sec) and astar (57.3 sec); path cost = 60, search nodes expanded = 4137 and score = 570 in all three cases 

    if fDistList:
        minFDist = min(fDistList)
        if minFDist > 0: res += 10/minFDist

        #if maxFDist: res += maxFDist #pacman dies if I don't take the reciprocal

    newGhostStates = currentGameState.getGhostStates()
    gList = [newGhostState.getPosition() for newGhostState in newGhostStates]
    gDistList = [manhattanDistance(newPos, ghost) for ghost in gList]

    minGDist = 0 #initializing nearest ghost distance
    if gDistList:
        minGDist = min(gDistList)

    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # now lets take into account whether the ghosts are scared or not; technically we would want pacman to run into ghosts when they are scared and run away from them when they are not and hence our score should alternate between adding and subtracting the ghost distance to itself to optimize the performance metric in each case; the same is the case for the number of power pellets remaining

    minSTime = 0 #initializing the least time that any ghost will remain scared for
    if newScaredTimes:
        minSTime = min(newScaredTimes)

    # to make sure pacman eats all/most of the food, lets just add to the score the number of pellets remaining
    #if fList: res +=  10/len(fList) #absolutely makes no difference

    # to give more incentive to pacman to eat the power pellets, lets add the number remaining from the score; but he shouldn't eat a power pellet if any of the ghosts is scared
    numPowPel = len(currentGameState.getCapsules())

    if (minSTime > 0):
        if minGDist: res += 10/minGDist + minSTime - numPowPel

    else:
        if minGDist: res += -10/minGDist + numPowPel
        #if minGDist: res -= minGDist #pacman dies if I don't take the reciprocal

    "*** YOUR CODE HERE ***"
    #print res
    return res


    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
