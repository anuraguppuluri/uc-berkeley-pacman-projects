#Anurag Uppuluri
#110352456
#2-24-2019
#cd C:\Users\anura\OneDrive - Fresno State\csufr\myfresnostate\artificial intelligence programming\ass2\search
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
    e = Directions.EAST
    return  [s, s, w, s, w, w, s, w]
    #return [w, w, w, w, s, s, e, s, s, w] 

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())
    
    #"*** YOUR CODE HERE ***"
    #"node = [state, action , parent, path_cost]"
    node = [problem.getStartState(), 'Stop', None, 0]
    actionList = [] #initialize return value
    frontier = util.Stack()
    frontier.push(node)
    explored = []

    while frontier:
      #print 'Frontier: ', [f[0] for f in frontier]
      #print 'Frontier:- '
      #frontier.display()
      #if (!problem.getSuccessors(frontier.pop())):
      node = frontier.pop()

      #frontier = frontier[1:]
      #print node[0], ': Is it a goal?'
      if problem.isGoalState(node[0]):
        #print 'Yes!'
        tempNode = node
        while(tempNode[2]):
            actionList.insert(0, tempNode[1])
            tempNode = tempNode[2]
        return actionList
      #else:
        #print 'No.'

      #if node[0] not in explored and not frontier.find(node):
      if node[0] not in explored:
        #print 'Expanding: ', node[0]
        #print '=>', problem.actions(node[0])
        explored.append(node[0])
        for succ in problem.getSuccessors(node[0]):
          newNode = [succ[0], succ[1], node, succ[2]]
          #if succ[0] not in explored and not frontier.find(newNode):
          if succ[0] not in explored:
            frontier.push(newNode)

    #util.raiseNotDefined()
    return []

def breadthFirstSearch(problem):
    #"""Search the shallowest nodes in the search tree first."""
    #"*** YOUR CODE HERE ***"
    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())

    '"actionList" should be generated when the solution is found by following the parent links from the Node with the goal state. This is why the Node includes a link to the parent.'

    "node = [state, action , parent, path_cost]"
    node = [problem.getStartState(), 'Stop', None, 0]
    actionList = [] #initialize return value
    frontier = util.Queue()
    frontier.push(node)
    explored = []

    while frontier:
      #print 'Frontier: ', [f[0] for f in frontier]
      #print 'Frontier:- '
      #frontier.display()
      node = frontier.pop()

      #frontier = frontier[1:]
      #print node[0], ': Is it a goal?'
      if problem.isGoalState(node[0]):
        #print 'Yes!'
        tempNode = node
        while (tempNode[2]):
          actionList.insert(0, tempNode[1])
          tempNode = tempNode[2]
        return actionList
      #else:
        #print 'No.'

      #if node[0] not in explored and not frontier.find(node):
      if node[0] not in explored:
        #print 'Expanding: ', node[0]
        #print '=>', problem.actions(node[0])
        explored.append(node[0])
        for succ in problem.getSuccessors(node[0]):
          newNode = [succ[0], succ[1], node, succ[2]]
          #if succ[0] not in explored and not frontier.find(newNode):
          if succ[0] not in explored:
            frontier.push(newNode)

    #util.raiseNotDefined()
    return []
    
def uniformCostSearch(problem):
    #"""Search the node of least total cost first."""
    #"*** YOUR CODE HERE ***"
    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())

    #"node = [state, action , parent, cost_function]"
    node = [problem.getStartState(), 'Stop', None, 0]
    actionList = [] #initialize return value
    frontier = util.PriorityQueue()
    frontier.push(node, 0)
    explored = []

    while frontier:
      #print 'Frontier: ', [f[0] for f in frontier]
      #print 'Frontier:- '
      #frontier.display()
      node = frontier.pop()

      #frontier = frontier[1:]
      #print node[0] , ', ' , str(node[3]) , ': Is it a goal?'
      if problem.isGoalState(node[0]):
        #print 'Yes!'
        tempNode = node
        while(tempNode[2]):
          actionList.insert(0, tempNode[1])
          tempNode = tempNode[2]
        return actionList
      #else:
        #print 'No.'

      #if node[0] not in explored and not frontier.find(node):
      if node[0] not in explored:
        #print 'Expanding: ', node[0]
        #print '=>', problem.actions(node[0])
        explored.append(node[0])
        for succ in problem.getSuccessors(node[0]):
          newNode = [succ[0], succ[1], node, succ[2]]
          if succ[0] not in explored:
            #frontier.push(newNode, succ[2])
            tempActionList = []
            tempNode = newNode
            while(tempNode[2]):
              tempActionList.insert(0, tempNode[1])
              tempNode = tempNode[2]
            frontier.push(newNode, problem.getCostOfActions(tempActionList))
          '''
          fSameNode = frontier.findAndRet(newNode)
          if succ[0] not in explored and not fSameNode:
            frontier.push(newNode, succ[2])
          elif fSameNode:
            if succ[2] < fSameNode[3]:
              frontier.delItem(fSameNode)
              frontier.push(newNode, succ[2])
          '''

    #util.raiseNotDefined()
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    #"""Search the node that has the lowest combined cost and heuristic first."""
    #"*** YOUR CODE HERE ***"
    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())

    "node = [state, action , parent, path_cost]"
    startState = problem.getStartState()
    node = [startState, 'Stop', None, 0]
    actionList = [] #initialize return value
    frontier = util.PriorityQueue()
    frontier.push(node, 0 + heuristic(startState, problem))
    explored = []

    while frontier:
      #print 'Frontier: ', [f[0] for f in frontier]
      #print 'Frontier:- '
      #frontier.display()
      node = frontier.pop()

      #frontier = frontier[1:]
      #print node[0] , ', ' , str(node[3]) , ': Is it a goal?'
      if problem.isGoalState(node[0]):
        #print 'Yes!'
        tempNode = node
        while(tempNode[2]):
          actionList.insert(0, tempNode[1])
          tempNode = tempNode[2]
        return actionList
      #else:
        #print 'No.'

      #if node[0] not in explored and not frontier.find(node):
      if node[0] not in explored:
        #print 'Expanding: ', node[0]
        #print '=>', problem.actions(node[0])
        explored.append(node[0])
        for succ in problem.getSuccessors(node[0]):
          newNode = [succ[0], succ[1], node, succ[2]]
          if succ[0] not in explored:
            tempActionList = []
            tempNode = newNode
            while(tempNode[2]):
              tempActionList.insert(0,tempNode[1])
              tempNode = tempNode[2]
            #frontier.push(newNode, succ[2] + nullHeuristic(succ[0], problem))
            frontier.push(newNode, problem.getCostOfActions(tempActionList) + heuristic(succ[0], problem))
          '''
          fSameNode = frontier.findAndRet(newNode)
          if succ[0] not in explored and not fSameNode:
            frontier.push(newNode, succ[2])
          elif fSameNode:
            if succ[2] < fSameNode[3]:
              frontier.delItem(fSameNode)
              frontier.push(newNode, succ[2])
          '''

    #util.raiseNotDefined()
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
