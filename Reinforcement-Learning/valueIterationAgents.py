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
        for itr in range(self.iterations):
          state_count = util.Counter()
          states =self.mdp.getStates()
          for state in states:
            max_val = -99999
            state_count[state] = 0.0
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
              qval = self.computeQValueFromValues(state, action)
              state_count[state] = max(max_val, qval)
              max_val = max(max_val, qval)
          self.values = state_count


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
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        qval = 0
        for pair in transitions:
            nxt = pair[0]
            prob = pair[1]
            reward = self.mdp.getReward(state, action, nxt)
            nxt_qval = self.values[nxt]
            qval += prob * (reward + self.discount * nxt_qval)
        return qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            self.values[state] = 0
            return None
        best_action = None
        max_val = -999999
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
          q_value = self.computeQValueFromValues(state, action)
          if q_value >= max_val:
            best_action = action
            max_val = max(max_val, q_value)
        return best_action
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for i in range(self.iterations):
            states = self.mdp.getStates()
            ind = i % len(states)
            state = states[ind]
            action = self.computeActionFromValues(state)
            if action is None:
                val = 0
            else:
                val = self.computeQValueFromValues(state, action)
            self.values[state] = val


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        predecessors = {}
        for s in states:
            predecessors[s] = set()
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for nxt, prob in transitions:
                    if prob != 0:
                            predecessors[nxt].add(state)
                            
        q = util.PriorityQueue()
        for state in states:
            terminal = self.mdp.isTerminal(state)
            if not terminal:
                q_vals = []
                curr = self.values[state]
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    val = self.computeQValueFromValues(state, action)
                    q_vals.append(val)
                max_qval = max(q_vals)
                delta = curr - max_qval
                if delta > 0:
                    delta = delta * -1
                q.push(state, delta)
        
        for i in range(self.iterations):
            if q.isEmpty():
                return
            s = q.pop()
            terminal = self.mdp.isTerminal(s)
            if not terminal:
                values = []
                actions = self.mdp.getPossibleActions(s)
                for action in actions:
                    value = 0
                    transitions = self.mdp.getTransitionStatesAndProbs(s, action)
                    for nxt, prob in transitions:
                        reward = self.mdp.getReward(s, action, nxt)
                        nxt_qval = self.values[nxt]
                        value += prob * (reward + self.discount * nxt_qval)
                    values.append(value)
                self.values[s] = max(values)
            allPrevious = predecessors[s]
            for prev in allPrevious:
                curr = self.values[prev]
                q_vals = []
                actions = self.mdp.getPossibleActions(prev)
                for action in actions:
                    val = self.computeQValueFromValues(prev, action)
                    q_vals.append(val)
                max_q = max(q_vals)
                delta = abs(curr - max_q)
                if (delta > self.theta):
                    delta = delta * -1 
                    q.update(prev, delta)