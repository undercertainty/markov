
# coding: utf-8

# # Markov Model class definition

# This notebook defines a constructor for a Markov model class.

# In[46]:

# Put all the imports at the beginning

import numpy as np


# Before the Markov model class, I also need a MarkovState class. The MarkovState class will contain the information about the state following one or more transitional steps from the MM.
# 
# This class needs the following data:
# 
# 1. An index (index_ls) of the states considered in the MarkovState
# 
# 2. A (logged) probabilistic distribution of the current state (currentState_ar)
# 
# 3. A structure containing the historical paths to each node in the index.

# In[47]:

class MarkovState:
    
    def __init__(self, index_ls, currentLogState_ar, paths_ls):
        
        self.myIndex_ls=index_ls
        self.myCurrentLogState_ar=currentLogState_ar
        self.myPaths_ls=paths_ls
        
    def get_log_current_state_distribution(self):
        return self.myCurrentLogState_ar
    
    def get_current_state_distribution(self):
        return np.exp(self.myCurrentLogState_ar)
    
    def get_index(self):
        return self.myIndex_ls
    
    def get_path_list(self):
        return self.myPaths_ls
        
    def most_likely_path(self, state):
        '''
        Find the most likely path to the current state,
        as found from the path history
        '''
        stateIndex_i=self.myIndex_ls.index(state)
        return [self.myIndex_ls[row[stateIndex_i]] for row in self.myPaths_ls] + [state]
    
    def most_likely_state(self, n=1):
        '''
        Return the n most likely states to have ended up in.
        '''
        return [y[1] for y in sorted([x for x in zip(self.get_current_state_distribution(), self.get_index())],
                                     reverse=True)
               ][:n]


# Although it's not the only way of defining a Markov model, for the moment, I'm going to do the definition by taking arguments in the constructor that represent a distribution of transitions.

# In[48]:

class MarkovModel:
    
    def __init__(self, transitions_ls, initialStates_ls=[], extraStates_ls=[]):
        '''
        Take a list of initial states, and a list of pairs of transitions
        between states. Create a Markov model based on the distribution of
        initial states, and distribution of transitions.
        
        extraStates_ls is a list of additional states which do not appear
        in the two lists initialStates_ls and transitions_ls.
        
        If initialStates_ls is empty, assume an equal distribution over all
        the states obtained from the transitions and the extra states.
        '''
        
        # First, build the list of all states in the model
        self.stateIndex_ls=list({x for x in initialStates_ls}.                                  union({x for (x, y) in transitions_ls}).                                  union({y for (x, y) in transitions_ls}).                                  union(set(extraStates_ls)))
        self.stateIndex_ls.sort() # just for aesthetics

        # Now build an array that contains the initial states
        if initialStates_ls==[]:
            initialStates_ls=self.stateIndex_ls        
        self.initialState_ar=np.array([initialStates_ls.count(state) 
                                       for state in self.stateIndex_ls])
        # and normalise the values so the prob.s sum to 1
        self.initialState_ar=self.initialState_ar/np.sum(self.initialState_ar)
        
        # Now build an array that encodes the transitions
        self.transitionMatrix_ar=np.zeros((len(self.stateIndex_ls), 
                                           len(self.stateIndex_ls)), 
                                          dtype=np.float)  # Normally int, but we're
                                                           # going to normalise
        for (x, y) in transitions_ls:
            self.transitionMatrix_ar[self.stateIndex_ls.index(x)][self.stateIndex_ls.index(y)]+=1
        for (i, r) in enumerate(self.transitionMatrix_ar):
            if np.sum(self.transitionMatrix_ar[i])>0:
                self.transitionMatrix_ar[i]=self.transitionMatrix_ar[i]/sum(self.transitionMatrix_ar[i])
                
        # Take the log of the transition matrix to make
        # calculations more accurate
        self.logTransitionMatrix_ar=np.log(self.transitionMatrix_ar)
        
        # Same for the initial states:
        self.logInitialState_ar=np.log(self.initialState_ar)
        

    def create_markov_state(self, statesIn_ls):
        '''
        Helper function to convert a list of states
        to a MarkovState object. Usually used as the
        first step of input to the apply method.
        '''
        
        initialStatesDist_ls=[statesIn_ls.count(s) for s in self.stateIndex_ls]
        dist_ar=np.array(initialStatesDist_ls)/np.sum(initialStatesDist_ls)
        
        return MarkovState(self.stateIndex_ls,
                           np.log(dist_ar),
                           [])
        
    def apply_1(self, stateIn_ms):
        '''
        Helper function to apply: applies the transition matrix
        for self to stateIn_ms. Returns the pair of the log 
        distribution of outputs and the previous state from 
        which the next state is arrived at.
        '''
        stateDistOut_ar=np.empty(len(self.stateIndex_ls))
        previousStateOut_ls=[0]*len(self.stateIndex_ls)
        
        # For each row in the transition matrix:
        for (i, row) in enumerate(self.logTransitionMatrix_ar):

            # multiply (logged) each of the transition probabilities
            # by the probability of being in that state
            calculateTransitions_ar=stateIn_ms.get_log_current_state_distribution() +                                     self.logTransitionMatrix_ar.transpose()[i]
            
            # Find the index of the largest value (most probable transition) 
            argmax_i=np.argmax(calculateTransitions_ar)

            # and add that probability and the previous state
            # to the output values
            stateDistOut_ar[i]=calculateTransitions_ar[argmax_i]
            previousStateOut_ls[i]=argmax_i
            
        # return {'logdist':stateDistOut_ar, 
        #         'prevstates': previousStateOut_ls}

        return MarkovState(stateIn_ms.get_index(),
                           stateDistOut_ar,
                           stateIn_ms.get_path_list() + [previousStateOut_ls])

    def apply(self, stateIn_ms, steps=1):
        '''
        Takes an input MarkovState, and returns the output
        MarkovState following steps applications.
        
        Can also take a list of states as an alternative to
        the input MarkovState, in which case it will be 
        converted as necessary.

        Both stateIn_ar and transitionIn_ar are expressed as logs.
        
        TODO: Raise an error if indices don't match, or if a
              list is input which contains nonexistent states.
        '''
        
        # First, if the given argument is not a MarkovState,
        # generate one based on the input
        if not isinstance(stateIn_ms, MarkovState):
            stateIn_ms=self.create_markov_state(stateIn_ms)
    
        # Next, let's assume that we're only doing one step
        # at the moment
        
        stateOut_ms=stateIn_ms
        
        for i in range(steps):
            stateOut_ms=self.apply_1(stateOut_ms)
        
        return stateOut_ms

