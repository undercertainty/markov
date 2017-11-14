
# coding: utf-8

# # Markov Model class definition

# This notebook defines a constructor for a Markov model class.

# In[1]:

# Put all the imports at the beginning

import numpy as np


# Although it's not the only way of defining a Markov model, for the moment, I'm going to do the definition by taking arguments in the constructor that represent a distribution of transitions.

# In[2]:

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
        self.states_ls=list({x for x in initialStates_ls}.                               union({x for (x, y) in transitions_ls}).                               union({y for (x, y) in transitions_ls}).                               union(set(extraStates_ls)))
        self.states_ls.sort() # just for aesthetics

        # Now build an array that contains the initial states
        if initialStates_ls==[]:
            initialStates_ls=self.states_ls        
        self.initialState_ar=np.array([initialStates_ls.count(state) 
                                       for state in self.states_ls])
        # and normalise the values so the prob.s sum to 1
        self.initialState_ar=self.initialState_ar/np.sum(self.initialState_ar)
        
        # Now build an array that encodes the transitions
        self.transitionMatrix_ar=np.zeros((len(self.states_ls), 
                                           len(self.states_ls)), 
                                          dtype=np.float)  # Normally int, but we're
                                                           # going to normalise
        for (x, y) in transitions_ls:
            self.transitionMatrix_ar[self.states_ls.index(x)][self.states_ls.index(y)]+=1
        for (i, r) in enumerate(self.transitionMatrix_ar):
            if np.sum(self.transitionMatrix_ar[i])>0:
                self.transitionMatrix_ar[i]=self.transitionMatrix_ar[i]/sum(self.transitionMatrix_ar[i])
                
        # Take the log of the transition matrix to make
        # calculations more accurate
        self.logTransitionMatrix_ar=np.log(self.transitionMatrix_ar)
        
        # Same for the initial states:
        self.logInitialState_ar=np.log(self.initialState_ar)
        
    def apply(self, stateIn_ar, transitionIn_ar):
        '''
        Takes an input state and a transition matrix, and returns
        an output state.

        Both stateIn_ar and transitionIn_ar are expressed as logs.
        '''

        stateOut_ar=np.empty(stateIn_ar.shape)
        transOut_ar=np.zeros(stateIn_ar.shape, dtype=np.int)

        for (i, x) in enumerate(stateIn_ar):
            calculateTransitions_ar=stateIn_ar + transitionMatrix_ar[i]
            argmax_i=np.argmax(calculateTransitions_ar)

            stateOut_ar[i]=calculateTransitions_ar[argmax_i]
            transOut_ar[i]=argmax_i


        return (stateOut_ar, transOut_ar)
    


# Now, as well as the Markov model class, I also want a MarkovState class. The MarkovState class will contain the information about the state following one or more transitional steps from the MM.
# 
# This class needs the following data:
# 
# 1. An index (index_ls) of the states considered in the MarkovState
# 
# 2. A probabilistic distribution of the current state (currentState_ar)
# 
# 3. A structure containing the historical paths to each node in the index.

# In[3]:

class MarkovState:
    
    def __init__(self, index_ls, currentState_ar, paths_ar):
        
        self.myIndex_ls=index_ls
        self.myCurrentState_ar=currentState_ar
        self.myPaths_ar=paths_ar
        
    


# In[ ]:



