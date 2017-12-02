
# coding: utf-8

# # Markov Model class definition

# This notebook defines a constructor for a Markov model class.

# In[1]:

# Put all the imports at the beginning

import pandas as pd
import numpy as np

from collections import Counter
from copy import deepcopy

np.seterr(divide='ignore')


# Before the Markov model class, I also need a MarkovState class. The MarkovState class will contain the information about the state following one or more transitional steps from the MM.
# 
# This class needs the following data:
# 
# 1. An index (index_ls) of the states considered in the MarkovState
# 
# 2. A (logged) probabilistic distribution of the current state (currentState_ar)
# 
# 3. A structure containing the historical paths to each node in the index.

# In[2]:

class MarkovState:
    
    def __init__(self, currentLogState_sr, paths_df):
        
        self.myCurrentLogState_sr=currentLogState_sr
        self.myPaths_df=paths_df
        
    def get_log_current_state_distribution(self):
        return self.myCurrentLogState_sr
    
    def get_current_state_distribution(self):
        return np.exp(self.myCurrentLogState_sr)
    
    def get_index(self):
        return list(self.myCurrentLogState_sr.index)
    
    def get_path_dataframe(self):
        return self.myPaths_df
        
    def most_likely_path(self, state):
        '''
        Find the most likely path to the current state,
        as found from the path history
        '''
        # Hacky, but seems to work
        s=state
        o=[state]
        for c in reversed(self.myPaths_df.columns):
            o.append(self.myPaths_df[c][s])
            s=self.myPaths_df[c][s]
        o.reverse()
        return o[1:]

    def most_likely_state(self, n=1):
        '''
        Return the n most likely states to have ended up in.
        '''
        return [y[1] for y in sorted([x for x in zip(self.get_current_state_distribution(), self.get_index())],
                                     reverse=True)
               ][:n]


# Although it's not the only way of defining a Markov model, for the moment, I'm going to do the definition by taking arguments in the constructor that represent a distribution of transitions.

# In[3]:

class MarkovModel:
    
    def __init__(self, transitions_ls, initialStates_ls=[]):
        '''
        Take a list of initial states, and a list of pairs of transitions
        between states. Create a Markov model based on the distribution of
        initial states, and distribution of transitions.
        
        If initialStates_ls is empty, assume an equal distribution over all
        the states obtained from the transitions and the extra states.
        '''

        # First, build the list of all states in the model
        self.stateIndex_ls=list({x for x in initialStates_ls}.                                  union({x for (x, y) in transitions_ls}).                                  union({y for (x, y) in transitions_ls}))
        self.stateIndex_ls.sort()  # Just for aesthetics

        # Now build a series that contains the initial states
        if initialStates_ls:
            self.initialState_sr=pd.Series(Counter(initialStates_ls),
                                           index=self.stateIndex_ls).fillna(0)
        else:
            self.initialState_sr=pd.Series(Counter(self.stateIndex_ls), 
                                           index=self.stateIndex_ls).fillna(0)
        
        # and normalise the values so the prob.s sum to 1
        self.initialState_sr=self.initialState_sr/np.sum(self.initialState_sr)
        
        # And convert to a Markov State
        self.initialState_ms=MarkovState(np.log(self.initialState_sr),
                                         pd.DataFrame(index=self.initialState_sr.index,
                                                      columns=[0]))

        # Now build an array that encodes the transitions
        self.transitionMatrix_df=pd.DataFrame(0,
                                              index=self.stateIndex_ls,
                                              columns=self.stateIndex_ls)
        
        for (x, y) in transitions_ls:
            self.transitionMatrix_df.loc[x][y]+=1
        for row_ix in self.transitionMatrix_df.index:
            if not all(self.transitionMatrix_df.loc[row_ix]==0):
                self.transitionMatrix_df.loc[row_ix]=                    self.transitionMatrix_df.loc[row_ix]/                      np.sum(self.transitionMatrix_df.loc[row_ix])
                
        # Take the log of the transition matrix to make
        # calculations more accurate
        self.logTransitionMatrix_df=np.log(self.transitionMatrix_df)

        # Same for the initial states:
        self.logInitialState_sr=np.log(self.initialState_sr)


    def create_markov_state(self, statesIn_ls):
        '''
        Helper function to convert a list of states
        to a MarkovState object. Usually used as the
        first step of input to the apply method.
        '''
        
        statesDist_sr=pd.Series(Counter(statesIn_ls),
                                index=self.transitionMatrix_df.index
                               ).fillna(0)
        statesDist_sr=statesDist_sr/np.sum(statesDist_sr)
        
        return MarkovState(np.log(statesDist_sr),
                           pd.DataFrame(index=self.transitionMatrix_df.index,
                                        columns=[0]))
        
    def apply_1(self, stateIn_ms):
        '''
        Helper function to apply: applies the transition matrix
        for self to stateIn_ms. Returns the pair of the log 
        distribution of outputs and the previous state from 
        which the next state is arrived at.
        '''
        logCurrentState_sr=stateIn_ms.get_log_current_state_distribution()
        logNextState_sr=pd.Series(index=logCurrentState_sr.index)
        previousState_sr=pd.Series(index=logCurrentState_sr.index)
        
        # For each column in the logged transition matrix:
        for c in self.logTransitionMatrix_df.index:

            # multiply (logged) each of the transition probabilities
            # by the probability of being in that state
            calculateTransitions_sr=logCurrentState_sr +                                     self.logTransitionMatrix_df[c]
            
            logNextState_sr[c]=calculateTransitions_sr.max()
            previousState_sr[c]=calculateTransitions_sr.idxmax()
            
        tmp_df=stateIn_ms.get_path_dataframe()
        tmp_df[max(tmp_df.columns)+1]=previousState_sr
            
        return MarkovState(logNextState_sr,
                           tmp_df)

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
    
        # Next, let's call a helper function apply_1
        # to make the application once
        
        stateOut_ms=deepcopy(stateIn_ms)

        for i in range(steps):
            stateOut_ms=self.apply_1(stateOut_ms)
        
        return stateOut_ms
    
    def transition_weight(self, state1, state2):
        '''
        Return the weight (usually a probability, if the weight
        of all leaving arcs sum to 1) of the arc from state1
        to state2
        '''
        return self.transitionMatrix_ar[self.stateIndex_ls.index(state1)][self.stateIndex_ls.index(state2)]
    
    def get_states(self):
        '''
        Return list of states in model
        '''
        return list(self.transitionMatrix_df.index)
    
    def get_initial_state(self):
        '''
        Return the initial states as a Markov State
        '''
        return self.initialState_ms


# Now we're going to try to write some code to merge two different markov models. This isn't mathematically particularly well founded, but what the heck. It's quite a bit easier now that we've got everything in terms of dataframes.

# Can't reconstruct the input arrays from the models themselves, so let's create a subclass of MM so we can override `__init__`:

# In[4]:

class MarkovModelFromTransitionMatrix(MarkovModel):
    def __init__(self, transitionMatrixIn_df):
        '''
        Give the arrays directly. Note, this is not generally
        intended to be called by the user.
        '''

        # Now store the array that encodes the transitions,
        self.transitionMatrix_df=transitionMatrixIn_df
        # and take the log
        self.logTransitionMatrix_df=np.log(self.transitionMatrix_df)


# In[5]:

def merge(model1, model2, weight, normalise=True):
    '''
    Combine the transition matrices of model1 and model2 into
    an averaged model, in which model1 has weight, and model2
    has (1-weight).
    
    If normalise==True, then adjust the model so that the
    outputs from all nodes sum to 1 (or zero if no leaving
    arcs).
    '''
    
    m1=model1.transitionMatrix_df*weight
    m2=model2.transitionMatrix_df*(1-weight)

    return MarkovModelFromTransitionMatrix((m1.add(m2, fill_value=0)).fillna(0))


# That's interesting... exporting as python creates a bug with raw cells. Better change them to markdown:

# mm1=MarkovModel([('a', 'b'), ('a', 'c'), ('a', 'c'), ('c', 'b'), ('c', 'a')])
# mm1.transitionMatrix_df

# mm2=MarkovModel([('b', 'd'), ('d', 'c'), ('d', 'd'), ('c', 'd'), ('b', 'd')])
# mm2.transitionMatrix_df

# combined_mm=merge(mm1, mm2, 0.3)

# combined_mm.transitionMatrix_df

# s0=combined_mm.create_markov_state(['a'])

# s0.get_current_state_distribution()

# s0.get_path_dataframe()

# s_ms=combined_mm.apply(s0, 5)

# s_ms.get_path_dataframe()

# s_ms.most_likely_path('a')

# So my minimal testing suggests that this is now working reasonably reliably.
