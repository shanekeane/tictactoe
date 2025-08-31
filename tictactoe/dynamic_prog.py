import numpy as np
import copy
from .utils import is_win, is_loss, int_to_state, state_to_int

def get_possible_actions_for_state(state_int):
    """
    Gets all possible states, pads with zeroes to length 9
    """
    state = int_to_state(state_int)
    poss_actions = [i for i, ltr in enumerate(state) if ltr == '0'] 
    next_states = list()
    for i in range(len(poss_actions)):
        next_state = state[:poss_actions[i]] + '1' + state[poss_actions[i]+1:]
        next_states.append(state_to_int(next_state))
    return np.asarray(next_states, dtype=int), poss_actions

def get_2_actions(state_int):
    """
    Gets all possible states, pads with zeroes to length 9
    """
    state = int_to_state(state_int)
    poss_actions = [i for i, ltr in enumerate(state) if ltr == '0'] 
    next_states = list()
    for i in range(len(poss_actions)):
        next_state = state[:poss_actions[i]] + '2' + state[poss_actions[i]+1:]
        next_states.append(state_to_int(next_state))
    return np.asarray(next_states, dtype=int), poss_actions

def get_pi_for_state(state_int, V, rewards, GAMMA, choose_one=False):
    """
    Gets an array of length 9 with all possible movements. 
    Padded with zeros (most states won't have 9 possible moves)
    """
    possible_actions, action_inds = get_possible_actions_for_state(state_int)
    pi_state = np.zeros(9)
    vs_for_as = V[possible_actions]
    rs_for_as = rewards[possible_actions]
    sum_values = rs_for_as + GAMMA*vs_for_as
    if choose_one==False:
        max_as = np.where(sum_values==np.max(sum_values))[0]
        for max_a in max_as:
            pi_state[action_inds[max_a]] += 1.0/len(max_as)
    else:
        max_a = np.argmax(sum_values)
        pi_state[action_inds[max_a]] = 1.0
    return pi_state

def get_policy_from_V(V, rewards, GAMMA, choose_one=False):
    """
    Returns policy pi from an inputted array of state-value functions.
    Policy is array of size 3^9 X 9 (max possible number of next states)
    """
    pi = np.zeros((len(V), 9))
    for state in range(0, len(V)-1):
        if valid_state(state):
            pi[state] = get_pi_for_state(state, V, rewards, GAMMA, choose_one)
    return pi

def swap_state(state_int):
    numbers = np.asarray(list(int_to_state(state_int)))
    x = np.asarray(list('0'*9))
    x[np.where(numbers=='1')[0]] = '2'
    x[np.where(numbers=='2')[0]] = '1'
    outp = state_to_int(''.join(x))
    return outp

def get_rewards():
    rewards = np.zeros(np.power(3,9))
    for i in range(len(rewards)):
        if is_win(int_to_state(i)):
            rewards[i] = 1.0
        if is_loss(int_to_state(i)):
            rewards[i] = -1.0
    return rewards

def valid_state(state_int):
    #takes int
    state = int_to_state(state_int)
    state+='012'
    entries, counts = np.unique(np.asarray(list(state)), return_counts=True)
    if np.abs(counts[2]-counts[1]) > 1 or counts[1]>counts[2] or counts[0]==1:
        return False
    else:
        return True

def twos_move(state_int):
    state = int_to_state(state_int)
    state+='012'
    entries, counts = np.unique(np.asarray(list(state)), return_counts=True)
    if counts[1] > counts[2] and np.abs(counts[1]-counts[2]) <= 1 and counts[0] != 0:
        return True
    else:
        return False

def policy_evaluation(V, pi, rewards, GAMMA, DELTA):
    """
    Returns policy pi from an inputted array of state-value functions.
    There are S states. 
    
    Parameters
    ----------
    V : numpy.ndarray, shape = (S)
        An array of floats containing the state-value functions.
    pi : numpy.ndarray, shape = (S, S)
        The policy. Each row corresponds to a state, and containgan array 
        indicating probability for moving to each of the S states.
    rewards : numpy.ndarray, shape = (S) or shape = (sqrt(S), sqrt(S))
        An array of floats containing the rewards for transitions to each state.
    GAMMA : float
        The discount rate.
    DELTA : float
        Sets maximum difference between old and new Vs during policy evaluation.
 
    Returns
    -------
    V : numpy.ndarray, shape = (S)
        An array of floats containing the updated state-value functions.
    """
    diff = DELTA+1.0
    while diff>DELTA:
        v = copy.deepcopy(V)
        for i in range(0, len(V)-1):
            if valid_state(i):
                if rewards[i] == 0.0:
                    poss_actions, action_inds = get_possible_actions_for_state(i)
                    V[i] = np.sum(pi[i, action_inds]*(rewards[poss_actions]+GAMMA*v[poss_actions]))
            elif twos_move(i):
                if rewards[i] == 0.0:
                    poss_2_actions, action_inds = get_2_actions(i)
                    alt_i = swap_state(i)
                    if len(poss_2_actions) > 0:
                    #V[i] = np.sum(pi[alt_i,action_inds]*(rewards[poss_2_actions]+GAMMA*v[poss_2_actions]))
                        V[i] = 0.5*np.sum((rewards[poss_2_actions]+GAMMA*v[poss_2_actions]))/len(poss_2_actions) + 0.5*np.sum(pi[alt_i,action_inds]*(rewards[poss_2_actions]+GAMMA*v[poss_2_actions]))
                        #V[i] = np.sum((rewards[poss_2_actions]+GAMMA*v[poss_2_actions]))/len(poss_2_actions)
                else:
                    V[i] = 0.0
        diff = np.linalg.norm(v-V)
    return V

def policy_iteration(GAMMA, DELTA):
    """
    Does policy iteration on grid defined by rewards.
    
    Parameters
    ----------
    rewards : numpy.ndarray, shape = (S) or shape = (sqrt(S), sqrt(S))
        An array of floats containing the rewards for transitions to each state.
    GAMMA : float
        The discount rate.
    DELTA : float
        Sets maximum difference between old and new Vs during policy evaluation.
 
    Returns
    -------
    V : numpy.ndarray, shape = (S)
        An array of floats containing the calculated state-value functions.
    pi : numpy.ndarray, shape = (S, S)
        The policy. Each row corresponds to a state, and containing an array 
        indicating probability for moving to each of the S states.     
    """
    rewards = get_rewards()
    V = np.zeros(len(rewards))
    pi = get_policy_from_V(V, rewards, GAMMA, choose_one=False)
    old_pi = np.ones_like(pi)
    while (old_pi==pi).all()==False:
        old_pi = copy.deepcopy(pi)
        V = policy_evaluation(V, pi, rewards, GAMMA, DELTA)
        pi = get_policy_from_V(V, rewards, GAMMA, choose_one=True)
    return V, pi

