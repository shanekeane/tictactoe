import numpy as np
import copy as deepcopy
import matplotlib.pyplot as plt

def get_xogrid(state):
    """
    From a state, obtain grid of Xs and Os
    
    Parameters
    ----------
    state : string
        Describes states in length 9 string, with empty as 0,
        X as 2 and O as 1. 
 
    Returns
    -------
    V : numpy.ndarray, shape = (3,3)
        An array of strings containing Xs and Os
    """
    grid = np.asarray(list(state))
    grid[grid=='1'] = 'O'
    grid[grid=='2'] = 'X'
    grid[grid=='0'] = ''
    return grid.reshape(3,3)

def print_grid(state):
    """
    Outputs board with Xs and Os
    
    Parameters
    ----------
    state : string
        Describes states in length 9 string, with empty as 0,
        X as 2 and O as 1.     
    """
    zero_grid = np.zeros(len(state))
    grid = get_xogrid(state)
    fig1, ax1 = plt.subplots(1, sharex = True, sharey = False, figsize=(1,1))
    ax1.imshow(zero_grid.reshape(3,3), cmap='Greys')
    ax1.grid(color='k', linewidth=2, which='major')
    for (j,i),label in np.ndenumerate(grid):
        ax1.text(i,j, label, size=10, ha='center', va='center')
    major_ticks = np.arange(-0.5,2.5,1)
    minor_ticks = np.arange(-0.5,2.5,1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    plt.show()

def check_array(state, pattern):
    """
    Returns two or false depending on whether there's a pattern of '111' or '222'
    indicating a win. 
    
    Parameters
    ----------
    state : string
        Describes states in length 9 string, with empty as 0,
        X as 2 and O as 1. 
    patten : string
        '111' or '222' indicating a win
        
    Returns
    -------
    Bool indicating if it's been found in the state or not
    """
    check_list = list()
    check_list.append(any(state[i]+state[i+3]+state[i+6]==pattern for i in range(3)))
    check_list.append(any(state[i]+state[i+1]+state[i+2]==pattern for i in [0,3,6]))
    check_list.append(state[0]+state[4]+state[8]==pattern)
    check_list.append(state[2]+state[4]+state[6]==pattern)
    return any(check_list)

def is_win(state):
    """
    Indicates whether a state has a win for '111' - i.e. three Os 
    
    Parameters
    ----------
    state : string
        Describes states in length 9 string, with empty as 0,
        X as 2 and O as 1. 
 
    Returns
    -------
    Bool indicating win or not
    """
    pattern = '111'
    return check_array(state, pattern)

def is_loss(state):
    """
    Indicates whether a state has a loss for '111' - i.e. three Xs  
    
    Parameters
    ----------
    state : string
        Describes states in length 9 string, with empty as 0,
        X as 2 and O as 1. 
 
    Returns
    -------
    Bool indicating loss or not
    """
    pattern = '222'
    return check_array(state, pattern)

def state_to_int(state):
    """
    Outputs a unique int for a string state.
    
    Parameters
    ----------
    state : string
        Describes states in length 9 string, with empty as 0,
        X as 2 and O as 1. 
 
    Returns
    -------
    Int corresponding to the state - i.e. the decimal number for the base
    3 number represented by the state.
    """
    return int(state, base=3)

def int_to_state(number):
    """
    Convert integer to a unique state.
    
    Parameters
    ----------
    number : int
        Integer representing a unique state
 
    Returns
    -------
    state : string
        Describes states in length 9 string, with empty as 0,
        X as 2 and O as 1. 
    """
    state = np.base_repr(number, base=3)
    state = '0' * (9 - len(state)) + state
    return state

def get_next_state(pi, state):
    """
    Get next state based on current state and policy pi
    
    Parameters
    ----------
    pi : numpy.ndarray, shape = (3^9, 9)
        The policy. Each row corresponds to a state, and contains an array 
        indicating probability of adding a move to each of the 9 squares.
    state : string
        Describes states in length 9 string, with empty as 0,
        X as 2 and O as 1. 
 
    Returns
    -------
    next_state : string
        Describes states in length 9 string, with empty as 0,
        X as 2 and O as 1. 
    """
    state_int = state_to_int(state)
    move = np.argmax(pi[state_int])
    next_state = state[:move]+'1'+state[move+1:]
    return next_state

def play_bot(pi):
    """
    Play a game with a bot defined by the policy pi
    
    Parameters
    ----------
    pi : numpy.ndarray, shape = (3^9, 9)
        The policy. Each row corresponds to a state, and contains an array 
        indicating probability of adding a move to each of the 9 squares.
    """
    player = int(input('Who will start, 0 for you, 1 for bot: '))
    while player not in (0,1):
        player = int(input('Must choose 0 for you OR 1 for bot: '))
    if player == 0:
        player = int(2)
        policy = pi[1]
    else:
        policy = pi[0]
    state = '0'*9
    #player = np.random.randint(1,3) #randomly choose player
    while '0' in state and not is_loss(state) and not is_win(state):
        if player == 2:
            zero_states = [i for i, ltr in enumerate(state) if ltr == '0']
            move = int(input('Your move (square 1-9):'))-1
            while move not in zero_states:
                move = int(input('Invalid move. Your move (square 1-9):'))-1
            state = state[:move]+'2'+state[move+1:]
        else:
            state = get_next_state(policy, state)
        print_grid(state)
        player = (player*2)%3 #changes 1<->2
    if is_win(state):
        print('You lose')
    elif is_loss(state):
        print('You win')
    else:
        print('DRAW')

def play_random(pi, games, bot_start=True):
    """
    Play a given number of games against a bot defined by pi
    
    Parameters
    ----------
    pi : numpy.ndarray, shape = (3^9, 9)
        The policy. Each row corresponds to a state, and contains an array 
        indicating probability of adding a move to each of the 9 squares.
    games : int
        Number of games to be played
    bot_start : bool
        True for bot start, False for player start
 
    Returns
    -------
    V : numpy.ndarray, shape = (S)
        An array of floats containing the updated state-value functions.
    """
    draws = 0
    wins = 0
    losses = 0
    for i in range(games):
        if bot_start == False:
            player = int(2)
            policy = pi[1]
        elif bot_start == True:
            player = int(1)
            policy = pi[0]
        else:
            raise Exception('Must choose True or False')
        state = '0'*9
        #player = np.random.randint(1,3) #randomly choose player
        while '0' in state and not is_loss(state) and not is_win(state):
            if player == 2:
                zero_states = [i for i, ltr in enumerate(state) if ltr == '0']
                move = np.random.choice(zero_states)
                state = state[:move]+'2'+state[move+1:]
            else:
                state = get_next_state(policy, state)
            player = (player*2)%3 #changes 1<->2
        if is_win(state):
            losses +=1
        elif is_loss(state):
            wins+=1
        else:
            draws+=1
    print(f'Bot wins: {losses}\nPlayer wins: {wins}\nDraws: {draws}\n\n')
    print(f'BOT WIN RATE: {losses/games}')
    
