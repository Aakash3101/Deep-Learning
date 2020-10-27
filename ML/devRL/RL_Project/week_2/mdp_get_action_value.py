'''
from mdp import MDP

transition_probs = {
    's0': {
        'a0': {'s0': 0.5, 's2': 0.5},
        'a1': {'s2': 1}
    },
    's1': {
        'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
        'a1': {'s1': 0.95, 's2': 0.05}
    },
    's2': {
        'a0': {'s0': 0.4, 's1': 0.6},
        'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}
    }
}
rewards = {
    's1': {'a0': {'s0': +5}},
    's2': {'a1': {'s0': -1}}
}


mdp = MDP(transition_probs, rewards, initial_state='s0')
'''
import numpy as np 

def get_action_value(mdp, state_values, state, action, gamma):
    """
    Q(i)(s,a) = Sum(s)( P(s'|s, a).[r(s,a,s') + gamma * V(i)(s')])
    Computes Q(s,a) as in formula above
    """
    action_value = 0
    for s in mdp.get_all_states():
        action_value += mdp.get_transition_prob(state, action, s)* \
            (mdp.get_reward(state, action, s) + gamma * state_values[s])
    return action_value

def get_new_state_value(mdp, state_values, state, gamma):
    """
    V(i+1)(s) = max Q(i)(s,a) = max (Sum(s)( P(s'|s, a).[r(s,a,s') + gamma * V(i)(s'))
    Computes next V(s) as in formula above. Please do not change state_values in process
    """
    new_V = []
    if mdp.is_terminal(state):
        return 0
    else:
        for a in mdp.get_possible_actions(state):
            new_V.append(get_action_value(mdp, state_values, state, a, gamma))
        return max(new_V)

def get_optimal_action(mdp, state_values, state, gamma=0.9):
    """ 
    Pi* (s) = argmax Q(s,a)
    Finds optimal action using the above formula
    """
    new_V = []
    if mdp.is_terminal(state):
        return None
    else:
        for a in mdp.get_possible_actions(state):
            new_V.append(get_action_value(mdp, state_values, state, a, gamma))
        return mdp.get_possible_actions(state)[np.argmax(new_V)]

def value_iteration(mdp, state_values=None, gamma=0.9, num_iter=1000,
                    min_difference=1e-5):
    """
    performs num_iter value iteration steps starting from state_values,
    Same as before but in a function
    """
    state_values = state_values or {s: 0 for s in mdp.get_all_states()}
    for i in range(num_iter):
        
        # Compute new state values using the functions you defined above
        # It must be a dict {state: new_V(state)}
        new_state_values = {s: get_new_state_value(mdp, state_values, s, gamma)
                        for s in mdp.get_all_states()}

        assert isinstance(new_state_values, dict)

        # Compute difference
        diff = max(abs(new_state_values[s] - state_values[s])
                   for s in mdp.get_all_states())

        #print("iter %4i   |   diff: %6.5f   |   V(start): %.3f " %
        #      (i, diff, new_state_values[mdp._initial_state]))

        state_values = new_state_values
        if diff < min_difference:
            break
        
    return state_values

