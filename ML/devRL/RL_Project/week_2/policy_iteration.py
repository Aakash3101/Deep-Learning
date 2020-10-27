"""
Policy Iteration

Initialize Pi(0) // random or fixed action

for n= 0, 1, 2
1. Compute the state-value function V^(pi)(n)
2. Using V^(pi)(n), compute the state-action-value function Q (pi)(n)
3. Compute new policy Pi (n+1) (s) = argamax a Q^(pi)(n) (s,a)
"""
from mdp import MDP
import numpy as np
from mdp_get_action_value import get_new_state_value, get_action_value
from mdp_get_action_value import get_optimal_action, value_iteration

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


mdp = MDP(transition_probs, rewards)

def compute_vpi(mdp, policy, gamma):
    """
    Computes V^pi(s) FOR ALL STATES under given policy.
    :param policy: a dict of currently chosen actions {s : a}
    :returns: a dict {state : V^pi(state) for all states}
    """
    res = {}
    state_values = {s: 0 for s in mdp.get_all_states()}
    for s in policy.keys():
        #res[s] = get_new_state_value(mdp, state_values, s, gamma)
        res[s] = get_action_value(mdp, state_values, s, policy[s], gamma)
           
    return res


def compute_new_policy(mdp, vpi, gamma):
    """
    Computes new policy as argmax of state values
    :param vpi: a dict {state : V^pi(state) for all states}
    :returns: a dict {state : optimal action for all states}
    """
    res = {}
    for s in vpi.keys():
        res[s] = get_optimal_action(mdp, vpi, s, gamma)
    
    return res




def policy_iteration(mdp, policy=None, gamma=0.9, num_iter=1000, min_difference=1e-5):
    """ 
    Run the policy iteration loop for num_iter iterations or till difference between V(s) is below min_difference.
    If policy is not given, initialize it at random.
    """
    vpi = {s: 0 for s in mdp.get_all_states()}
    if policy == None:
        
        policy = compute_new_policy(mdp, vpi, gamma)
    
    else:

        policy = policy
        
    for i in range(num_iter):
        
        new_vpi = compute_vpi(mdp, policy, gamma)
        new_policy = compute_new_policy(mdp, new_vpi, gamma)
        # Compute difference
        diff = max(abs(new_vpi[s] - vpi[s]) for s in mdp.get_all_states())
        
    
        #print("iter %4i   |   diff: %6.5f   |   V(start): %.3f " %
        #      (i, diff, new_vpi[mdp._initial_state]))
        
        vpi = new_vpi
        policy = new_policy
        if diff < min_difference:
            break

    return vpi, policy


print(policy_iteration(mdp))
'''
test_policy = {s: np.random.choice(
    mdp.get_possible_actions(s)) for s in mdp.get_all_states()}
test_policy = {'s0': 'a0', 's1': 'a1', 's2':'a0'}

new_vpi = compute_vpi(mdp, test_policy, gamma=0.9)

print(new_vpi)

assert type(new_vpi) is dict, "compute_vpi must return a dict {state : V^pi(state) for all states}"

new_policy = compute_new_policy(mdp, new_vpi, gamma=0.9)

print(new_policy)

assert type(new_policy) is dict, "compute_new_policy must return a dict {state : optimal action for all states}"
'''