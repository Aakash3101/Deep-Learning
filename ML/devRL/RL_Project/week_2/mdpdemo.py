from mdp import MDP
from mdp import has_graphviz, plot_graph_with_state_values
#from IPython.display import display
import numpy as np
from mdp_get_action_value import get_action_value, get_new_state_value
from mdp_get_action_value import get_optimal_action
from mdp import plot_graph_optimal_strategy_and_state_values

#print("Graphviz available:", has_graphviz)

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
print('initial state =', mdp.reset())
next_state, reward, done, info = mdp.step('a1')
print('next_state = %s, reward = %s, done = %s' % (next_state, reward, done))

print("mdp.get_all_states =", mdp.get_all_states())
print("mdp.get_possible_actions('s1') = ", mdp.get_possible_actions('s1'))
print("mdp.get_next_states('s1', 'a0') = ", mdp.get_next_states('s1', 'a0'))
print("mdp.get_reward('s1', 'a0', 's0') = ", mdp.get_reward('s1', 'a0', 's0'))
print("mdp.get_transition_prob('s1', 'a0', 's0') = ", mdp.get_transition_prob('s1', 'a0', 's0'))
'''

#Visualizing the MDPs
'''
if has_graphviz:
    from mdp import plot_graph, plot_graph_with_state_values, plot_graph_optimal_strategy_and_state_values
    
    plot_graph(mdp).render(view=True)
'''


test_Vs = {s: i for i, s in enumerate(sorted(mdp.get_all_states()))}
#print(test_Vs)
assert np.allclose(get_action_value(mdp, test_Vs, 's2', 'a1', 0.9), 0.69)
assert np.allclose(get_action_value(mdp, test_Vs, 's1', 'a0', 0.9), 3.95)

test_Vs_copy = dict(test_Vs)
#print(get_new_state_value(mdp, test_Vs, 's2', 0.9))
assert np.allclose(get_new_state_value(mdp, test_Vs, 's0', 0.9), 1.8)
assert np.allclose(get_new_state_value(mdp, test_Vs, 's2', 0.9), 0.69)
assert test_Vs == test_Vs_copy, "please do not change state_values in get_new_state_value"


#parameters
gamma = 0.9             # discount for MDP
num_iter = 100          # maximum iterations, excluding initialization
# stop VI if new values are this close to old values (or closer)
min_difference = 0.001

#initialize V(s)
state_values = {s: 0 for s in mdp.get_all_states()}

#if has_graphviz:
#    plot_graph_with_state_values(mdp, state_values).render(view=True)

for i in range(num_iter):

    # Compute new state values using the functions you defined above
    # It must be a dict (state : float V_new(state))

    new_state_values = {s: get_new_state_value(mdp, state_values, s, gamma)
                        for s in mdp.get_all_states()}

    assert isinstance(new_state_values, dict)

    # Compute difference
    diff = max(abs(new_state_values[s] - state_values[s])
               for s in mdp.get_all_states())

    print("iter %4i   |   diff: %6.5f   |   " % (i, diff), end="")

    print('   '.join("V(%s) = %.3f" % (s, v)
                     for s, v in state_values.items()), end='\n\n')

    state_values = new_state_values

    if diff < min_difference:
        print("Terminated")
        break

if has_graphviz:
    plot_graph_with_state_values(mdp, state_values).render(view=True)


print("Final state values:", state_values)

assert abs(state_values['s0'] - 8.032) < 0.01
assert abs(state_values['s1'] - 11.169) < 0.01
assert abs(state_values['s2'] - 8.921) < 0.01


assert get_optimal_action(mdp, state_values, 's0', gamma) == 'a1'
assert get_optimal_action(mdp, state_values, 's1', gamma) == 'a0'
assert get_optimal_action(mdp, state_values, 's2', gamma) == 'a0'

if has_graphviz:
    try:
        plot_graph_optimal_strategy_and_state_values(mdp, state_values).render(view=True)
    except ImportError:
        raise ImportError("Run the cell that starts with \"%%writefile mdp_get_action_value.py\"")

# Measure agent's average reward

s = mdp.reset()
rewards = []
for _ in range(10000):
    s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
    rewards.append(r)

print("average reward: ", np.mean(rewards))

assert(0.85 < np.mean(rewards) < 1.0)