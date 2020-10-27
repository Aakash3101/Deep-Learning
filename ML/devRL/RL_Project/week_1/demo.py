import gym
from gym import wrappers
from IPython import display
import matplotlib.pyplot as plt

# create env manually to set time limit. Please don't change this.
TIME_LIMIT = 250
env = gym.wrappers.TimeLimit(
    gym.make("MountainCar-v0"),
    max_episode_steps=TIME_LIMIT + 1,
)
s0 = env.reset()
actions = {'left': 0, 'stop': 1, 'right': 2}

plt.figure(figsize=(4, 3))
display.clear_output(wait=True)

for t in range(TIME_LIMIT):
    plt.gca().clear()
    
    # change the line below to reach the flag
    if t<50:
        s, r, done, _ = env.step(actions['left'])
    else:
        s, r, done, _ = env.step(actions['right'])
    print(s, t)
        

    # draw game image on display
    plt.imshow(env.render('rgb_array'))
    
    display.clear_output(wait=True)
    display.display(plt.gcf())

    if done:
        print("Well done!")
        break
    else:
        print("Time limit exceeded. Try again.")

display.clear_output(wait=True)
env.close()