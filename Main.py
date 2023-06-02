import numpy as np
from Environment import Environment
from IQL import IQL


"""
This is the main execution file 
"""

# Let's load our environment 
env = Environment(grid_length=50, grid_width=50, num_agents=1000, image_path="Projet pfa/images/helloo.png")

# Now we instanciate our algorithm
learning_algo = IQL(env, learning_rate=0.6, discount_rate=0.9, epsilon=1,  max_iter=2000 ,  exploration_depth = 2000)

learning_algo.main(1000)




























