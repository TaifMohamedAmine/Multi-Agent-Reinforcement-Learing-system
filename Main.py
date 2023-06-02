import numpy as np
from Environment import Environment
from IQL import IQL

# Let's load our environment 
env = Environment(grid_length=25, grid_width=25, num_agents=300, image_path="Projet pfa/images/hello_1.png")

# Now we instanciate our algorithm
learning_algo = IQL(env, learning_rate=0.5, discount_rate=0.9, epsilon=0.3,  max_iter=100 ,  exploration_depth = 1000)


learning_algo.main(100)




























