from Environment import Environment
from IQL import IQL

"""
This is the excutable file to test the algorithm, we load the environment then we run the IQL algorithm
"""


#Let's load our environment 
env = Environment(grid_length=25, grid_width=25, num_agents=250, image_path="PFA/IQL/image/hello_1.png", extra_padding=True)

# Now we instanciate our algorithm
learning_algo = IQL(env, learning_rate=0.7, discount_rate=0.9, epsilon=0.9,  max_iter=100 ,  exploration_depth = 100)
learning_algo.main(10)














