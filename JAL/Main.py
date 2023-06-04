import numpy as np
from Environment_JAL import Environment
from JAL import JointActionLearning


# Let's load our environment 
env = Environment(grid_length=25, grid_width=25, num_agents=5, image_path="/home/amine/Desktop/VS CODE /PFA/JAL/images/hello_1.png", extra_padding=False)

joint = JointActionLearning(env, learning_rate=0.4, discount_factor=0.9, exploration_ratio=0.5, max_iter=500, exploration_depth=50)

joint.main(10)
























