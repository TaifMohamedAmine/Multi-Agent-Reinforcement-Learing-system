from Environment import Environment
from IQL import IQL


# Let's load our environment 
env = Environment(grid_length=25, grid_width=25, num_agents=300, image_path="/home/amine/Desktop/VS CODE /Projet pfa/IQL/image/hello_1.png", extra_padding=True)

# Now we instanciate our algorithm
learning_algo = IQL(env, learning_rate=0.5, discount_rate=0.8, epsilon=0.3,  max_iter=1000 ,  exploration_depth = 400)
learning_algo.main(100)























