import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Agent import Agent
from MDP import MDP



class Environment : 
    
    def __init__(self, dimension, num_agents , positions_gools):
        self.dimension = dimension
        self.num_agents = num_agents
        self.old_agents = []
        self.positions_gools = positions_gools[:]
        self.positions_gools_achieved = []
        self.agents = []
        self.grid_reward = np.zeros((self.dimension[0], self.dimension[1]))
        self.mdp = MDP(self)
        self.V, self.policy = self.mdp.value_iteration()
       
        # we initialize the agents in random unique positions and we need to make sure that the agents are not initialized in the same position
        agent_id = 0
        agent_position = []
        while agent_id < self.num_agents :
            x = np.random.randint(0, self.dimension[0])
            y = np.random.randint(0, self.dimension[1])
            if [x,y] not in agent_position :
                agent = Agent([x, y], agent_id)
                self.agents.append(agent)
                agent_id+=1
                agent_position.append([x,y])
        self.update_grid_reward(True)
        #self.plot_policy()
    
    # function of grid reward where 1 in the position of agent and -1 in the position of gool and 0 in the rest of the grid
    def update_grid_reward(self , inital = False) :
        if inital :
            for gool in self.positions_gools :
                self.grid_reward[gool[0], gool[1]] = -1
            for agent in self.agents :
                self.grid_reward[agent.position[0], agent.position[1]] = 1
            self.update_agents_gools()
        else :
            self.grid_reward = np.zeros((self.dimension[0], self.dimension[1]))
            for gool in self.positions_gools :
                self.grid_reward[gool[0], gool[1]] = -1
            for agent in self.agents :
                self.grid_reward[agent.position[0], agent.position[1]] = 1
            for agent in self.old_agents :
                self.grid_reward[agent.position[0], agent.position[1]] = 1
            

        
       
    
    # function of update agents gools where if the agent is in the position of gool then gool = True and remove the gool from the list of gools
    def update_agents_gools(self):
        for agent in self.agents :
            if tuple(agent.position) in self.positions_gools :
                agent.gool = True
                self.positions_gools.remove(tuple(agent.position))
                self.positions_gools_achieved.append(tuple(agent.position))
                # remove the agent from the list of agents
                self.old_agents.append(agent)
                self.agents.remove(agent)



    # function of update env where we update the position of the agents and the grid reward
    def update_env(self):
        A = {'up':[-1, 0], 'down':[ 1, 0], 'right':[0,  1], 'left':[0,  - 1], 'stay':[0, 0]}
        for agent in self.agents :
            i = 0 
            while agent.gool == False and i < 20 :
                agent_policy = self.policy[tuple(agent.position)]
                deplacement = A[agent_policy]
                move = agent.update_position(deplacement , self.dimension[0] , self.dimension[1])
                if move : 
                    agent.reward += self.V[tuple(agent.position)]
                self.update_agents_gools()
                self.update_grid_reward()
                i+=1


    def plot_policy(self):
        # create matrix of reward from the grid reward convert 1 to 0 
        reward_matrix = self.grid_reward.copy()
        reward_matrix[reward_matrix == 1] = 0

        # Create a mapping of policy directions to arrow components (x, y)
        direction_map = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0),
            'stay': (0, 0)
        }

        # Create two matrices for the x and y components of the arrow directions
        arrow_x = np.zeros((self.dimension[0], self.dimension[1]))
        arrow_y = np.zeros((self.dimension[0], self.dimension[1]))

        for i in range(self.dimension[0]):
            for j in range(self.dimension[1]):
                direction = self.policy[(i, j)]
                arrow_x[i, j], arrow_y[i, j] = direction_map[direction]

        # Create a new figure
        plt.figure()

        # Plot the heatmap using the reward matrix
        plt.imshow(reward_matrix, cmap='hot', interpolation='nearest', alpha=0.7)

        # Plot the arrows
        X, Y = np.meshgrid(np.arange(0, self.dimension[1]), np.arange(0, self.dimension[0]))
        plt.quiver(X, Y, arrow_x, arrow_y, angles='xy', scale_units='xy', scale=1, color='black')

        # Plot the blue points for the 'stay' action
        for i in range(self.dimension[0]):
            for j in range(self.dimension[1]):
                if self.policy[(i, j)] == 'stay':
                    plt.plot(j, i, 'bo', markersize=10)

        # Add a legend to explain the plot
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Arrows: Move actions'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Blue dot: Stay action'),
            Line2D([0], [0], color='w', lw=0, label='Heatmap: Reward matrix')
        ]

        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        # Show the plot
        plt.show()