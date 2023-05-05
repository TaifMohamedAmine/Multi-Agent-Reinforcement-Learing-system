import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


class Agent:
    '''
    agent class where we initialize each agent of our environement
    '''
    def __init__(self, pos, id):
        self.id = id # we give each agent a unique id
        self.pos = pos # initial pos of the agent
        self.vel = [0, 0]  # velocity for each of x and y
        self.radius = 1  # radius of the agent ball

    def move(self):
        self.pos += self.vel # update the velocity


class Environment :
    '''
    environement class where we create the env where the agents interact
    '''
    def __init__(self, grid_size, num_agents):
        
        self.num_agents = num_agents # the number of agents
        self.grid_size = grid_size #  the height of the environment 
        self.agents = [] # list of the env agents (initialized with zero agents)
        
        x_idx = list(range(self.grid_size))
        y_idx = list(range(self.grid_size))

        # randomly scatter my agents in the grid
        agent_id = 0
        for agent in range(self.num_agents) :
            x, y = random.choice(x_idx), random.choice(y_idx)
            x_idx.remove(x)
            y_idx.remove(y)
            agent = Agent([x, y], id)
            self.agents.append(agent) # add the agent to the list of agents with pos (x,y) and id = agent_id
            agent_id+=1

    def update_env(self):
        '''
        this function updates the environement following a policy. 
        !!! needs work :=)
        '''
        


        return 

class Visualization :

    def __init__(self, env):
        self.env = env # we initialize our environement
        self.fig, self.ax = plt.subplots()
        self.scat = self.ax.scatter([], []) # empty figure at first
        self.ax.set_xlim(0, self.env.grid_size) 
        self.ax.set_ylim(0, self.env.grid_size)
    
    def update(self, i):
        '''
        updates the plot with env updates 
        '''
        # we need the new point positions from env.update

        self.env.update_env()
        new_agent_data = self.env.agents
        # let's retrive the new positions
        x = [agent.pos[0] for agent in new_agent_data]
        y = [agent.pos[1] for agent in new_agent_data] 
        self.scat.set_offsets(np.c_[x, y])
        return self.scat, 

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, interval=50)
        plt.show()


class MARL : 
    '''
    our system's training class
    '''
    def __init__(self, env) :
        self.env = env # we initialize our environment (methods : )
        
    
    
        
        














if __name__ == '__main__':
    
    """
    make the each agent start in a random position of the grid
    """

    grid_size = 1000
    num_agents = 100

    env = Environment(grid_size, num_agents)

    vis = Visualization(env)
    vis.run()






