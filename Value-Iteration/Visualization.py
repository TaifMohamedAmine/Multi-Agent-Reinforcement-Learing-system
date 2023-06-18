from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from Environment import Environment




class Visualization:
    def __init__(self, env):
        self.env = env
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.env.dimension[0])
        self.ax.set_ylim(0, self.env.dimension[1])
        self.i = 0
        self.max_i = 15
        

    def init(self):
        '''
        Initializes the plot
        '''
        #grid_reward = self.env.grid_reward
        # we need the positions and colors for 1 in the grid
        x, y = np.where(self.env.grid_reward == 1)
        color_1 = 'black'
        scatter_1 = self.ax.scatter(x, y, c=color_1, marker='s')

        # we need the positions and colors for -1 in the grid
        x1, y1 = np.where(self.env.grid_reward == -1)
        color_2 = 'green'
        scatter_2 = self.ax.scatter(x1, y1, c=color_2, marker='s')
        print('i = ',self.i)
        self.i += 1

        return scatter_1, scatter_2

    def update(self, frame):
        '''
        Updates the plot with env updates
        ''' 
        # Clear the previous plot
        self.ax.clear()

        # we need the new point positions from env.update
        self.env.update_env()
        grid_reward = self.env.grid_reward

        # Get the positions and colors for 1 in the grid
        x, y = np.where(grid_reward == 1)
        color_1 = 'black'
        scatter_1 = self.ax.scatter(x, y, c=color_1, marker='s')

        # Get the positions and colors for -1 in the grid
        x1, y1 = np.where(grid_reward == -1)
        color_2 = 'green'
        scatter_2 = self.ax.scatter(x1, y1,c=color_2, marker='s')
        print('i = ',self.i)
        self.i += 1

        return scatter_1, scatter_2

    def run(self):
        print('I am here 2')
        ani = FuncAnimation(self.fig, self.update, init_func=self.init ,frames=range(15), interval=200, blit=True)
        ani.save('animation.gif', writer='pillow' , fps=10)
        plt.show()
