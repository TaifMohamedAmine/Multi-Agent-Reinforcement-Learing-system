import numpy as np
import random
from Visualization import Visualization
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
Notes : 
        - for each move the agents dont occupy the target position, the transition reward is : -0.1
        - for each move where the agent will occupy the target position, the transition reward is : +1
        - a position can only be occupied by one agent only
        - the target position is stationnary and doesnt change gradually in the game
        - we still need to evaluate each agent !! 
        - probability to move from one state to another is the same amongst all the agents
        - all agents share the same reward function
        
'''

"""
stuff i need to pay attention to : 
    - when two agent's new state is the same ( we cant afford that cuz we're working with positions )
    ==> can be solved with the sequential aspect of this approach
"""


class IQL : 
    """
    Independent Q learning algorithm
    """
    def __init__(self, env , learning_rate, discount_rate, epsilon, max_iter, exploration_depth):

        ''' 
        we need to create our reward function to a grid initialized with an empty array
        the size of the reward function is (number of states * number of states)
        '''

        # our game environment :
        self.env = env

        # some hyperparameters for our algo :
        self.lr, self.discount_rate, self.eps, self.max_iter = learning_rate, discount_rate, epsilon, max_iter

        # we set a max number of iterations in explorations for each episode
        self.depth = exploration_depth

        # some algo params 
        self.num_states = self.env.grid_length * self.env.grid_width
        self.action_list = list(self.env.actions.values())

        # we create a list of Q_matrices for all our agents
        self.Q_matrices_list = []

        for agent in self.env.agents :
            tmp_Q_matrix = np.zeros((self.num_states, len(self.env.actions)))
            self.Q_matrices_list.append([agent,tmp_Q_matrix])

        self.fig, self.ax = plt.subplots()
        self.scat = self.ax.scatter([], []) # empty figure at first
        self.ax.set_xlim(0, self.env.grid_length) 
        self.ax.set_ylim(0, self.env.grid_width)

        
    
    def convert_state_idx(self, state):
        """
        its a method we need to convert a position state to a table index
        """
        return state[0]*self.env.grid_length + state[1]



    def convert_action_idx(self, action):
        """
        a method to get the index of the action in order to access the Q tabel
        """
        index = self.action_list.index(action)
        return index


    def check_invalid_index(self, state):
        """
        this method adds a big penalty if an updated state is out of our grid
        """
        x, y = state[0], state[1]
        
        if (x >= self.env.grid_length or x < 0) or (y >= self.env.grid_width or y < 0) : 
            return True 
        return False
        


    def eps_greedy_policy(self, Q_matrix , state, exploration):
        """
        this method is to select an action using the q matrix and an exploration proba
        """
        u = np.random.uniform()
        if u < exploration : 
            action = random.choice(self.action_list)

        else : 
            action_idx = np.argmax(Q_matrix[state])
            action = self.action_list[action_idx]

        return action
    

    def train(self):
        """
            Our main training function 
        """

        print("training has begun !!")

        for episode in range(self.max_iter):

            print(f"***************** episode {episode} ******************")

            # reset the env in each ep for agents to test multiple  :
            self.env.reset_env()

            # for each tick, each agent continues exploring his state
            for tick in range(self.depth) :
        
                agent_itr = 0

                for agent in self.env.agents : 

                    #print(agent.pos)

                    # this agent's Q matrix :
                    agent_Q_matrix = self.Q_matrices_list[agent_itr][1]

                    if not agent.reached_end_state :     

                        # Let's get the current state (position) of the agent
                        current_state = agent.pos
                        curr_state_idx = self.convert_state_idx(current_state)
                        
                        # now we select and action using an epsilon greedy policy
                        action = self.eps_greedy_policy(agent_Q_matrix, curr_state_idx, self.lr)
                        
                        # update the agent's action and state :
                        agent.action = action

                        # we now need to update the Q table
                        curr_state_idx, action_idx = self.convert_state_idx(current_state) ,self.convert_action_idx(action)

                        old_Q_value = agent_Q_matrix[curr_state_idx][action_idx]
                        
                        agent.next_state = [x + y for x, y in zip(agent.pos, agent.action)]

                        new_state = agent.next_state

                        #print(agent.pos, agent.action , agent.next_state)


                        if self.check_invalid_index(new_state): 
                            
                            reward = -100 # big penalty if the agent goes out of bounds
                            
                            Best_new_state_Q

                            self.Q_matrices_list[agent_itr][1][curr_state_idx][action_idx] = (1 - self.lr) * old_Q_value + self.lr*(reward + 
                                                                                                        self.discount_rate * Best_new_state_Q)



                        else :
                            new_move_updates = self.env.reward_list[new_state[0]][new_state[1]]# the reward of the new state
                        
                            reward, agent.reached_end_state = new_move_updates[0], new_move_updates[1]

                            """if agent.reached_end_state : 
                                print("this agent reached an end state")"""


                            Best_new_state_Q = max(agent_Q_matrix[self.convert_state_idx(new_state)])

                            self.Q_matrices_list[agent_itr][1][curr_state_idx][action_idx] = (1 - self.lr) * old_Q_value + self.lr*(reward + 
                                                                                                        self.discount_rate * Best_new_state_Q)

                                
                            #print(self.Q_matrices_list[agent_itr][1][curr_state_idx])
                            # the agent excecutes the action and updates to new state 
                            agent.move()
                    
                    agent_itr += 1
            
            self.env.check_target()

            # we need to decay the exploration rate (self.eps) by 1 / max_iter
            self.eps = self.eps * ((self.max_iter - episode) / self.max_iter)

        return self.Q_matrices_list

    def update_agents(self, i):
        """
        a function that updates the states of agents with no exploration
        """

        #print(f"this method is ran in frame {i}")

        agent_idx = 0 

        x, y = [], []

        for agent in self.env.agents : 
                
            if not agent.reached_end_state :

                # this agent's Q matrix :
                agent_Q_matrix = self.Q_matrices_list[agent_idx][1]

                # Let's get the agents current state
                curr_state = agent.pos
                curr_state_idx = self.convert_state_idx(curr_state)

                # get the propriate action with no exploration : 
                action = self.eps_greedy_policy(agent_Q_matrix, curr_state_idx, 0)

                print(agent_Q_matrix[curr_state_idx], action)

                # update the action and positon of the agent 
                agent.action = action
                
                
                agent.next_state = [a + b for a, b in zip(agent.pos, agent.action)]

                if self.check_invalid_index(agent.next_state):
                    agent.next_state = agent.pos 

                agent.reached_end_state = self.env.reward_list[agent.next_state[0]][agent.next_state[1]][1]

                x.append(agent.next_state[0])
                y.append(agent.next_state[1])

                agent.move()

            else : 
                x.append(agent.pos[0])
                y.append(agent.pos[1])

            agent_idx += 1 

        self.scat.set_offsets(np.c_[x, y])
        
        return self.scat, 



    def simulate(self, max_mov):
        """
        this method is to simulate the trained and final Q tables
        """  
        # reset the env in each ep for agents to test multiple  :
        self.env.reset_env()

        # we create a list to check if all agents have reached the target pixels
        reached_target = [agent.reached_end_state for agent in self.env.agents]

        itr = 0
        
        # in each episode, the agent updates its position one time, we repeat until all the agents reach a target pixel
        while( (not all(reached_target)) and itr < max_mov) :      
            
            self.update_agents()
            
            # update the the bool list
            reached_target = [agent.reached_end_state for agent in self.env.agents]
            itr += 1

        
    def main(self, max_mov):
        """
        a function where we execute all the functions
        """
        self.train()
        
        self.env.reset_env()

        animation = FuncAnimation(self.fig, self.update_agents, frames = max_mov, interval = 200, blit = True)
        plt.show()



            












