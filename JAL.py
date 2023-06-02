import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import matplotlib.animation as animation

"""
in this file we will be trying to implement the Joint action learners MARL algorithm
"""

class JointActionLearning : 
    '''
    Joint action learning class on a subgrid
    '''
    def __init__(self,env ,num_agents, grid_length , grid_width, learning_rate, discount_factor, exploration_ratio):
        
        self.num_agents = num_agents
        self.width, self.length = grid_width, grid_length
        self.num_states = self.width * self.length
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.exploration_ratio = exploration_ratio
        # our action space
        self.actions = {
            'STOP':[0, 0], # the agents stays still
            'UP':[0, 1],   # increase y by 1
            'DOWN':[0, -1], # decrease y by 1
            'RIGHT':[1, 0],# increase x by 1
            'LEFT':[-1, 0] # decrease x by 1
        }

        # a list of our actions :
        self.action_list = list(self.actions.values())

        # initialize the Q matrix to 0
        q_matrix_shape = tuple([self.num_states] + [len(self.actions)] * self.num_agents)
        self.Q_matrix = np.zeros(q_matrix_shape)     

        # initialize the value vector to 0
        self.V = np.zeros(shape=(self.num_states, 1))

        # we initialize the policy matrix with equals probability to transition from each state to all other states
        self.pi = np.full(shape=(self.num_states, len(self.actions)), fill_value=1/(len(self.actions)))

        # array C that does something which i dont know yet
        self.C = np.zeros(q_matrix_shape)

        # array n that counts the numbers of transitions to each state
        self.n = np.zeros(shape=(self.num_states, 1))

        # the game environement 
        self.env = env

        # a list of the environement agents
        self.agents = self.env.agents

        # a list of rewards in each state 
        self.reward_list = env.reward_list

    def observe_current_state(self):
        """
        function to observe the states of the agents from the environment
        ==> to be updated later
        """
    
    def convert_state_to_idx(self, state):
        """
        this method is the link between the grid and trainin matrices
        
        given a state like (i ,j) we need to convert it to an index idx: 
        """
        index = state[0]*self.length + state[1]
        return index
    

    def choose_action(self, state):
        """
        this function is used to explore and choose an action with the given probabilities
        """
        transition_proba = self.pi[state]

        if len(np.unique()) == 1 :
            # only for the first iteration since the initial action probas is the same
            return random.choice(list(self.actions.values()))
        else : 

            # we add an exploration term
            if np.random.uniform() < self.exploration_ratio : # epsilon greedy approach
                print("exploring a random action")
                action = random.choice(list(self.actions.values()))
                return action
            else : 
                # we choose an action following the given policy
                u = np.random.uniform()
                cum_weights = list(np.cumsum(transition_proba))
                for item in cum_weights : 
                    if u < item :
                        return list(self.actions.values())[cum_weights.index(item)]    
        

    """
    Notes : the actions and states used will be in the form of indexes
    """

    def train(self):
        """
        the JAL training method
        """
        max_iter = 100
        for itr in range(max_iter):
            agent_idx = 0
            for agent in self.agents : 
                state = agent.pos
                state_idx = self.convert_state_to_idx(state)
                agent.action = self.choose_action(state_idx) # chosen action with the current policy 
                action_idx = self.action_list.index(agent.action)
                
                # let's create a list of all other agents indexes:
                idx_list = [item for item in list(range(self.num_agents)) if item !=agent] # equivalent to -i 

                # we observe some informations about the other agents
                action_observ = [self.action_list.index(item.action)  if item != agent else self.action_list(agent.action) for item in self.agents]
                """reward_observ = [item.reward for item in self.agents if item != agent]
                nextstate_observ = [item.next_state for item in self.agents if item != agent]"""

                # the Q value for the current state and joint actions of the other agents
                current_Q_value = self.Q_matrix[state_idx][action_observ]   
                
                agent_next_state  = agent.next_state
                agent_next_state_idx = self.convert_state_to_idx(agent_next_state)

                # we extract the reward of the current agent.
                reward = self.reward_list[agent_next_state]

                # update the Q value : 
                self.Q_matrix[state_idx][action_observ] = current_Q_value + self.lr * (reward + self.discount_factor*self.V[agent_next_state_idx]
                                                                                        - current_Q_value) 

                # update the C matrix : 
                self.C[state_idx][action_observ] = self.C[state_idx][action_observ] + 1 
                
                # update the n array : 
                self.n[state_idx] = self.n[state_idx] + 1

                # Let's now update the policy matrix pi : 

                self.pi[state_idx, :] = None

                # Let's update V  :

                sum_var = 0
                for action_id in range(len(self.actions)):
                    prob = self.pi[state_idx,action_id]
                    tmp_action_observ = [self.action_list.index(item.action) if item != agent else action_id for item in self.agents]
                    C_k = self.C[state_idx][tmp_action_observ]
                    sum_var += (prob * C_k * self.Q_matrix[state_idx][action_observ]) / self.n[state_idx]
                
                self.V[state_idx] = sum_var

                # we decay our learning rate with each:  
                self.lr = self.lr * ((max_iter - itr) / max_iter)

                # move agent

                agent_idx += 1






if __name__ == '__main__':
    print("test")
