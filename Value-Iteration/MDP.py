import numpy as np  
from itertools import product



class MDP :
    def __init__(self,env) :
        self.env = env
        self.dimension = self.env.dimension
        self.states = list(product(range(self.dimension[0] ), range(self.dimension[1] )))
        self.actions = ['up', 'down', 'left', 'right' , 'stay']


    def count_neighbors(self, state):
        n, m = self.dimension[0], self.dimension[1]
        # Apply the 4 possible movements on the state and store the results in a list
        neighbors = [((state[0] + 1), state[1]), ((state[0] - 1), state[1]), (state[0], (state[1] + 1)), (state[0], (state[1] - 1))]
        # Count the number of neighbors that do not exceed the grid limits
        count = 0
        for i, neighbor in enumerate(neighbors):
            if neighbor[0] >= 0 and neighbor[0] < n and neighbor[1] >= 0 and neighbor[1] < m:
                count += 1

        return count



    

    def get_transition_probabilities(self, state, new_state ,action):
        # creer un dict contient chaque action et (i,j) correspondant
        A = {'up':(-1, 0), 'down':( 1, 0), 'right':(0,  1), 'left':(0,  - 1), 'stay':(0, 0)}
        # tester si on applique l'action 'action' sur l'etat 'state' on obtient l'etat 'new_state'
        if (state[0] + A[action][0] == new_state[0]) and (state[1] + A[action][1] == new_state[1]):
            nbr_neighbors  = self.count_neighbors(state)
            return 1 / (nbr_neighbors+1)
        else :
            return 0.0
        
        
    def get_reward(self, state, new_state ,action):
        if self.get_transition_probabilities(state, new_state ,action) != 0.0:
            if new_state in self.env.positions_gools_achieved:
                return -1
            elif new_state in self.env.positions_gools:
                return 1
            else:
                return 0
        else:
            return 0
    
    def value_iteration(self, discount_factor=1, epsilon=1e-4 , max_iterations=100):
        # Initialize the value function for all states
        V = {state: 0 for state in self.states}

        for i in range(max_iterations):
            delta = 0
            # Update the value function for each state
            for state in self.states:
                old_value = V[state]
                new_value = float('-inf')

                # Calculate the maximum expected value for each action
                for action in self.actions:
                    action_value = 0
                    for new_state in self.states:
                        transition_prob = self.get_transition_probabilities(state, new_state, action)
                        reward = self.get_reward(state, new_state, action)
                        action_value += transition_prob * (reward + discount_factor * V[new_state])
                        #print('action_value : ',action_value)

                    new_value = max(new_value, action_value)

                # Update the value function for the current state
                V[state] = new_value
                delta = max(delta, abs(old_value - new_value))

            # afficher l'etat d'avancement de l'algo
            #print('iteration : ',i)

            # Check for convergence
            if delta < epsilon:
                break

        # Compute the optimal policy
        policy = {state: None for state in self.states}
        for state in self.states:
            best_action = None
            best_value = float('-inf')

            # Find the best action for each state
            for action in self.actions:
                action_value = 0
                for new_state in self.states:
                    transition_prob = self.get_transition_probabilities(state, new_state, action)
                    reward = self.get_reward(state, new_state, action)
                    action_value += transition_prob * (reward + discount_factor * V[new_state])
                
                    

                if action_value > best_value:
                    #print('action_value : ',action_value)
                    best_value = action_value
                    best_action = action

            policy[state] = best_action

        return V, policy