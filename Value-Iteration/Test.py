from Environment import Environment
from FromImageToEnivroment import FromImageToEnivroment
import matplotlib.pyplot as plt
import timeit



class Test :
    def __init__(self, path):
        self.path = path
        from_image_to_env = FromImageToEnivroment(path)
        self.env = from_image_to_env.get_environment()
        self.positions_gools = self.env.positions_gools

    # make a plot of the number of achieved gools in function of the number of agents
    def plot_achieved_gools(self):
        psition_gools = self.env.positions_gools
        number_agents = []
        number_achieved_gools = []
        for i in range(0,11):
            nb = self.env.num_agents + int(self.env.num_agents * i * 0.1)
            number_agents.append(nb)
            env2 = Environment(self.env.dimension, nb , psition_gools)
            for j in range(4):
                env2.update_env()
            number_achieved_gools.append(len(env2.positions_gools_achieved))

        # plot the number of achieved gools in function of the number of agents
        plt.plot(number_agents, number_achieved_gools)
        plt.xlabel('number of agents')
        plt.ylabel('number of achieved gools')
        plt.show()


    # make function of the number of achives gools in function iteations
    def plot_achieved_gools_in_function_iterations(self):
        # make copie of number of agents and positions of gool
        nb_agents = self.env.num_agents
        psition_gools = self.env.positions_gools
        number_iter = []
        number_achieved_gools = []
        for i in range(15) :
            number_iter.append(i)
            env2 = Environment(self.env.dimension, nb_agents , psition_gools)
            for j in range(i):
                env2.update_env()
            number_achieved_gools.append(len(env2.positions_gools_achieved))

        # plot the number of achieved gools in function of the number of agents
        plt.plot(number_iter, number_achieved_gools)
        plt.xlabel('number of iterations')
        plt.ylabel('number of achieved gools')
        plt.show()

    # function the time in fucntion of iteration 
    def time_iteration(self) :
        # make copie of number of agents and positions of gool
        nb_agents = self.env.num_agents
        psition_gools = self.env.positions_gools
        number_iter = []
        time = []
        for i in range(15) :
            number_iter.append(i)
            env2 = Environment(self.env.dimension, nb_agents , psition_gools)
            start = timeit.default_timer()
            for j in range(i):
                env2.update_env()
            stop = timeit.default_timer()
            time.append(stop - start)

        # plot the number of achieved gools in function of the number of agents
        plt.plot(number_iter, time)
        plt.xlabel('number of iterations')
        plt.ylabel('time')
        plt.show()

     # function plot the number of achieved gools in function of the parametres of value iteration
    def plot_achieved_gools_in_function_parametres(self):
        discount_factor = np.arange(0, 1, 0.2)
        # epsilon form 10^-5 to 10^-1
        epsilon = np.arange(0.00001, 0.1, 0.02)
        # max_iterations from 10 to 100
        max_iterations = np.arange(10, 100, 10)
        number_achieved_gools1 = []
        number_achieved_gools2 = []
        number_achieved_gools3 = []
        for i in discount_factor :
            self.env.V, self.env.policy = self.env.mdp.value_iteration(discount_factor=i)
            for j in range(4):
                self.env.update_env()
            number_achieved_gools1.append(len(env.positions_gools_achieved))

        for i in epsilon :
            self.env.V, self.env.policy = self.env.mdp.value_iteration(epsilon=i)
            for j in range(4):
                self.env.update_env()
            number_achieved_gools2.append(len(env.positions_gools_achieved))
    
        for i in max_iterations :
            self.env.V, self.env.policy = self.env.mdp.value_iteration(max_iterations=i)
            for j in range(4):
                self.env.update_env()
            number_achieved_gools3.append(len(self.env.positions_gools_achieved))

        # Create a figure with 3 subplots in a single row
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the number of achieved goals for each parameter in separate subplots
        axes[0].plot(discount_factor, number_achieved_gools1, label='discount_factor')
        axes[0].set_xlabel('discount_factor')
        axes[0].set_ylabel('number of achieved goals')
        axes[0].legend()

        axes[1].plot(epsilon, number_achieved_gools2, label='epsilon')
        axes[1].set_xlabel('epsilon')
        axes[1].set_ylabel('number of achieved goals')
        axes[1].legend()

        axes[2].plot(max_iterations, number_achieved_gools3, label='max_iterations')
        axes[2].set_xlabel('max_iterations')
        axes[2].set_ylabel('number of achieved goals')
        axes[2].legend()

        # Display the figure with 3 subplots
        plt.show()

    # function plot the total of rewards in function of the number of iterations and the number of agents
    def plot_total_rewards(self):
        nb_agents = self.env.num_agents
        psition_gools = self.env.positions_gools
        number_agents = []
        iterations = []
        num_gools_achieved = []
        total_reward3 = []
        total_reward2 = []
        total_reward1 = []
        for i in range(0,11):
            nb = self.env.num_agents + int(self.env.num_agents * i * 0.1)
            number_agents.append(nb)
            env2 = Environment(self.env.dimension, nb , psition_gools)
            for j in range(4):
                env2.update_env()
            
            total_reward1.append(sum([agent.reward for agent in env2.old_agents]))

        for i in range(15) :
            iterations.append(i)
            env2 = Environment(self.env.dimension, nb_agents , psition_gools)
            for j in range(i):
                env2.update_env()
            total_reward2.append(sum([agent.reward for agent in env2.old_agents]))

        env2 = Environment(self.env.dimension, nb_agents , psition_gools)   
        num_gools_achieved.append(len(env2.positions_gools_achieved))
        total_reward3.append(sum([agent.reward for agent in env2.old_agents]))
        for j in range(15):
            env2.update_env()
            num_gools_achieved.append(len(env2.positions_gools_achieved))
            total_reward3.append(sum([agent.reward for agent in env2.old_agents]))
    
        # Create a figure with 3 subplots (1 row, 3 columns)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the first graph: total_reward1 vs number_agents
        ax1.plot(number_agents, total_reward1)
        ax1.set_title('Total Reward vs Number of Agents')
        ax1.set_xlabel('Number of Agents')
        ax1.set_ylabel('Total Reward')

        # Plot the second graph: total_reward2 vs iterations
        ax2.plot(iterations, total_reward2)
        ax2.set_title('Total Reward vs Iterations')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Total Reward')

        # Plot the third graph: total_reward3 and num_gools_achieved vs iterations
        #ax3.plot(iterations, total_reward3, label='Total Reward')
        ax3.plot(total_reward3, num_gools_achieved, label='Number of Goals Achieved')
        ax3.set_title('Total Reward and Goals Achieved ')
        ax3.set_xlabel('Total Reward')
        ax3.set_ylabel('Goals Achieved')
        ax3.legend()

        # Display the plots
        plt.tight_layout()
        plt.show()


    