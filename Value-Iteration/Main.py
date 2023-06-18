from FromImageToEnivroment import FromImageToEnivroment
from Visualization import Visualization
from Test import Test
class Main :
    path = r'C:\Users\viet\Desktop\S4\Multi-Agent-Reinforcement-Learning-system-main\star.png'
    #from_image_to_env = FromImageToEnivroment(path)
    #env = from_image_to_env.get_environment()
    #vis = Visualization(env)
    #vis.run()
    test = Test(path)
    #test.plot_achieved_gools()
    #test.plot_achieved_gools_in_function_iterations()
    #test.time_iteration()
    #test.plot_achieved_gools_in_function_iterations()
    test.plot_total_rewards()