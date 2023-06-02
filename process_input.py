import cv2

'''
what we want to do in this code is : 

test different ways of how we want our input for our environment to be, example :
    - an image that we can turn to black and white then agents should occupy black pixels, and white pixels should remain empty

ideas : 
    - split the grid into 4/9/16 sub grids so we can reduce the complexity of the model, and dont take into count interactions 
    that wont even happen

''' 

class Input :

    def __init__(self, path, reward = 1, sanction = -1):
        
        # our input image path
        self.path = path

        # Let's read our img then turn it to black and white
        tmp_img = cv2.imread(path, cv2.IMREAD_COLOR) #
        bl_wh_img = cv2.threshold(tmp_img, 125, 255, cv2.THRESH_BINARY)[1]
        
        # now we retrieve the size the our grid for our environment
        self.grid_size = tmp_img.shape[0]
        img = []
        for i in range(self.grid_size) :
            l = [max(item) for item in bl_wh_img[i]]
            img.append(l)
        self.img = img

        # we define what are the rewards and sanctions in our environement
        self.reward = reward
        self.sanction = sanction

        # we can calculate the number of black pixels so we can make it our number of agents
        self.num_targets = 0
        for row in self.img : 
            self.num_targets += row.count(0)


    def reward_grid(self):
        '''
            If the agent is in a target black pixel, it gets rewarded. If its in a white pixel, the agents gets sanctionned. 
        '''
        reward_list = []
        for row in self.img : 
            rew_row = [[self.reward, True] if item == 0 else [self.sanction, False] for item in row]
            reward_list.append(rew_row)

        return reward_list       





