import cv2

'''
what we want to do in this code is : 

test different ways of how we want our input for our environment to be, example :
    - an image that we can turn to black and white then agents should occupy black pixels, and white pixels should remain empty

'''

class Input :

    def __init__(self, path, reward, sanction):
        
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
        self.num_agents = 0
        for row in self.img : 
            self.num_agents += row.count(0)


        

    
    def reward_grid(self):
        '''
        i thought of creating an environment like mine sweep : 
            if the agent is in a black pixel, it gets rewarded. If its in a white pixel, the agents gets sanctionned. 
        '''
        reward_list = []
        for row in self.img : 
            rew_row = [self.reward if item == 0 else self.sanction for item in row]
            reward_list.append(rew_row)        




# we test the class 

test = Input("projet pfa/star.png", 100, -99)
print(test.num_agents) # prints the number of black pixels in the input image




