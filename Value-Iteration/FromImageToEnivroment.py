
import numpy as np
import cv2
from Environment import Environment

class FromImageToEnivroment:
    def __init__(self, path):
        self.path = path
        self.dimension = (0,0)
        self.num_agents = 0
        self.positions_agents = []
        self.read_image()

    def read_image(self):
        image = cv2.imread(self.path)
        image = cv2.resize(image, (25, 35))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
        _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV) # create a binary thresholded image
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # contours from the thresholded image
        mask = np.zeros_like(gray) # create a mask image for drawing purposes
        cv2.drawContours(mask, contours, -1, (255), 1)
        binary = mask   #cv2.bitwise_not(mask)
        binary = (binary / 255.0).round()
    
        

        
        self.dimension = binary.shape
        self.positions_gools = [ tuple(position) for position in np.argwhere(binary == 1).tolist()]
        self.num_agents =  len(self.positions_gools) * 2

    def get_environment(self):
        return Environment(self.dimension, self.num_agents, self.positions_gools)