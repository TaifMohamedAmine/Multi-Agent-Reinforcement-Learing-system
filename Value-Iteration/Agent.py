class Agent :
    def __init__(self, position, id):
        self.position = position
        self.id = id
        self.deplacement = [0,0]
        self.gool = False 
        self.reward = 0

    def update_position(self, deplacement , n , m):
        # chek if the agent doesn't go out of the grid with the deplacement
        self.deplacement = deplacement
        if self.position[0] + deplacement[0] < 0 or self.position[0] + deplacement[0] >= n or self.position[1] + deplacement[1] < 0 or self.position[1] + deplacement[1] >= m :
            return False
        else :
            self.position[0] += deplacement[0]
            self.position[1] += deplacement[1]
            return True