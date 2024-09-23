import numpy as np
import math

class Utility:
    """
    Utility function for the ants
    """
    def __init__(self, u_exploration = 0, u_exploitation = 0, alpha = 0):
        """
        u_exploration: utility for exploration
        u_exploitation: utility for exploitation
        alpha: weight for the utility function
        """
        self.u_exploration = u_exploration
        self.u_exploitation = u_exploitation
        self.alpha = alpha
    
    def calculate_utility(self):
        """
        Calculates the utility of the ant

        Returns
        -------
        float
            utility of the ant
        """
        return self.alpha*self.u_exploitation + (1-self.alpha)*self.u_exploration

class Ant:
    """
    Ant class
    """

    def __init__(self, position, u_exploration=0,u_exploitation=0,alpha=0):
        """
        position: position of the ant
        u_exploration: utility for exploration
        u_exploitation: utility for exploitation
        alpha: weight for the utility function
        """
        self.S = position
        self.F = Utility(u_exploration, u_exploitation, alpha)
        self.U = self.F.calculate_utility()

    def __lt__(self, other):
         return self.U < other.U

class Table:
    """
    Table class
    """
    def __init__(self, num_ants):
        """
        num_ants: number of ants
        """
        self.num_ants = num_ants
        self.ants = []
        self.W = []
 
    def initialize_ants(self):
        """
        Initializes the ants
        """
        width = 10
        height = 10
        horizontal_map = [0, width]
        vertical_map = [0, height]

        q = 50
        k = self.num_ants
        for i in range(self.num_ants):
            x = np.random.uniform(low=horizontal_map[0], high=horizontal_map[1])
            y = np.random.uniform(low=vertical_map[0], high=vertical_map[1])

            ant = Ant(np.array([x,y]))
            self.ants.append(ant)

            w_l = (1/(q * k* math.sqrt(2*math.pi))) * math.exp((-((i+1)-1)**2)/(2*q**2*k**2))
            self.W.append(w_l)

    def sample_ant(self):
        """
        Samples an ant from the table
        """
        P = []
        for i in range(len(self.ants)):
            p = self.W[i]/sum(self.W)
            P.append(p)

        index = np.random.choice(len(self.ants), p=P)
        positions = [ant.S for ant in self.ants]
        std_x = np.std(np.array(positions)[:, 0])
        std_y = np.std(np.array(positions)[:, 1])

        x_new = np.random.normal(self.ants[index].S[0], std_x)
        y_new = np.random.normal(self.ants[index].S[1], std_y)

        return np.array([x_new, y_new]), index

    def update_ants(self, P, cost_graph_sl, cost_euclidean_sl_xb, cost_graph_xb):
        """
        Updates the ants

        Parameters
        ----------
        P: tuple
            tuple containing the following:
                u_exploration: utility for exploration
                u_exploitation: utility for exploitation
                new_node: new node
                end: boolean indicating if the ant has reached the end
                alpha: weight for the utility function
                l: index of the ant
                path: path of the ant
                c_path: cost of the path
        cost_graph_sl: cost of the graph from the current node to the new node
        cost_euclidean_sl_xb: euclidean distance from the current node to the end node
        cost_graph_xb: cost of the graph from the new node to the end node

        Returns
        -------
        None
        """
        u_exploration, u_exploitation, new_node, end, alpha, l, path, c_path = P

        if (c_path is math.inf) and (path is not None):
            print("Resetting the utility of the ants")
            for ant in self.ants:
                ant.F.u_exploitation = 0
                ant.F.u_exploration = 0
                ant.U = 0
        else:   
            F = Utility()
            F.u_exploration = u_exploration
            sl = self.ants[l].S
            F.alpha = self.ants[l].F.alpha
            F.u_exploitation = self.ants[l].F.u_exploitation

            if path is not None:
                if (cost_graph_sl + cost_euclidean_sl_xb) > cost_graph_xb: 
                    F.u_exploitation = 0
                    F.u_exploration = 0
                    
            del self.ants[l]
            ant = Ant(sl, F.u_exploration, F.u_exploitation, F.alpha)
            self.ants.append(ant)
            self.sort_table()
                
            new_node = (new_node.x, new_node.y)
            ant = Ant(new_node, u_exploration, u_exploitation, alpha)
            self.ants.append(ant)
            self.sort_table()
            del self.ants[-1]

    def sort_table(self):
        """
        Sorts the table
        """
        self.ants.sort(reverse=True)