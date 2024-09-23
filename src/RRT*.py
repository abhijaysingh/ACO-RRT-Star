import math
import sys
import matplotlib.pyplot as plt
import pathlib
import random
import time
import numpy as np
from ACO import Table

sys.path.append(str(pathlib.Path(__file__).parent.parent))

show_animation = True

class RRTStar():
    """
    Class for RRT Star planning
    """

    class Node():
        """
        RRT Node
        """
        def __init__(self, x, y):
            """
            Node class constructor
            x: x position
            y: y position
            """
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            self.cost = 0.0

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=30.0, path_resolution=1.0, 
                 goal_sample_rate=20, max_iter=600, connect_circle_dist=50.0, search_until_max_iter=False, robot_radius=0.0):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius

        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter
        self.node_list = []

        self.Table = Table(25)
        self.R = 0.1
        self.eta = 5
        self.d = 2

    def solver(self, animation=True):
        """
        Solves the planning problem

        Parameters:
        ----------
        animation: boolean
            boolean indicating if the animation should be shown

        Returns:
        -------
        path: list
            list of the path
        """

        self.node_list = [self.start]
        self.Table.initialize_ants()
        c_path = math.inf
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            
            new_node = self.update_tree(rnd)
            
            path = None
            if path is not None:
                self.draw_graph(rnd, path)
                c_path = self.node_list[last_index].cost
            elif animation:
                self.draw_graph(rnd)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        print("Cost to goal: ", self.node_list[last_index].cost)
        if last_index is not None:
            return self.generate_final_course(last_index)


    def update_tree(self, rnd, animation=True):
        """
        Updates the tree

        Parameters
        ----------
        rnd: Node
            randomly generated node
        
        Returns
        -------
        new_node: Node
            new node
        """
        nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
        new_node = self.steer(self.node_list[nearest_ind], rnd,
                              self.expand_dis)
        
        near_node = self.node_list[nearest_ind]
        new_node.cost = near_node.cost + \
            math.hypot(new_node.x-near_node.x,
                       new_node.y-near_node.y)
        
        if self.check_collision(
                new_node, self.obstacle_list, self.robot_radius):
            near_inds = self.find_near_nodes(new_node)
            node_with_updated_parent = self.choose_parent(
                new_node, near_inds)
            if node_with_updated_parent:
                self.rewire(node_with_updated_parent, near_inds)
                self.node_list.append(node_with_updated_parent)
            else:
                self.node_list.append(new_node)

        return new_node

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Steer method

        Parameters
        ----------
        from_node: Node
            node from which to steer
        to_node: Node
            node to which to steer
        extend_length: float
            length to extend

        Returns
        -------
        new_node: Node
            new node
        """
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def draw_graph(self, rnd=None, path=None):
        """
        Draws the graph

        Parameters
        ----------
        rnd: Node
            randomly generated node
        path: list
            list of the path

        """
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")
              
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')

        plt.grid(True)
        plt.pause(0.01)

    def choose_parent(self, new_node, near_inds):
        """
        Finds the cheapest point to new_node among the nodes in near_inds

        Parameters
        ----------
        new_node: Node
            new node
        near_inds: list
            list of the indices of the nodes near the new node

        Returns
        -------
        new_node: Node
            new node
        """
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        """
        Returns the index of the node with the lowest cost towards the goal

        Returns
        -------
        i: int
            index of the node with the lowest cost towards the goal
        """
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        Finds the nodes near the new node

        Parameters
        ----------
        new_node: Node
            new node
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
        Rewires the tree

        Parameters
        ----------
        new_node: Node
            new node
        near_inds: list
            list of the indices of the nodes near the new node
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(
                edge_node, self.obstacle_list, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                for node in self.node_list:
                    if node.parent == self.node_list[i]:
                        node.parent = edge_node
                self.node_list[i] = edge_node
                self.propagate_cost_to_leaves(self.node_list[i])

    def calc_new_cost(self, from_node, to_node):
        """
        Calculates the new cost

        Parameters
        ----------
        from_node: Node
            node from which to calculate the new cost
        to_node: Node
            node to which to calculate the new cost

        Returns
        -------
        new_cost: float
            new cost
        """
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        """
        Propagates the cost to the leaves

        Parameters
        ----------
        parent_node: Node
            parent node
        """
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def get_random_node(self):
        """
        Returns a random node

        Returns
        -------
        rnd: Node
            random node
        """
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd
    
    def generate_final_course(self, goal_ind):
        """
        Generates the final course

        Parameters
        ----------
        goal_ind: int   
            index of the goal node

        Returns
        -------
        path: list
            list of the path
        """
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        """
        Calculates the distance to the goal

        Parameters
        ----------
        x: float
            x position
        y: float
            y position

        Returns
        -------
        float
            distance to the goal
        """
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)
    
    @staticmethod
    def plot_circle(x, y, size, color="-b"):  
        """
        Plots a circle

        Parameters
        ----------
        x: float
            x position
        y: float
            y position
        size: float
            size of the circle
        color: string
            color of the circle
        """
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        """
        Returns the index of the nearest node

        Parameters
        ----------
        node_list: list
            list of the nodes
        rnd_node: Node
            randomly generated node

        Returns
        -------
        minind: int
            index of the nearest node
        """
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):
        """
        Checks if the node is outside the play area

        Parameters
        ----------
        node: Node
            node to check
        play_area: PlayArea
            play area

        Returns
        -------
        bool
            True if the node is outside the play area, False otherwise
        """
        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList, robot_radius):
        """
        Checks if the node collides with an obstacle

        Parameters
        ----------
        node: Node
            node to check
        obstacleList: list
            list of the obstacles
        robot_radius: float
            robot radius

        Returns
        -------
        bool
            True if the node collides with an obstacle, False otherwise
        """
        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size+robot_radius)**2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """
        Calculates the distance and angle between two nodes

        Parameters
        ----------
        from_node: Node
            node from which to calculate the distance and angle
        to_node: Node
            node to which to calculate the distance and angle

        Returns
        -------
        d: float
            distance between the two nodes
        theta: float
            angle between the two nodes
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

def main():
    print("Start " + __file__)

    obstacle_list = [
        (3, 6, 2),
        (3, 10, 2),
        (8, 10, 1),
        (6, 12, 1),
    ]  
    rrt_star = RRTStar(
        start=[0, 0],
        goal=[5, 4],
        rand_area=[-2, 15],
        obstacle_list=obstacle_list,
        expand_dis=1,
        robot_radius=0.1)
    
    start_time = time.perf_counter()
    path = rrt_star.solver(animation=show_animation)
    end_time = time.perf_counter()
    print("Time taken: ", end_time - start_time)

    print("Path: ", path)
    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        if show_animation:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
            
            plt.grid(True)
            plt.show()

    path.reverse()

if __name__ == '__main__':
    main()
