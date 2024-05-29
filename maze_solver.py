import _tkinter
import heapq
import math
import time
import turtle
import random
from PIL import Image  # install package pillow
import matplotlib.pyplot as plt

evaporation_rate = 0.5


def resize_image(original_path, resized_path, new_size):
    image = Image.open(original_path)
    resized_image = image.resize(new_size)
    resized_image.save(resized_path)
    print(f"Resized image saved to {resized_path}")


def move_picture(x, y, pic):
    pic.penup()
    pic.goto(x, y)


def ant_colony_optimization_with_params(root, goal, nodes, alpha, beta, Q, num_ants, num_iterations, pen_for_pheromone,
                                        picture, analyser):
    pheromone_map = {node: 1.0 for node in nodes}
    best_path = None
    best_path_length = float('inf')

    for iteration in range(num_iterations):
        all_paths = []
        for ant in range(num_ants):
            path, path_length = find_path(root, goal, pheromone_map, alpha, beta, picture, analyser)
            if path and path_length < best_path_length:
                best_path = path
                best_path_length = path_length
            all_paths.append((path, path_length))
        update_pheromones(pheromone_map, all_paths, Q, pen_for_pheromone, analyser)

    exec_time = time.time()
    for node in best_path:
        pen_for_pheromone.goto(node.x, node.y)
        pen_for_pheromone.color('pink')
        pen_for_pheromone.stamp()
    return best_path, best_path_length, exec_time


# Function for ants to find a path
def find_path(root, goal, pheromone_map, alpha, beta, picture, analyser):
    current_node = root
    path = [current_node]
    path_length = 0
    visited = set()
    visited.add(current_node)

    while current_node != goal:
        next_node = select_next_node(current_node, goal, pheromone_map, visited, alpha, beta)
        if not next_node:
            return None, float('inf')
        path_length += heuristic(current_node.x, current_node.y, next_node)
        path.append(next_node)
        if not analyser:
            move_picture(current_node.x, current_node.y, picture)
        visited.add(next_node)
        current_node = next_node

    return path, path_length


def select_next_node(current_node, goal, pheromone_map, visited, alpha, beta):
    neighbors = []
    probabilities = []
    for neighbor in current_node.friend:
        if neighbor not in visited and neighbor.data != 'X':
            pheromone_level = pheromone_map[neighbor] ** alpha
            heuristic_value = (1 / (heuristic(neighbor.x, neighbor.y, goal) + 1e-6)) ** beta
            probabilities.append(pheromone_level * heuristic_value)
            neighbors.append(neighbor)

    total = sum(probabilities)
    if total == 0:
        return None

    probabilities = [p / total for p in probabilities]
    chosen_neighbor = random.choices(neighbors, probabilities)[0]
    return chosen_neighbor


# Function to update pheromones
def update_pheromones(pheromone_map, all_paths, Q, pen_for_pheromone, analyser):
    for node in pheromone_map:
        pheromone_map[node] *= (1 - evaporation_rate)

    for path, path_length in all_paths:
        if path:
            pheromone_deposit = Q / path_length
            for node in path:
                pheromone_map[node] += pheromone_deposit
                if node.data != 'p' and node.data != 'G' and not analyser:
                    pen_for_pheromone.goto(node.x, node.y)
                    pen_for_pheromone.color('black')
                    pen_for_pheromone.stamp()
                    pen_for_pheromone.color('green')
                    pen_for_pheromone.goto(node.x, node.y)
                    pen_for_pheromone.write(f'{pheromone_map[node]:.2f}', align="center", font=("Arial", 12, "normal"))


def display_pheromone_levels(maze, nodes, pheromone_map):
    maze_copy = [list(row) for row in maze]
    max_pheromone = max(pheromone_map.values())
    for node in nodes:
        if maze_copy[node.row][node.col] not in ('p', 'G', 'X'):
            pheromone_level = pheromone_map[node] / max_pheromone
            if pheromone_level > 0:
                maze_copy[node.row][node.col] = f"{pheromone_level:.2f}"
            else:
                maze_copy[node.row][node.col] = ' '
    print("\nMaze with pheromone levels:")
    for row in maze_copy:
        print(' '.join(row))


# Function to display the maze with the best path
def display_maze_with_path(maze, path):
    maze_copy = [list(row) for row in maze]
    for node in path:
        if maze_copy[node.row][node.col] not in ('p', 'G'):
            maze_copy[node.row][node.col] = 'F'
    print("\nMaze with the final path:")
    for row in maze_copy:
        print(''.join(row))


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def is_empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

    def peek(self):
        if self.is_empty():
            return None
        return self.elements[0][1]


class Node:
    def __init__(self, data, x, y, row, col):
        self.x = x
        self.y = y
        self.data = data
        self.row = row
        self.col = col
        self.friend = []
        self.parent = None
        self.g_cost = float('inf')
        self.h_cost = float('inf')
        self.pheromone = 1.0

    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost() < other.f_cost()


class Draw(turtle.Turtle):
    def __init__(self, is_player):
        turtle.Turtle.__init__(self)
        self.shape("square")
        self.shapesize(2, 2)
        self.penup()
        self.speed(0)
        self.hideturtle()
        if is_player == "p":
            self.color("orange")
        elif is_player == "W":
            self.color("red")
        elif is_player == "B":
            self.color("orange")
        elif is_player == "F":
            self.shape("square")
            self.shapesize(1)
            self.color("blue")
            self.hideturtle()
        elif is_player == "pheromone":
            self.shape("square")
            self.shapesize(1)
            self.color("green")
            self.hideturtle()
        else:
            self.shape("circle")
            self.color("gold")
            self.hideturtle()
            self.gold = 100

    def change_color(self):
        self.color("green")


def setup_maze(maze, pen, player, goal):
    grid_drawer = turtle.Turtle()
    grid_drawer.color("white")
    grid_drawer.speed(0)
    grid_drawer.penup()
    num_cells_y = 15
    num_cells_x = 25
    cell_size = 50
    start_x = -625  # Starting x coordinate
    start_y = 375
    for i in range(num_cells_x + 1):
        x = start_x + i * cell_size
        grid_drawer.goto(x, start_y)
        grid_drawer.pendown()
        grid_drawer.goto(x, start_y - num_cells_y * cell_size)
        grid_drawer.penup()
    for i in range(num_cells_y + 1):
        y = start_y - i * cell_size
        grid_drawer.goto(start_x, y)
        grid_drawer.pendown()
        grid_drawer.goto(start_x + num_cells_x * cell_size, y)
        grid_drawer.penup()

    grid_drawer.hideturtle()
    try:
        cell_info = turtle.Turtle()
        cell_info.hideturtle()
        cell_info.penup()
        pen.speed(100000)
        cell_info.speed(1000000)
        for y in range(len(maze)):
            for x in range(len(maze[y])):
                character = maze[y][x]
                screen_x = -600 + (x * 50)
                screen_y = 350 - (y * 50)
                if character == "G":
                    goal.goto(screen_x, screen_y)
                    goal.stamp()
                    cell_info.goto(screen_x, screen_y)
                    cell_info.color("black")
                    cell_info.write(f"Goal", align="center", font=("Arial", 12, "normal"))
                if character == "p":
                    player.goto(screen_x, screen_y)
                    player.stamp()
                    cell_info.goto(screen_x, screen_y)
                    cell_info.color("black")
                    cell_info.write(f"Start", align="center", font=("Arial", 12, "normal"))
                if character == "X":
                    pen.goto(screen_x, screen_y)
                    pen.stamp()
                    cell_info.goto(screen_x, screen_y)
                    cell_info.color("black")
                    cell_info.write(f"Wall", align="center", font=("Arial", 12, "normal"))
    except _tkinter.TclError as e:
        pass


def createNodes(maze):
    listOfNodes = []
    for y in range(len(maze)):
        for x in range(len(maze[y])):
            character = maze[y][x]
            screen_x = -600 + (x * 50)
            screen_y = 350 - (y * 50)
            node = Node(character, screen_x, screen_y, y, x)
            listOfNodes.append(node)
    return listOfNodes


def createFriendsList(listOfNodes):
    root = None
    goal = None
    node_dict = {(node.x, node.y): node for node in listOfNodes}
    for node in listOfNodes:
        if node.data == "p":
            root = node
            node.g_cost = 0
        else:
            node.g_cost = 20
        if node.data == "G":
            goal = node
        positions = [(50, 0), (-50, 0), (0, 50), (0, -50)]
        for dx, dy in positions:
            neighbor_x = node.x + dx
            neighbor_y = node.y + dy
            if (neighbor_x, neighbor_y) in node_dict:
                neighbor = node_dict[(neighbor_x, neighbor_y)]
                node.friend.append(neighbor)
                neighbor.parent = node
    return root, goal


def findIndex(list, node):
    for ff in range(len(list)):
        if node.data == list[ff].data and node.x == list[ff].x and node.y == list[ff].y and node.row == list[
            ff].row and node.col == list[ff].col:
            return ff


def heuristic(x, y, goal):
    a = math.pow((goal.x - x), 2)
    b = math.pow((goal.y - y), 2)
    sum = a + b
    sum = math.sqrt(sum)
    return sum


def mahnattan(x,y, node2):
    return abs(x - node2.x) + abs(y - node2.y)


def read_File_Create_List(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
    data_2d = []
    for line in content:
        row = list(line.strip())
        data_2d.append(row)
    return data_2d


def A_star_Search(start, goal, mazeList, path, finalPath, goal_pen):
    queue = PriorityQueue()
    visited = []
    parent = []
    index2 = 0
    queue.put(start, 0)
    for i in range(len(mazeList)):
        visited.append(0)
    for i in range(len(mazeList)):
        parent.append(None)
    visited[0] = 1
    while queue.peek() != goal:
        vertex = queue.get()
        if vertex.data != "p":
            path.goto(vertex.x, vertex.y)
            path.speed(0)
            time.sleep(0.1)
            path.stamp()
        for neighbours in vertex.friend:
            neighbours.g_cost = vertex.g_cost + 20
            neighbours.h_cost = heuristic(neighbours.x, neighbours.y, goal)
            f_cost = neighbours.f_cost()
            index = findIndex(mazeList, neighbours)
            if visited[index] != 1 and neighbours.data != "X":
                visited[index] = 1
                queue.put(neighbours, f_cost)
                parent[index] = vertex
                if neighbours.data == "G":
                    print(True)
                    print(index)
                    index2 = index
                    goal_pen.color("red")
    while parent[index2].data != root.data:
        node = parent[index2]
        index2 = findIndex(n, node)
        finalPath.goto(node.x, node.y)
        finalPath.stamp()
        finalPath.speed(0)


def reset_maze(path, finalPath):
    path.clear()
    finalPath.clear()


def analyze_aco_parameters(root, goal, nodes, param_name, param_values, alpha, beta, Q, num_ants, num_iterations, pen,
                           ant_pic):
    path_lengths = []
    execution_times = []

    for value in param_values:
        if param_name == 'alpha':
            a, b, q, na, ni = value, beta, Q, num_ants, num_iterations
        elif param_name == 'beta':
            a, b, q, na, ni = alpha, value, Q, num_ants, num_iterations
        elif param_name == 'Q':
            a, b, q, na, ni = alpha, beta, value, num_ants, num_iterations
        elif param_name == 'num_ants':
            a, b, q, na, ni = alpha, beta, Q, value, num_iterations
        elif param_name == 'num_iterations':
            a, b, q, na, ni = alpha, beta, Q, num_ants, value

        start_time = time.time()
        best_path, best_path_length, exec_time = ant_colony_optimization_with_params(root, goal, nodes, a, b, q, na, ni,
                                                                                     pen, ant_pic, True)
        end_time = time.time()

        path_lengths.append(best_path_length)
        execution_times.append(end_time - start_time)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(param_values, path_lengths, marker='o')
    plt.title(f'Impact of {param_name} on Path Length')
    plt.xlabel(param_name)
    plt.ylabel('Path Length')
    plt.subplot(1, 2, 2)
    plt.plot(param_values, execution_times, marker='o')
    plt.title(f'Impact of {param_name} on Execution Time')
    plt.xlabel(param_name)
    plt.ylabel('Execution Time (s)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    user_input = int(input("Enter 1 to run the aco algorithm. Enter 2 for graphical analysis of aco algorithm.Enter 3 to run A* algorithm: "))
    filename = "maze.txt"
    print(filename)
    Wall = Draw("W")
    Start = Draw("p")
    goal = Draw("G")
    path = Draw("B")
    finalPath = Draw("F")
    pen_for_pheromone = Draw("pheromone")
    wn = turtle.Screen()
    wn.bgcolor("black")
    wn.title("A Maze Game")
    wn.setup(1400, 800)
    maze_list = read_File_Create_List(filename)
    n = createNodes(maze_list)
    root, goal_node = createFriendsList(n)
    wn.onkey(lambda: reset_maze(path, finalPath), 'r')
    original_image_path = "ezgif.com-animated-gif-maker.gif"
    resized_image_path = "ezgif.com-animated-gif-maker.gif"
    try:
        wn.register_shape(resized_image_path)
        print(f"Shape {resized_image_path} registered successfully.")
    except turtle.TurtleGraphicsError as e:
        print(f"Error registering shape: {e}")
        exit(1)
    resize_image(resized_image_path, original_image_path, (50, 50))
    picture = turtle.Turtle()
    picture.shape(resized_image_path)

    if user_input == 1:
        setup_maze(maze_list, Wall, Start, goal)
        best_path = ant_colony_optimization_with_params(root, goal_node, n, 1.0, 5.0, 100, 3, 5, pen_for_pheromone,
                                                        picture, False)
        print(best_path)

    elif user_input == 2:
        alpha_values = [0.5, 1.0, 1.5, 2.0]
        beta_values = [2.0, 5.0, 8.0, 10.0]
        Q_values = [50, 100, 150, 200]
        num_ants_values = [10, 30, 50, 70]
        num_iterations_values = [5, 10, 20, 30]
        print("Analyzing impact of alpha values...")
        analyze_aco_parameters(root, goal_node, n, 'alpha', alpha_values, 1.0, 5.0, 100, 50, 10, pen_for_pheromone,
                               picture)
        print("Analyzing impact of beta values...")
        analyze_aco_parameters(root, goal_node, n, 'beta', beta_values, 1.0, 5.0, 100, 50, 10, pen_for_pheromone,
                               picture)
        print("Analyzing impact of Q values...")
        analyze_aco_parameters(root, goal_node, n, 'Q', Q_values, 1.0, 5.0, 100, 50, 10, pen_for_pheromone, picture)
        print("Analyzing impact of num_ants values...")
        analyze_aco_parameters(root, goal_node, n, 'num_ants', num_ants_values, 1.0, 5.0, 100, 50, 10,
                               pen_for_pheromone, picture)
        print("Analyzing impact of num_iterations values...")
        analyze_aco_parameters(root, goal_node, n, 'num_iterations', num_iterations_values, 1.0, 5.0, 100, 50, 10,
                               pen_for_pheromone, picture)
    elif user_input == 3:
        setup_maze(maze_list, Wall, Start, goal)
        A_star_Search(root, goal_node, n, path, finalPath, goal)

    wn.mainloop()
