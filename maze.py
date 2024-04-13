import pygame
import sys
import random
import time
import matplotlib.pyplot as plt
from queue import PriorityQueue
from collections import defaultdict

class BaseMaze:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        self.cells = {(x, y): (x, y) for y in range(height) for x in range(width)}

        self.edges = {}
        self.start_node = (0, 0)  
        self.end_node = (width - 1, height - 1) 
        self.visited = [[False for _ in range(width)] for _ in range(height)]
        self.solution = []

    def add_edge(self, node1, node2):
        self.edges[(node1, node2)] = 1
        self.edges[(node2, node1)] = 1

    def remove_edge(self, node1, node2):
        self.edges[(node1, node2)] = 0
        self.edges[(node2, node1)] = 0    

    def is_edge(self, node1, node2):
        return self.edges.get((node1, node2), 0)

    def solve_maze(self, start=None, end=None):
        if start is None:
            start = self.start_node
        if end is None:
            end = self.end_node

        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {cell: float('inf') for cell in self.cells}
        g_score[start] = 0
        f_score = {cell: float('inf') for cell in self.cells}
        f_score[start] = self.heuristic(start, end)

        while not open_set.empty():
            current = open_set.get()[1]

            if current == end:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1  

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, end)
                    if not any(neighbor == item[1] for item in open_set.queue):
                        open_set.put((f_score[neighbor], neighbor))

        return None  

    def heuristic(self, a, b):
        
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        valid_neighbors = []
        for nx, ny in neighbors:
            if self.is_edge((x, y), (nx, ny)):
                valid_neighbors.append((nx, ny))
        return valid_neighbors

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        self.solution = total_path
        return total_path

    def draw_maze(self, screen):
        background_color = (255, 255, 255)  # White
        wall_color = (0, 0, 0)  # Black
        path_color = (0, 0, 255) # Blue
        screen.fill(background_color)

        start_rect = pygame.Rect(self.start_node[0]*self.cell_size, self.start_node[1]*self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, (0, 255, 0), start_rect)  # Green 
        
        end_rect = pygame.Rect(self.end_node[0]*self.cell_size, self.end_node[1]*self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, (255, 0, 0), end_rect) #Red


        for y in range(self.height):
            for x in range(self.width):
                for dy, dx in [(-1, 0), (0, -1)]:  
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if not self.is_edge((x, y), (nx, ny)):
                            if dx == -1:  
                                pygame.draw.line(screen, wall_color, (x*self.cell_size, y*self.cell_size), (x*self.cell_size, (y+1)*self.cell_size), 2)
                            if dy == -1:  
                                pygame.draw.line(screen, wall_color, (x*self.cell_size, y*self.cell_size), ((x+1)*self.cell_size, y*self.cell_size), 2)

        for i in range(len(self.solution) - 1):
                start = self.solution[i]
                end = self.solution[i + 1]
                start_pos = (start[0] * self.cell_size + self.cell_size // 2, start[1] * self.cell_size + self.cell_size // 2)
                end_pos = (end[0] * self.cell_size + self.cell_size // 2, end[1] * self.cell_size + self.cell_size // 2)
                pygame.draw.line(screen, path_color, start_pos, end_pos, 5)
        
        pygame.draw.rect(screen, wall_color, pygame.Rect(0, 0, self.width*self.cell_size, self.height*self.cell_size), 2)

class DFSMaze(BaseMaze):

    def get_unvisited_neighbors(self, x, y):
        neighbors = []
        for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if 0 <= nx < self.width and 0 <= ny < self.height and not self.visited[nx][ny]:
                neighbors.append((nx, ny))
        return neighbors

    def dfs(self, x, y):
        self.visited[x][y] = True
        neighbors = self.get_unvisited_neighbors(x, y)
        random.shuffle(neighbors)
        for nx, ny in neighbors:
            if not self.visited[nx][ny]:
                self.add_edge((x, y), (nx, ny))
                self.dfs(nx, ny)

    def generate_maze(self):
        self.dfs(self.start_node[0], self.start_node[1])

class PrimsMaze(BaseMaze):
    def prims(self):
    
        current_cell = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        self.visited[current_cell[0]][current_cell[1]] = True
        walls = self.get_cell_walls(current_cell)

        while walls:
            wall = random.choice(walls)
            cell1, cell2 = self.get_cells_adjacent_to_wall(wall)

            if cell1 and cell2:
                if self.visited[cell1[0]][cell1[1]] != self.visited[cell2[0]][cell2[1]]:
                    self.add_edge(cell1, cell2) 
                    for cell in (cell1, cell2):
                        if not self.visited[cell[0]][cell[1]]:
                            self.visited[cell[0]][cell[1]] = True
                            walls.extend(self.get_cell_walls(cell))

            walls.remove(wall)

    def get_cell_walls(self, cell):
        x, y = cell
        walls = []
        if x > 0: walls.append(((x, y), (x-1, y)))
        if x < self.width - 1: walls.append(((x, y), (x+1, y)))
        if y > 0: walls.append(((x, y), (x, y-1)))
        if y < self.height - 1: walls.append(((x, y), (x, y+1)))
        return walls

    def is_valid_cell(self, cell):
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cells_adjacent_to_wall(self, wall):
        cell1, cell2 = wall
        if self.is_valid_cell(cell1) and self.is_valid_cell(cell2):
            return cell1, cell2
        return None, None

    def generate_maze(self):
        self.prims()

class KruskalsMaze(BaseMaze):

    def __init__(self, width, height, cell_size):
        super().__init__(width, height, cell_size)
        self.parent = {(x, y): (x, y) for y in range(self.height) for x in range(self.width)}
        self.rank = {(x, y): 0 for y in range(height) for x in range(width)} 
        self.walls = []

    def find_set(self, cell):
        if self.parent[cell] != cell:
            self.parent[cell] = self.find_set(self.parent[cell])
        return self.parent[cell]

    def union_sets(self, cell1, cell2):
        root1 = self.find_set(cell1)
        root2 = self.find_set(cell2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def kruskals(self):
        for y in range(self.height):
            for x in range(self.width):
                if x < self.width - 1:
                    self.walls.append(((x, y), (x+1, y)))
                if y < self.height - 1:
                    self.walls.append(((x, y), (x, y+1)))

        random.shuffle(self.walls)

        edge_count = 0

        for wall in self.walls:
            if edge_count == self.width * self.height - 1:
                break
            
            cell1, cell2 = wall
            set1 = self.find_set(cell1)
            set2 = self.find_set(cell2)

            if set1 != set2:
                self.union_sets(set1, set2)
                self.add_edge(cell1, cell2)
                edge_count += 1

    def generate_maze(self):
        self.kruskals()

             

class RecursiveDivisionMaze(BaseMaze):
    def divide(self, x, y, width, height, orientation):
        if width < 2 or height < 2 :
            return

        if orientation == 'horizontal':
            divide_south_of = random.randint(y, y + height - 2)
            passage_at = random.randint(x, x + width - 1)
            for i in range(x, x + width):
                if i == passage_at:
                    self.add_edge((divide_south_of, i), (divide_south_of + 1, i))
                else:
                    self.remove_edge((divide_south_of, i), (divide_south_of + 1, i))

            self.divide(x, y, width, divide_south_of - y + 1, self.choose_orientation(width, divide_south_of - y + 1))
            self.divide(x, divide_south_of + 1, width, y + height - divide_south_of - 1, self.choose_orientation(width, y + height - divide_south_of - 1))

        elif orientation == 'vertical':
                
            divide_east_of = random.randint(x, x + width - 2)
            passage_at = random.randint(y, y + height - 1)
            for i in range(y, y + height):
                if i == passage_at:
                    self.add_edge((i, divide_east_of), (i, divide_east_of + 1))
                else:
                    self.remove_edge((i, divide_east_of), (i, divide_east_of + 1))

            self.divide(x, y, divide_east_of - x + 1, height, self.choose_orientation(divide_east_of - x + 1, height))
            self.divide(divide_east_of + 1, y, x + width - divide_east_of - 1, height, self.choose_orientation(x + width - divide_east_of - 1, height))

    def initialize_maze(self):
        for x in range(self.width):
            for y in range(self.height): 
                if x < self.width - 1:
                    self.add_edge((x, y), (x+1, y)) 
                if y < self.height - 1:
                    self.add_edge((x, y), (x, y+1))   

    def choose_orientation(self, width, height):
        if width < height:
            return 'horizontal'
        elif height < width:
            return 'vertical'
        else:
            return 'horizontal' if random.randint(0, 1) == 0 else 'vertical'

    def recursive_division(self):
        self.initialize_maze()  
        self.divide(0, 0, self.width, self.height, self.choose_orientation(self.width, self.height))

    def generate_maze(self):
        return self.recursive_division()

class BinaryTreeMaze(BaseMaze):
    def binary_tree(self):
        for y in range(self.height):
            for x in range(self.width):
                neighbors = []

                if y < 0:
                    neighbors.append((x, y - 1))
                if x > 0:
                    neighbors.append((x - 1, y))

                if neighbors:
                    nx, ny = random.choice(neighbors)
                    self.add_edge((x, y), (nx, ny))

    def generate_maze(self):
        return self.binary_tree()

class SidewinderMaze(BaseMaze):
    def sidewinder(self):
        for y in range(self.height):
            run = []
            for x in range(self.width):
                run.append((x, y))
                at_eastern_boundary = (x == self.width - 1)
                at_northern_boundary = (y == 0)
                
                should_close_run = at_eastern_boundary or (not at_northern_boundary and random.choice([True, False]))

                if should_close_run:
                    nx, ny = random.choice(run)
                    if ny > 0:  
                        self.add_edge((nx, ny), (nx, ny - 1)) 
                        self.add_edge((nx, ny - 1), (nx, ny))  

                    run = []  
                else:
                    self.add_edge((x, y), (x + 1, y))
                    self.add_edge((x + 1, y), (x, y))
    
    def generate_maze(self):
        return self.sidewinder()   

class AldousBroderMaze(BaseMaze):
    def aldous_broder(self):
        visited_cells = 1
        total_cells = self.width * self.height

        current_x, current_y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
        self.visited[current_y][current_x] = True

        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        while visited_cells < total_cells:
            
            dx, dy = random.choice(directions)
            next_x, next_y = current_x + dx, current_y + dy

            
            if 0 <= next_x < self.width and 0 <= next_y < self.height:
                if not self.visited[next_y][next_x]:  
                    
                    self.add_edge((current_x, current_y), (next_x, next_y))
                    self.add_edge((next_x, next_y), (current_x, current_y))
                    visited_cells += 1
             
                current_x, current_y = next_x, next_y
                self.visited[next_y][next_x] = True

    def generate_maze(self):
        return self.aldous_broder()

class WilsonsMaze(BaseMaze):
    def wilsons(self):
       
        unvisited = {(x, y) for x in range(self.width) for y in range(self.height)}
        
        initial = random.choice(list(unvisited))
        unvisited.remove(initial)

        while unvisited:
           
            cell = start = random.choice(list(unvisited))
            path = [start]
            while cell in unvisited:
                x, y = cell
                neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
              
                neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.width and 0 <= ny < self.height]
                next_cell = random.choice(neighbors)
                if next_cell in path:
                    path = path[:path.index(next_cell) + 1]  # Erase loop
                else:
                    path.append(next_cell)
                cell = next_cell

            for first, second in zip(path, path[1:]):
                self.add_edge((first[0], first[1]), (second[0], second[1]))
                self.add_edge((second[0], second[1]), (first[0], first[1]))
                unvisited.discard(first)
                unvisited.discard(second) 

    def generate_maze(self):
        return self.wilsons()

def measure_execution_times():
    
    cell_size = 20
    execution_times = defaultdict(list)
    dims = []

    for dim in range(20, 40):

        width, height = dim, dim
        dims.append(dim)

        start_time = time.time()
        dfs_maze = DFSMaze(dim, dim, cell_size)
        dfs_maze.generate_maze()
        execution_times['dfs'].append(time.time() - start_time);

        start_time = time.time()
        prims_maze = PrimsMaze(dim, dim, cell_size)
        prims_maze.generate_maze()
        execution_times['prim\'s'].append(time.time() - start_time);
        
        start_time = time.time()
        kruskals_maze = KruskalsMaze(dim, dim, cell_size)
        kruskals_maze.generate_maze()
        execution_times['kruskal\'s'].append(time.time() - start_time);
        

        start_time = time.time()
        recursive_division_maze = RecursiveDivisionMaze(dim, dim, cell_size)
        recursive_division_maze.generate_maze()
        execution_times['recursive_division'].append(time.time() - start_time);

        start_time = time.time()
        binary_tree_maze = BinaryTreeMaze(dim, dim, cell_size)
        binary_tree_maze.generate_maze()
        execution_times['binary_tree'].append(time.time() - start_time);

        start_time = time.time()
        sidewinder_maze = SidewinderMaze(dim, dim, cell_size)
        sidewinder_maze.generate_maze()
        execution_times['sidewinder'].append(time.time() - start_time);

        start_time = time.time()
        aldous_broder_maze = AldousBroderMaze(dim, dim, cell_size)
        aldous_broder_maze.generate_maze()
        execution_times['aldous-broder'].append(time.time() - start_time);

        start_time = time.time()
        wilsons_maze = WilsonsMaze(dim, dim, cell_size)
        wilsons_maze.generate_maze()
        execution_times['wilson\'s'].append(time.time() - start_time);

    for key in execution_times.keys():
        x_labels = list(dims)
        y_values = list(execution_times[key])

        plt.plot(x_labels, y_values, label = key)
        plt.xlabel('Maze Size')
        plt.ylabel('Execution Time')
        plt.title('Maze Generation Algorithm Execution Times')
        plt.xticks(rotation=45)
        
    plt.legend(title='Algorithm Names', loc='best', shadow=True, fancybox=True)    
    plt.show()

def run_game():
    pygame.init()
    cell_size = 20
    width, height = 30, 30
    screen_size = width * cell_size, height * cell_size
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Maze Generator")

    dfs_maze = DFSMaze(width, height, cell_size)
    dfs_maze.generate_maze()
    dfs_maze.solve_maze()

    prims_maze = PrimsMaze(width, height, cell_size)
    prims_maze.generate_maze()
    prims_maze.solve_maze()
    
    kruskals_maze = KruskalsMaze(width, height, cell_size)
    kruskals_maze.generate_maze()
    kruskals_maze.solve_maze()

    recursive_division_maze = RecursiveDivisionMaze(width, height, cell_size)
    recursive_division_maze.generate_maze()
    recursive_division_maze.solve_maze()

    binary_tree_maze = BinaryTreeMaze(width, height, cell_size)
    binary_tree_maze.generate_maze()
    binary_tree_maze.solve_maze()

    sidewinder_maze = SidewinderMaze(width, height, cell_size)
    sidewinder_maze.generate_maze()
    sidewinder_maze.solve_maze()

    aldous_broder_maze = AldousBroderMaze(width, height, cell_size)
    aldous_broder_maze.generate_maze()
    aldous_broder_maze.solve_maze()

    wilsons_maze = WilsonsMaze(width, height, cell_size)
    wilsons_maze.generate_maze()
    wilsons_maze.solve_maze()

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        #dfs_maze.draw_maze(screen)
        #prims_maze.draw_maze(screen)
        #kruskals_maze.draw_maze(screen)
        #recursive_division_maze.draw_maze(screen)
        #binary_tree_maze.draw_maze(screen)
        #sidewinder_maze.draw_maze(screen)
        #aldous_broder_maze.draw_maze(screen)
        wilsons_maze.draw_maze(screen)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
#   measure_execution_times()
    run_game()

