"""
CS311 Programming Assignment 1: Search

Full Name: Kusuma Kumar

Brief description of my heuristic:

TODO Briefly describe your heuristic and why it is more efficient

I chose to implement the linear conflict heuristic. I think this is more effiecient compared to manhattan since it accounrts
for those situations two tile are in reversed order of this desired positions and I feel that this was a pretty common scenaario. 
It also remains admissible since it calculates more accurately compared to manhattan and it never overestimates the cost of reaching the goal.
"""

import argparse, itertools, random, sys, heapq
from collections import deque
from typing import Callable, List, Optional, Sequence, Tuple


# You are welcome to add constants, but do not modify the pre-existing constants

# Problem size 
BOARD_SIZE = 3

# The goal is a "blank" (0) in bottom right corner
GOAL = tuple(range(1, BOARD_SIZE**2)) + (0,)

TOP_EDGE = list(range(0, BOARD_SIZE))
LEFT_EDGE = list(range(0, BOARD_SIZE**2, BOARD_SIZE))
RIGHT_EDGE = list(range(BOARD_SIZE - 1, BOARD_SIZE**2, BOARD_SIZE))
BOTTOM_EDGE = list(range(BOARD_SIZE**2 - BOARD_SIZE, BOARD_SIZE**2))

def inversions(board: Sequence[int]) -> int:
    """Return the number of times a larger 'piece' precedes a 'smaller' piece in board"""
    return sum(
        (a > b and a != 0 and b != 0) for (a, b) in itertools.combinations(board, 2)
    )


class Node:
    def __init__(self, state: Sequence[int], parent: "Node" = None, cost=0):
        """Create Node to track particular state and associated parent and cost

        State is tracked as a "row-wise" sequence, i.e., the board (with _ as the blank)
        1 2 3
        4 5 6
        7 8 _
        is represented as (1, 2, 3, 4, 5, 6, 7, 8, 0) with the blank represented with a 0

        Args:
            state (Sequence[int]): State for this node, typically a list, e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8]
            parent (Node, optional): Parent node, None indicates the root node. Defaults to None.
            cost (int, optional): Cost in moves to reach this node. Defaults to 0.
        """
        self.state = tuple(state)  # To facilitate "hashable" make state immutable
        self.parent = parent
        self.cost = cost

    def is_goal(self) -> bool:
        """Return True if Node has goal state"""
        return self.state == GOAL

    def expand(self) -> List["Node"]:
        """Expand current node into possible child nodes with corresponding parent and cost"""
        # TODO: Implement this function to generate child nodes based on the current state
        # move 0 left, right , top , bottom
        #cost: cost for parent + 1(for the move we just made)
        children = []
        pos0 = self.state.index(0)
        
        if pos0 not in LEFT_EDGE:
            positions = self.calc_pos(pos0, "left")
            child_left = Node(self._swap(positions[0],positions[1], positions[2], positions[3]), self , self.cost + 1)
            children.append(child_left)
        
        if pos0 not in TOP_EDGE:
            positions = self.calc_pos(pos0, "top")
            child_top = Node(self._swap(positions[0],positions[1], positions[2], positions[3]), self , self.cost + 1)
            children.append(child_top)
        
        if pos0 not in RIGHT_EDGE:
            positions = self.calc_pos(pos0, "right")
            child_right = Node(self._swap(positions[0],positions[1], positions[2], positions[3]), self , self.cost + 1)
            children.append(child_right)
        
        if pos0 not in BOTTOM_EDGE:
            positions = self.calc_pos(pos0, "bottom")
            child_bottom = Node(self._swap(positions[0],positions[1], positions[2], positions[3]), self , self.cost + 1)
            children.append(child_bottom)
        
        return children
    
    def calc_pos(self, position, move) -> List[int]:
        positions = [0, 0, 0, 0]
        positions[0] = position // BOARD_SIZE
        positions[1] = position % BOARD_SIZE
        
        if move == "left":
            positions[2] = positions[0]
            positions[3] = positions[1] - 1
        elif move == "right":
            positions[2] = positions[0]
            positions[3] = positions[1] + 1
        elif move == "top":
            positions[2] = positions[0] - 1
            positions[3] = positions[1]
        elif move == "bottom":
            positions[2] = positions[0] + 1
            positions[3] = positions[1]
        
        return positions
    
    def _swap(self, row1: int, col1: int, row2: int, col2: int) -> Sequence[int]:
        """Swap values in current state between row1,col1 and row2,col2, returning new "state" to construct a Node"""
        state = list(self.state)
        state[row1 * BOARD_SIZE + col1], state[row2 * BOARD_SIZE + col2] = (
            state[row2 * BOARD_SIZE + col2],
            state[row1 * BOARD_SIZE + col1],
        )
        return state

    def __str__(self):
        return str(self.state)

    # The following methods enable Node to be used in types that use hashing (sets, dictionaries) or perform comparisons. Note
    # that the comparisons are performed exclusively on the state and ignore parent and cost values.

    def __hash__(self):
        return self.state.__hash__()

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


def bfs(initial_board: Sequence[int], max_depth=12) -> Tuple[Optional[Node], int]:
    """Perform breadth-first search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 12.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """
    # TODO: Implement BFS. Your function should return a tuple containing the solution node and number of unique node explored
    # create the priority queue. Add self. the get its children. keep addind and popping until you get goal state
    # if you reach max_depth return None, 0
    # if you reach goal state return the goal state and number of nodes explored
    # if you reach the end of the queue and no goal state is found return None, 0
    root = Node(initial_board)
    if root.is_goal():
        return root, 1

    queue = deque([root])
    reached = set()
    reached.add(root)

    while queue:
        current = queue.popleft()
        
        if current.cost >= max_depth:
            return None, 0
        
        for child in current.expand():
            if child.is_goal():
                return child, len(reached) + 1
            
            if child not in reached:
                reached.add(child)
                queue.append(child)

    return None, len(reached)


class PQueue(object):
    def __init__(self):
        self.heap = []
    
    def push(self, heuristic: int, node: Node):
        heapq.heappush(self.heap, (heuristic, node))

    def pop(self) -> Node:
        return heapq.heappop(self.heap)[1]
    
    def isEmpty(self) -> bool:
        return len(self.heap) == 0


def manhattan_distance(node: Node) -> int:
    """Compute manhattan distance f(node), i.e., g(node) + h(node)"""
    # TODO: Implement the Manhattan distance heuristic (sum of Manhattan distances to goal location)
    man_dist = 0
    for i in range(1,BOARD_SIZE**2):
        if node.state.index(i) + 1 == i:
            continue
        curr_row = node.state.index(i) // BOARD_SIZE
        curr_col = node.state.index(i) % BOARD_SIZE
        goal_row = (i - 1 ) // BOARD_SIZE
        goal_col = (i - 1) % BOARD_SIZE
        man_dist += abs(curr_row - goal_row) + abs(curr_col - goal_col)
    return man_dist + node.cost


def custom_heuristic(node: Node) -> int:
    # TODO: Implement and document your _admissable_ heuristic function
    # implementing the linear conflict heuristic

    conflict = 0
    #linear conflicts in rows
    for row in range(BOARD_SIZE):
        row_tiles = [node.state[row * BOARD_SIZE + col] for col in range(BOARD_SIZE) if node.state[row * BOARD_SIZE + col] != 0]
        for i in range(len(row_tiles)):
            for j in range(i + 1, len(row_tiles)):
                # condition to check if an greater number tile comes before smaller number tile in a row and both their goal rows are the current row
                if row_tiles[i] > row_tiles[j]:  # Only check for greater tile
                    # Determine the goal positions of the tiles
                    goal_i = (row_tiles[i] - 1) // BOARD_SIZE
                    goal_j = (row_tiles[j] - 1) // BOARD_SIZE
                    if goal_i == goal_j == row:  # Both tiles should be in the current row
                        conflict += 2

    # linear conflicts in columns 
    for col in range(BOARD_SIZE):
        col_tiles = [node.state[row * BOARD_SIZE + col] for row in range(BOARD_SIZE) if node.state[row * BOARD_SIZE + col] != 0]
        for i in range(len(col_tiles)):
            for j in range(i + 1, len(col_tiles)):
                # condition to check if an greater number tile comes before smaller number tile in a column and both their goal columns are the current col
                if col_tiles[i] > col_tiles[j]:  # Only check for greater tile
                    # Determine the goal positions of the tiles
                    goal_i = (col_tiles[i] - 1) % BOARD_SIZE
                    goal_j = (col_tiles[j] - 1) % BOARD_SIZE
                    if goal_i == goal_j == col:  # Both tiles should be in the current column
                        conflict += 2
    return manhattan_distance(node) + conflict 



def astar(
    initial_board: Sequence[int],
    max_depth=12,
    heuristic: Callable[[Node], int] = custom_heuristic,
) -> Tuple[Optional[Node], int]:
    """Perform astar search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 12.
        heuristic (_Callable[[Node], int], optional): Heuristic function. Defaults to manhattan_distance.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """
    # TODO: Implement A* search. Make sure that your code uses the heuristic function provided as
    # an argument so that the test code can switch in your custom heuristic (i.e., do not "hard code"
    # manhattan distance as the heuristic)
    heap = PQueue()
    root = Node(initial_board)
    if root.is_goal():
        return root, 1
    
    heap.push(heuristic(root), root)
    reached = {}
    reached[root] = root.cost
    
    while not heap.isEmpty():
        current_node = heap.pop()
        
        if current_node.cost > max_depth:
            return None, len(reached)
        
        if current_node.is_goal():
            return current_node, len(reached)
        
        for child in current_node.expand():
            # maintain the cost as heuristic value so you can pop according to the lowest cost
            if child in reached:
                if reached[child] > child.cost:
                    reached[child] = child.cost
                    heap.push(heuristic(child), child)
            else:
                reached[child] = child.cost
                heap.push(heuristic(child), child)
    
    return None, len(reached)

if __name__ == "__main__":

    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(
        description="Run search algorithms in random inputs"
    )
    parser.add_argument(
        "-a",
        "--algo",
        default="bfs",
        help="Algorithm (one of bfs, astar, astar_custom)",
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=1000,
        help="Number of iterations",
    )
    parser.add_argument(
        "-s",
        "--state",
        type=str,
        default=None,
        help="Execute a single iteration using this board configuration specified as a string, e.g., 123456780",
    )

    args = parser.parse_args()

    num_solutions = 0
    num_cost = 0
    num_nodes = 0

    if args.algo == "bfs":
        algo = bfs
    elif args.algo == "astar":
        algo = astar
    elif args.algo == "astar_custom":
        algo = lambda board: astar(board, heuristic=custom_heuristic)
    else:
        raise ValueError("Unknown algorithm type")

    if args.state is None:
        iterations = args.iter
        while iterations > 0:
            init_state = list(range(BOARD_SIZE**2))
            random.shuffle(init_state)

            # A problem is only solvable if the parity of the initial state matches that
            # of the goal.
            if inversions(init_state) % 2 != inversions(GOAL) % 2:
                continue

            solution, nodes = algo(init_state)
            if solution:
                num_solutions += 1
                num_cost += solution.cost
                num_nodes += nodes

            iterations -= 1
    else:
        # Attempt single input state
        solution, nodes = algo([int(s) for s in args.state])
        if solution:
            num_solutions = 1
            num_cost = solution.cost
            num_nodes = nodes

    if num_solutions:
        print(
            "Iterations:",
            args.iter,
            "Solutions:",
            num_solutions,
            "Average moves:",
            num_cost / num_solutions,
            "Average nodes:",
            num_nodes / num_solutions,
        )
    else:
        print("Iterations:", args.iter, "Solutions: 0")
