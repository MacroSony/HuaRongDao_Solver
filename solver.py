from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
import sys

# ====================================================================================

char_goal = '1'
char_single = '2'


class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_goal, is_single, coord_x, coord_y, orientation):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v')
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single, \
                                       self.coord_x, self.coord_y, self.orientation)


class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.empty_pos is a list of tuple that noted the (x, y) index of empty
        # location of the current board
        self.empty_poses = []

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()

    def __hash__(self):
        """
        Overwrite hash by turning self.grid to tuple and hashing it

        """
        return hash(tuple(map(tuple, self.grid)))

    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.

        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

        for y in range(self.height):
            for x in range(self.width):
                if (self.grid[y][x] == '.'):
                    self.empty_poses.append((x, y))

    def display(self):
        """
        Print out the current board.

        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()
        


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces.
    State has a Board and some extra information that is relevant to the search:
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.

    def __lt__(self, other):
        """
        Overwrite < operator to compare f value of the states
        """
        return self.f < other.f


def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^':  # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<':  # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if g_found == False:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)

    return board


def generate_board(board_string):
    """
    Load initial board from a given file.

    :param board_string: String of the board as input, 4x5 board with \n to separate each row.
    :type board_string: str
    :return: A loaded board
    :rtype: Board
    """

    line_index = 0
    pieces = []
    g_found = False

    lines = board_string.split('\n')
    for l in lines:
        for x, char in enumerate(l):
            if char == '^':  # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif char == '<':  # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif char == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif char == char_goal:
                if g_found == False:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1
    board = Board(pieces)
    return board


def test_goal(state):
    """
    :param state: State to be tested
    :type state: State
    """
    goal_piece = None
    for piece in state.board.pieces:
        if piece.is_goal:
            goal_piece = piece
    if goal_piece.coord_x == 1 and goal_piece.coord_y == 3:
        return True
    return False


def find_piece(state, pieces, pos):
    """
    :param state: The state that included pieces
    :type state: State
    :param pieces: The list of Pieces
    :type pieces: List[Piece]
    :param pos: Position of target piece (x, y)
    :type pos: Tuple[int, int]
    """
    if (state.board.grid[pos[1]][pos[0]] == '.'):
        return None
    for i in range(0, len(pieces)):
        if (pieces[i].coord_x == pos[0] and pieces[i].coord_y == pos[1]):
            return i
    return None


def create_state(parent_state, pos, new_pos):
    """
    :param parent_state: The parent state of the new state to be created
    :type parent_state: State
    :param pos: Position of target piece (x, y)
    :type pos: Tuple[int, int]
    :param new_pos: New position of target piece (x, y)
    :type new_pos: Tuple[int, int]
    """
    new_pieces = deepcopy(parent_state.board.pieces)
    piece_idx = find_piece(parent_state, new_pieces, pos)
    new_pieces[piece_idx].coord_x = new_pos[0]
    new_pieces[piece_idx].coord_y = new_pos[1]
    new_board = Board(new_pieces)
    new_f = parent_state.f - heuristic_manhattan(parent_state.board) + heuristic_manhattan(new_board) + 1
    new_state = State(new_board, new_f, parent_state.depth + 1, parent_state)
    return new_state


def get_single(state, piece):
    """
    :param state: The state to be checked
    :type state: State
    :param piece: Piece to check
    :type piece: Piece
    """
    new_states = []
    pos = (piece.coord_x, piece.coord_y)
    if (pos[0] - 1 >= 0):
        if (state.board.grid[pos[1]][pos[0] - 1] == '.'):
            new_states.append(create_state(state, pos, (pos[0] - 1, pos[1])))
    if (pos[0] + 1 < state.board.width):
        if (state.board.grid[pos[1]][pos[0] + 1] == '.'):
            new_states.append(create_state(state, pos, (pos[0] + 1, pos[1])))
    if (pos[1] - 1 >= 0):
        if (state.board.grid[pos[1] - 1][pos[0]] == '.'):
            new_states.append(create_state(state, pos, (pos[0], pos[1] - 1)))
    if (pos[1] + 1 < state.board.height):
        if (state.board.grid[pos[1] + 1][pos[0]] == '.'):
            new_states.append(create_state(state, pos, (pos[0], pos[1] + 1)))
    return new_states


def get_goal(state, piece):
    """
    :param state: The state to be checked
    :type state: State
    :param piece: Piece to check
    :type piece: Piece
    """
    new_states = []
    pos = (piece.coord_x, piece.coord_y)
    if (pos[0] - 1 >= 0):
        if (state.board.grid[pos[1]][pos[0] - 1] == '.' and state.board.grid[pos[1] + 1][pos[0] - 1] == '.'):
            new_states.append(create_state(state, pos, (pos[0] - 1, pos[1])))
    if (pos[0] + 2 < state.board.width):
        if (state.board.grid[pos[1]][pos[0] + 2] == '.' and state.board.grid[pos[1] + 1][pos[0] + 2] == '.'):
            new_states.append(create_state(state, pos, (pos[0] + 1, pos[1])))
    if (pos[1] - 1 >= 0):
        if (state.board.grid[pos[1] - 1][pos[0]] == '.' and state.board.grid[pos[1] - 1][pos[0] + 1] == '.'):
            new_states.append(create_state(state, pos, (pos[0], pos[1] - 1)))
    if (pos[1] + 2 < state.board.height):
        if (state.board.grid[pos[1] + 2][pos[0]] == '.' and state.board.grid[pos[1] + 2][pos[0] + 1] == '.'):
            new_states.append(create_state(state, pos, (pos[0], pos[1] + 1)))
    return new_states


def get_h(state, piece):
    """
    :param state: The state to be checked
    :type state: State
    :param piece: Piece to check
    :type piece: Piece
    """
    new_states = []
    pos = (piece.coord_x, piece.coord_y)
    if (pos[0] - 1 >= 0):
        if (state.board.grid[pos[1]][pos[0] - 1] == '.'):
            new_states.append(create_state(state, pos, (pos[0] - 1, pos[1])))
    if (pos[0] + 2 < state.board.width):
        if (state.board.grid[pos[1]][pos[0] + 2] == '.'):
            new_states.append(create_state(state, pos, (pos[0] + 1, pos[1])))
    if (pos[1] - 1 >= 0):
        if (state.board.grid[pos[1] - 1][pos[0]] == '.' and state.board.grid[pos[1] - 1][pos[0] + 1] == '.'):
            new_states.append(create_state(state, pos, (pos[0], pos[1] - 1)))
    if (pos[1] + 1 < state.board.height):
        if (state.board.grid[pos[1] + 1][pos[0]] == '.' and state.board.grid[pos[1] + 1][pos[0] + 1] == '.'):
            new_states.append(create_state(state, pos, (pos[0], pos[1] + 1)))
    return new_states


def get_v(state, piece):
    """
    :param state: The state to be checked
    :type state: State
    :param piece: Piece to check
    :type piece: Piece
    """
    new_states = []
    pos = (piece.coord_x, piece.coord_y)
    if (pos[0] - 1 >= 0):
        if (state.board.grid[pos[1]][pos[0] - 1] == '.' and state.board.grid[pos[1] + 1][pos[0] - 1] == '.'):
            new_states.append(create_state(state, pos, (pos[0] - 1, pos[1])))
    if (pos[0] + 1 < state.board.width):
        if (state.board.grid[pos[1]][pos[0] + 1] == '.' and state.board.grid[pos[1] + 1][pos[0] + 1] == '.'):
            new_states.append(create_state(state, pos, (pos[0] + 1, pos[1])))
    if (pos[1] - 1 >= 0):
        if (state.board.grid[pos[1] - 1][pos[0]] == '.'):
            new_states.append(create_state(state, pos, (pos[0], pos[1] - 1)))
    if (pos[1] + 2 < state.board.height):
        if (state.board.grid[pos[1] + 2][pos[0]] == '.'):
            new_states.append(create_state(state, pos, (pos[0], pos[1] + 1)))
    return new_states


def get_successor(state):
    """
    :param state: The state to find succesors
    :type state: State
    """
    new_states = []
    for piece in state.board.pieces:
        if (piece.is_single):
            new_states.extend(get_single(state, piece))
        elif (piece.is_goal):
            new_states.extend(get_goal(state, piece))
        elif (piece.orientation == 'h'):
            new_states.extend(get_h(state, piece))
        elif (piece.orientation == 'v'):
            new_states.extend(get_v(state, piece))
    return new_states


def dfs(state):
    """
    :param state: The state to perform DFS on
    :type state: State
    """
    stack = [state]
    explored = []
    while (stack):
        curr_state = stack.pop()
        if (not curr_state.board.grid in explored):
            explored.append(curr_state.board.grid)
            if (test_goal(curr_state)):
                return curr_state
            stack.extend(get_successor(curr_state))
    return None


def heuristic_manhattan(board):
    """
    :param board: The state to find succesors
    :type board: Board
    """
    for piece in board.pieces:
        if piece.is_goal:
            return abs(piece.coord_x - 1) + abs(piece.coord_y - 3)


def astar(state):
    """
    :param state: The state to perform astar search on
    :type state: State
    """
    heap = []
    heappush(heap, state)
    explored = set()
    while heap:
        curr_state = heappop(heap)
        if not hash(curr_state.board) in explored:
            explored.add(hash(curr_state.board))
            if test_goal(curr_state):
                return curr_state
            for new_state in get_successor(curr_state):
                heappush(heap, new_state)
    return None


def solve(board_string):
    """
    :param board_string: String of the board as input, 4x5 board with \n to separate each row.
    :type board_string: str
    """
    board = generate_board(board_string)
    state = State(board, 0, 0)
    result = astar(state)

    return result



if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file

    file = open(args.inputfile, "r")
    content = file.read()
    file.close()

    result = solve(content)

    steps = []

    while result.parent is not None:
        steps.append(result.board)
        result = result.parent

    target = open(args.outputfile, 'w')

    count = 0
    while steps:
        step = steps.pop()
        count += 1
        for i, line in enumerate(step.grid):
            for ch in line:
                target.write(ch)
            target.write('\n')
        target.write('\n')
    print("--- %s seconds ---" % (time.time() - start_time))
    print(count)
