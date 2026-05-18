class Player(Enum):
    BLACK = 'black'
    WHITE = 'white'
    # returns the opponent of the current player
    def get_opponent(self):
        if self == Player.BLACK:
            return Player.WHITE
        else:
            return Player.BLACK
    
ROW, COL = 6, 6
INF = 90129012
WIN = 21092109
MOVE_NONE = (-1, -1), (-1, -1)
TIME_LIMIT = 10

Score = Union[int, float]
Move = tuple[tuple[int, int], tuple[int, int]] 
Board = list[list[str]]
State = tuple[Board, Player]
Action = tuple[tuple[int, int], tuple[int, int]]

def print_state(board: Board) -> None:
    horizontal_rule = "+" + ("-" * 5 + "+") * COL
    for row in board:
        print(horizontal_rule)
        print(f"|  {'  |  '.join(' ' if tile == '_' else tile for tile in row)}  |")
    print(horizontal_rule)


def heuristic(state: State) -> Score:
    bcount = 0
    wcount = 0
    board = state[0]
    for r, row in enumerate(board):
        for tile in row:
            if tile == "B":
                if r == 5:
                    return WIN
                bcount += 1
            elif tile == "W":
                if r == 0:
                    return -WIN
                wcount += 1
    if wcount == 0:
        return WIN
    if bcount == 0:
        return -WIN
    return bcount - wcount

### Task 2.1: Implementing Breakthrough for Minimax
def valid_actions(
    state: State
) -> set[Action]:
    """ YOUR CODE HERE """
    board, agent = state
    rows, cols = len(board), len(board[0])
    result = set()
    player = "W"
    if agent == Player.BLACK:
        player = "B"

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == player:
                if player == "B":
                    if (r + 1 < rows and c - 1 >= 0) and board[r + 1][c - 1] != player:
                        result.add(((r, c), (r + 1, c - 1)))
                    if (r + 1 < rows and c + 1 < cols) and board[r + 1][c + 1] != player:
                        result.add(((r, c), (r + 1, c + 1)))
                    if (r + 1 < rows) and board[r + 1][c] == "_":
                        result.add(((r, c), (r + 1, c)))
                if player == "W":
                    if (r - 1 >= 0 and c - 1 >= 0) and board[r - 1][c - 1] != player:
                        result.add(((r, c), (r - 1, c - 1)))
                    if (r - 1 >= 0 and c + 1 < cols) and board[r - 1][c + 1] != player:
                        result.add(((r, c), (r - 1, c + 1)))
                    if (r - 1 >= 0) and board[r - 1][c] == "_":
                        result.add(((r, c), (r - 1, c)))
    return result
    """ YOUR CODE END HERE """

def transition(
        state: State,
        action: Action,
    ) -> State:
    """ YOUR CODE HERE """
    board, agent = state
    board = copy.deepcopy(board)  # import copy

    src, dest = action[0], action[1]
    player = "W"
    if agent == Player.BLACK:
        player = "B"

    board[src[0]][src[1]] = "_" 
    board[dest[0]][dest[1]] = player
    
    return (board, agent)
    """ YOUR CODE END HERE """

def is_terminal(state: State) -> bool:
    """ YOUR CODE HERE """
    board, agent = state
    rows, cols = len(board), len(board[0])
    countB, countW = 0, 0

    for r in range(rows):
        for c in range(cols):
            player = board[r][c]
            if player == "B":
                countB += 1
                if r == rows - 1:
                    return True
            if player == "W":
                countW += 1
                if r == 0:
                    return True
    return countW == 0 or countB == 0
    """ YOUR CODE END HERE """

def utility(state: State) -> int:
    """ YOUR CODE HERE """
    board, agent = state
    rows, cols = len(board), len(board[0])
    countB, countW = 0, 0

    for r in range(rows):
        for c in range(cols):
            player = board[r][c]
            if player == "B":
                countB += 1
                if r == rows - 1:
                    return WIN
            if player == "W":
                countW += 1
                if r == 0:
                    return -WIN
                
    if countB == 0:
        return -WIN
    if countW == 0:
        return WIN

    return countB - countW
    """ YOUR CODE END HERE """

def evaluate(state: State) -> int:
    if not is_terminal(state):
        return heuristic(state)
    return utility(state)

def minimax(
    board: Board,
    depth: int,
    max_depth: int,
    current_player: Player,
) -> tuple[Score, Action]:
    state = (board,current_player)
    
    if depth == max_depth or is_terminal(state):
        return heuristic(state), MOVE_NONE

    if current_player == Player.BLACK:
        best_score = -INF
    else:
        best_score = INF

    best_move = MOVE_NONE
    next_player = current_player.get_opponent()

    for action in valid_actions(state):
        child = transition(state, action)
        new_board = child[0]
        score = minimax(new_board, depth + 1, max_depth, next_player)[0]

        if current_player == Player.BLACK:
            if score > best_score:
                best_score = score
                best_move = action
        else:
            if score < best_score:
                best_score = score
                best_move = action

    return best_score, best_move


### Task 2.2: Integrate alpha-beta pruning into minimax
def minimax_alpha_beta(
    board: Board,
    depth: int,
    max_depth: int,
    alpha: Score,
    beta: Score,
    current_player: Player
) -> tuple[Score, Move]:
    """ YOUR CODE HERE """
    state = (board,current_player)

    if depth == max_depth or is_terminal(state):
        return heuristic(state), MOVE_NONE

    if current_player == Player.BLACK:
        best_score = -INF
    else:
        best_score = INF

    best_move = MOVE_NONE
    next_player = current_player.get_opponent()

    for action in valid_actions(state):
        child = transition(state, action)
        currScore = heuristic(child)

        new_board = child[0]
        score = minimax_alpha_beta(new_board, depth + 1, max_depth, alpha, beta, next_player)[0]

        if current_player == Player.BLACK:
            if score > best_score:
                best_score = score
                best_move = action
            
            alpha = max(alpha, best_score)
            if alpha >= beta:
                return best_score, best_move

        else:
            if score < best_score:
                best_score = score
                best_move = action
                
            beta = max(beta, best_score)
            if beta <= alpha:
                return best_score, best_move

    return best_score, best_move
    """ YOUR CODE END HERE """

def invoke_search_fn(search_fn, board, max_depth, current_player):
    if "alpha_beta" in search_fn.__name__:
        return search_fn(board, 0, max_depth, -INF, INF, current_player)
    else:
        return search_fn(board, 0, max_depth, current_player)

### Task 2.3: Implement an improved heuristic function
def improved_evaluate(state: State) -> Score:
    """ YOUR CODE HERE """
    board, agent = state
    rows, cols = len(board), len(board[0])
    countB, countW = 0, 0
    weightB, weightW = 0, 0
    for r, row in enumerate(board):
        for tile in row:
            if tile == "B":
                weight = math.floor(r/2)


                if r == 5:
                    return WIN
                countB += 1
                weightB += weight
            elif tile == "W":
                
                weight = math.floor((rows - 1 - r)/2)
                
                if r == 0:
                    return -WIN
                countW += 1
                weightW += weight
    if countW == 0:
        return WIN
    if countB == 0:
        return -WIN
    return weightB - weightW
    """ YOUR CODE END HERE """
