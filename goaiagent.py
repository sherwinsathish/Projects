import numpy as np
import random
import time
from read import readInput
from write import writeOutput
from host import GO

INF = float('inf') #used for alpha beta

def libcheck(board, x, y):
    visited = set()
    player = board[x][y]#1 or 2
    return dfs(board, x, y, player, visited) #check liberties with dfs

def dfs(board, x, y, player, visited):
    if (x, y) in visited:
        return False
    visited.add((x, y))#so no repeats
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] #orthogonal directions
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(board) and 0 <= ny < len(board): # within board bounds
            if board[nx][ny] == 0:
                return True
            if board[nx][ny] == player and dfs(board, nx, ny, player, visited): #adjacent piece
                return True
    return False

def capture(board, player):
    opponent = 3 - player#1 or 2
    captured = False
    captured_count = 0
    for x in range(len(board)):
        for y in range(len(board)):
            if board[x][y] == opponent and not libcheck(board, x, y):
                board[x][y] = 0  #captured stone 0
                captured = True
                captured_count += 1
    return captured, captured_count

def freespots(board, x, y, player):
    if board[x][y] != 0:
        return False  #taken
    test_board = np.copy(board)
    test_board[x][y] = player
    if libcheck(test_board, x, y) or capture(test_board, player)[0]:
        return True
    return False

def possiblemoves(board, player):
    board_size = len(board)
    center = board_size // 2
    distance_buckets = {}
    for i in range(board_size):
        for j in range(board_size):
            if freespots(board, i, j, player):
                distance = abs(i - center) + abs(j - center)#aim is to go closest to the center
                if distance not in distance_buckets:
                    distance_buckets[distance] = []
                distance_buckets[distance].append((i, j))

    sorted_moves = []
    for distance in sorted(distance_buckets):
        sorted_moves.extend(distance_buckets[distance])

    return sorted_moves

def makemove(board, move, player):#new board
    new_board = np.copy(board)
    new_board[move[0]][move[1]] = player
    return new_board

def calcscore(board, player, komi=2.5):#calc score
    opponent = 3 - player
    player_score = np.sum(board == player)
    opponent_score = np.sum(board == opponent)
    if player == 2:
        player_score+=komi
    if opponent ==2:
        opponent_score+=komi
    return player_score - opponent_score

def alphabetaprune(board, depth, alpha, beta, maximizing_player, player, start_time):
    if time.time() - start_time > 9.5:
        return calcscore(board, player)

    valid_moves = possiblemoves(board, player)
    if depth == 0 or len(valid_moves) == 0:
        return calcscore(board, player)

    if maximizing_player:
        max_eval = -INF
        for move in valid_moves:
            new_board = makemove(board, move, player)
            captured, captured_count = capture(new_board, player)
            eval = alphabetaprune(new_board, depth - 1, alpha, beta, False, 3 - player, start_time)#recursively switch to min
            eval += captured_count * 10
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = INF
        for move in valid_moves:
            new_board = makemove(board, move, 3 - player)
            captured, captured_count = capture(new_board, 3 - player)
            eval = alphabetaprune(new_board, depth - 1, alpha, beta, True, player, start_time)#switch to max
            eval -= captured_count * 10
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def ideep_alphabeta(board, max_depth, player, start_time):
    best_move = None
    best_score = -float('inf')
    visited_moves = set()

    for depth in range(1, max_depth + 1):
        for move in possiblemoves(board, player):
            if time.time() - start_time > 9.5:# add time limit
                return best_move
            if move in visited_moves:
                continue

            visited_moves.add(move)
            new_board = makemove(board, move, player)
            captured, captured_count = capture(new_board, player)
            score = alphabetaprune(new_board, depth - 1, -float('inf'), float('inf'), False, 3 - player, start_time)
            score += captured_count * 10
            if score > best_score:
                best_score = score
                best_move = move

    return best_move

def get_input(go, piece_type, steps, start_time):
    middle_spot = (2, 2)

    if steps <= 2 and go.valid_place_check(middle_spot[0], middle_spot[1], piece_type, test_check=True):
        return middle_spot
    max_depth = 3
    move = ideep_alphabeta(go.board, max_depth, piece_type, start_time)
    return "PASS" if move is None else move

if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    steps = sum(board[i][j] != 0 for i in range(N) for j in range(N))

    start_time = time.time()
    action = get_input(go, piece_type, steps, start_time)
    writeOutput(action)
