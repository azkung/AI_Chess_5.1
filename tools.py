import numpy as np
import chess
import chess.pgn
from numpy import save
import numpy



def boardToArray(board: chess.Board):
    for i in range(6):
        myboard = []
    
def board_to_np(board):
    pgn = board.epd()
    foo = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for i in range(6):
        foo2 = []
        for row in rows:
            foo3 = []
            for thing in row:
                if thing.isdigit():
                    for x in range(0, int(thing)):
                        foo3.append(0)
                else:
                    if thing == 'P' and i == 0:
                        foo3.append(1)
                    elif thing == 'p' and i == 0:
                        foo3.append(-1)
                    elif thing == 'N' and i == 1:
                        foo3.append(1)
                    elif thing == 'n' and i == 1:
                        foo3.append(-1)
                    elif thing == 'B' and i == 2:
                        foo3.append(1)
                    elif thing == 'b' and i == 2:
                        foo3.append(-1)
                    elif thing == 'R' and i == 3:
                        foo3.append(1)
                    elif thing == 'r' and i == 3:
                        foo3.append(-1)
                    elif thing == 'Q' and i == 4:
                        foo3.append(1)
                    elif thing == 'q' and i == 4:
                        foo3.append(-1)
                    elif thing == 'K' and i == 5:
                        foo3.append(1)
                    elif thing == 'k' and i == 5:
                        foo3.append(-1)
                    else:
                        foo3.append(0)
            foo2.append(foo3)

        foo.append(foo2)
    return np.array(foo)


def format(path : str, idx : int, total_games : int, skip_ties : bool = False):
    with open("datasets/ficsgamesdb_2021_standard2000_nomovetimes_264388.pgn") as pgn:
        x = []
        y = []
        for i in range(idx):
            game = chess.pgn.read_game(pgn)
        for i in range(total_games):
            game = chess.pgn.read_game(pgn)
            result = game.headers["Result"]
            if(result == "1-0"):
                winner = 1
            if(result == "0-1"):
                winner = 0
            if(result == "1/2-1/2"):
                if(skip_ties):
                    continue
                winner = 0.5

            board = game.board()
            x.append(board_to_np(board))
            y.append(winner)
            for move in game.mainline_moves():
                board.push(move)
                x.append(board_to_np(board))
                y.append(winner)
        x = np.array(x)
        y = np.array(y)
        print(x.shape)
        print(y.shape)
        save(path + "_x.npy", x)
        save(path + "_y.npy", y)



