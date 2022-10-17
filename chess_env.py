import bots.bot_attack as bot_attack
import bots.bot_basic as bot_basic
import bots.bot_inference as bot_inference
import bots.bot_middle as bot_middle
import bots.bot_random as bot_random
import chess

board = chess.Board()

    
print(board)
print()
moves = 0
while board.is_game_over() == False:
    print(board.turn)
    bestMove = bot_inference.bestMove(board)
    board.push(bestMove)
    print(board)
    print()

    if(board.is_game_over()):
        break

    print(board.turn)
    bestMove = bot_attack.bestMove(board)
    board.push(bestMove)
    print(board)
    print()

    moves += 1

    
print(board)
print(moves)
if(board.turn):
    print("Black Wins")
else:
    print("White Wins")