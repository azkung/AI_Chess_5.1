import chess
import math


def scoreCalcAttack(board: chess.Board):
    currentScore = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        
        if piece != None:
            
            if piece.piece_type == 1:
                if piece.color == True:
                    currentScore += (4 + len(board.attacks(square)))
                else:
                    currentScore -= (4 + len(board.attacks(square)))
                    
            if piece.piece_type == 2:
                if piece.color == True:
                    currentScore += (12 + len(board.attacks(square)))
                else:
                    currentScore -= (12 + len(board.attacks(square)))
                    
            if piece.piece_type == 3:
                if piece.color == True:
                    currentScore += (12 + len(board.attacks(square)))
                else:
                    currentScore -= (12 + len(board.attacks(square)))
                    
            if piece.piece_type == 4:
                if piece.color == True:
                    currentScore += (20 + len(board.attacks(square)))
                else:
                    currentScore -= (20 + len(board.attacks(square)))
            
            if piece.piece_type == 5:
                if piece.color == True:
                    currentScore += (32 + len(board.attacks(square)))
                else:
                    currentScore -= (32 + len(board.attacks(square)))
                    
    return currentScore


def scoreCalcBoard(board):
    return scoreCalcAttack(board)


def AI(board : chess.Board, depth, alpha, beta):
    #Minimax AI that beats you
    #White maximizing, Black minimizing
    
    if board.is_checkmate():
        if board.turn == True:
            return float('-inf'), None
        else:
            return float('inf'), None
    
    if board.is_stalemate():
        return 0, None
    
    if board.is_game_over():
        return 0, None
        

    if depth == 0:
        return scoreCalcBoard(board), None

    if board.turn == True:
        #Maximizing White
        maxScore = float('-inf')
        maxMove = None
        
        for move in board.legal_moves:
            board.push(move)
            score = AI(board, depth-1, alpha, beta)[0]
            board.pop()
            
            previousMaxScore = maxScore
            maxScore = max(score, maxScore)

            if maxScore != previousMaxScore:
                maxMove = move
                
            alpha = max(alpha, score)
            if beta <= alpha:
                break

        return maxScore, maxMove

    else:
        #Minimizing Black
        minScore = float('inf')
        
        minMove = None
        
        for move in board.legal_moves:
            board.push(move)
            score = AI(board, depth-1, alpha, beta)[0]
            board.pop()
            
            previousMinScore = minScore
            minScore = min(score, minScore)

            if minScore != previousMinScore:
                minMove = move 
            
            beta = min(beta, score)

            if beta <= alpha:
                break

        return minScore, minMove
    

def bestMove(board: chess.Board):
    score, bestMove = AI(board, 3, float('-inf'), float('inf'))
    if bestMove == None:
        for move in board.legal_moves:
            return move
    return bestMove