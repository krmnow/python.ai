
#metoda wywyłoana przed rozpoczęciem rozgrywki
def new_board():
  return((0,0,0),
        (0,0,0),
        (0,0,0))

def aplly_move(board_state, move, side):
  move_x, move_y = move
  state_list = list(list(s) for s in borad_state)
  state_list[move_x][move_y] = side
  return tuple(tuple(s) for s in state_list)

import itertools
def available_moves(board_state):
  for x, y in itertools.product(range(3), range(3)):
    if board_state[x][y] ==0:
      yield(x, y)
      
def has__in_a_line(line):
  return all(x==-1 for x in line) | all(x==1 for x in line)
