
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
