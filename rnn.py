def steps(s, x, U, W):
  return x * U + s * W
  
def forward(X, U, W):
  #Inicjalizacja stanu aktywacji dla każdej próbki w sekwencji
  S = np.zeros((number_of_samples, sequence_lenght+1))
  #aktualizacja stanów sekwencji
  for t in range(0, sequence_lenght):
    S[:,t+1] = step(S[:,t], X[:,t], U, W) #funkcja kroku
  return S
