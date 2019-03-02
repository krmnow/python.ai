
#bufory dla partii danych wejściowych i dolelowych. Pierwszy wymiar - rozmiar partii(liczba próbek), drugi wymiar - liczba oznaczająca długośc sekwencji tekstu
inputs = tf.placeholder(tf.int32, (batch_size, sequece_length))
targets = tf.placeholder(tf.int32, (batch_size, sequence_length))0

#każdy znak przekształcamy w wektor. Wekotr skłąda się z samych 0, poza komórką odpowiadającą indeksowi (1)
one_hot_inputs = tf.one_hot(inputs, depth=number_of_characters)

#definicja architektury LSTM. Najpierw trzeba określić komórki LSTM dla każdej warstwy [lstm_sizes to lista rozmiarów dla każdej warstwy]
cell_list = (tf.nn.rnn_cell.LSTMCell(lstm_size) for lstm_size in lstm_sizes)

#opakowanie komórek w jendowarstwową komórkę RNN
multi_cell_lstm = tf.nn_cell,MultiRNNCell(cell_list)
