
#bufory dla partii danych wejściowych i dolelowych. Pierwszy wymiar - rozmiar partii(liczba próbek), drugi wymiar - liczba oznaczająca długośc sekwencji tekstu
inputs = tf.placeholder(tf.int32, (batch_size, sequece_length))
targets = tf.placeholder(tf.int32, (batch_size, sequence_length))0

#każdy znak przekształcamy w wektor. Wekotr skłąda się z samych 0, poza komórką odpowiadającą indeksowi (1)
one_hot_inputs = tf.one_hot(inputs, depth=number_of_characters)

#definicja architektury LSTM. Najpierw trzeba określić komórki LSTM dla każdej warstwy [lstm_sizes to lista rozmiarów dla każdej warstwy]
cell_list = (tf.nn.rnn_cell.LSTMCell(lstm_size) for lstm_size in lstm_sizes)

#opakowanie komórek w jendowarstwową komórkę RNN
multi_cell_lstm = tf.nn_cell,MultiRNNCell(cell_list)


initial_state = self.multicell_lstm.zero_state(batch_sizem tf.float32)
#konwersja do zmiennych, aby zapamiętać stan między partiami
state_variables = tf.python.util.nest.flatten(initial_state)))

#dynamin_rnn zwraca krotkę składającą się z tensora reprezentujące wyjście LSTM oraz stan końcowy
lstm_output, final_state = tf.nn.dynamin_rnn(
    cell=multi_cell_lstm, inputs=one_hot_inputs,
    initial_state=state_variable)

#metoda control_dependencies jest używana, by wymusić aktualizację stanu przed zwroceniem wyjścia LSTM
store_variables = (
    state_variable.assign(new_state)
    for(state_variable, new_state) in zip(
        tf.python.util.nest.flatten(self.state_variables),
        tf.python.util.nest.flattern(final_state)
    with tf.control_dependencies(store_states):
        lstm_output = tf.identity(stm_output)
    #spłaszczenie wyjścia do macierzy o rozmiarach liczba wyjśc * liczba cech wyjściowych
    output_flat = tf.reshape(lstm_output, (-1, lstm_sizes(-1)))
