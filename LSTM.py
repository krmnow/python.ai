
#bufory dla partii danych wejściowych i dolelowych. Pierwszy wymiar - rozmiar partii(liczba próbek), drugi wymiar - liczba oznaczająca długośc sekwencji tekstu
inputs = tf.placeholder(tf.int32, (batch_size, sequece_length))
targets = tf.placeholder(tf.int32, (batch_size, sequence_length))0

#każdy znak przekształcamy w wektor. Wekotr skłąda się z samych 0, poza komórką odpowiadającą indeksowi (1)
one_hot_inputs = tf.one_hot(inputs, depth=number_of_characters)
