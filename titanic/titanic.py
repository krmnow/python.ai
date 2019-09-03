import os
TITANIC_PATH = os.path.join("zestawy danych", "titanic")

import pandas as pd

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)

train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")

#


#Dane są już rozdzielone na zbiory uczący i testowy.
# Jednak dane testowe nie zawierają etykiet. Naszym zadaniem jest wytrenowanie jak najlepszego modelu
# za pomocą za pomocą danych uczących, a następnie uzyskanie prognoz wobec danych testowych i przesłanie
# wyników do serwisu Kaggle w celu uzyskania ostatecznej oceny.

#Przyjrzyjmy się kilku pierwszym rzędom zestawu uczącego:
print(train_data.head(5))

#Atrybuty mają następujące znaczenie:
#•Survived (przetrwał): jest to nasza klasa docelowa; 0 oznacza, że pasażer nie przeżył, a 1, że przetrwał.
#•Pclass (klasa): klasa pasażera.
#•Name, Sex, Age (imię i nazwisko), (płeć), (wiek): nie wymagają wyjaśnienia.
#•SibSp (rodzeństwo i partnerzy): liczba rodzeństwa i partnerów pasażera obecnych na statku.
#•Parch (wstępni i zstępni): liczba dzieci i rodziców pasażera obecnych na statku.
#•Ticket (bilet): identyfikator biletu.
#•Fare (opłata): zapłacona cena za podróż (w funtach).
#•Cabin (kajuta): numer kajuty pasażera.
#•Embarked (miejsce wejścia): miejsce wejścia na pokład Titanica.

#Przyjrzyjmy się dokładniej temu zestawowi, aby ustalić, jak wiele brakuje danych:

print(train_data.info())

#No dobrze, zatem część atrybutów Age, Cabin i Embarked jest pustych (mniej niż 891 jest niezerowych);
# dotyczy to zwłaszcza atrybutu Cabin (brakuje w nim 77% wartości).
# Zignorujemy na razie atrybut Cabin i zajmiemy się pozostałymi.
# Atrybut Age zawiera około 19% pustych wartości, zatem musimy zdecydować, co z nimi zrobić.
# Rozsądnym pomysłem wydaje się zastąpienie brakujących wartości medianą wieku.

#Atrybuty Name i Ticket mogą mieć jakąś wartość, ciężko jednak będzie przekształcić jej w
# przydatne cyfry zrozumiałe dla modelu. Dlatego również je na razie zignorujemy.
#Sprawdźmy atrybuty numeryczne:


print(train_data.describe())

#•Ojej, przeżyło zaledwie 38% osób (Survived). :( To niemal 40%, zatem dokładność będzie odpowiednią metryką wydajności naszego modelu.
#•Średnia opłata (Fare) wynosiła 32.20 funtów, co nie wydaje się zbyt drogo (aczkolwiek w tamtych czasach było to pewnie mnóstwo pieniędzy).
#•Średni wiek (Age) wynosił mniej, niż 30 lat.
#Sprawdźmy, czy istnieją tylko dwie wartości docelowe: 0 i 1:

print(train_data["survived"].value_counts())

#Teraz przyjrzyjmy się pobieżnie wszystkim atrybutom kategorialnym:
print(train_data["Pclass"].value_counts())
print(train_data["Sex"].value_counts())
print(train_data["Embarked"].value_counts())

#Atrybut Embarked mówi nam, w którym porcie pasażera wszedł na pokład: C=Cherbourg, Q=Queenstown, S=Southampton.
#Klasa CategoricalEncoder pozwoli nam przekształcić atrybuty kategorialne do postaci wektorów "gorącojedynkowych".
# Zostanie ona wkrótce dodana do modułu Scikit-Learn, obecnie zaś możemy skorzystać z ponższego kodu (skopiowanego z prośby #9151).

# Definicja klasy CategoricalEncoder, skopiowana z prośby PR #9151.
# Wystarczy uruchomić tę komórkę albo wkleić ją do własnego kodu.
# Nie musisz próbować zrozumieć każdego wiersza

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Koduje cechy kategorialne w postaci macierzy numerycznej.
        Danymi wejściowymi dostarczanymi do tego transformatora powinna być macierz
        zawierająca liczby stałoprzecinkowe lub ciągi znaków, symbolizujące
        wartości przechowywane przez cechy kategorialne (dyskretne).
        Możemy kodować cechy za pomocą schematu "gorącojedynkowego" (jeden-z-K)
        (``encoding='onehot'``, domyślne rozwiązanie) lub przekształcać je do postaci
        liczb porządkowych (``encoding='ordinal'``).
        Tego typu kodowanie jest wymagane podczas dostarczania danych kategorialnych do wielu
        etymatorów modułu Scikit-Learn, mianowicie w modelach liniowych i maszynach
        SVM wykorzystujących standardowe jądra. Więcej informacji znajdziesz w:
        :ref:`User Guide <preprocessing_categorical_features>`.
        Parametry
        ----------
        encoding : ciąg znaków, 'onehot', 'onehot-dense' lub 'ordinal'
            Rodzaj stosowanego kodowania (domyślna wartość to 'onehot'):
            - 'onehot': koduje cechy za pomocą schematu "gorącojedynkowego" (jeden-z-K,
               bywa również nazywany kodowaniem 'sztucznym'). Zostaje utworzona kolumna
               binarna dla każdej kategorii, a zwracana jest macierz rzadka.
            - 'onehot-dense': to samo, co wartość 'onehot', ale zwraca macierz gęstą zamiast rzadkiej.
            - 'ordinal': koduje cechy w postaci liczb porządkowych. Uzyskujemy w ten sposób
              pojedynczą kolumną zawierającą liczby stałoprzecinkowe (od 0 do n_kategorii - 1)
              dla każdej cechy.
        categories : 'auto' lub lista list/tablic wartości.
            Kategorie (niepowtarzalne wartości) na każdą cechę:
            - 'auto' : Automatycznie określa kategorie za pomocą danych uczących.
            - lista : ``categories[i]`` przechowuje kategorie oczekiwane w i-tej kolumnie.
              Przekazane kategorie zostają posortowanie przed zakodowaniem danych
              (użyte kategorie można przejrzeć w atrybucie ``categories_``).
        dtype : typ liczby, domyślnie np.float64
            Wymagany typ wartości wyjściowej.
        handle_unknown : 'error' (domyślnie) lub 'ignore'
            Za jego pomocą określamy, czy w przypadku obecności nieznanej cechy w czasie
            wykonywania transformacji ma być wyświetlany komunikat o błędzie (wartość
            domyślna) lub czy ma zostać zignorowana. Po wybraniu wartości 'ignore'
            i natrafieniu na nieznaną kategorię w trakcie przekształceń, wygenerowane
            kolumny "gorącojedynkowe" dla tej cechy będą wypełnione zerami.
            Ignorowanie nieznanych kategorii nie jest obsługiwane w parametrze
            ``encoding='ordinal'``.
        Atrybuty
        ----------
        categories_ : lista tablic
            Kategorie każdej cechy określone podczas uczenia. W przypadku ręcznego
            wyznaczania kategorii znajdziemy tu listę posortowanych kategorii
            (w kolejności odpowiadającej wynikowi operacji 'transform').
        Przykłady
        --------
        Mając zbiór danych składający się z trzech cech i dwóch próbek pozwalamy koderowi
        znaleźć maksymalną wartość każdej cechy i przekształcić dane do postaci
        binarnego kodowania "gorącojedynkowego".
        >>> from sklearn.preprocessing import CategoricalEncoder
        >>> enc = CategoricalEncoder(handle_unknown='ignore')
        >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
        ... # doctest: +ELLIPSIS
        CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
                  encoding='onehot', handle_unknown='ignore')
        >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
        array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
               [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
        Powiązane materiały
        --------
        sklearn.preprocessing.OneHotEncoder : przeprowadzana kodowanie "gorącojedynkowe"
          stałoprzecinkowych cech porządkowych. Klasa ``OneHotEncoder zakłada``, że cechy wejściowe
          przechowują wartości w zakresie ``[0, max(cecha)]`` zamiast korzystać z
          niepowtarzalnych wartości.
        sklearn.feature_extraction.DictVectorizer : przeprowadzana kodowanie "gorącojedynkowe"
          elementów słowanika (a także cech przechowujących ciągi znaków).
        sklearn.feature_extraction.FeatureHasher : przeprowadzana przybliżone kodowanie "gorącojedynkowe"
          elementów słownika lub ciągów znaków.
        """
    def __init__(self, encoding="onehot", categories="auto", dtype=np.float64, handel_unknow='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknow = handle_unknow

    def fit(self, X, y=None):
        """Dopasowuje klasę CategoricalEncoder do danych wejściowych X.
                Parametry
                ----------
                X : tablicopodobny, postać [n_próbek, n_cech]
                    Dane służące do określania kategorii każdej cechy.
                Zwraca
                -------
                self
                """
        if self.encoding not in ['onehot', 'oneghot-dense', 'ordinal']:
            template = ("Należy wybrać jedno z nastepujacych kodowan: 'onehot', 'onehot-dense, lub 'ordinal;, wybrano %s")
            raise ValueError(template % self.handle_unknow)
        if self.encoding == 'ordinal' and self.handle_unknow == 'igonre':
            raise  ValueError("Wartosc handle_unknow='ignore' nie jest obslugiana przez parametr encoding='original'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self.__label__encoders_ = [LabelEncoder() for __ in range(n_features)]

        for i in range(n_features):
            le = self.__label__encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknow == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("znaleziono nieznane kategorie {0} w kolumnie {1} podczas dopasowywania".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self.__label__encoders_]

        return self

    def transform(self, X):
        """Przekształca X za pomocą kodowania "gorącojedynkowego".
                Parametry
                ----------
                X : tablicopodobny, postać [n_próbek, n_cech]
                    Kodowane dane.
                Zwraca
                -------
                X_out : macierz rzadka lub dwuwymiarowa tablica
                    Przekształcone dane wejściowe.
                """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknow == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("znalexiono nieznane kategorie {0} w kolumnie {1} podczas przkszralcania".format(diff, i))

                    raise  ValueError(msg)
                else:
            # Wyznaczamy akceptowalną wartość rzędom sprawiającym problem i
            # kontynuujemy. Rzędy te zostają oznaczone jako `X_mask` i zostaną
            # później usunięte.
                    X_mask[:, i] = valid_mask
            X[:, i][~valid_mask] = self.categories_[i][0]
        X_int[:, i] = self.__label__encoders_[i].transform(X[:, i])

    if self.encoding == 'ordinal':
        return X_int.astype(self.dtype, copy=False)

    mask = X_mask.ravel()
    n_values = [cats.shape[0] for cats in self.categories_]
    n_values = np.array([0] + n_values)
    indices = np.cumsum(n_values)

    column_indices = (X_int + indices[:-1].revael()[mask])
    row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                            n_features)[mask]
    data = np.ones(n_samples * n_features)[mask]

    out = sparse.csc_matrix((data, (row_indices, column_indices)),
                            shape=(n_samples, indices[-1]),
                            dtype=self.dtype).tocsr()
    if self.encoding == 'onehot-dense':
        return out.toarray()
    else:
        return out

from sklearn.base import BaseEstimator, TransformerMixin

# Tworzy klasę wybierającą numeryczne i kategorialne kolumny,
# gdyż moduł Scikit-Learn nie zawiera jeszcze obsługi obiektów DataFrame

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


#Stwórzmy potok dla atrybutów numerycznych:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
    ("imputer", Imputer(strategy="median")),
])

print(num_pipeline.fit_transform(train_data))

#Potrzebujemy również klasy przypisującej dla kolumn zawierających
# kategorialne ciągi znaków (standardowa klasa Imputer nie działa wobec nich):

#Inspirację stanowiło pytanie umieszczone na stronie stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent = pd.Series([X[c].value_counts().index[0] for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent)

#Teraz możemy stworzyć potok dla atrybutów kategorialnych:

cat_pipeline = Pileline([
    ('select_cat', DataFrameSelector(["Pclass", "Sex", "Embarked"])),
    ('imputer', MostFrequentImputer()),
    ("cat_encoder", CategoricalEncoder(encoding='onehot-dense')),
])

print(cat_pipeline.fit_transform(train_data))

#W końcu możemy połączyć potoki numeryczne i kategorialne:

from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

#Świetnie! Mamy teraz do dyspozycji dobry potok przetwarzania wstępnego,
# który pobiera nieprzetworzone dane i na wyjściu umieszcza numeryczne cechy wejściowe, które możemy przesłać do dowolnego modelu uczenia maszynowego.

X_train = preprocess_pipeline.fit_transform((train_data))
print(X_train)

#i etykiety
y_train = train_data["Survived"]

#możemy przystąpić do uczenia klasyfikatora. Zaczynamy od klasySVC

from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train, y_train)

#i już jest wyuczony. Można go użyć do obliczania prognoz

X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)

#Teraz wystarczyłoby wygenerować plik CSV przechowujący te prognozy (przy zachowaniu formatu zdefiniowanego w serwisie Kaggle),
# a następnie przesłać go i liczyć na dobre wyniki. Ale chwila! Możemy jeszcze dopomóc szczęściu.
# Możemy przecież skorzystać ze sprawdzianu krzyżowego, aby sprawdzić, jak dobry jest nasz model

from sklearn.model_selection import cross_val_score

scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print(scores.mean())

#Poszukajmy metody na ~80%. Spróbujmy RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print(scores.mean())

