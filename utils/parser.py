import argparse


def cmd_parser():
    parser =  argparse.ArgumentParser()
    
    parser.add_argument("--csv-path", action="store", required=True,
                        help="Ścieżka do pliku csv na bazę danych.")
    parser.add_argument("--mxml-files", action="store", required=False,
                        help="Ścieżka do folderu z plikami MusicXML.")
    parser.add_argument("--train", action="store_true", required=False,
                        help="Rozpoczyna trening modelu.")
    parser.add_argument("--test-size", action="store", required=False, default=0.4, type=float,
                        help="Ułamek grupy testowej z bazy danych.")
    parser.add_argument("--kernel", action="store", required=False, default="rbf",
                        help="Jądro modelu.")
    parser.add_argument("--gamma", action="store", required=False, default=0.1,
                        help="Obszar graniczny strefy Gaussa, tym większa gamma tym "
                        "mniej sztywne są granice decyzyjne.")
    parser.add_argument("-C", action="store", required=False, default=1.0,
                        help="Odwrotność parametru regularyzacji, zapobiegającej zbyt "
                        "dużej wartości wag cech.")
    parser.add_argument("--random-state", action="store", required=False, default=None,
                        help="Generator losowości, domyślnie generator biblioteki Numpy")
    parser.add_argument("--save-model", action="store", required=False, default=False,
                        help="Save model to the file with .joblib extension.")
    parser.add_argument("--classify", action="store", required=False, default=False,
                        help="Path to csv data with test songs.")
    args = parser.parse_args()
    
    return args