import argparse


def cmd_parser():
    parser =  argparse.ArgumentParser()
    
    parser.add_argument("--csv-path", action="store", required=True,
                        help="Ścieżka do pliku csv na bazę danych.")
    parser.add_argument("--mxml-files", action="store", required=False,
                        help="Ścieżka do folderu z plikami MusicXML.")
    parser.add_argument("--train", action="store_true", required=False,
                        help="Rozpoczyna trening modelu.")
    parser.add_argument("--test-size", action="store", required=False, default=0.4,
                        help="Ułamek grupy testowej z bazy danych.")
    parser.add_argument("--eta", action="store", required=False, default=0.1,
                        help="Współczynnik uczenia.")
    parser.add_argument("--max_iter", action="store", required=False, default=200,
                        help="Maxymalna liczba epok.")
    parser.add_argument("--random-state", action="store", required=False, default=None,
                        help="Generator losowości, domśylnie generator biblioteki Numpy")
    args = parser.parse_args()
    
    return args