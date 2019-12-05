# import pandas as pd
# df = pd.read_csv(r'D:\Studies\Sem7\bachelor\mxml_recognizer\database_100.csv', sep=";", index_col=0)
# params = df.fillna(0).iloc[:, :]
# # new_data = params.iloc[:, 0:6]

# _20 = params.loc[:, "2.0"]
# stdpitch = params.loc[:, "pitch_std"]
# avg_note_duration = params.loc[:, "avg_note_duration"]
# full = params.loc[:, "4.0"]
# # half = params.loc[:, "0.5"]
# # onehalf = params.loc[:, "1.5"]
# # quarter = params.loc[:, "0.25"]
# # number = params.loc[:, "783.990871963499"]
# # eight = params.loc[:, "0.0"]
# labels = params.loc[:, "class_label"]

# # params = pd.concat([new_data, _20, full, half, onehalf, quarter, eight, labels, number], axis=1)
# params = pd.concat([_20, stdpitch, avg_note_duration, full, labels], axis=1)
# params.to_csv("database_100_smaller.csv", sep=";")
import music21

stream = music21.converter.parse(r'D:\Studies\Sem7\bachelor\mxml_recognizer\mxml_copies\classicism\CPEBach_Gross_ist_der_Herr_AltJDB_EnglText.mxl')
stream = stream.flat
notes = stream.notes
for note in notes:
    if isinstance(note, music21.chord.Chord):
        for note_in_chord in note.notes:
            print(note.quarterLength)
            if note.quarterLength == 2.0:
                print("")
            
    else:
        print(note.quarterLength)
    if note.quarterLength == 2.0:
                print("")
# music21.environment.set("musicxmlPath", "C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe")
# stream.show()
print("")