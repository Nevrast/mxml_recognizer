from music21 import meter, key, clef, note, stream


def song():
    """
    Simple helper function that returns "Wlazł kotek na płotek" music stream
    """
    s = stream.Stream()
    ts1 = meter.TimeSignature("3/4")

    p1 = stream.Part(number=1)
    p1.insert(0, ts1)
    p1.insert(0, key.KeySignature(0))
    p1.insert(0, clef.TrebleClef())
    m1 = stream.Measure(number=1)
    m1.append(note.Note("G"))
    m1.append(note.Note("E"))
    m1.append(note.Note("E", type="quarter"))
    m2 = stream.Measure(number=2)
    m2.append(note.Note("F"))
    m2.append(note.Note("D"))
    m2.append(note.Note("D"))
    m3 = stream.Measure(number=3)
    m3.append(note.Note("C", type="eighth"))
    m3.append(note.Note("E", type="eighth"))
    m3.append(note.Note("G", type="half"))
    p1.append(m1)
    p1.append(m2)
    p1.append(m3)
    m4 = stream.Measure(number=4)
    m4.append(note.Note("G"))
    m4.append(note.Note("E"))
    m4.append(note.Note("E", type="quarter"))
    m5 = stream.Measure(number=5)
    m5.append(note.Note("F"))
    m5.append(note.Note("D"))
    m5.append(note.Note("D"))
    m6 = stream.Measure(number=6)
    m6.append(note.Note("C4", type="eighth"))
    m6.append(note.Note("E4", type="eighth"))
    m6.append(note.Note("C4", type="half"))
    p1.append(m4)
    p1.append(m5)
    p1.append(m6)
    s.insert(0, p1)
    return s