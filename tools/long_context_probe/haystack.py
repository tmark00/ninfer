# -*- coding: utf-8 -*-
"""Builds a filler corpus with control values planted at chosen depths.

The filler is deterministic word salad, so a value only survives if the model actually attends to
its position: nothing in the surrounding text implies it. Depths are given as a percentage of the
paragraph count, which tracks token position closely enough because paragraphs are uniform.
"""
import random

WORDS = ("rejestr iteracja wpis transfer pomiar pozycja limit znacznik wektor modul krok bufor prog "
         "odczyt wiersz zakres etap przebieg etykieta cykl blok kanal ramka sekwencja wskaznik "
         "segment tablica kolumna wartosc licznik indeks obszar warstwa wezel sciezka").split()

# (name, depth percent, value). The spread matters more than the values: one near the very start
# and one near the very end force any answer that combines them to span the whole context.
VALUES = [("ALFA", 5, 4173), ("BETA", 45, 2856), ("GAMMA", 70, 9014),
          ("DELTA", 82, 6392), ("EPSILON", 92, 1587), ("OMEGA", 98, 7245)]

# The rule sits at its own depth and points at three values held elsewhere, so answering it is a
# hop across the context rather than a point lookup.
RULE_DEPTH = 88
RULE = ("Regula wiazaca: parametr pochodny ZETA jest rowny sumie parametrow DELTA i EPSILON, "
        "pomniejszonej o parametr ALFA. Zapamietaj te regule. ")


def build(target_bytes, seed=20260908):
    rng = random.Random(seed)
    paragraphs, size, index = [], 0, 0
    while size < target_bytes:
        line = "Akapit %d. %s" % (index, " ".join(rng.choice(WORDS) for _ in range(60)))
        paragraphs.append(line)
        size += len(line) + 1
        index += 1
    for name, depth, value in VALUES:
        at = max(1, min(len(paragraphs) - 1, (len(paragraphs) * depth) // 100))
        paragraphs[at] = "Zapis kontrolny: parametr %s ma wartosc %d. " % (name, value) + paragraphs[at]
    at = (len(paragraphs) * RULE_DEPTH) // 100
    paragraphs[at] = RULE + paragraphs[at]
    return "\n".join(paragraphs)


NEEDLE_QUESTION = (
    "\n\nKoniec rejestru. W powyzszym tekscie ukryto szesc zapisow kontrolnych. Podaj wartosc "
    "kazdego parametru, jedna linia na parametr, dokladnie w formacie NAZWA=WARTOSC, bez zadnego "
    "innego tekstu. Parametry: " + ", ".join(n for n, _, _ in VALUES) + ".")

REASONING_QUESTION = (
    "\n\nKoniec rejestru. Odpowiedz na szesc pytan o powyzszy tekst. Wypisz dokladnie szesc linii "
    "w formacie PN=ODPOWIEDZ, bez zadnego innego tekstu.\n"
    "P1: suma wartosci parametrow ALFA i OMEGA.\n"
    "P2: roznica wartosci GAMMA minus BETA.\n"
    "P3: nazwa parametru o najwiekszej wartosci.\n"
    "P4: nazwa parametru o najmniejszej wartosci.\n"
    "P5: ile z szesciu parametrow ma wartosc parzysta.\n"
    "P6: wartosc parametru ZETA wedlug reguly wiazacej podanej w tekscie.")


def needle_expected():
    return {name: str(value) for name, _, value in VALUES}


def reasoning_expected():
    v = {n: value for n, _, value in VALUES}
    return {"P1": str(v["ALFA"] + v["OMEGA"]),
            "P2": str(v["GAMMA"] - v["BETA"]),
            "P3": max(v, key=v.get),
            "P4": min(v, key=v.get),
            "P5": str(sum(1 for x in v.values() if x % 2 == 0)),
            "P6": str(v["DELTA"] + v["EPSILON"] - v["ALFA"])}
