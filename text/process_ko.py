from jamo import h2j

from .symbols_ko import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def cleaned_ko_text_to_sequence(cleaned_text):
    sequence = [_symbol_to_id[symbol] for symbol in h2j(cleaned_text)]
    return sequence


def sequence_to_ko_text(sequence):
    result = "".join([_id_to_symbol[s] for s in sequence if s in _id_to_symbol])
    return result
