from typing import Optional

Sequence = 'list[float]'

class Buffer():
    def __init__(self):
        self.elements: list = []
        self.weight: int = 0
        self.full: bool = False

    def populate(self, elements: list, initial:bool=False):
        self.elements = elements
        if initial:
            self.weight = 1
            self.full = True


class MRL99:
    k = 10
    def __init__(self, input_sequence: Sequence):
        self.input_sequence = input_sequence

    def new(self, buffer: Buffer):
        assert len(buffer.elements) == 0, 'the buffer should be empty'
        try:
            buffer.populate(self.input_sequence[:self.k])
        except IndexError:
            # adding an equal number of +inf and -inf elements
            # Q: what if it cannot be equal?
            pass