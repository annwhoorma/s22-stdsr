'''
By implementing MRL98 I wanted to show how it's built upon MRL98.
The only difference between them is `r` parameter in NEW steph (see section 3.1 from MRL99 paper)

From the original paper MRL99:
Notice that choosing r = 1 amounts to no sampling.
If r is larger then sampling is introduced. The larger r is the sparser the sample.
The algorithm will dynamically change the value of r during execution leading to a
variable rate of sampling.
'''


import random
from typing import Optional

from mrl98 import MRL98, Buffer, Element, Sequence, Fullness

class MRL99(MRL98):
    def __init__(self, input_sequence: Sequence, b: int, be: int):
        super().__init__(input_sequence, b, be)

    def _choose_one_from_next_r(self, sequence: Sequence, r: int) -> Element:
        '''
        chooses one element from `r` next elements of some sequence; the element will be removed from the sequence
        @param sequence: sequence to sample from
        @param r: sampling rate
        @returns element: chosen element from the sequence
        '''
        # randomly generate index
        index = random.randint(0, min(r, len(sequence)-1))
        element = sequence[index]
        del sequence[index]
        return element

    def new(self, buffer: Buffer, r: int) -> Buffer:
        '''
        NEW step
        @param buffer: an empty buffer to fill with next `self.be` values
        @param r: an integer that represent the sampling rate (see the original paper MRL99)
        @return buffer: input buffer filled with values
        '''
        assert buffer.full == Fullness.EMPTY, 'the buffer should be empty'
        assert len(self.input_sequence) >= 1, 'the input sequence must have at least one element'
        # if original sequence still has enough elements
        if self.be * r < len(self.input_sequence):
            population = [self._choose_one_from_next_r(self.input_sequence, r) for _ in range(self.be)]
            buffer.populate(population, is_mrl98=False, weight=r, full=Fullness.FULL)
        else:
            population = [self._choose_one_from_next_r(self.input_sequence, r) for _ in range(len(self.input_sequence))]
            buffer.populate(population, is_mrl98=False, weight=r, full=Fullness.PARTIAL)
        assert (buffer.full == Fullness.FULL or buffer.full == Fullness.PARTIAL), 'after NEW step, resulting buffer must be marked as full or partially full'
        assert buffer.weight == r, f'after NEW step, resulting buffer must have weight r={r}'
        return buffer

    def _calculate_phi_tick(self, phi: float) -> float:
        return phi