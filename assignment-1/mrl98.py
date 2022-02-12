'''
"From the original paper about MRL98 (cited in README)
> An algorithm for computing approximate quartiles consists of a series of
invocations of NEW and COLLAPSE, terminating with OUTPUT. NEW populates
empty buffers with input and COLLAPSE reclaims some of them by collapsing
a chosen subset of full buffers. OUTPUT is invoked on the final set of full
buffers. Different buffer collapsing policies correspond to different algorithms.
'''

from math import ceil
from typing import Tuple, Optional

Element = int
Sequence = 'list[Element]'

plus_inf  = float('inf')
minus_inf = -plus_inf

class Buffer:
    '''
    A class to describe one buffer
    '''
    def __init__(self, be, elements:Optional[list]=None):
        self.to_initial()
        self.be = be
        # for testing
        self.elements = [None] * self.be if elements is None else elements
        self.level = 0

    def update_level(self, new_level: int):
        '''
        assigns a new level to a buffer, used in new_algorithm.py
        '''
        assert new_level >= 0, 'buffer level must be a non-negative number'
        self.level = new_level

    def populate(self, elements: list, is_initial:bool=False, weight:int=0, full:bool=False, is_mrl98:bool=True):
        '''
        populate a buffer
        @param elements: a list of elements
        @param is_initial: True if the population happens during NEW step
        @param weight: weight for this buffer (if is_initial is True, it will be set to 1 regardless)
        @full: a label - whether the buffer is considered full or not
        '''
        self.elements = elements
        if is_initial:
            self.weight = 1 if is_mrl98 else 2
            self.full = True
        else:
            self.weight = weight
            self.full = full
        self.cursor = 0

    def to_initial(self):
        '''
        sets all the properties to their initial state
        '''
        self.weight = 0
        self.full = False
        self.cursor = 0

    def incr_cursor(self):
        '''
        increment the cursor; this operation makes sense during COLLAPSE step
        '''
        if self.cursor is not None and self.cursor < len(self.elements) - 1:
            self.cursor += 1
        else:
            self.cursor = None

    def sort(self):
        '''
        sorts the elements of the buffer in ascending order
        '''
        self.elements.sort()

    def len(self) -> int:
        '''
        returns the length of the elements of the buffer
        '''
        return len(self.elements)

    def get_elem(self, index) -> Optional[Element]:
        '''
        returns an element at `index` position from the buffer if this element exists; returns None otherwise
        @param index: index to look at
        '''
        try:
            return self.elements[index] if not index is None else plus_inf
        except:
            return None


class MRL98:
    '''
    A class to describe the steps of MRL99 algorithm
    '''
    y_idx = 0
    def __init__(self, input_sequence: Sequence, b: int, be:int):
        '''
        @param input_sequence: the original dataset of numbers
        @param b: number of buffers to use
        @param be: number of elements per buffer
        '''
        self.input_sequence = input_sequence
        self.input_seq_len = len(self.input_sequence)
        self.b = b # number of buffers
        self.be = be # number of elements per buffer
        self.buffers = []
        self.infs_added = 0


    def new(self, buffer: Buffer) -> Buffer:
        '''
        NEW step
        @param buffer: an empty buffer to fill with next `self.be` values
        @return buffer: input buffer filled with values
        '''
        assert buffer.full == False, 'the buffer should be empty'
        assert len(self.input_sequence) >= 1, 'the input sequence must have at least one element'
        if self.be < len(self.input_sequence):
            buffer.populate(self.input_sequence[:self.be], is_initial=True)
            del self.input_sequence[:self.be]
        else:
            seq_len = len(self.input_sequence)
            # adding an equal number of +inf and -inf elements
            more_elems = self.be - seq_len
            if more_elems % 2 == 1:
                more_elems += 1
            self.infs_added += more_elems
            infs = [plus_inf] * (more_elems // 2) + [minus_inf] * (more_elems // 2)
            elems = self.input_sequence[:seq_len] + infs
            buffer.populate(elems, is_initial=True)
            del self.input_sequence[:seq_len]
        assert buffer.full and buffer.weight == 1, 'after NEW step, resulting buffer must be marked as full and have a weight of 1'
        return buffer


    def collapse(self, buffers: 'list[Buffer]') -> Buffer:
        '''
        COLLAPSE step: takes at least 2 buffers and returns a new buffer
        @param buffers: list of full buffers with assigned weights
        @return buffer: Y (see section 3.2 from the original paper)
        '''
        assert len(buffers) >= 2, 'should be 2 or more buffers'
        assert all(buffer.full and buffer.len() >= self.be for buffer in buffers), f'all buffers must be full and contain {self.be} or more elements'

        sequence = [] # new sequence, future Y
        sum_of_weights = 0

        for buffer in buffers:
            buffer.sort()
            sum_of_weights += buffer.weight
        k = sum_of_weights # step (in the original paper's example k=9 - Fig.1)
        i = 0 # counter
        min_len = sum_of_weights * self.be
        offset = ceil(sum_of_weights / 2)
        while i <= min_len:
            minimum, buffer = self._get_min_from_buffers(buffers)
            i += buffer.weight
            if minimum == plus_inf or (i >= offset and self._divides(i-buffer.weight, i, offset+k*len(sequence))):
                sequence.append(minimum)

        for buffer in buffers:
            buffer.to_initial()
        # populate the first buffer from `buffers` with Y, other buffers will be empty: see section 3.2 from the original paper
        y = buffers[self.y_idx]
        y.populate(sequence, weight=sum_of_weights, full=True)
        assert y.full and y.len() >= self.be, f'Y must be full and contain >= {self.be} elements at the end of the COLLAPSE step'
        return y


    def output(self, phi: float, buffers: 'list[Buffer]') -> Element:
        '''
        OUTPUT step
        @param phi: the quantile to find a value at
        @param buffers: a list of buffers marked as full and containing `self.be` elements
        @return estimated element at phi-th precentile
        '''
        assert 0 <= phi <= 1, 'phi must be from [0, 1]'
        assert len(buffers) >= 2, 'should be 2 or more buffers'
        assert all(buffer.full and buffer.len() >= self.be for buffer in buffers), f'all buffers must be full and contain >= {self.be} elements'

        phi_tick = self._calculate_phi_tick(phi)
        y = self.collapse(buffers)
        position = max(0, ceil(phi_tick * self.be - 1))
        return y.get_elem(position)


    def _divides(self, start: Element, end: Element, divider: int) -> bool:
        '''
        checks if any number from range [`start`, `end`] divides by `divider` without a remainder
        '''
        for num in range(start, end+1):
            if num % divider == 0:
                return True
        return False


    def _get_min_from_buffers(self, buffers: 'list[Buffer]') -> Tuple[Element, Buffer]:
        '''
        returns the min from all buffers with consideration that that min wasn't returned before
        @param buffers: a list of buffers
        @return minimum: the min Element
        @return buffer: the buffer that minimum was found in
        '''
        elements = map(lambda i, buffer: (i, buffer.get_elem(buffer.cursor)), range(len(buffers)), buffers)
        buffer_number, minimum = min(elements, key=lambda elem: elem[1])
        buffer = buffers[buffer_number]
        buffer.incr_cursor()
        return minimum, buffer


    def _calculate_phi_tick(self, phi: float):
        '''
        calculates phi' from the paper
        @param phi: original percentile
        @return phi_tick: phi'
        '''
        self.beta = (self.input_seq_len + self.infs_added) / self.input_seq_len
        assert self.beta >=1, 'beta must be >= 1'
        phi_tick = (2 * phi + self.beta - 1) / (2 * self.beta)
        return phi_tick

if __name__ == '__main__':
    pass