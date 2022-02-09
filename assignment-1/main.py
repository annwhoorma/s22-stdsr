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
    def __init__(self, elements:list=[]):
        self.to_initial()
        self.elements = elements

    def populate(self, elements: list, is_initial:bool=False, weight:int=0, full:bool=False):
        '''
        populate a buffer
        @param elements: a list of elements
        @param is_initial: True if the population happens during NEW step
        @param weight: weight for this buffer (if is_initial is True, it will be set to 1 regardless)
        @full: a label - whether the buffer is considered full or not
        '''
        self.elements = elements
        if is_initial:
            self.weight = 1
            self.full = True
        else:
            self.weight = weight
            self.full = full
        self.cursor = 0

    def to_initial(self):
        '''
        sets all the properties to their initial state
        '''
        self.elements = []
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
    b = 3 # number of buffers
    be = 5 # number of elements per buffer
    offset = 5
    y_idx = 0
    def __init__(self, input_sequence: Sequence, b: int=10):
        '''
        @param input_sequence: the original dataset of numbers
        @param b: number of buffers to use; default is 10
        '''
        self.input_sequence = input_sequence
        self.input_seq_len = len(self.input_sequence)
        self.b = b # number of buffers
        self.be = ceil(self.input_seq_len / b) # number of elements per buffer
        self.buffers = []


    def new(self, buffer: Buffer) -> Buffer:
        '''
        NEW step
        @param buffer: an empty buffer to fill with next `self.be` values
        @return buffer: input buffer filled with values
        '''
        assert len(buffer.elements) == 0, 'the buffer should be empty'
        try:
            buffer.populate(self.input_sequence[:self.be])
            del self.input_sequence[:self.be]
        except: # IndexError
            seq_len = len(self.input_sequence)
            # adding an equal number of +inf and -inf elements
            more_elems = seq_len - self.be
            if len(more_elems) % 2 == 1:
                more_elems += 1
            infs = [plus_inf] * (more_elems // 2) + [minus_inf] * (more_elems // 2)
            elems = self.input_sequence[:seq_len] + infs
            buffer.populate(elems)
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
        while i <= min_len:
            minimum, buffer = self._get_min_from_buffers(buffers)
            i += buffer.weight
            if i >= self.offset and self._divides(i-buffer.weight, i, self.offset+k*len(sequence)):
                sequence.append(minimum)

        # see Fig. 1 from the original paper and formulas from section 3.1
        self.beta = min_len / self.input_seq_len
        assert self.beta >= 1, 'beta must be >= 1'

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
        assert len(buffers) >= 2, 'should be 2 or more buffers'
        assert all(buffer.full and len(buffer) >= self.k for buffer in buffers), f'all buffers must be full and contain >= {self.be} elements'
        
        phi_tick = self._calculate_phi_tick(phi)
        y = self.collapse(buffers)
        position = ceil(phi_tick * self.k * y.weight)
        return y[position]


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


    # def _get_elements_from_cursors_v2(self, buffers: 'list[Buffer]'):
    #     minimum = None
    #     buffer_number = None
    #     for i, buffer in enumerate(buffers):
    #         element = buffer[buffer.cursor]
    #         if element < minimum or minimum is None:
    #             minimum = element
    #             buffer_number = i
    #     buffers[buffer_number].incr_cursor()
    #     return minimum, buffer_number


    def _calculate_phi_tick(self, phi: float):
        '''
        calculates phi' from the paper
        @param phi: original percentile
        @return phi_tick: phi'
        '''
        assert self.beta >=1, 'beta must be >= 1'
        phi_tick = (2 * phi + self.beta - 1) / (2 * self.beta)
        return phi_tick


