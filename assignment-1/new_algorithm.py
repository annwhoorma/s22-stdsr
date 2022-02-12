from numpy import empty
from mrl98 import Buffer, Element, Fullness, Sequence, MRL98, plus_inf, Fullness
from mrl99 import MRL99

'''
This .py file implements the third (most efficient at 1998) version: New Algorithm:

"Associate with esch buffer X an integer C(X) that denotes its level.
Let .! be the smallest among the levels of currently full buffers.
If there is exactly one empty buffer, invoke NEW and assign it level L.
If there are at least two empty buffers, invoke NEW on each and assign level 0 to each one.
If there 8x-e no empty buffers, invoke COLLAPSE on the set of buffers with level L.
Assign the output buffer, level e + 1"
'''

class NewAlgorithm:
    b = 3
    def __init__(self, mrl_type: str, input_sequence: Sequence, b: int, be: int, phi: float):
        assert '99' in mrl_type or '98' in mrl_type, 'mrl_type must contain 98 or 99'
        self.mrl = MRL98(input_sequence, b, be) if '98' in mrl_type else MRL99(input_sequence, b, be)
        # for MRL98 r is always 1 and not used
        self.r = 2 if '99' in mrl_type else None
        self.input_sequence = input_sequence
        self.b = b
        self.mrl_type = mrl_type
        self.be = be
        self.phi = phi
        self.buffers: list[Buffer] = []
        self.l = 0 # at any time, the smallest level among all self.buffers with full==True (!)
        self._create_buffers()

    def _create_buffers(self):
        for _ in range(self.b):
            self.buffers.append(Buffer(self.be))

    def _count_empty_buffers(self) -> 'list[Buffer]':
        empty_buffers = []
        for buffer in self.buffers:
            if not buffer.full == Fullness.FULL:
                empty_buffers.append(buffer)
        return empty_buffers, len(empty_buffers)

    def _update_l(self):
        self.l = min(self.buffers, key=lambda buffer: buffer.level if buffer.full == Fullness.FULL else plus_inf).level

    def _step(self):
        empty_buffers, empty_buffers_amount = self._count_empty_buffers()
        if empty_buffers_amount == 1:
            # invoke NEW on the empty buffer and assign it level `self.l`
            buffer = empty_buffers[0]
            self.mrl.new(buffer) if '98' in self.mrl_type else self.mrl.new(buffer, self.r)
            buffer.update_level(self.l)
        elif empty_buffers_amount >= 2:
            # invoke NEW on each and assign level 0 to each one
            for buffer in empty_buffers:
                self.mrl.new(buffer) if '98' in self.mrl_type else self.mrl.new(buffer, self.r)
                buffer.update_level(0 if '98' in self.mrl_type else 1)
        else:
            # invoke COLLAPSE on the set of buffers with level `self.l` and assign level `self.l`+1 to the output buffer
            min_buffers = list(filter(lambda buffer: buffer.level == self.l, self.buffers))
            output_buffer = self.mrl.collapse(min_buffers)
            output_buffer.update_level(self.l + 1)
            # tree height increases
            if '99' in self.mrl_type:
                self.r *= 2
        self._update_l()

    def run(self):
        while len(self.input_sequence) > 0:
            self._step()
        
        result = self.mrl.output(self.phi, self.buffers)
        return result