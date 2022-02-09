from main import Buffer, Element, Sequence, MRL98

'''
From the original paper about MRL98 (cited in README)
An algorithm for computing approximate quartiles consists of a series of
invocations of NEW and COLLAPSE, terminating with OUTPUT. NEW populates
empty buffers with input and COLLAPSE reclaims some of them by collapsing
a chosen subset of full buffers. OUTPUT is invoked on the final set of full
buffers. Different buffer collapsing policies correspond to different algorithms.
We now describe three interesting policies.
'''

