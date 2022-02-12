from mrl98 import Buffer, MRL98

def imitate_buffers():
    buffers = [
        Buffer([12, 52, 72, 102, 132]),
        Buffer([23, 33, 83, 143, 153]),
        Buffer([44, 64, 94, 114, 124]),
        ]
    # artifitical
    buffers[0].weight = 1
    buffers[0].full = True
    buffers[1].weight = 1
    buffers[1].full = True
    buffers[2].weight = 1
    buffers[2].full = True
    return buffers

mrl98 = MRL98([0]*15, 3, 5)
mrl98.beta = 1
print(f'Y: [52, 83, 114, 143, inf]')
for phi in [0, 0.05, 0.2, 0.5, 0.7, 0.95, 1]:
    res = mrl98.output(phi, imitate_buffers())
    phi_tick = round(mrl98._calculate_phi_tick(phi), 5)
    print(f'at {phi}: phi\': {phi_tick}, {res}')