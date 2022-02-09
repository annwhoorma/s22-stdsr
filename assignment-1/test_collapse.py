from main import MRL99, Buffer

buffers = [
    Buffer([12, 52, 72, 102, 132]),
    Buffer([23, 33, 83, 143, 153]),
    Buffer([44, 64, 94, 114, 124]),
    ]

buffers[0].weight = 2
buffers[0].full = True
buffers[1].weight = 3
buffers[1].full = True
buffers[2].weight = 4
buffers[2].full = True

mrl99 = MRL99([0]*15, 3)
res = mrl99.collapse(buffers)

for buffer in buffers:
    print(buffer.__dict__)