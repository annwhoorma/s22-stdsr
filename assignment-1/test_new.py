from mrl98 import Buffer, MRL98

def test_case1():
    '''
    mrl98's sequence has more elements than one buffer
    '''
    sequence = [1, 2, 3, 4, 5]
    mrl98 = MRL98(input_sequence=sequence, b=2, be=3)
    print(f'sequence: {sequence}')
    print(f'# of elements per buffer: {mrl98.be}')
    buffer = Buffer()
    return mrl98.new(buffer)

def test_case2():
    '''
    mrl98's sequence has less elements than one buffer
    '''
    sequence = [1, 2, 3]
    mrl98 = MRL98(input_sequence=sequence, b=1, be=3)
    mrl98.be = 4 # artificial
    print(f'sequence: {sequence}')
    print(f'# of elements per buffer: {mrl98.be}')
    buffer = Buffer()
    return mrl98.new(buffer)


print('test case 1'), print(test_case1().__dict__)
print()
print('test case 2'), print(test_case2().__dict__)