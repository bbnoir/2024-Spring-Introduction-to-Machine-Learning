import numpy as np
import os

def gen_demo():
    # sample lines from the file to generate the demo
    with open('HW1.csv', 'r') as f:
        # copy the header
        header = next(f)
        # sample 8000 lines ignoring the header
        lines = [next(f) for _ in range(8000)]
        # write the sampled lines to a new file
        with open('HW1_demo.csv', 'w') as f:
            f.write(header)
            f.write(''.join(lines))

if __name__ == '__main__':
    gen_demo()
    print('Demo generated successfully')




