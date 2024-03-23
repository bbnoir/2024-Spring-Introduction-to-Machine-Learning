import numpy as np
import os

def gen_demo():
    # sample lines from the file to generate the demo
    with open('HW1.csv', 'r') as f:
        # copy the header
        header = next(f)
        # randomly sample 8000 lines
        lines = np.random.choice(list(f), 8000, replace=False)
        # write the sampled lines to a new file
        with open('HW1_demo.csv', 'w') as f:
            f.write(header)
            f.write(''.join(lines))

if __name__ == '__main__':
    gen_demo()
    print('Demo generated successfully')




