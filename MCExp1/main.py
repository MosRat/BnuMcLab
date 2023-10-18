# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import numpy as np

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_capability())
# print(torch.cuda.device.mro())

t1 = torch.tensor([
    # np.arange(1,9)
    1,
    2,
    4,
]).reshape(-1, 1).to('cuda')
print(t1)
t2 = torch.arange(1, 10).reshape(3, -1).to('cuda')
# print(torch.nn.Conv2d())
print(t1 + t2)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
