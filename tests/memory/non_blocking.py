# https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html

import torch
from torch.utils.benchmark import Timer


def timer(cmd):
    median = (
        Timer(cmd, globals=globals())
        .adaptive_autorange(min_run_time=1.0, max_run_time=20.0)
        .median
        * 1000
    )
    print(f"{cmd}: {median: 4.4f} ms")
    return median


# A simple loop that copies all tensors to cuda
def copy_to_device(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.to("cuda:0"))
    return result


# A loop that copies all tensors to cuda asynchronously
def copy_to_device_nonblocking(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.to("cuda:0", non_blocking=True))
    # We need to synchronize
    torch.cuda.synchronize()
    return result


# Create a list of tensors
tensors = [torch.randn(1000) for _ in range(1000)]
tensors_pinned = [torch.randn(1000, pin_memory=True) for _ in range(1000)]
to_device = timer("copy_to_device(*tensors)")
to_device_nonblocking = timer("copy_to_device_nonblocking(*tensors)")
to_device_pinned = timer("copy_to_device(*tensors_pinned)")
to_device_pinned_nonblocking = timer("copy_to_device_nonblocking(*tensors_pinned)")
