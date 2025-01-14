# https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html

# # x1 NVIDIA A100-SXM4-40GB, IC Cluster
# copy_to_device(*tensors):  17.6453 ms
# copy_to_device_nonblocking(*tensors):  9.7758 ms
# copy_to_device(*tensors_pinned):  17.7954 ms
# copy_to_device_nonblocking(*tensors_pinned):  9.6278 ms
#
#
# # x1 NVIDIA GH200
# copy_to_device(*tensors):  10.9591 ms
# copy_to_device_nonblocking(*tensors):  5.0752 ms
# copy_to_device(*tensors_pinned):  10.8342 ms
# copy_to_device_nonblocking(*tensors_pinned):  4.7570 ms


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
