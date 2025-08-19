from tinygrad.device import Compiled, Buffer
# import ttnn backend
import ttnn

# torch tensors for buffer management
import torch

class TTNNDevice(Compiled):
    devices = []
    # Only support single device for now
    def __init__(self, device: str):
        self.ttnn_device = ttnn.Device(device_id=int(device))
        TTNNDevice.devices.append(self)

    def synchronize(self):
        ttnn.synchronize_device(self.ttnn_device)

class TTNNBuffer:
    def __init__(self, size):
        self.size = size
        self._tensor = None
    def tensor(self):
        if self._tensor is None:
            raise Exception("Tensor doesn't exist")
        return self._tensor
    def _free(self):
        ttnn.deallocate(self.tensor())
    def _from_buffer(self, buff: memoryview):
        # Create torch tensor from buffer, and then load TTNN tensor from there
        torch_tensor = torch.frombuffer(buff, dtype=torch.uint8)
        self._tensor = ttnn.from_torch(torch_tensor)
    def _to_device(self):
        torch_tensor = ttnn.to_torch(self.tensor())
        return memoryview(torch_tensor.numpy())

class TTNNAllocator(Allocator['TTNNDevice']):
    def _alloc(self, size, options):
        self._buf = TTNNBuffer(size)
        return self._buf
    def _free(self, buf, options):
        buf._free()
    def _copyin(self, dest, src: memoryview):
        dest._from_buffer(src)
    def _copyout(self, dest: memoryview, src):
        return src._to_device()
