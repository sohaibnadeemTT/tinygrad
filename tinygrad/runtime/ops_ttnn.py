from tinygrad.device import Compiled
# import ttnn backend
import ttnn

class TTNNDevice(Compiled):
    devices = []
    # Only support single device for now
    def __init__(self, device: str):
        self.ttnn_device = ttnn.Device(device_id=int(device))
        TTNNDevice.devices.append(self)

    def synchronize(self):
        ttnn.synchronize_device(self.ttnn_device)
