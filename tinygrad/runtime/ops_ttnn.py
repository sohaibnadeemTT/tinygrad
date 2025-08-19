from __future__ import annotations
import functools, time, ctypes, platform
from typing import Any
from tinygrad.helpers import getenv, DEBUG, prod
from tinygrad.device import Compiled, Compiler, LRUAllocator, Buffer, BufferSpec
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import UOp
from tinygrad.dtype import DType
# import ttnn backend
import ttnn

# torch tensors for buffer management
import torch

class TTNNRenderer(CStyleLanguage):
    """Custom renderer for TTNN backend"""
    device = "TTNN"
    
    def __init__(self):
        super().__init__()
        # Add TTNN-specific rendering options here
        self.lang = "C"  # Use C-style language for now
    
    # ESSENTIAL: Define workitem code generation for TTNN  
    # This fixes the KeyError: 'l' issue
    code_for_workitem = {
        "g": lambda x: f"get_global_id({x})",  # Global workitem ID 
        "l": lambda x: f"get_local_id({x})",   # Local workitem ID
        "i": lambda x: f"get_global_id({x})",  # Combined index
    }
    
    def render(self, uops: list[UOp]) -> str:
        """Render UOPs to TTNN-compatible code"""
        # For now, use the base C-style rendering
        # You can customize this to generate TTNN-specific code
        src = super().render(uops)
        
        # Add TTNN-specific headers or modifications
        ttnn_header = """
// TTNN Backend Generated Code
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Work-item ID functions (to be implemented by TTNN runtime)
static inline int get_global_id(int dim) { return 0; }  // Simplified for now
static inline int get_local_id(int dim) { return 0; }   // Simplified for now

"""
        return ttnn_header + src

class TTNNProgram:
    """Simple TTNN program wrapper"""
    def __init__(self, device: TTNNDevice, function_name: str, lib: bytes):
        self.device = device
        self.function_name = function_name
        self.lib = lib
        # In a real implementation, you'd compile lib for TTNN
        if DEBUG >= 1:
            print(f"TTNN program '{function_name}' created with {len(lib)} bytes")
    
    def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
        """Execute the program on TTNN"""
        if DEBUG >= 2:
            print(f"TTNN executing {self.function_name} with {len(bufs)} buffers")
        
        # For now, just simulate execution
        # In a real implementation, you'd execute the compiled kernel on TTNN
        start_time = time.time()
        
        # Simulate some work
        time.sleep(0.001)  # 1ms simulation
        
        exec_time = time.time() - start_time
        if DEBUG >= 1:
            print(f"TTNN executed {self.function_name} in {exec_time*1000:.2f}ms")
        
        return exec_time if wait else None

class TTNNDevice(Compiled):
    devices = []
    # Only support single device for now
    def __init__(self, device: str = ""):
        # Initialize ttnn device
        device_id = int(device.split(":")[1]) if ":" in device else 0
        self.ttnn_device = ttnn.open_device(device_id=device_id)
        TTNNDevice.devices.append(self)
        
        # Initialize the Compiled base class
        super().__init__(
            device=device,
            allocator=TTNNAllocator(self),
            renderer=TTNNRenderer(),    # Use TTNN-specific renderer
            compiler=Compiler(),        # Use basic compiler for now  
            runtime=functools.partial(TTNNProgram, self),  # TTNN program runtime
            graph=None                  # No graph support for now
        )

    def synchronize(self):
        ttnn.synchronize_device(self.ttnn_device)

class TTNNBuffer:
    def __init__(self, size, ttnn_device):
        self.size = size
        self.ttnn_device = ttnn_device
        # Always allocate a TTNN tensor immediately
        self._allocate_tensor()
        
    def _allocate_tensor(self):
        """Allocate empty TTNN tensor for this buffer"""
        if DEBUG >= 1:
            print(f"TTNNBuffer: allocating TTNN tensor of size {self.size}")
        
        # Create torch tensor with zeros, then convert to TTNN
        torch_tensor = torch.zeros(self.size, dtype=torch.uint8)
        self._tensor = ttnn.from_torch(torch_tensor, device=self.ttnn_device)
        
        if DEBUG >= 1:
            print(f"TTNNBuffer: successfully allocated TTNN tensor")
    
    def tensor(self):
        return self._tensor
        
    def _free(self):
        if self._tensor is not None:
            ttnn.deallocate(self._tensor)
            self._tensor = None
            
    def _from_buffer(self, buff: memoryview):
        """Copy data from CPU buffer into existing TTNN tensor"""
        if DEBUG >= 1:
            print(f"TTNNBuffer: copying {len(buff)} bytes from CPU to TTNN")
            
        # Create torch tensor from buffer
        torch_tensor = torch.frombuffer(buff, dtype=torch.uint8)
        
        # Replace our existing tensor with new data
        if self._tensor is not None:
            ttnn.deallocate(self._tensor)
        self._tensor = ttnn.from_torch(torch_tensor, device=self.ttnn_device)
        
        if DEBUG >= 1:
            print(f"TTNNBuffer: successfully copied data to TTNN tensor")
    
    def _to_device(self):
        """Copy data from TTNN tensor back to CPU memoryview"""
        if DEBUG >= 1:
            print(f"TTNNBuffer: copying TTNN tensor back to CPU")
            
        torch_tensor = ttnn.to_torch(self._tensor)
        return memoryview(torch_tensor.numpy())

class TTNNAllocator(LRUAllocator):
    def __init__(self, device: TTNNDevice):
        super().__init__(device)
        
    def _alloc(self, size, options):
        return TTNNBuffer(size, self.dev.ttnn_device)
        
    def _free(self, buf, options):
        buf._free()
        
    def _copyin(self, dest, src: memoryview):
        dest._from_buffer(src)
        
    def _copyout(self, dest: memoryview, src):
        # Copy data from device buffer to dest memoryview
        device_data = src._to_device()
        dest[:] = device_data