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
    def __init__(self, size, ttnn_device, dtype):
        self.size = size
        self.ttnn_device = ttnn_device
        self.dtype = dtype  # tinygrad DType
        self._tensor = None  # Lazy allocation - create when we get actual data
        
        if DEBUG >= 1:
            print(f"TTNNBuffer: created buffer of size {size} with dtype {dtype} (lazy allocation)")
    
    def _tinygrad_to_torch_dtype(self, tinygrad_dtype):
        """Convert tinygrad DType to torch dtype"""
        from tinygrad.dtype import dtypes
        mapping = {
            dtypes.float32: torch.float32,
            dtypes.float16: torch.float16,
            dtypes.bfloat16: torch.bfloat16,
            dtypes.int32: torch.int32,
            dtypes.int16: torch.int16,
            dtypes.int8: torch.int8,
            dtypes.uint8: torch.uint8,
            dtypes.bool: torch.bool,
        }
        return mapping.get(tinygrad_dtype, torch.float32)
    
    def tensor(self):
        if self._tensor is None:
            # For output buffers that haven't received data yet, create a placeholder
            torch_dtype = self._tinygrad_to_torch_dtype(self.dtype)
            if DEBUG >= 1:
                print(f"TTNNBuffer: creating placeholder tensor with dtype {self.dtype} -> {torch_dtype}")
            torch_tensor = torch.zeros(self.size, dtype=torch_dtype)
            self._tensor = ttnn.from_torch(torch_tensor, device=self.ttnn_device)
        return self._tensor
        
    def _free(self):
        if self._tensor is not None:
            ttnn.deallocate(self._tensor)
            self._tensor = None
            
    def _from_buffer(self, buff: memoryview):
        """Copy data from CPU buffer into TTNN tensor"""
        torch_dtype = self._tinygrad_to_torch_dtype(self.dtype)
        
        if DEBUG >= 1:
            print(f"TTNNBuffer: copying {len(buff)} bytes from CPU to TTNN")
            print(f"TTNNBuffer: tinygrad dtype: {self.dtype} -> torch dtype: {torch_dtype}")
            print(f"TTNNBuffer: memoryview format: '{buff.format}', itemsize: {buff.itemsize}")
        
        # Convert raw bytes to tensor with correct dtype
        # memoryview format is always 'B' (raw bytes), but we know the logical dtype
        torch_tensor = torch.frombuffer(buff, dtype=torch_dtype)
        
        if DEBUG >= 1:
            print(f"TTNNBuffer: created torch_tensor: {torch_tensor} (dtype: {torch_tensor.dtype})")
        
        # Replace/create TTNN tensor
        if self._tensor is not None:
            ttnn.deallocate(self._tensor)
        self._tensor = ttnn.from_torch(torch_tensor, device=self.ttnn_device)
        
        if DEBUG >= 1:
            print(f"TTNNBuffer: successfully created TTNN tensor")
    
    def _to_device(self):
        """Copy data from TTNN tensor back to CPU memoryview"""
        if DEBUG >= 1:
            print(f"TTNNBuffer: copying TTNN tensor back to CPU")
            
        torch_tensor = ttnn.to_torch(self._tensor)
        if DEBUG >= 1:
            print(f"TTNNBuffer: torch_tensor from TTNN: {torch_tensor} (dtype: {torch_tensor.dtype})")
        return memoryview(torch_tensor.numpy())

class TTNNAllocator(LRUAllocator):
    def __init__(self, device: TTNNDevice):
        super().__init__(device)
        
    def _alloc(self, size, options):
        # Get dtype from BufferSpec options
        dtype = getattr(options, 'dtype', None)
        
        if DEBUG >= 1:
            print(f"TTNNAllocator: allocating buffer with size {size}, dtype: {dtype}")
        
        if dtype is None:
            # Fallback - should not happen with BufferSpec enhancement
            from tinygrad.dtype import dtypes
            dtype = dtypes.float32
            if DEBUG >= 1:
                print(f"TTNNAllocator: no dtype in options, using fallback {dtype}")
        
        return TTNNBuffer(size, self.dev.ttnn_device, dtype)
        
    def _free(self, buf, options):
        buf._free()
        
    def _copyin(self, dest, src: memoryview):
        dest._from_buffer(src)
        
    def _copyout(self, dest: memoryview, src):
        # Copy data from device buffer to dest memoryview
        device_data = src._to_device()
        if DEBUG >= 1:
            print(f"TTNNAllocator: _copyout: {device_data}")
            print(f"TTNNAllocator: dest format: '{dest.format}', device_data format: '{device_data.format}'")
        
        # Cast dest to match device_data structure for assignment
        if dest.format != device_data.format:
            if DEBUG >= 1:
                print(f"TTNNAllocator: casting dest from '{dest.format}' to '{device_data.format}'")
            dest_cast = dest.cast(device_data.format)
            dest_cast[:] = device_data
        else:
            # Formats already match
            dest[:] = device_data