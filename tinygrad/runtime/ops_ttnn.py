from __future__ import annotations
import functools, time, ctypes, platform
from typing import Any
from tinygrad.helpers import getenv, DEBUG, prod
from tinygrad.device import Compiled, Compiler, LRUAllocator, Buffer, BufferSpec
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import UOp
from tinygrad.dtype import DType

import json
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
    """TTNN program wrapper that executes a JSON instruction array describing tensor ops."""
    def __init__(self, device: TTNNDevice, function_name: str, lib: bytes):
        self.device = device
        self.function_name = function_name
        self.lib = lib
        if DEBUG >= 1:
            print(f"TTNN program '{function_name}' created with {len(lib)} bytes")

    def execute_instruction(self, instruction: dict, tensors: dict):
        op = instruction.get("op")
        if op == "store":
            # Allocate a tensor and store it in the tensors dictionary
            tensor_info = instruction["tensor"]
            tensor_name = tensor_info["name"]
            size = tensor_info["size"]
            dtype = tensor_info["dtype"]
            tensors[tensor_name] = TTNNBuffer(size, self.device.ttnn_device) # May need to adjust this based on actual tensor creation
            if DEBUG >= 1:
                print(f"Allocated tensor '{tensor_name}' with size {size} and dtype {dtype}")
        elif op == "add":
            # Perform elementwise addition
            a_name = instruction["inputs"][0]
            b_name = instruction["inputs"][1]
            out_name = instruction["output"]
            a_tensor = tensors[a_name].tensor()
            b_tensor = tensors[b_name].tensor()
            out_tensor = tensors[out_name].tensor()
            out_tensor = ttnn.add(a_tensor, b_tensor)
            if DEBUG >= 1:
                print(f"Performed addition: {a_name} + {b_name} -> {out_name}")
        elif op == "if":
            # Conditional execution
            condition = instruction["condition"]
            if condition:  # Replace with actual condition logic as needed
                if DEBUG >= 1:
                    print(f"Condition met, executing nested instructions")
                for nested_instruction in instruction["instructions"]:
                    self.execute_instruction(nested_instruction, tensors)
            else:
                if DEBUG >= 1:
                    print(f"Condition not met, skipping nested instructions")
        else:
            raise ValueError(f"Unsupported operation: {op}")

    def __call__(self, json_array: str, wait=False):
        """Execute the program on TTNN"""
        if DEBUG >= 2:
            print(f"TTNN executing {self.function_name} with JSON array")
        instructions = json.loads(json_array)
        tensors = {}
        for instruction in instructions:
            self.execute_instruction(instruction, tensors)
        if DEBUG >= 1:
            print(f"TTNN program '{self.function_name}' executed successfully")
        return None

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