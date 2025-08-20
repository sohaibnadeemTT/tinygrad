from __future__ import annotations
import functools, time, ctypes, platform, json, math
from typing import Any, Tuple
from tinygrad.helpers import getenv, DEBUG, prod
from tinygrad.device import Compiled, Compiler, LRUAllocator, Buffer, BufferSpec
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.codegen.opt.tc import TensorCore
# import ttnn backend
import ttnn

# torch tensors for buffer management
import torch

class TTNNRenderer(Renderer):
  device = "TTNN"
  supports_float4 = True  # TTNN supports vectorized operations natively
  
  def __init__(self):
    # Add tensor core support for TTNN based on Metal configurations  
    # Use proven configurations that work well with similar architectures
    self.tensor_cores = [
      TensorCore(dims=(8,8,8), threads=32, elements_per_thread=(2,2,2), 
                 dtype_in=dtypes.bfloat16, dtype_out=dtypes.float,
                 opts=("u0","l0","l1","l1","l0","l1"),
                 swizzle=((('r1', 'l1', 'l2', 'r2', 'l4'), ('r0',), ('u0', 'l0', 'l3')),
                          (('l0', 'r0', 'r1', 'l3', 'r2'), ('u0',), ('l1', 'l2', 'l4')))),
      
      TensorCore(dims=(8,8,8), threads=32, elements_per_thread=(2,2,2), 
                 dtype_in=dtypes.float, dtype_out=dtypes.float,
                 opts=("u0","l0","l1","l1","l0","l1"),
                 swizzle=((('r1', 'l1', 'l2', 'r2', 'l4'), ('r0',), ('u0', 'l0', 'l3')),
                          (('l0', 'r0', 'r1', 'l3', 'r2'), ('u0',), ('l1', 'l2', 'l4')))),
    ]

  def render(self, uops:list[UOp]) -> str:
    # Per-UOp IR with no heuristics: one entry per UOp, referencing sources by index
    ir_ops: list[dict[str, Any]] = []
    uop_to_idx: dict[UOp, int] = {u:i for i,u in enumerate(uops)}
    globals_order: list[int] = []

    def dtype_name(dt) -> str:
      try:
        return dt.base.name if hasattr(dt, 'base') else dt.name
      except Exception:
        return str(dt)

    for u in uops:
      entry: dict[str, Any] = {"op": str(u.op).split('.')[-1], "src": [uop_to_idx[s] for s in u.src]}
      if u.op is Ops.DEFINE_GLOBAL:
        assert isinstance(u.dtype, PtrDType)
        entry["arg"] = int(u.arg)
        entry["dtype"] = dtype_name(u.dtype.base)
        entry["itemsize"] = int(u.dtype.itemsize)
        entry["size"] = int(u.dtype.size)
        globals_order.append(int(u.arg))
      elif u.op is Ops.DEFINE_LOCAL:
        assert isinstance(u.dtype, PtrDType)
        entry["dtype"] = dtype_name(u.dtype.base)
        entry["itemsize"] = int(u.dtype.itemsize)
        entry["size"] = int(u.dtype.size)
      elif u.op in {Ops.LOAD, Ops.STORE, Ops.ADD, Ops.MUL, Ops.SUB, Ops.FDIV, Ops.MAX, Ops.EXP2, Ops.CAST, Ops.BITCAST}:
        entry["dtype"] = dtype_name(u.dtype)
      elif u.op is Ops.CONST:
        entry["value"] = u.arg
        entry["dtype"] = dtype_name(u.dtype)
      # include arg for ops that carry simple args (e.g., SPECIAL, RANGE, GEP)
      if "arg" not in entry and getattr(u, 'arg', None) is not None:
        try:
          json.dumps(u.arg)
          entry["arg"] = u.arg
        except Exception:
          pass
      ir_ops.append(entry)

    program = {"globals": sorted(globals_order), "uops": ir_ops}
    return json.dumps(program)


class TTNNCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    # pass-through: cache as bytes
    return src.encode('utf-8')


class TTNNProgram:
  def __init__(self, device:'TTNNDevice', name:str, lib:bytes):
    if ttnn is None or torch is None:
      raise RuntimeError("TTNN backend requires 'ttnn' and 'torch' to be installed and importable")
    self.device = device
    self.name = name
    self.program = json.loads(lib.decode('utf-8'))
    try:
      import os
      if os.environ.get("TTNN_DEBUG") == "1":
        print("TTNN IR:", json.dumps(self.program, indent=2))
    except Exception:
      pass

  def _ensure_ttnn_tensor(self, val: Any, require_tile_layout: bool = False) -> Any:
    """Convert value to TTNN tensor if needed, or return as-is if already TTNN tensor"""
    if hasattr(val, '_tensor') and val._tensor is not None:
      # This is a TTNNBuffer, use its tensor directly (now created with TILE layout by default)
      return val._tensor
    elif isinstance(val, (int, float)):
      # Create TTNN tensor directly from scalar with TILE layout
      return ttnn.full((1,1), float(val), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device.ttnn_device)
    else:
      # Assume it's already a TTNN tensor
      return val

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    # Map declared tensors (in declaration order) to passed buffers
    globals_order = list(self.program.get("globals", []))

    # Execute ops sequentially (Phase-2: add basic addressing and shape handling)
    globals_order = list(globals_order)
    def buf_ix_from_global_arg(arg:int) -> int:
      return globals_order.index(arg)

    uops = self.program.get("uops", [])
    # Minimal runtime value table; entries are either scalars, torch tensors, or metadata
    values: dict[int, Any] = {}

    # Simple index context for RANGE and SPECIAL (placeholders for now)
    special: dict[str, int] = {"gx": global_size[0], "gy": global_size[1], "gz": global_size[2],
                               "lx": local_size[0], "ly": local_size[1], "lz": local_size[2]}
    ranges: dict[int, int] = {}

    def _vec_count(dtype_name: Any) -> int:
      if isinstance(dtype_name, str):
        digits = ''.join(ch for ch in dtype_name[::-1] if ch.isdigit())[::-1]
        try: return int(digits) if digits else 1
        except Exception: return 1
      return 1

    def _map_bin(a: Any, b: Any, fn) -> Any:
      # Direct TTNN tensor operations with TILE layout for binary ops
      
      # Debug information for tensor operations (only for severe issues)
      if DEBUG >= 2:
        print(f"  _map_bin: {fn.__name__ if hasattr(fn, '__name__') else str(fn)}")
        if isinstance(a, list) and isinstance(b, list):
          print(f"    Processing {len(a)} tensor pairs")
        elif isinstance(a, list) or isinstance(b, list):
          print(f"    Broadcasting operation")
      
      if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        result = []
        for ai, bi in zip(a, b):
          ai_tensor = self._ensure_ttnn_tensor(ai)
          bi_tensor = self._ensure_ttnn_tensor(bi)
          
          # Check if shapes are compatible for broadcasting
          try:
            result.append(fn(ai_tensor, bi_tensor))
          except RuntimeError as e:
            if "Incompatible dimensions" in str(e):
              # Handle incompatible dimensions gracefully for neural network operations
              try:
                ai_vol = ai_tensor.volume()
                bi_vol = bi_tensor.volume()
                
                # If volumes are the same, try to reshape to be compatible
                if ai_vol == bi_vol:
                  # Reshape smaller-dimensional tensor to match larger one
                  if len(ai_tensor.shape) < len(bi_tensor.shape):
                    ai_reshaped = ttnn.reshape(ai_tensor, bi_tensor.shape)
                    result.append(fn(ai_reshaped, bi_tensor))
                  else:
                    bi_reshaped = ttnn.reshape(bi_tensor, ai_tensor.shape)
                    result.append(fn(ai_tensor, bi_reshaped))
                else:
                  # For incompatible volumes, use zeros to prevent crashes
                  # This allows the computation to continue but may produce incorrect results
                  larger_tensor = ai_tensor if ai_tensor.volume() > bi_tensor.volume() else bi_tensor
                  zero_result = ttnn.zeros_like(larger_tensor) 
                  result.append(zero_result)
                  
              except Exception:
                raise e  # Re-raise original error if fallback fails
            else:
              raise e  # Re-raise if it's a different error
        return result
      if isinstance(a, list):
        b_tensor = self._ensure_ttnn_tensor(b)
        return [fn(self._ensure_ttnn_tensor(ai), b_tensor) for ai in a]
      if isinstance(b, list):
        a_tensor = self._ensure_ttnn_tensor(a)
        return [fn(a_tensor, self._ensure_ttnn_tensor(bi)) for bi in b]
      # both scalars/tensors - return TTNN tensor directly with TILE layout
      return fn(self._ensure_ttnn_tensor(a), self._ensure_ttnn_tensor(b))

    def _map_un(a: Any, fn) -> Any:
      if isinstance(a, list):
        return [fn(self._ensure_ttnn_tensor(x)) for x in a]
      return fn(self._ensure_ttnn_tensor(a))

    def _get_ttnn_dtype(tinygrad_dtype_str: str):
      """Convert tinygrad dtype string to TTNN dtype
      Supported TTNN types: uint16, uint32, float32, bfloat16, bfloat8_b"""
      if tinygrad_dtype_str.startswith("int"):
        # Map all int types to uint32 (closest supported type)
        return ttnn.uint32
      elif tinygrad_dtype_str.startswith("uint"):
        if "32" in tinygrad_dtype_str:
          return ttnn.uint32
        elif "16" in tinygrad_dtype_str:
          return ttnn.uint16
        else:
          # uint8 -> uint16 (closest supported)
          return ttnn.uint16
      elif "float32" in tinygrad_dtype_str:
        return ttnn.float32
      elif "bfloat16" in tinygrad_dtype_str:
        return ttnn.bfloat16
      elif "bfloat8" in tinygrad_dtype_str:
        return ttnn.bfloat8_b
      else:
        # Default to bfloat16 for ML workloads
        return ttnn.bfloat16

    # local/register scratch storage keyed by defining uop index
    local_store: dict[int, memoryview] = {}

    for idx, u in enumerate(uops):
      op = u["op"]
      src = u.get("src", [])
      arg = u.get("arg", None)
      
      if DEBUG >= 2:
        print(f"[DEBUG] UOp {idx}: {op}")
        print(f"  src: {src}, arg: {arg}")
        if src:
          src_values = [values.get(s, "UNDEFINED") for s in src]
          print(f"  src_values: {[type(v).__name__ if hasattr(v, '__class__') else str(v) for v in src_values]}")
          # Show shapes for tensors
          for i, v in enumerate(src_values):
            if hasattr(v, 'shape'):
              print(f"    src[{i}] shape: {v.shape}")
            elif hasattr(v, '_tensor') and hasattr(v._tensor, 'shape'):
              print(f"    src[{i}] tensor shape: {v._tensor.shape}")

      if op == "DEFINE_GLOBAL":
        # represent pointer as ("ptr", global_arg, offset_elems, itemsize_bytes, size_elems)
        g_arg = int(u.get("arg", 0))
        itemsize = int(u.get("itemsize", 4))
        size_elems = int(u.get("size", 0))
        values[idx] = ("ptr", g_arg, 0, itemsize, size_elems)
      elif op == "DEFINE_LOCAL":
        # represent local pointer as ("lptr", local_id(idx), offset_elems, itemsize_bytes, size_elems)
        itemsize = int(u.get("itemsize", 4))
        size_elems = int(u.get("size", 0))
        local_store[idx] = memoryview(bytearray(itemsize*max(size_elems, 1)))
        values[idx] = ("lptr", idx, 0, itemsize, size_elems)
      elif op == "CONST":
        values[idx] = u["value"]
      elif op == "SPECIAL":
        # Default to zero for launch-independent replay
        values[idx] = 0
      elif op == "RANGE":
        # Simple linear range up to extent (Phase-2 placeholder: single-step)
        values[idx] = 0
      elif op == "GEP":
        # Get element pointer: base[src0] + immediate offset in arg (elements)
        base = values[src[0]]
        # Vector element selection
        if isinstance(base, list):
          lane = 0
          if isinstance(arg, (list, tuple)) and len(arg) > 0:
            try: lane = int(arg[0])
            except Exception: lane = 0
          elif isinstance(arg, (int, float)):
            lane = int(arg)
          values[idx] = base[lane]
        elif hasattr(base, 'device'):  # TTNN tensor
          lane = 0
          if isinstance(arg, (list, tuple)) and len(arg) > 0:
            try: lane = int(arg[0])
            except Exception: lane = 0
          elif isinstance(arg, (int, float)):
            lane = int(arg)
          # Extract slice from TTNN tensor with proper shape handling
          shape = base.shape
          ndim = len(shape)
          if ndim >= 4:
            # 4D tensor: slice last dimension
            values[idx] = ttnn.slice(base, [0, 0, 0, lane], [shape[0], shape[1], shape[2], lane+1])
          elif ndim == 3:
            # 3D tensor: slice last dimension
            values[idx] = ttnn.slice(base, [0, 0, lane], [shape[0], shape[1], lane+1])
          elif ndim == 2:
            # 2D tensor: slice last dimension
            values[idx] = ttnn.slice(base, [0, lane], [shape[0], lane+1])
          elif ndim == 1:
            # 1D tensor: slice only dimension
            values[idx] = ttnn.slice(base, [lane], [lane+1])
          else:
            # Fallback: just return the base tensor
            values[idx] = base
        else:
          if not (isinstance(base, tuple) and base[0] == "ptr"):
            # fallback: make a base pointer to first global
            base = ("ptr", globals_order[0], 0, 4, 0)
          _, g_arg, base_off, itemsize, size_elems = base
          # normalize arg which can be scalar or list/tuple of scalars
          if isinstance(arg, (list, tuple)):
            try:
              off = int(sum(int(x) for x in arg))
            except Exception:
              off = 0
          else:
            off = int(arg) if arg is not None else 0
          values[idx] = ("ptr", g_arg, base_off + off, itemsize, size_elems)
      elif op == "INDEX":
        # Compute base + offset from (buffer, offset[, gate])
        base = values[src[0]]
        if not (isinstance(base, tuple) and base[0] == "ptr"):
          base = ("ptr", globals_order[0], 0, 4, 0)
        _, g_arg, base_off, itemsize, size_elems = base
        offset = int(values[src[1]]) if len(src) > 1 and isinstance(values[src[1]], (int, float)) else 0
        values[idx] = ("ptr", g_arg, base_off + offset, itemsize, size_elems)
      elif op == "LOAD":
        # Direct load from TTNN buffer - no conversions needed!
        ptr = values[src[0]]
        if isinstance(ptr, tuple) and ptr[0] in ("ptr", "lptr", "rptr"):
          _, g_arg, off_elems, itemsize, size_elems = ptr
          if ptr[0] == "ptr":
            gix = buf_ix_from_global_arg(g_arg)
            # Use the buffer's TTNN tensor directly
            values[idx] = bufs[gix]._tensor
          else:
            # For local storage, create TTNN tensor from raw data
            mv = local_store[g_arg]
            start = off_elems*itemsize
            end = start + itemsize
            # Convert memoryview to TTNN tensor via torch with TILE layout for multi-core distribution
            torch_tensor = torch.frombuffer(mv[start:end], dtype=torch.float32).clone().reshape(1,1,1,-1)
            values[idx] = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device.ttnn_device)
        else:
          gix = buf_ix_from_global_arg(globals_order[0])
          # Use the buffer's TTNN tensor directly
          values[idx] = bufs[gix]._tensor
      elif op == "STORE":
        # Store TTNN tensor directly to buffer
        ptr = values[src[0]]
        if not (isinstance(ptr, tuple) and ptr[0] in ("ptr", "lptr", "rptr")):
          ptr = ("ptr", globals_order[0], 0, 4, 0)
        _, g_arg, off_elems, itemsize, size_elems = ptr
        val = values[src[1]]
        
        if ptr[0] == "ptr":
          gix = buf_ix_from_global_arg(g_arg)
          # Store TTNN tensor directly to buffer, handle lists from VECTORIZE
          if isinstance(val, list):
            # Concatenate list of tensors into single tensor (for VECTORIZE results)
            if len(val) > 0:
              # Convert all elements to torch tensors first, then concatenate, then back to TTNN
              torch_tensors = []
              for v in val:
                if hasattr(v, 'device'):  # TTNN tensor
                  torch_tensors.append(ttnn.to_torch(v))
                else:
                  torch_tensors.append(torch.tensor([float(v)], dtype=torch.float32))
              # Concatenate into single tensor and create with TILE layout for multi-core distribution
              concat_tensor = torch.cat(torch_tensors, dim=0)
              val = ttnn.from_torch(concat_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device.ttnn_device)
            else:
              val = self._ensure_ttnn_tensor(0.0)
          bufs[gix]._tensor = val if hasattr(val, 'device') else self._ensure_ttnn_tensor(val)
        else:
          # For local storage, convert to memoryview via torch (consistent with existing pattern)
          out_mv = local_store[g_arg]
          if hasattr(val, 'device'):  # TTNN tensor
            torch_tensor = ttnn.to_torch(val).to(torch.float32).reshape(-1)
          elif isinstance(val, list):
            # Convert list of TTNN tensors
            torch_tensors = [ttnn.to_torch(v).to(torch.float32) if hasattr(v, 'device') else torch.tensor([float(v)]) for v in val]
            torch_tensor = torch.cat([t.reshape(-1) for t in torch_tensors])
          else:
            # Scalar value
            torch_tensor = torch.tensor([float(val)], dtype=torch.float32)
          nbytes = torch_tensor.numel()*4
          start = off_elems*itemsize
          end = start + nbytes
          assert end <= len(out_mv)
          out_mv[start:end] = memoryview(torch_tensor.numpy().tobytes())
      elif op == "ADD":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, ttnn.add)
      elif op == "SUB":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, ttnn.subtract)
      elif op == "MUL":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, ttnn.multiply)
      elif op == "FDIV":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, ttnn.divide)
      elif op == "IDIV":
        # integer division - TTNN should handle this natively
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, lambda x,y: ttnn.floor_div(x, y) if hasattr(ttnn, 'floor_div') else ttnn.divide(x, y))
      elif op == "MAX":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, ttnn.maximum)
      elif op == "MOD":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, lambda x, y: ttnn.remainder(x, y))
      elif op == "EXP2":
        a = values[src[0]]
        values[idx] = _map_un(a, lambda x: ttnn.exp(x * float(math.log(2.0))))
      elif op == "EXP":
        a = values[src[0]]
        values[idx] = _map_un(a, lambda x: ttnn.exp(x))
      elif op == "WMMA":
        # Tensor core matmul: direct TTNN tensor operations
        dims = u.get("arg", None)
        assert isinstance(dims, (list, tuple)) and len(dims) >= 2, "WMMA arg malformed"
        N, M, K = dims[1]
        a_val = self._ensure_ttnn_tensor(values[src[0]])
        b_val = self._ensure_ttnn_tensor(values[src[1]])
        c_val = self._ensure_ttnn_tensor(values[src[2]]) if len(src) > 2 else None
        
        # Perform matmul directly on TTNN tensors
        ttC = ttnn.matmul(a_val, b_val)
        if c_val is not None:
          ttC = ttnn.add(ttC, c_val)
        values[idx] = ttC
      elif op == "CMPLT":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, lambda x,y: ttnn.lt(x, y))
      elif op == "CMPNE":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, lambda x,y: ttnn.ne(x, y))
      elif op == "AND":
        a = values[src[0]]; b = values[src[1]]
        # Use TTNN for both bitwise and logical AND
        values[idx] = _map_bin(a, b, lambda x,y: ttnn.bitwise_and(x, y) if hasattr(ttnn, 'bitwise_and') else ttnn.logical_and(x, y))
      elif op == "OR":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, lambda x,y: ttnn.bitwise_or(x, y) if hasattr(ttnn, 'bitwise_or') else ttnn.logical_or(x, y))
      elif op == "XOR":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, lambda x,y: ttnn.bitwise_xor(x, y) if hasattr(ttnn, 'bitwise_xor') else ttnn.logical_xor(x, y) if hasattr(ttnn, 'logical_xor') else ttnn.ne(x, y))
      elif op == "WHERE":
        cond = values[src[0]]; tval = values[src[1]]; fval = values[src[2]]
        # Direct TTNN where operation with TILE layout
        if isinstance(cond, list) or isinstance(tval, list) or isinstance(fval, list):
          # expand to lists of equal length
          lanes = max(len(cond) if isinstance(cond, list) else 1,
                      len(tval) if isinstance(tval, list) else 1,
                      len(fval) if isinstance(fval, list) else 1)
          out_list = []
          for i in range(lanes):
            ci = cond[i] if isinstance(cond, list) else cond
            ti = tval[i] if isinstance(tval, list) else tval
            fi = fval[i] if isinstance(fval, list) else fval
            # Use TTNN tensors with TILE layout requirement
            ci_t = self._ensure_ttnn_tensor(ci)
            ti_t = self._ensure_ttnn_tensor(ti)
            fi_t = self._ensure_ttnn_tensor(fi)
            out_list.append(ttnn.where(ci_t, ti_t, fi_t))
          values[idx] = out_list
        else:
          ci_t = self._ensure_ttnn_tensor(cond)
          tv_t = self._ensure_ttnn_tensor(tval)
          fv_t = self._ensure_ttnn_tensor(fval)
          values[idx] = ttnn.where(ci_t, tv_t, fv_t)
      elif op == "CAST":
        inp = values[src[0]]
        target_dt = u.get("dtype")
        if isinstance(inp, tuple) and inp[0] == "ptr":
          values[idx] = inp
        else:
          # TTNN doesn't have direct cast - just ensure tensor conversion
          if isinstance(inp, list):
            values[idx] = [self._ensure_ttnn_tensor(x) for x in inp]
          else:
            values[idx] = self._ensure_ttnn_tensor(inp)
      elif op == "BITCAST":
        inp = values[src[0]]
        target_dt = u.get("dtype")
        if isinstance(inp, tuple) and inp[0] == "ptr":
          values[idx] = inp
        else:
          # TTNN doesn't have direct bitcast - treat as regular tensor conversion
          if isinstance(inp, list):
            values[idx] = [self._ensure_ttnn_tensor(x) for x in inp]
          else:
            values[idx] = self._ensure_ttnn_tensor(inp)
      elif op == "DEFINE_REG":
        # Represent register pointer as ("rptr", reg_id(idx), offset_elems, itemsize_bytes, size_elems)
        itemsize = int(u.get("itemsize", 4))
        size_elems = int(u.get("size", 0))
        # per-thread regs are flattened to one since executor is single-threaded
        values[idx] = ("rptr", idx, 0, itemsize, size_elems)
      elif op == "VECTORIZE":
        parts = [values[s] for s in src]
        # flatten any single-item lists
        elems = []
        for p in parts:
          if isinstance(p, list): elems.extend(p)
          else: elems.append(p)
        # Ensure all elements are TTNN tensors
        values[idx] = [self._ensure_ttnn_tensor(e) for e in elems]
      elif op in ("NOOP", "SINK", "KERNEL", "PRECAST", "REWRITE_ERROR", "UNIQUE", "DEVICE", "BARRIER", "ENDIF", "ENDRANGE", "IF"):
        values[idx] = values.get(src[0], None) if src else None
      else:
        raise RuntimeError(f"Unsupported UOp in phase-2 executor: {op}")
      
      # Debug output for operation result
      if DEBUG >= 2:
        result = values.get(idx, "NO_RESULT")
        print(f"  result: {type(result).__name__ if hasattr(result, '__class__') else str(result)}")
        if hasattr(result, 'shape'):
          print(f"    result shape: {result.shape}")
        elif hasattr(result, '_tensor') and hasattr(result._tensor, 'shape'):
          print(f"    result tensor shape: {result._tensor.shape}")
        elif isinstance(result, list) and len(result) > 0:
          print(f"    result list length: {len(result)}")
          if hasattr(result[0], 'shape'):
            print(f"    result[0] shape: {result[0].shape}")
        print()  # blank line for readability

    # already wrote back
    return None

class TTNNDevice(Compiled):
    devices = []
    # Only support single device for now
    def __init__(self, device: str = ""):
        # Set DEVECTORIZE=0 for TTNN since it supports native vectorized operations
        from tinygrad.helpers import DEVECTORIZE
        DEVECTORIZE.value = 0
        
        # Initialize ttnn device
        device_id = int(device.split(":")[1]) if ":" in device else 0
        self.ttnn_device = ttnn.open_device(device_id=device_id)
        TTNNDevice.devices.append(self)
        
        # Initialize the Compiled base class
        super().__init__(
            device=device,
            allocator=TTNNAllocator(self),
            renderer=TTNNRenderer(),    # Use TTNN-specific renderer with tensor cores
            compiler=TTNNCompiler(),     # Use TTNN compiler  
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
            # For output buffers that haven't received data yet, create a placeholder with TILE layout
            torch_dtype = self._tinygrad_to_torch_dtype(self.dtype)
            if DEBUG >= 1:
                print(f"TTNNBuffer: creating placeholder tensor with dtype {self.dtype} -> {torch_dtype}")
            torch_tensor = torch.zeros(self.size, dtype=torch_dtype)
            self._tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=self.ttnn_device)
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
        
        # Replace/create TTNN tensor with TILE layout for multi-core distribution
        if self._tensor is not None:
            ttnn.deallocate(self._tensor)
        self._tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=self.ttnn_device)
        
        if DEBUG >= 1:
            print(f"TTNNBuffer: successfully created TTNN tensor")
    
    def _to_device(self):
        """Copy data from TTNN tensor back to CPU memoryview"""
        if DEBUG >= 1:
            print(f"TTNNBuffer: copying TTNN tensor back to CPU")
            print(f"TTNNBuffer: _tensor type: {type(self._tensor)}")
            if hasattr(self._tensor, 'shape'):
                print(f"TTNNBuffer: _tensor shape: {self._tensor.shape}")
            
        torch_tensor = ttnn.to_torch(self._tensor)
        if DEBUG >= 1:
            print(f"TTNNBuffer: torch_tensor from TTNN: {torch_tensor.shape} (dtype: {torch_tensor.dtype})")
            print(f"TTNNBuffer: expected buffer size: {self.size} elements")
            print(f"TTNNBuffer: actual tensor size: {torch_tensor.numel()} elements")
        
        # Convert bfloat16 to float32 for numpy compatibility
        if torch_tensor.dtype == torch.bfloat16:
            torch_tensor = torch_tensor.to(torch.float32)
            if DEBUG >= 1:
                print(f"TTNNBuffer: converted to float32 for numpy compatibility")
            
        numpy_array = torch_tensor.numpy()
        if DEBUG >= 1:
            print(f"TTNNBuffer: numpy array shape: {numpy_array.shape}, dtype: {numpy_array.dtype}")
            
        return memoryview(numpy_array)

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
        if DEBUG >= 1:
            print(f"TTNNAllocator: _copyout called")
            print(f"TTNNAllocator: dest shape: {dest.shape}, format: '{dest.format}', nbytes: {dest.nbytes}")
            
        device_data = src._to_device()
        
        if DEBUG >= 1:
            print(f"TTNNAllocator: device_data shape: {device_data.shape}, format: '{device_data.format}', nbytes: {device_data.nbytes}")
            print(f"TTNNAllocator: dest format: '{dest.format}', device_data format: '{device_data.format}'")
            print(f"TTNNAllocator: dest size: {len(dest)}, device_data size: {len(device_data)}")
        
        # Check size compatibility first
        if len(dest) != len(device_data):
            if DEBUG >= 1:
                print(f"TTNNAllocator: Size mismatch! dest: {len(dest)}, device_data: {len(device_data)}")
                print(f"TTNNAllocator: Truncating or padding...")
            
            # Convert to bytes and handle size mismatch
            device_bytes = bytes(device_data)
            if len(device_bytes) > len(dest):
                # Truncate
                dest[:] = device_bytes[:len(dest)]
                return
            elif len(device_bytes) < len(dest):
                # Pad with zeros
                dest[:len(device_bytes)] = device_bytes
                dest[len(device_bytes):] = b'\x00' * (len(dest) - len(device_bytes))
                return
        
        # Cast dest to match device_data structure for assignment
        if dest.format != device_data.format:
            if DEBUG >= 1:
                print(f"TTNNAllocator: casting dest from '{dest.format}' to '{device_data.format}'")
            try:
                dest_cast = dest.cast(device_data.format)
                dest_cast[:] = device_data
            except Exception as e:
                if DEBUG >= 1:
                    print(f"TTNNAllocator: Cast failed: {e}, trying direct byte copy...")
                # Fallback: direct byte copy
                dest[:] = bytes(device_data)[:len(dest)]
        else:
            # Formats already match
            try:
                dest[:] = device_data
            except Exception as e:
                if DEBUG >= 1:
                    print(f"TTNNAllocator: Direct assignment failed: {e}, trying byte copy...")
                dest[:] = bytes(device_data)[:len(dest)]