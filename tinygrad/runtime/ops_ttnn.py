from __future__ import annotations
import functools, time, ctypes, platform, json, math
from typing import Any, Tuple
from tinygrad.helpers import getenv, DEBUG, prod
from tinygrad.device import Compiled, Compiler, LRUAllocator, Buffer, BufferSpec
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import DType, PtrDType
# import ttnn backend
import ttnn

# torch tensors for buffer management
import torch

class TTNNRenderer(Renderer):
  device = "TTNN"

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
      # This is a TTNNBuffer, use its tensor directly
      tensor = val._tensor
      # Convert to TILE layout if required for operations
      if require_tile_layout and hasattr(tensor, 'get_layout'):
        if tensor.get_layout() != ttnn.TILE_LAYOUT:
          return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
      return tensor
    elif isinstance(val, (int, float)):
      # Create TTNN tensor directly from scalar in TILE layout
      layout = ttnn.TILE_LAYOUT if require_tile_layout else ttnn.ROW_MAJOR_LAYOUT
      return ttnn.full((1,1,1,1), float(val), dtype=ttnn.bfloat16, layout=layout, device=self.device.ttnn_device)
    else:
      # Assume it's already a TTNN tensor, convert layout if needed
      if require_tile_layout and hasattr(val, 'get_layout'):
        if val.get_layout() != ttnn.TILE_LAYOUT:
          return ttnn.to_layout(val, ttnn.TILE_LAYOUT)
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
      if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        return [fn(self._ensure_ttnn_tensor(ai, require_tile_layout=True), self._ensure_ttnn_tensor(bi, require_tile_layout=True)) for ai, bi in zip(a, b)]
      if isinstance(a, list):
        b_tensor = self._ensure_ttnn_tensor(b, require_tile_layout=True)
        return [fn(self._ensure_ttnn_tensor(ai, require_tile_layout=True), b_tensor) for ai in a]
      if isinstance(b, list):
        a_tensor = self._ensure_ttnn_tensor(a, require_tile_layout=True)
        return [fn(a_tensor, self._ensure_ttnn_tensor(bi, require_tile_layout=True)) for bi in b]
      # both scalars/tensors - return TTNN tensor directly with TILE layout
      return fn(self._ensure_ttnn_tensor(a, require_tile_layout=True), self._ensure_ttnn_tensor(b, require_tile_layout=True))

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

    def _map_bin_ttnn_bool(a: Any, b: Any, combiner) -> Any:
      # combiner takes two boolean TTNN tensors and returns TTNN tensor
      def to_bool_ttnn(x: Any):
        xt = self._ensure_ttnn_tensor(x)
        zero_tensor = ttnn.zeros_like(xt)
        return ttnn.ne(xt, zero_tensor)
      if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        return [combiner(to_bool_ttnn(x), to_bool_ttnn(y)) for x, y in zip(a, b)]
      if isinstance(a, list):
        return [combiner(to_bool_ttnn(x), to_bool_ttnn(b)) for x in a]
      if isinstance(b, list):
        return [combiner(to_bool_ttnn(a), to_bool_ttnn(y)) for y in b]
      return combiner(to_bool_ttnn(a), to_bool_ttnn(b))



    # local/register scratch storage keyed by defining uop index
    local_store: dict[int, memoryview] = {}

    for idx, u in enumerate(uops):
      op = u["op"]
      src = u.get("src", [])
      arg = u.get("arg", None)

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
          # Extract slice from TTNN tensor
          values[idx] = ttnn.slice(base, [0, 0, 0, lane], [base.shape[0], base.shape[1], base.shape[2], lane+1])
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
            # Convert memoryview to TTNN tensor via torch (simpler path)
            torch_tensor = torch.frombuffer(mv[start:end], dtype=torch.float32).clone().reshape(1,1,1,-1)
            values[idx] = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, device=self.device.ttnn_device)
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
          # Store TTNN tensor directly to buffer
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
            ci_t = self._ensure_ttnn_tensor(ci, require_tile_layout=True)
            ti_t = self._ensure_ttnn_tensor(ti, require_tile_layout=True)
            fi_t = self._ensure_ttnn_tensor(fi, require_tile_layout=True)
            out_list.append(ttnn.where(ci_t, ti_t, fi_t))
          values[idx] = out_list
        else:
          ci_t = self._ensure_ttnn_tensor(cond, require_tile_layout=True)
          tv_t = self._ensure_ttnn_tensor(tval, require_tile_layout=True)
          fv_t = self._ensure_ttnn_tensor(fval, require_tile_layout=True)
          values[idx] = ttnn.where(ci_t, tv_t, fv_t)
      elif op == "CAST":
        inp = values[src[0]]
        target_dt = u.get("dtype")
        if isinstance(inp, tuple) and inp[0] == "ptr":
          values[idx] = inp
        else:
          # Use TTNN's native casting with proper dtype
          ttnn_dtype = _get_ttnn_dtype(target_dt) if isinstance(target_dt, str) else ttnn.bfloat16
          if isinstance(inp, list):
            values[idx] = [ttnn.cast(self._ensure_ttnn_tensor(x), ttnn_dtype) for x in inp]
          else:
            values[idx] = ttnn.cast(self._ensure_ttnn_tensor(inp), ttnn_dtype)
      elif op == "BITCAST":
        inp = values[src[0]]
        target_dt = u.get("dtype")
        if isinstance(inp, tuple) and inp[0] == "ptr":
          values[idx] = inp
        else:
          # TTNN bitcast if available, otherwise use regular cast
          ttnn_dtype = _get_ttnn_dtype(target_dt) if isinstance(target_dt, str) else ttnn.bfloat16
          if isinstance(inp, list):
            values[idx] = [ttnn.bitcast(self._ensure_ttnn_tensor(x), ttnn_dtype) if hasattr(ttnn, 'bitcast') else ttnn.cast(self._ensure_ttnn_tensor(x), ttnn_dtype) for x in inp]
          else:
            values[idx] = ttnn.bitcast(self._ensure_ttnn_tensor(inp), ttnn_dtype) if hasattr(ttnn, 'bitcast') else ttnn.cast(self._ensure_ttnn_tensor(inp), ttnn_dtype)
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

    # already wrote back
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