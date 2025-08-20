from __future__ import annotations

import json, math
from typing import Any, Tuple

try:
    import torch  # type: ignore
    import ttnn   # type: ignore
except Exception as e:  # pragma: no cover - import checked at runtime
    torch = None  # type: ignore
    ttnn = None   # type: ignore

from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.renderer import Renderer
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.uop.ops import Ops, UOp
# NOTE: Python fallback has been removed per requirements. All execution must use TTNN or simple copy.


class TTNNAllocator(Allocator['TTNNDevice']):
  def _alloc(self, size:int, options):
    # host-side staging buffer; TT-NN tensors will be constructed at program run
    return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src


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
      elif u.op in {Ops.LOAD, Ops.STORE, Ops.ADD, Ops.MUL, Ops.SUB, Ops.FDIV, Ops.IDIV, Ops.MAX, Ops.EXP2, Ops.CAST, Ops.BITCAST, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR}:
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
    
    # Always log program factory JSON to tinygrad-artifacts
    try:
      import os, time
      artifacts_dir = "tinygrad-artifacts"
      os.makedirs(artifacts_dir, exist_ok=True)
      
      # Create unique filename with timestamp
      timestamp = int(time.time() * 1000)
      json_filename = os.path.join(artifacts_dir, f"ttnn_program_{self.name}_{timestamp}.json")
      
      with open(json_filename, 'w') as f:
        json.dump(self.program, f, indent=2)
      
      print(f"TTNN Program Factory saved to: {json_filename}")
      
      # Also print to console if debug is enabled
      if os.environ.get("TTNN_DEBUG") == "1":
        print("TTNN IR:", json.dumps(self.program, indent=2))
    except Exception as e:
      print(f"Warning: Could not save program factory to disk: {e}")
      pass

  def _mv_to_torch_vec(self, mv:memoryview, dtype_bytes:int) -> torch.Tensor:
    numel = len(mv)//dtype_bytes
    # Interpret as float32 buffer by default, clone to avoid aliasing
    t = torch.frombuffer(mv, dtype=torch.float32, count=numel).clone()
    return t.reshape(1, 1, 1, numel).contiguous()

  def _to_tnnn_with_layout(self, torch_tensor:torch.Tensor, layout) -> Any:
    # move to device with requested layout
    return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=layout, device=self.device.tt_device)
  def _to_ttnn(self, torch_tensor:torch.Tensor):
    # default ROW_MAJOR
    return self._to_tnnn_with_layout(torch_tensor, ttnn.ROW_MAJOR_LAYOUT)

  def _from_ttnn(self, tt_tensor) -> torch.Tensor:
    # back to torch on host
    return ttnn.to_torch(tt_tensor).to(torch.float32).contiguous()

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    # Map declared tensors (in declaration order) to passed buffers
    globals_order = list(self.program.get("globals", []))

    # Debug: print launch parameters
    import os
    if os.environ.get("TTNN_DEBUG") == "1":
      print(f"TTNN Program launch: global_size={global_size}, local_size={local_size}")

    # Execute ops sequentially (Phase-2: add basic addressing and shape handling)
    globals_order = list(globals_order)
    def buf_ix_from_global_arg(arg:int) -> int:
      return globals_order.index(arg)

    uops = self.program.get("uops", [])
    # Minimal runtime value table; entries are either scalars, torch tensors, or metadata
    values: dict[int, Any] = {}

    # For matrix operations, we need to execute for each work item in the local_size grid
    total_work_items = local_size[0] * local_size[1] * local_size[2]
    
    # If this is a matrix operation (local_size > 1), execute for each work item
    if total_work_items > 1:
      if os.environ.get("TTNN_DEBUG") == "1":
        print(f"Executing {total_work_items} work items: local_size={local_size}")
      
      # Execute the program for each work item in the local_size grid
      for lidx2 in range(local_size[2]):
        for lidx1 in range(local_size[1]):
          for lidx0 in range(local_size[0]):
            if os.environ.get("TTNN_DEBUG") == "1":
              print(f"  Work item: lidx0={lidx0}, lidx1={lidx1}, lidx2={lidx2}")
            
            # Set override values for this work item
            self._lidx_override = {"lidx0": lidx0, "lidx1": lidx1, "lidx2": lidx2}
            self._debug_work_item = True  # Enable per-work-item debugging
            
            # Execute the main logic for this work item
            self._execute_single_thread(*bufs, global_size=global_size, local_size=(1,1,1), vals=vals)
            
            # Clear override
            self._lidx_override = None
            self._debug_work_item = False
      
      return None
    
    # Execute as single thread
    return self._execute_single_thread(*bufs, global_size=global_size, local_size=local_size, vals=vals)

  def _execute_single_thread(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    """Execute for a single thread"""
    globals_order = list(self.program.get("globals", []))
    def buf_ix_from_global_arg(arg:int) -> int:
      return globals_order.index(arg)

    uops = self.program.get("uops", [])
    values: dict[int, Any] = {}
    
    # Simple index context for RANGE and SPECIAL (placeholders for single work item)
    special: dict[str, int] = {"gx": global_size[0], "gy": global_size[1], "gz": global_size[2],
                               "lx": local_size[0], "ly": local_size[1], "lz": local_size[2]}
    ranges: dict[int, int] = {}

    def _vec_count(dtype_name: Any) -> int:
      if isinstance(dtype_name, str):
        digits = ''.join(ch for ch in dtype_name[::-1] if ch.isdigit())[::-1]
        try: return int(digits) if digits else 1
        except Exception: return 1
      return 1

    def _to_tensor(val: Any, like: torch.Tensor|None=None) -> torch.Tensor:
      if isinstance(val, torch.Tensor):
        return val
      if isinstance(val, (int, float)):
        if like is not None:
          return torch.full_like(like, float(val))
        return torch.tensor(float(val), dtype=torch.float32).reshape(1,1,1,1)
      raise RuntimeError(f"Expected scalar or tensor, got {type(val)}")

    def _map_bin(a: Any, b: Any, fn) -> Any:
      # lane-wise mapping for vector lists
      if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        outs = []
        for ai, bi in zip(a, b):
          t_out = fn(self._to_ttnn(_to_tensor(ai)), self._to_ttnn(_to_tensor(bi, _to_tensor(ai))))
          outs.append(self._from_ttnn(t_out))
        return outs
      if isinstance(a, list):
        outs = []
        for ai in a:
          ai_t = _to_tensor(ai)
          bi_t = _to_tensor(b, ai_t)
          t_out = fn(self._to_ttnn(ai_t), self._to_ttnn(bi_t))
          outs.append(self._from_ttnn(t_out))
        return outs
      if isinstance(b, list):
        outs = []
        for bi in b:
          bi_t = _to_tensor(bi)
          ai_t = _to_tensor(a, bi_t)
          t_out = fn(self._to_ttnn(ai_t), self._to_ttnn(bi_t))
          outs.append(self._from_ttnn(t_out))
        return outs
      # both scalars/tensors
      a_t = _to_tensor(a)
      b_t = _to_tensor(b, a_t)
      return self._from_ttnn(fn(self._to_ttnn(a_t), self._to_ttnn(b_t)))

    def _map_un(a: Any, fn) -> Any:
      if isinstance(a, list):
        return [self._from_ttnn(fn(self._to_ttnn(_to_tensor(x)))) for x in a]
      return self._from_ttnn(fn(self._to_ttnn(_to_tensor(a))))

    def _is_int_dtype(u: dict) -> bool:
      dt = u.get("dtype")
      return isinstance(dt, str) and dt.startswith("int")

    def _as_int_tensor(v: Any) -> torch.Tensor:
      t = _to_tensor(v)
      return t.to(torch.int32)

    def _as_float_tensor(v: Any) -> torch.Tensor:
      t = _to_tensor(v)
      return t.to(torch.float32)

    def _bitcast_tensor(t: torch.Tensor, target: str) -> torch.Tensor:
      import numpy as np
      cpu = t.detach().contiguous().view(-1).cpu()
      arr = cpu.numpy()
      if target and isinstance(target, str) and target.startswith("int"):
        # reinterpret float32 -> int32
        if arr.dtype != np.int32:
          arr = arr.view(np.int32)
      elif target and isinstance(target, str) and target.startswith("float"):
        # reinterpret int32 -> float32
        if arr.dtype != np.float32:
          arr = arr.view(np.float32)
      return torch.from_numpy(arr.reshape(cpu.shape)).contiguous()

    def _map_bin_int(a: Any, b: Any, torch_op) -> Any:
      if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        return [torch_op(_as_int_tensor(x), _as_int_tensor(y)) for x, y in zip(a, b)]
      if isinstance(a, list):
        return [torch_op(_as_int_tensor(x), _as_int_tensor(b)) for x in a]
      if isinstance(b, list):
        return [torch_op(_as_int_tensor(a), _as_int_tensor(y)) for y in b]
      return torch_op(_as_int_tensor(a), _as_int_tensor(b))

    def _map_bin_host(a: Any, b: Any, torch_op) -> Any:
      if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        return [torch_op(_to_tensor(x), _to_tensor(y)) for x, y in zip(a, b)]
      if isinstance(a, list):
        return [torch_op(_to_tensor(x), _to_tensor(b, _to_tensor(x))) for x in a]
      if isinstance(b, list):
        return [torch_op(_to_tensor(a, _to_tensor(bi)), _to_tensor(bi)) for bi in b]
      return torch_op(_to_tensor(a), _to_tensor(b, _to_tensor(a)))

    def _map_bin_ttnn_bool(a: Any, b: Any, combiner) -> Any:
      # combiner takes two boolean TTNN tensors and returns TTNN tensor
      def to_bool_ttnn(x: Any):
        xt = _to_tensor(x)
        return ttnn.ne(self._to_ttnn(xt), self._to_ttnn(torch.zeros_like(xt)))
      if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        return [self._from_ttnn(combiner(to_bool_ttnn(x), to_bool_ttnn(y))) for x, y in zip(a, b)]
      if isinstance(a, list):
        return [self._from_ttnn(combiner(to_bool_ttnn(x), to_bool_ttnn(b))) for x in a]
      if isinstance(b, list):
        return [self._from_ttnn(combiner(to_bool_ttnn(a), to_bool_ttnn(y))) for y in b]
      return self._from_ttnn(combiner(to_bool_ttnn(a), to_bool_ttnn(b)))

    def _coerce_to_single_tensor(val: Any) -> torch.Tensor:
      if isinstance(val, list):
        parts = [x if isinstance(x, torch.Tensor) else _to_tensor(x) for x in val]
        return torch.cat([p.reshape(1,1,1,-1) for p in parts], dim=-1)
      return val if isinstance(val, torch.Tensor) else _to_tensor(val)

    def _as_2d(val: Any, shape: tuple[int,int]) -> torch.Tensor:
      t = _coerce_to_single_tensor(val).reshape(-1)
      m, n = shape
      assert t.numel() == m*n, f"WMMA shape mismatch: expected {m*n} elems got {t.numel()}"
      return t.reshape(m, n).contiguous()

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
        # Get special operation name (e.g., "lidx0", "lidx1")
        if isinstance(arg, list) and len(arg) >= 1:
          special_name = arg[0]
          # Use override value if available, otherwise default to 0
          if hasattr(self, '_lidx_override') and self._lidx_override and special_name in self._lidx_override:
            values[idx] = self._lidx_override[special_name]
          elif special_name in special:
            values[idx] = special[special_name]
          else:
            values[idx] = 0
        else:
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
          # Ensure lane is within bounds and correctly extract the element
          if 0 <= lane < len(base):
            values[idx] = base[lane]
          else:
            # Create a zero tensor with the same format as the base elements
            if base:
              values[idx] = _to_tensor(0.0)
            else:
              values[idx] = _to_tensor(0.0)
        elif isinstance(base, torch.Tensor):
          lane = 0
          if isinstance(arg, (list, tuple)) and len(arg) > 0:
            try: lane = int(arg[0])
            except Exception: lane = 0
          elif isinstance(arg, (int, float)):
            lane = int(arg)
          # Handle tensor element extraction more carefully
          if base.dim() >= 2:
            # For matrices, preserve the shape and extract the element correctly
            flat_base = base.view(-1)
            if 0 <= lane < flat_base.numel():
              values[idx] = flat_base[lane:lane+1].reshape(1,1,1,1)
            else:
              values[idx] = _to_tensor(0.0)
          else:
            # For 1D tensors (4D with singleton dims), use original logic
            if 0 <= lane < base.shape[-1]:
              values[idx] = base[..., lane:lane+1]
            else:
              values[idx] = _to_tensor(0.0)
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
        
        # Debug INDEX computation for work items
        if hasattr(self, '_debug_work_item') and self._debug_work_item:
          import os
          if os.environ.get("TTNN_DEBUG") == "1":
            lidx_info = getattr(self, '_lidx_override', {})
            print(f"    INDEX: lidx={lidx_info}, src={src}, base_off={base_off}, offset={offset}, final_offset={base_off + offset}")
        
        values[idx] = ("ptr", g_arg, base_off + offset, itemsize, size_elems)
      elif op == "LOAD":
        # Contiguous load from a global buffer referenced by pointer expression in src[0]
        ptr = values[src[0]]
        if isinstance(ptr, tuple) and ptr[0] in ("ptr", "lptr", "rptr"):
          _, g_arg, off_elems, itemsize, size_elems = ptr
          if ptr[0] == "ptr":
            gix = buf_ix_from_global_arg(g_arg)
            mv = bufs[gix]
          else:
            mv = local_store[g_arg]
          vecn = _vec_count(u.get("dtype"))
          if vecn > 1:
            start = off_elems*itemsize
            end = start + vecn*itemsize
            # Load multiple elements directly from memory without reshaping
            flat_data = torch.frombuffer(mv[start:end], dtype=torch.float32, count=vecn).clone()
            # Create individual tensors for each element, maintaining proper 4D format
            values[idx] = [flat_data[i:i+1].reshape(1,1,1,1) for i in range(vecn)]
          else:
            start = off_elems*itemsize
            end = start + itemsize
            values[idx] = self._mv_to_torch_vec(mv[start:end], itemsize)
        else:
          gix = buf_ix_from_global_arg(globals_order[0])
          values[idx] = self._mv_to_torch_vec(bufs[gix], 4)
      elif op == "STORE":
        # Write back computed tensor to the pointer in src[0]
        ptr = values[src[0]]
        if not (isinstance(ptr, tuple) and ptr[0] in ("ptr", "lptr", "rptr")):
          ptr = ("ptr", globals_order[0], 0, 4, 0)
        _, g_arg, off_elems, itemsize, size_elems = ptr
        if ptr[0] == "ptr":
          gix = buf_ix_from_global_arg(g_arg)
          out_mv = bufs[gix]
        else:
          out_mv = local_store[g_arg]
        val = values[src[1]]
        if isinstance(val, list):
          tensor = torch.cat([x.reshape(1,1,1,-1) for x in val], dim=-1)
        else:
          tensor = val if isinstance(val, torch.Tensor) else _to_tensor(val)
        tensor = tensor.reshape(-1)
        nbytes = tensor.numel()*4
        start = off_elems*itemsize
        end = start + nbytes
        assert end <= len(out_mv)
        
        # Debug output for work item computation
        if hasattr(self, '_debug_work_item') and self._debug_work_item:
          import os
          if os.environ.get("TTNN_DEBUG") == "1":
            lidx_info = getattr(self, '_lidx_override', {})
            print(f"    STORE: lidx={lidx_info}, computed={tensor.numpy()}, storing at buf[{gix if ptr[0] == 'ptr' else 'local'}][{start}:{end}] (off_elems={off_elems})")
        
        out_mv[start:end] = memoryview(tensor.numpy().tobytes())
      elif op == "ADD":
        a = values[src[0]]; b = values[src[1]]
        if _is_int_dtype(u):
          result = _map_bin_int(a, b, torch.add)
          # For index calculations, convert tensor results back to scalar
          if isinstance(result, torch.Tensor) and result.numel() == 1:
            result = int(result.item())
          values[idx] = result
          
          # Debug ADD computation for work items
          if hasattr(self, '_debug_work_item') and self._debug_work_item:
            import os
            if os.environ.get("TTNN_DEBUG") == "1":
              lidx_info = getattr(self, '_lidx_override', {})
              print(f"    ADD: lidx={lidx_info}, a={a}, b={b}, result={result}")
        else:
          values[idx] = _map_bin(a, b, lambda x,y: x + y)
      elif op == "SUB":
        a = values[src[0]]; b = values[src[1]]
        if _is_int_dtype(u):
          values[idx] = _map_bin_int(a, b, torch.sub)
        else:
          values[idx] = _map_bin(a, b, lambda x,y: ttnn.subtract(x, y))
      elif op == "MUL":
        a = values[src[0]]; b = values[src[1]]
        if _is_int_dtype(u):
          result = _map_bin_int(a, b, torch.mul)
          # For index calculations, convert tensor results back to scalar
          if isinstance(result, torch.Tensor) and result.numel() == 1:
            result = int(result.item())
          values[idx] = result
          
          # Debug MUL computation for work items
          if hasattr(self, '_debug_work_item') and self._debug_work_item:
            import os
            if os.environ.get("TTNN_DEBUG") == "1":
              lidx_info = getattr(self, '_lidx_override', {})
              print(f"    MUL: lidx={lidx_info}, a={a}, b={b}, result={result}")
        else:
          values[idx] = _map_bin(a, b, lambda x,y: ttnn.multiply(x, y))
      elif op == "FDIV":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, lambda x,y: ttnn.divide(x, y))
      elif op == "IDIV":
        # integer division used for index math; do on host with integer tensors
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin_int(a, b, lambda x,y: torch.div(x, y, rounding_mode='floor').to(torch.float32))
      elif op == "MAX":
        a = values[src[0]]; b = values[src[1]]
        if _is_int_dtype(u):
          values[idx] = _map_bin_int(a, b, torch.maximum)
        else:
          values[idx] = _map_bin(a, b, lambda x,y: ttnn.maximum(x, y))
      elif op == "EXP2":
        a = values[src[0]]
        values[idx] = _map_un(a, lambda x: ttnn.exp(x * float(math.log(2.0))))
      elif op == "EXP":
        a = values[src[0]]
        values[idx] = _map_un(a, lambda x: ttnn.exp(x))
      elif op == "WMMA":
        # Tensor core matmul: sources are A, B, and accumulator C
        # arg layout: (..., (N,M,K), dtype_in, dtype_out, ...)
        dims = u.get("arg", None)
        assert isinstance(dims, (list, tuple)) and len(dims) >= 2, "WMMA arg malformed"
        N, M, K = dims[1]
        a_val = values[src[0]]
        b_val = values[src[1]]
        c_val = values[src[2]] if len(src) > 2 else None
        A2 = _as_2d(a_val, (M, K))
        B2 = _as_2d(b_val, (K, N))
        # Use TILE_LAYOUT for matmul (required by TTNN)
        ttA = self._to_tnnn_with_layout(A2, ttnn.TILE_LAYOUT)
        ttB = self._to_tnnn_with_layout(B2, ttnn.TILE_LAYOUT)
        ttC = ttnn.matmul(ttA, ttB)
        if c_val is not None:
          C2 = _as_2d(c_val, (M, N))
          ttC = ttnn.add(ttC, self._to_tnnn_with_layout(C2, ttnn.TILE_LAYOUT))
        values[idx] = self._from_ttnn(ttC)
        print(f"✅ Used WMMA operation for {M}x{K} @ {K}x{N} matrix multiplication")
      elif op == "CMPLT":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, lambda x,y: ttnn.lt(x, y))
      elif op == "CMPNE":
        a = values[src[0]]; b = values[src[1]]
        values[idx] = _map_bin(a, b, lambda x,y: ttnn.ne(x, y))
      elif op == "AND":
        a = values[src[0]]; b = values[src[1]]
        if _is_int_dtype(u):
          values[idx] = _map_bin_int(a, b, torch.bitwise_and)
        else:
          # boolean AND using TTNN: a!=0 AND b!=0 => NOT ( (NOT a) OR (NOT b) )
          values[idx] = _map_bin_ttnn_bool(a, b, lambda xa, xb: ttnn.logical_and(xa, xb) if hasattr(ttnn, 'logical_and') else ttnn.ne(ttnn.or_(ttnn.eq(xa, self._to_ttnn(torch.zeros_like(_to_tensor(a)))), ttnn.eq(xb, self._to_ttnn(torch.zeros_like(_to_tensor(b))))), self._to_ttnn(torch.zeros_like(_to_tensor(a)))))
      elif op == "OR":
        a = values[src[0]]; b = values[src[1]]
        if _is_int_dtype(u):
          values[idx] = _map_bin_int(a, b, torch.bitwise_or)
        else:
          values[idx] = _map_bin_ttnn_bool(a, b, lambda xa, xb: ttnn.logical_or(xa, xb) if hasattr(ttnn, 'logical_or') else ttnn.ne(ttnn.and_(ttnn.eq(xa, self._to_ttnn(torch.zeros_like(_to_tensor(a)))), ttnn.eq(xb, self._to_ttnn(torch.zeros_like(_to_tensor(b))))), self._to_ttnn(torch.ones_like(_to_tensor(a)))))
      elif op == "XOR":
        a = values[src[0]]; b = values[src[1]]
        if _is_int_dtype(u):
          values[idx] = _map_bin_int(a, b, torch.bitwise_xor)
        else:
          # boolean XOR via TTNN if available else NE as xor for bools
          values[idx] = _map_bin_ttnn_bool(a, b, lambda xa, xb: ttnn.logical_xor(xa, xb) if hasattr(ttnn, 'logical_xor') else ttnn.ne(xa, xb))
      elif op == "SHL":
        a = values[src[0]]; b = values[src[1]]
        # Shift operations are integer-only
        values[idx] = _map_bin_int(a, b, torch.bitwise_left_shift)
      elif op == "SHR":
        a = values[src[0]]; b = values[src[1]]
        # Shift operations are integer-only
        values[idx] = _map_bin_int(a, b, torch.bitwise_right_shift)
      elif op == "WHERE":
        cond = values[src[0]]; tval = values[src[1]]; fval = values[src[2]]
        # broadcast lane-wise if lists
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
            # ttnn.where requires TILE layout for non-sharded inputs
            ci_t = self._to_tnnn_with_layout(_to_tensor(ci), ttnn.TILE_LAYOUT)
            ti_t = self._to_tnnn_with_layout(_to_tensor(ti, _to_tensor(ci)), ttnn.TILE_LAYOUT)
            fi_t = self._to_tnnn_with_layout(_to_tensor(fi, _to_tensor(ci)), ttnn.TILE_LAYOUT)
            t_out = ttnn.where(ci_t, ti_t, fi_t)
            out_list.append(self._from_ttnn(t_out))
          values[idx] = out_list
        else:
          ci_t = self._to_tnnn_with_layout(_to_tensor(cond), ttnn.TILE_LAYOUT)
          tv_t = self._to_tnnn_with_layout(_to_tensor(tval, _to_tensor(cond)), ttnn.TILE_LAYOUT)
          fv_t = self._to_tnnn_with_layout(_to_tensor(fval, _to_tensor(cond)), ttnn.TILE_LAYOUT)
          values[idx] = self._from_ttnn(ttnn.where(ci_t, tv_t, fv_t))
      elif op == "CAST":
        inp = values[src[0]]
        target_dt = u.get("dtype")
        if isinstance(inp, tuple) and inp[0] == "ptr":
          values[idx] = inp
        elif isinstance(inp, list):
          if _is_int_dtype(u): values[idx] = [_as_int_tensor(x) for x in inp]
          else: values[idx] = [_as_float_tensor(x) for x in inp]
        else:
          values[idx] = _as_int_tensor(inp) if _is_int_dtype(u) else _as_float_tensor(inp)
      elif op == "BITCAST":
        inp = values[src[0]]
        target_dt = u.get("dtype")
        if isinstance(inp, tuple) and inp[0] == "ptr":
          values[idx] = inp
        elif isinstance(inp, list):
          values[idx] = [_bitcast_tensor(_to_tensor(x), target_dt) for x in inp]
        else:
          values[idx] = _bitcast_tensor(_to_tensor(inp), target_dt)
      elif op == "DEFINE_REG":
        # Represent register pointer as ("rptr", reg_id(idx), offset_elems, itemsize_bytes, size_elems)
        itemsize = int(u.get("itemsize", 4))
        size_elems = int(u.get("size", 0))
        # per-thread regs are flattened to one since executor is single-threaded
        values[idx] = ("rptr", idx, 0, itemsize, size_elems)
      elif op == "VECTORIZE":
        parts = [values[s] for s in src]
        # flatten any single-item lists and ensure all are tensors
        elems = []
        for p in parts:
          if isinstance(p, list): 
            elems.extend([_to_tensor(x) if not isinstance(x, torch.Tensor) else x for x in p])
          else: 
            elems.append(_to_tensor(p) if not isinstance(p, torch.Tensor) else p)
        values[idx] = [e for e in elems]
      elif op in ("NOOP", "SINK", "KERNEL", "PRECAST", "REWRITE_ERROR", "UNIQUE", "DEVICE", "BARRIER", "ENDIF", "ENDRANGE", "IF"):
        values[idx] = values.get(src[0], None) if src else None
      else:
        raise RuntimeError(f"Unsupported UOp in phase-2 executor: {op}")

    # already wrote back
    return None


class TTNNDevice(Compiled):
  def __init__(self, device:str):
    if ttnn is None or torch is None:
      raise RuntimeError("TTNN backend requires 'ttnn' and 'torch' to be installed and importable")
    # parse device id: "TTNN" or "TTNN:idx"
    parts = device.split(":", 1)
    device_id = int(parts[1]) if len(parts) == 2 and parts[1] != "" else 0
    self.tt_device = ttnn.open_device(device_id=device_id)
    super().__init__(device, TTNNAllocator(self), TTNNRenderer(), TTNNCompiler(), lambda name, lib: TTNNProgram(self, name, lib))

  def finalize(self):
    if hasattr(self, 'tt_device') and self.tt_device is not None:
      try:
        ttnn.close_device(self.tt_device)
      except Exception:
        pass

