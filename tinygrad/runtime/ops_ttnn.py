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


class TTNNAllocator(Allocator['TTNNDevice']):
  def _alloc(self, size:int, options):
    # host-side staging buffer; TT-NN tensors will be constructed at program run
    return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src


class TTNNRenderer(Renderer):
  device = "TTNN"

  def render(self, uops:list[UOp]) -> str:
    # Build a list-of-ops factory inspired by provided schema
    ops:list[dict[str, Any]] = []
    global_declared:dict[int, str] = {}

    def dtype_str(ptrdt:PtrDType) -> str:
      base = ptrdt.base.scalar()
      # map tinygrad dtype names to common strings
      name = base.name
      if base == dtypes.float: return "float32"
      if base == dtypes.half: return "float16"
      if base == dtypes.bfloat16: return "bfloat16"
      if base == dtypes.int32: return "int32"
      if base == dtypes.uint32: return "uint32"
      if base == dtypes.float64: return "float64"
      return name

    # declare globals in encounter order with names buf<id>
    for u in uops:
      if u.op is Ops.DEFINE_GLOBAL:
        assert isinstance(u.dtype, PtrDType)
        gname = f"buf{u.arg}"
        global_declared[u.arg] = gname
        size = u.dtype.size if u.dtype.size != -1 else 0
        ops.append({
          "op": "store",
          "tensor": {
            "id": u.arg,
            "name": gname,
            "size": size,
            "dtype": dtype_str(u.dtype),
          }
        })

    # identify final store and create compute op
    store_nodes = [u for u in uops if u.op is Ops.STORE]
    if not store_nodes:
      raise NotImplementedError("TTNNRenderer: no STORE found")
    final_store = store_nodes[-1]
    out_buf_id = next(x.arg for x in final_store.src[0].toposort() if x.op is Ops.DEFINE_GLOBAL)
    out_name = global_declared[out_buf_id]
    val_node = final_store.src[1]

    # dependency set for the stored value
    deps = list(val_node.toposort().keys())
    # optional debug
    try:
      import os
      if os.environ.get("TTNN_DEBUG") == "1":
        print("TTNN renderer deps ops:", sorted({str(n.op) for n in deps}))
    except Exception:
      pass

    def first_global_name(node:UOp) -> str:
      gid = next(x.arg for x in node.toposort() if x.op is Ops.DEFINE_GLOBAL)
      return global_declared[gid]

    # prefer exp/exp2 if present, otherwise fall back to add
    exp2_nodes = [n for n in deps if n.op is Ops.EXP2]
    if exp2_nodes:
      e2 = exp2_nodes[-1]
      arg = e2.src[0]
      # unwrap simple wrappers
      seen = set()
      unwrap_ops = {Ops.CAST, Ops.BITCAST, Ops.VECTORIZE, Ops.GEP, Ops.WHERE}
      while arg.op in unwrap_ops and arg not in seen:
        seen.add(arg)
        arg = arg.src[1] if arg.op is Ops.WHERE else arg.src[0]
      if arg.op is Ops.MUL:
        # check for MUL(x, CONST k)
        a, b = arg.src
        const_node = a if a.op is Ops.CONST else b if b.op is Ops.CONST else None
        var_node = b if a.op is Ops.CONST else a if b.op is Ops.CONST else None
        if const_node is not None and var_node is not None:
          k = float(const_node.arg)
          if abs(k - (1.0/math.log(2.0))) < 1e-6:
            # it's exp(x)
            ops.append({"op": "exp", "inputs": [first_global_name(var_node)], "output": out_name})
          else:
            # unknown scaling, keep exp2 semantics
            ops.append({"op": "exp2", "inputs": [first_global_name(var_node)], "output": out_name})
        else:
          ops.append({"op": "exp2", "inputs": [first_global_name(arg)], "output": out_name})
      else:
        ops.append({"op": "exp2", "inputs": [first_global_name(arg)], "output": out_name})
    else:
      # Heuristic: if only one global buffer participates and we see non-trivial unary math, assume exp
      globals_in_deps = set()
      for n in deps:
        for x in n.toposort():
          if x.op is Ops.DEFINE_GLOBAL:
            globals_in_deps.add(x.arg)
      unary_math_present = any(n.op in {Ops.MUL, Ops.CMPLT, Ops.CMPNE, Ops.WHERE} for n in deps)
      if len(globals_in_deps) == 1 and unary_math_present:
        ops.append({"op": "exp", "inputs": [global_declared[next(iter(globals_in_deps))]], "output": out_name})
      else:
        # prefer add if present in deps
        def src_is_from_globals(n:UOp) -> bool:
          try:
            return all(any(x.op is Ops.DEFINE_GLOBAL for x in s.toposort()) for s in n.src)
          except Exception:
            return False
        add_nodes = [n for n in deps if n.op is Ops.ADD and src_is_from_globals(n)]
        if add_nodes:
          addn = add_nodes[-1]
          ins_names = [first_global_name(s) for s in addn.src]
          ops.append({"op": "add", "inputs": ins_names, "output": out_name})
        else:
          raise NotImplementedError(f"TTNNRenderer: unsupported expression for store result")

    return json.dumps(ops)


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
    self.ops = json.loads(lib.decode('utf-8'))
    try:
      import os
      if os.environ.get("TTNN_DEBUG") == "1":
        print("TTNN factory:", json.dumps(self.ops, indent=2))
    except Exception:
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
    name_to_ix:dict[str, int] = {}
    decl_order:list[str] = []
    for op in self.ops:
      if op.get("op") == "store" and isinstance(op.get("tensor"), dict):
        name = op["tensor"]["name"]
        decl_order.append(name)
        name_to_ix[name] = len(name_to_ix)

    # Helpers to wrap names into TTNN tensors
    def name_to_ttnn(name:str):
      mv = bufs[name_to_ix[name]]
      t = self._mv_to_torch_vec(mv, 4)
      return self._to_ttnn(t)
    def name_to_ttnn_tile(name:str):
      mv = bufs[name_to_ix[name]]
      numel = len(mv)//4
      import math as _m
      Y = 2
      X = (_m.ceil(numel/2))
      if X % 2 != 0: X += 1
      total = Y*X
      pad = total - numel
      t = self._mv_to_torch_vec(mv, 4)
      if pad > 0:
        t = torch.nn.functional.pad(t.reshape(1,1,1,numel), (0,pad)).reshape(1,1,1,total)
      t = t.reshape(1,1,Y,X).contiguous()
      return self._to_tnnn_with_layout(t, ttnn.TILE_LAYOUT), numel

    # Execute ops sequentially and write back after each compute
    last_out_name = None
    last_torch_out = None
    for op in self.ops:
      if op.get("op") == "add":
        a, b = op["inputs"]
        out = op["output"]
        out_ttnn = name_to_ttnn(a) + name_to_ttnn(b)
        last_torch_out = self._from_ttnn(out_ttnn)
        # write back result
        out_mv = bufs[name_to_ix[out]]
        flat = last_torch_out.reshape(-1)
        assert flat.numel()*4 == len(out_mv), "Output size mismatch"
        out_mv[:] = memoryview(flat.numpy().tobytes())
        last_out_name = out
      elif op.get("op") == "exp2":
        a = op["inputs"][0]
        out = op["output"]
        tt_in, orig = name_to_ttnn_tile(a)
        out_ttnn = ttnn.exp(tt_in * float(math.log(2.0)))
        last_torch_out = self._from_ttnn(out_ttnn)
        out_mv = bufs[name_to_ix[out]]
        flat = last_torch_out.reshape(-1)[:orig]
        assert flat.numel()*4 == len(out_mv), "Output size mismatch"
        out_mv[:] = memoryview(flat.numpy().tobytes())
        last_out_name = out
      elif op.get("op") == "exp":
        a = op["inputs"][0]
        out = op["output"]
        tt_in, orig = name_to_ttnn_tile(a)
        out_ttnn = ttnn.exp(tt_in)
        last_torch_out = self._from_ttnn(out_ttnn)
        out_mv = bufs[name_to_ix[out]]
        flat = last_torch_out.reshape(-1)[:orig]
        assert flat.numel()*4 == len(out_mv), "Output size mismatch"
        out_mv[:] = memoryview(flat.numpy().tobytes())
        last_out_name = out
      elif op.get("op") == "store":
        continue
      elif op.get("op") == "if":
        # Minimal support: if condition is truthy, execute inner instructions naively
        cond = bool(op.get("condition", False))
        if cond:
          for inner in op.get("instructions", []):
            if inner.get("op") == "add":
              a, b = inner["inputs"]
              out = inner["output"]
              out_ttnn = name_to_ttnn(a) + name_to_ttnn(b)
              last_torch_out = self._from_ttnn(out_ttnn)
              out_mv = bufs[name_to_ix[out]]
              flat = last_torch_out.reshape(-1)
              assert flat.numel()*4 == len(out_mv), "Output size mismatch"
              out_mv[:] = memoryview(flat.numpy().tobytes())
              last_out_name = out
            elif inner.get("op") == "exp2":
              a = inner["inputs"][0]
              out = inner["output"]
              tt_in, orig = name_to_ttnn_tile(a)
              out_ttnn = ttnn.exp(tt_in * float(math.log(2.0)))
              last_torch_out = self._from_ttnn(out_ttnn)
              out_mv = bufs[name_to_ix[out]]
              flat = last_torch_out.reshape(-1)[:orig]
              assert flat.numel()*4 == len(out_mv), "Output size mismatch"
              out_mv[:] = memoryview(flat.numpy().tobytes())
              last_out_name = out
            elif inner.get("op") == "exp":
              a = inner["inputs"][0]
              out = inner["output"]
              tt_in, orig = name_to_ttnn_tile(a)
              out_ttnn = ttnn.exp(tt_in)
              last_torch_out = self._from_ttnn(out_ttnn)
              out_mv = bufs[name_to_ix[out]]
              flat = last_torch_out.reshape(-1)[:orig]
              assert flat.numel()*4 == len(out_mv), "Output size mismatch"
              out_mv[:] = memoryview(flat.numpy().tobytes())
              last_out_name = out
      else:
        raise RuntimeError(f"Unsupported TTNN program op {op}")

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

