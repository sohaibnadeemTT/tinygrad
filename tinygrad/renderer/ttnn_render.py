from typing import Dict, List, Any, Optional, Set, DefaultDict
from collections import defaultdict, Counter
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, GroupOp
from tinygrad.dtype import DType, dtypes, PtrDType
from tinygrad.helpers import prod

class TTNNRenderer(Renderer):
    """
    TTNN Renderer - Generates PyTorch function calls as artifacts
    
    This renderer processes UOps and converts them into PyTorch tensor operations,
    specifically targeting TTNN (TensorTrain Neural Network) backend.
    """
    
    device = "TTNN"
    suffix = "TTNN"
    supports_float4 = True
    has_local = True
    has_shared = True
    
    def __init__(self):
        # Initialize result tracking and naming systems
        self.uop_results: Dict[UOp, str] = {}
        self.tensor_counter: DefaultDict[str, int] = defaultdict(int)
        self.function_calls: List[str] = []
        self.imports: Set[str] = set()
        self.tensor_definitions: List[str] = []
        
        # Add default imports
        self.imports.add("import torch")
        self.imports.add("import ttnn")
        
    def _get_tensor_name(self, prefix: str, uop: Optional[UOp] = None) -> str:
        """Generate unique tensor variable names"""
        count = self.tensor_counter[prefix]
        self.tensor_counter[prefix] += 1
        
        if uop and hasattr(uop, 'dtype'):
            dtype_suffix = self._get_dtype_suffix(uop.dtype)
            return f"{prefix}_{dtype_suffix}_{count}"
        return f"{prefix}_{count}"
    
    def _get_dtype_suffix(self, dtype: DType) -> str:
        """Convert DType to string suffix for naming"""
        dtype_map = {
            dtypes.float32: "f32",
            dtypes.float16: "f16", 
            dtypes.bfloat16: "bf16",
            dtypes.int32: "i32",
            dtypes.int64: "i64",
            dtypes.uint32: "u32",
            dtypes.bool: "bool",
        }
        return dtype_map.get(dtype, str(dtype).replace("dtypes.", ""))
    
    def _get_pytorch_dtype(self, dtype: DType) -> str:
        """Convert tinygrad DType to PyTorch dtype string"""
        dtype_map = {
            dtypes.float32: "torch.float32",
            dtypes.float16: "torch.float16",
            dtypes.bfloat16: "torch.bfloat16", 
            dtypes.int32: "torch.int32",
            dtypes.int64: "torch.int64",
            dtypes.uint32: "torch.uint32",
            dtypes.bool: "torch.bool",
        }
        return dtype_map.get(dtype, "torch.float32")
    
    def _process_buffer_ops(self, uop: UOp) -> str:
        """Process buffer-related operations (DEFINE_GLOBAL, LOAD, STORE)"""
        if uop.op == Ops.DEFINE_GLOBAL:
            # Global buffer definition
            buffer_name = f"buffer_{uop.arg}"
            dtype_str = self._get_pytorch_dtype(uop.dtype.base if isinstance(uop.dtype, PtrDType) else uop.dtype)
            size = uop.dtype.size if isinstance(uop.dtype, PtrDType) else 1
            
            result = f"{buffer_name} = torch.zeros({size}, dtype={dtype_str}, device='cuda')"
            self.tensor_definitions.append(result)
            return buffer_name
            
        elif uop.op == Ops.DEFINE_LOCAL:
            # Local/shared memory definition
            tensor_name = self._get_tensor_name("local", uop)
            dtype_str = self._get_pytorch_dtype(uop.dtype.base)
            size = uop.dtype.size
            
            result = f"{tensor_name} = torch.zeros({size}, dtype={dtype_str}, device='cuda')"
            self.tensor_definitions.append(result)
            return tensor_name
            
        elif uop.op == Ops.LOAD:
            # Memory load operation
            tensor_name = self._get_tensor_name("loaded", uop)
            src_tensor = self.uop_results[uop.src[0]]
            
            if len(uop.src) > 1:
                # Conditional load with gate
                gate = self.uop_results[uop.src[1]]
                result = f"{tensor_name} = torch.where({gate}, {src_tensor}, torch.zeros_like({src_tensor}))"
            else:
                # Simple load
                result = f"{tensor_name} = {src_tensor}.clone()"
                
            self.function_calls.append(result)
            return tensor_name
            
        elif uop.op == Ops.STORE:
            # Memory store operation
            dest_tensor = self.uop_results[uop.src[0]]
            value_tensor = self.uop_results[uop.src[1]]
            
            if len(uop.src) > 2:
                # Conditional store with gate
                gate = self.uop_results[uop.src[2]]
                result = f"{dest_tensor} = torch.where({gate}, {value_tensor}, {dest_tensor})"
            else:
                # Simple store
                result = f"{dest_tensor} = {value_tensor}"
                
            self.function_calls.append(result)
            return dest_tensor
            
        return ""
    
    def _process_arithmetic_ops(self, uop: UOp) -> str:
        """Process arithmetic operations (ADD, MUL, etc.)"""
        tensor_name = self._get_tensor_name("alu", uop)
        
        # Binary operations
        if uop.op in GroupOp.Binary:
            lhs = self.uop_results[uop.src[0]]
            rhs = self.uop_results[uop.src[1]]
            
            op_map = {
                Ops.ADD: f"torch.add({lhs}, {rhs})",
                Ops.MUL: f"torch.mul({lhs}, {rhs})", 
                Ops.SUB: f"torch.sub({lhs}, {rhs})",
                Ops.FDIV: f"torch.div({lhs}, {rhs})",
                Ops.POW: f"torch.pow({lhs}, {rhs})",
                Ops.MAX: f"torch.maximum({lhs}, {rhs})",
                Ops.CMPLT: f"torch.lt({lhs}, {rhs})",
                Ops.CMPEQ: f"torch.eq({lhs}, {rhs})",
                Ops.CMPNE: f"torch.ne({lhs}, {rhs})",
                Ops.AND: f"torch.logical_and({lhs}, {rhs})",
                Ops.OR: f"torch.logical_or({lhs}, {rhs})",
                Ops.XOR: f"torch.logical_xor({lhs}, {rhs})",
            }
            
            if uop.op in op_map:
                result = f"{tensor_name} = {op_map[uop.op]}"
                self.function_calls.append(result)
                return tensor_name
        
        # Unary operations  
        if uop.op in GroupOp.Unary:
            src = self.uop_results[uop.src[0]]
            
            op_map = {
                Ops.NEG: f"torch.neg({src})",
                Ops.SQRT: f"torch.sqrt({src})",
                Ops.EXP2: f"torch.pow(2.0, {src})",
                Ops.LOG2: f"torch.log2({src})",
                Ops.SIN: f"torch.sin({src})",
                Ops.RECIP: f"torch.reciprocal({src})",
                Ops.TRUNC: f"torch.trunc({src})",
            }
            
            if uop.op in op_map:
                result = f"{tensor_name} = {op_map[uop.op]}"
                self.function_calls.append(result) 
                return tensor_name
        
        # Ternary operations
        if uop.op in GroupOp.Ternary:
            if uop.op == Ops.WHERE:
                cond = self.uop_results[uop.src[0]]
                true_val = self.uop_results[uop.src[1]]  
                false_val = self.uop_results[uop.src[2]]
                result = f"{tensor_name} = torch.where({cond}, {true_val}, {false_val})"
                self.function_calls.append(result)
                return tensor_name
            elif uop.op == Ops.MULACC:
                a = self.uop_results[uop.src[0]]
                b = self.uop_results[uop.src[1]]
                c = self.uop_results[uop.src[2]]
                result = f"{tensor_name} = torch.addmm({c}, {a}, {b})"
                self.function_calls.append(result)
                return tensor_name
        
        return tensor_name
    
    def _process_const_ops(self, uop: UOp) -> str:
        """Process constant operations"""
        tensor_name = self._get_tensor_name("const", uop)
        dtype_str = self._get_pytorch_dtype(uop.dtype)
        
        if uop.op == Ops.CONST:
            result = f"{tensor_name} = torch.tensor({uop.arg}, dtype={dtype_str}, device='cuda')"
            self.function_calls.append(result)
            return tensor_name
        elif uop.op == Ops.VCONST:
            # Vector constant
            result = f"{tensor_name} = torch.full({uop.dtype.shape}, {uop.arg}, dtype={dtype_str}, device='cuda')"
            self.function_calls.append(result) 
            return tensor_name
        
        return tensor_name
    
    def _process_cast_ops(self, uop: UOp) -> str:
        """Process casting operations"""
        tensor_name = self._get_tensor_name("cast", uop)
        src = self.uop_results[uop.src[0]]
        target_dtype = self._get_pytorch_dtype(uop.dtype)
        
        if uop.op == Ops.CAST:
            result = f"{tensor_name} = {src}.to({target_dtype})"
            self.function_calls.append(result)
            return tensor_name
        elif uop.op == Ops.BITCAST:
            # Bitcast - view with different type
            result = f"{tensor_name} = {src}.view({target_dtype})"
            self.function_calls.append(result)
            return tensor_name
        
        return tensor_name
    
    def _process_reduce_ops(self, uop: UOp) -> str:
        """Process reduction operations"""
        tensor_name = self._get_tensor_name("reduce", uop)
        
        if uop.op == Ops.REDUCE_AXIS:
            src = self.uop_results[uop.src[0]]
            axis = uop.arg if uop.arg is not None else None
            
            # TODO: Determine reduction type from context
            result = f"{tensor_name} = torch.sum({src}, dim={axis}, keepdim=True)"
            self.function_calls.append(result)
            return tensor_name
        
        return tensor_name
    
    def _process_control_flow_ops(self, uop: UOp) -> str:
        """Process control flow operations (IF, RANGE, etc.)"""
        if uop.op == Ops.IF:
            condition = self.uop_results[uop.src[0]]
            result = f"# if {condition}:"
            self.function_calls.append(result)
            return ""
        elif uop.op == Ops.ENDIF:
            result = "# endif"
            self.function_calls.append(result)
            return ""
        elif uop.op == Ops.RANGE:
            limit = self.uop_results[uop.src[0]] if uop.src else uop.arg
            result = f"# for i in range({limit}):"
            self.function_calls.append(result)
            return f"i{uop.arg}" if hasattr(uop, 'arg') else "i"
        elif uop.op == Ops.ENDRANGE:
            result = "# endfor"
            self.function_calls.append(result)
            return ""
        elif uop.op == Ops.BARRIER:
            result = "# torch.cuda.synchronize()  # barrier"
            self.function_calls.append(result)
            return ""
        
        return ""
    
    def _process_special_ops(self, uop: UOp) -> str:
        """Process special operations (workgroup IDs, etc.)"""
        if uop.op == Ops.SPECIAL:
            # Special operations like thread/block indices
            special_type, value = uop.arg[0], uop.arg[1] if len(uop.arg) > 1 else 0
            
            if special_type[0] == 'g':  # global index
                tensor_name = f"gidx{special_type[-1]}"
                result = f"{tensor_name} = torch.tensor({value}, device='cuda')  # global index"
            elif special_type[0] == 'l':  # local index  
                tensor_name = f"lidx{special_type[-1]}"
                result = f"{tensor_name} = torch.tensor({value}, device='cuda')  # local index"
            else:
                tensor_name = f"idx_{special_type}"
                result = f"{tensor_name} = torch.tensor({value}, device='cuda')  # special index"
            
            self.function_calls.append(result)
            return tensor_name
        
        return ""
    
    def _process_uop(self, uop: UOp) -> str:
        """Main UOp processing dispatcher"""
        # Skip NOOP and SINK operations
        if uop.op in {Ops.NOOP, Ops.SINK}:
            return ""
        
        # Dispatch to appropriate processor based on UOp type
        if uop.op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.LOAD, Ops.STORE}:
            return self._process_buffer_ops(uop)
        elif uop.op in GroupOp.ALU:
            return self._process_arithmetic_ops(uop)
        elif uop.op in {Ops.CONST, Ops.VCONST}:
            return self._process_const_ops(uop)
        elif uop.op in {Ops.CAST, Ops.BITCAST}:
            return self._process_cast_ops(uop)
        elif uop.op in {Ops.REDUCE_AXIS, Ops.REDUCE}:
            return self._process_reduce_ops(uop)
        elif uop.op in {Ops.IF, Ops.ENDIF, Ops.RANGE, Ops.ENDRANGE, Ops.BARRIER}:
            return self._process_control_flow_ops(uop)
        elif uop.op == Ops.SPECIAL:
            return self._process_special_ops(uop)
        else:
            # Fallback for unhandled operations
            tensor_name = self._get_tensor_name("unknown", uop)
            result = f"# TODO: Implement {uop.op} operation"
            self.function_calls.append(result)
            return tensor_name
    
    def render(self, uops: List[UOp]) -> str:
        """
        Main render method - processes UOps and generates PyTorch function calls
        
        Args:
            uops: List of UOp objects to process
            
        Returns:
            String containing generated PyTorch code
        """
        # Reset state for new render
        self.uop_results.clear()
        self.tensor_counter.clear()
        self.function_calls.clear()
        self.tensor_definitions.clear()
        
        # Process each UOp in order
        for uop in uops:
            result = self._process_uop(uop)
            if result:
                self.uop_results[uop] = result
        
        # Generate final output
        output_lines = []
        
        # Add imports
        output_lines.extend(sorted(self.imports))
        output_lines.append("")
        
        # Add function definition
        output_lines.append("def ttnn_kernel():")
        output_lines.append("    \"\"\"Generated TTNN kernel from tinygrad UOps\"\"\"")
        output_lines.append("")
        
        # Add tensor definitions
        if self.tensor_definitions:
            output_lines.append("    # Buffer definitions")
            for defn in self.tensor_definitions:
                output_lines.append(f"    {defn}")
            output_lines.append("")
        
        # Add function calls
        if self.function_calls:
            output_lines.append("    # Computation")
            for call in self.function_calls:
                if call.startswith("#"):
                    output_lines.append(f"    {call}")
                else:
                    output_lines.append(f"    {call}")
            output_lines.append("")
        
        # Add return statement
        output_lines.append("    return  # TODO: specify return values")
        output_lines.append("")
        
        # Add main execution
        output_lines.append("if __name__ == '__main__':")
        output_lines.append("    ttnn_kernel()")
        
        return "\n".join(output_lines)


# Example usage and testing utilities
def create_example_uops() -> List[UOp]:
    """Create example UOps for testing the renderer"""
    uops = []
    
    # Example: Simple addition of two tensors
    # DEFINE_GLOBAL for input tensors
    buf_a = UOp(Ops.DEFINE_GLOBAL, PtrDType(dtypes.float32, 1024), (), 0)
    buf_b = UOp(Ops.DEFINE_GLOBAL, PtrDType(dtypes.float32, 1024), (), 1)
    buf_c = UOp(Ops.DEFINE_GLOBAL, PtrDType(dtypes.float32, 1024), (), 2)
    
    # LOAD operations
    val_a = UOp(Ops.LOAD, dtypes.float32, (buf_a,))
    val_b = UOp(Ops.LOAD, dtypes.float32, (buf_b,))
    
    # ADD operation
    result = UOp(Ops.ADD, dtypes.float32, (val_a, val_b))
    
    # STORE operation
    store_op = UOp(Ops.STORE, dtypes.void, (buf_c, result))
    
    return [buf_a, buf_b, buf_c, val_a, val_b, result, store_op]


def test_ttnn_renderer():
    """Test the TTNN renderer with example UOps"""
    renderer = TTNNRenderer()
    example_uops = create_example_uops()
    
    output = renderer.render(example_uops)
    print("Generated PyTorch code:")
    print("=" * 50)
    print(output)
    print("=" * 50)
    
    return output


if __name__ == "__main__":
    # Run test when script is executed directly
    test_ttnn_renderer()