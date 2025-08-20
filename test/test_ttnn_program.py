import json
from tinygrad.runtime.ops_ttnn import TTNNDevice, TTNNProgram
import ttnn

# Example JSON instruction array (as in the prompt)
instructions = [
    {
        "op": "store",
        "tensor": {
            "name": "a",
            "size": 3,
            "dtype": "float32"
        }
    },
    {
        "op": "store",
        "tensor": {
            "name": "b",
            "size": 3,
            "dtype": "float32"
        }
    },
    {
        "op": "store",
        "tensor": {
            "name": "c",
            "size": 3,
            "dtype": "float32"
        }
    },
    {
        "op": "add",
        "inputs": ["a", "b"],
        "output": "c"
    },
    {
        "op": "if",
        "condition": True,
        "instructions": [
            {
                "op": "add",
                "inputs": ["c", "a"],
                "output": "b"
            }
        ]
    }
]

# Create TTNN device and program (dummy lib for test)
device = TTNNDevice()
program = TTNNProgram(device, "test_func", b"dummy_lib")

# Convert instructions to JSON string
json_array = json.dumps(instructions)

# Run the program
program(json_array)

print("TTNNProgram test completed.")
