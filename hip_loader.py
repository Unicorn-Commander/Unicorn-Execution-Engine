
import hip
import os

class HIPLoader:
    def __init__(self, kernel_file="hip_kernels.cpp"):
        self.kernel_file = os.path.join(os.path.dirname(__file__), kernel_file)
        self.module = self._compile_kernels()

    def _compile_kernels(self):
        if not os.path.exists(self.kernel_file):
            raise FileNotFoundError(f"Kernel file not found: {self.kernel_file}")

        try:
            # Compile the HIP kernel using hipcc
            module = hip.hipModuleLoad(self.kernel_file)
            return module
        except Exception as e:
            print(f"Error compiling HIP kernels: {e}")
            return None

    def get_kernel(self, kernel_name):
        if self.module:
            try:
                return hip.hipModuleGetFunction(self.module, kernel_name)
            except Exception as e:
                print(f"Error getting kernel '{kernel_name}': {e}")
        return None
