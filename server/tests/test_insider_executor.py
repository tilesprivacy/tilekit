import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from server.mem_agent.engine import VenvStackExecutor

def test_executor_system_python():
    """Verify VenvStackExecutor uses system python when configured."""
    print("Testing default executor (venv based)...")
    # Case 1: Default behavior (isolated venv)
    executor = VenvStackExecutor("test_default", "/tmp")
    executor._ensure_venv()
    assert executor.use_system_python is False
    print("  [OK] Default executor configured correctly")
    
    print("Testing system python executor override...")
    # Case 2: System python override
    executor_sys = VenvStackExecutor("test_sys", "/tmp", use_system_python=True)
    assert executor_sys.use_system_python is True
    
    # Execute code and check sys.executable
    print(f"Server sys.executable: {sys.executable}")
    code_path = "import sys; exec_path = sys.executable"
    
    # We expect this to use the exact same python interpreter as the test runner
    result, error = executor_sys.execute(code_path)
    
    if error:
        print(f"  [FAIL] Execution failed: {error}")
        sys.exit(1)
        
    exec_path = result.get('exec_path')
    print(f"Sandbox sys.executable: {exec_path}")
    
    # Verify paths match (resolving symlinks if needed)
    real_sys = os.path.realpath(sys.executable)
    real_exec = os.path.realpath(exec_path)
    
    assert real_sys == real_exec, f"Expected {real_sys}, got {real_exec}"
    print("  [PASS] Regression verification passed: Executor correctly used system python")

if __name__ == "__main__":
    test_executor_system_python()
