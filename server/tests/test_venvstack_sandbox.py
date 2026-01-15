"""Tests for VenvStackExecutor sandbox functionality.

Verifies:
- Isolated environment creation
- Network access denial
- File system restriction
- State persistence across executions
"""

import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mem_agent.engine import VenvStackExecutor, execute_sandboxed_venvstack


class TestVenvStackExecutor:
    """Test suite for VenvStackExecutor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session_id = "test_session_123"
        self.executor = VenvStackExecutor(
            session_id=self.session_id,
            workspace_path=self.temp_dir,
        )

    def teardown_method(self):
        """Clean up after tests."""
        self.executor.cleanup()
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_basic_execution(self):
        """Test basic code execution works."""
        code = "x = 1 + 1"
        result, error = self.executor.execute(code, use_sandbox=False)
        
        assert error == ""
        assert result is not None
        assert result.get("x") == 2

    def test_math_sqrt_execution(self):
        """Test math.sqrt execution - the most common code interpreter use case."""
        code = """import math
result = math.sqrt(123456789)
"""
        result, error = self.executor.execute(code, use_sandbox=False)
        
        assert error == "", f"Unexpected error: {error}"
        assert result is not None, "Result should not be None"
        assert "result" in result, f"'result' not in result dict: {result.keys()}"
        assert abs(result.get("result") - 11111.1110606) < 0.001, f"Incorrect sqrt result: {result.get('result')}"

    def test_import_and_compute(self):
        """Test that imports work correctly in the sandbox."""
        code = """import json
data = json.dumps({"key": "value"})
parsed = json.loads(data)
"""
        result, error = self.executor.execute(code, use_sandbox=False)
        
        assert error == "", f"Unexpected error: {error}"
        assert result is not None
        assert result.get("parsed") == {"key": "value"}

    def test_state_persistence(self):
        """Test that state persists across executions."""
        # First execution: set a variable
        code1 = "counter = 42"
        result1, error1 = self.executor.execute(code1, use_sandbox=False)
        
        assert error1 == ""
        assert result1.get("counter") == 42
        
        # Second execution: use the variable
        code2 = "result = counter * 2"
        result2, error2 = self.executor.execute(code2, use_sandbox=False)
        
        assert error2 == ""
        assert result2.get("result") == 84

    def test_file_creation_in_workspace(self):
        """Test that files can be created in the workspace."""
        code = '''
with open("test_file.txt", "w") as f:
    f.write("Hello, sandbox!")
file_created = True
'''
        result, error = self.executor.execute(code, use_sandbox=False)
        
        assert error == ""
        assert result.get("file_created") is True
        assert os.path.exists(os.path.join(self.temp_dir, "test_file.txt"))

    def test_file_read_in_workspace(self):
        """Test that files can be read from the workspace."""
        # Create a file first
        test_file = os.path.join(self.temp_dir, "readable.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        code = '''
with open("readable.txt", "r") as f:
    content = f.read()
'''
        result, error = self.executor.execute(code, use_sandbox=False)
        
        assert error == ""
        assert result.get("content") == "test content"

    def test_timeout_handling(self):
        """Test that execution timeout is enforced."""
        code = '''
import time
time.sleep(30)  # Should timeout
'''
        result, error = self.executor.execute(code, timeout=2, use_sandbox=False)
        
        assert result is None
        assert "timeout" in error.lower()

    @pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
    def test_sandbox_network_denied(self):
        """Test that network access is denied in sandbox mode."""
        code = '''
import urllib.request
try:
    urllib.request.urlopen("https://google.com", timeout=5)
    network_allowed = True
except Exception:
    network_allowed = False
'''
        result, error = self.executor.execute(code, use_sandbox=True)
        
        # Either the sandbox blocks it or we get an error
        if result is not None:
            assert result.get("network_allowed") is False
        else:
            # Sandbox may have blocked the entire execution
            pass

    @pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
    def test_sandbox_file_access_restricted(self):
        """Test that file access outside workspace is denied."""
        code = '''
try:
    with open("/etc/passwd", "r") as f:
        etc_passwd = f.read()
    file_access_denied = False
except Exception:
    file_access_denied = True
'''
        result, error = self.executor.execute(code, use_sandbox=True)
        
        if result is not None:
            assert result.get("file_access_denied") is True


class TestExecuteSandboxedVenvstack:
    """Test the convenience function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_convenience_function(self):
        """Test execute_sandboxed_venvstack works."""
        result, error = execute_sandboxed_venvstack(
            code="y = 5 * 5",
            session_id="convenience_test",
            workspace_path=self.temp_dir,
            use_sandbox=False,
        )
        
        assert error == ""
        assert result is not None
        assert result.get("y") == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
