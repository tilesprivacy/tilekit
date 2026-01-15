import logging
import os
import sys
import pickle
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


SANDBOX_TIMEOUT = 10
SANDBOX_PROFILE_PATH = Path(__file__).parent / "sandbox.sb"

# Configure a logger for the sandbox (in real use, configure handlers/level as needed)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # or DEBUG for more verbosity


class VenvStackExecutor:
    """Secure code executor using venvstacks and Apple sandbox-exec.
    
    Provides:
    - Isolated Python virtual environments via venvstacks
    - Kernel-level sandboxing via Apple's sandbox-exec
    - Persistent state across tool calls within a session
    - Automatic package installation
    """
    
    def __init__(
        self,
        session_id: str,
        workspace_path: str,
        base_venv_path: Optional[str] = None,
    ):
        """Initialize the executor.
        
        Args:
            session_id: Unique identifier for this execution session
            workspace_path: Directory the code is allowed to access
            base_venv_path: Optional base path for virtual environments
        """
        self.session_id = session_id
        self.workspace_path = os.path.abspath(workspace_path)
        self.base_venv_path = base_venv_path or os.path.join(
            tempfile.gettempdir(), "tiles_venvstacks"
        )
        self.venv_path = os.path.join(self.base_venv_path, session_id)
        self._session_state: Dict[str, Any] = {}
        self._initialized = False
        
    def _ensure_venv(self) -> None:
        """Ensure the virtual environment exists."""
        if self._initialized:
            return
            
        os.makedirs(self.venv_path, exist_ok=True)
        
        # Check if venv already exists
        python_path = os.path.join(self.venv_path, "bin", "python3")
        if not os.path.exists(python_path):
            # Create venv using system Python
            subprocess.run(
                [sys.executable, "-m", "venv", self.venv_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"Created venv at {self.venv_path}")
            
        self._initialized = True
        
    def _get_sandbox_profile(self) -> str:
        """Generate sandbox profile with paths substituted."""
        if not SANDBOX_PROFILE_PATH.exists():
            logger.warning("Sandbox profile not found, running without sandbox")
            return ""
            
        profile = SANDBOX_PROFILE_PATH.read_text()
        profile = profile.replace("${VENV_PATH}", self.venv_path)
        profile = profile.replace("${WORKSPACE_PATH}", self.workspace_path)
        return profile
        
    def install_package(self, package: str) -> Tuple[bool, str]:
        """Install a package in the session's venv.
        
        Args:
            package: Package name to install
            
        Returns:
            Tuple of (success, message)
        """
        self._ensure_venv()
        pip_path = os.path.join(self.venv_path, "bin", "pip")
        
        try:
            result = subprocess.run(
                [pip_path, "install", package],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return True, f"Successfully installed {package}"
            else:
                return False, f"Failed to install {package}: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, f"Timeout installing {package}"
        except Exception as e:
            return False, f"Error installing {package}: {e}"
            
    def execute(
        self,
        code: str,
        timeout: int = SANDBOX_TIMEOUT,
        use_sandbox: bool = False,  # Disabled by default until sandbox profile is fixed
    ) -> Tuple[Optional[Dict[str, Any]], str]:

        """Execute code in the sandboxed environment.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            use_sandbox: Whether to use Apple sandbox-exec (macOS only)
            
        Returns:
            Tuple of (locals_dict, error_message)
        """
        logger.info(f"[SANDBOX] execute() called with code:\n{code[:500]}...")
        
        self._ensure_venv()
        
        python_path = os.path.join(self.venv_path, "bin", "python3")
        logger.debug(f"[SANDBOX] Using python at: {python_path}")
        
        # Prepare code with state restoration/saving
        wrapped_code = f'''
import pickle
import sys
import os

# Restore session state
_session_state = {repr(self._session_state)}

# Make session state available as globals
globals().update(_session_state)

# Change to workspace
os.chdir({repr(self.workspace_path)})

# Execute user code
{code}

# Capture updated state (only picklable values)
_new_state = {{}}
for k, v in dict(locals()).items():
    if not k.startswith("_"):
        try:
            pickle.dumps(v)
            _new_state[k] = v
        except:
            pass

# Output state
sys.stdout.buffer.write(pickle.dumps((_new_state, None)))
'''
        
        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=self.workspace_path
        ) as f:
            f.write(wrapped_code)
            temp_script = f.name
        
        logger.debug(f"[SANDBOX] Wrote wrapped code to: {temp_script}")
            
        try:
            # Build command
            cmd = [python_path, temp_script]
            
            # Use sandbox-exec on macOS if available and enabled
            if use_sandbox and sys.platform == "darwin":
                profile = self._get_sandbox_profile()
                if profile:
                    # Write profile to temp file
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".sb", delete=False
                    ) as pf:
                        pf.write(profile)
                        profile_path = pf.name
                    cmd = ["sandbox-exec", "-f", profile_path] + cmd
                    logger.debug(f"[SANDBOX] Using sandbox profile: {profile_path}")
                    
            logger.info(f"[SANDBOX] Executing command: {' '.join(cmd)}")
            
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                cwd=self.workspace_path,
            )
            
            logger.debug(f"[SANDBOX] Return code: {result.returncode}")
            logger.debug(f"[SANDBOX] Stderr: {result.stderr.decode()[:500]}")
            
            if result.returncode != 0:
                error_msg = result.stderr.decode().strip()
                logger.error(f"[SANDBOX] Execution failed: {error_msg}")
                return None, error_msg
                
            # Parse output
            try:
                new_state, error = pickle.loads(result.stdout)
                if new_state:
                    self._session_state.update(new_state)
                logger.info(f"[SANDBOX] Execution success: vars={list(new_state.keys())}, error={error}")
                return new_state, error or ""
            except Exception as e:
                error_msg = f"Failed to parse output: {e}\nStderr: {result.stderr.decode()}"
                logger.error(f"[SANDBOX] Parse error: {error_msg}")
                return None, error_msg
                
        except subprocess.TimeoutExpired:
            logger.error(f"[SANDBOX] Timeout after {timeout}s")
            return None, f"Execution timeout ({timeout}s)"
        except Exception as e:
            logger.exception(f"[SANDBOX] Exception during execution: {e}")
            return None, f"Execution error: {e}"
        finally:
            # Cleanup temp files
            try:
                os.unlink(temp_script)
                if "profile_path" in locals():
                    os.unlink(profile_path)
            except:
                pass

                
    def cleanup(self) -> None:
        """Clean up the session's virtual environment."""
        if os.path.exists(self.venv_path):
            try:
                shutil.rmtree(self.venv_path)
                logger.info(f"Cleaned up venv at {self.venv_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup venv: {e}")


# Session executor cache
_executor_cache: Dict[str, VenvStackExecutor] = {}


def get_or_create_executor(
    session_id: str, workspace_path: str
) -> VenvStackExecutor:
    """Get or create an executor for a session."""
    if session_id not in _executor_cache:
        _executor_cache[session_id] = VenvStackExecutor(session_id, workspace_path)
    return _executor_cache[session_id]


def execute_sandboxed_venvstack(
    code: str,
    session_id: str,
    workspace_path: str,
    timeout: int = SANDBOX_TIMEOUT,
    use_sandbox: bool = True,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Execute code using VenvStackExecutor.
    
    This is the new recommended entry point for sandboxed code execution.
    
    Args:
        code: Python code to execute
        session_id: Session identifier for state persistence
        workspace_path: Directory the code is allowed to access
        timeout: Maximum execution time
        use_sandbox: Whether to use Apple sandbox-exec
        
    Returns:
        Tuple of (locals_dict, error_message)
    """
    executor = get_or_create_executor(session_id, workspace_path)
    return executor.execute(code, timeout=timeout, use_sandbox=use_sandbox)



