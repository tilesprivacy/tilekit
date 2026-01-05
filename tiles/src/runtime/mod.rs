#[allow(unused_imports)]
use crate::runtime::cpu::CPURuntime;
use crate::runtime::mlx::MLXRuntime;
use anyhow::Result;
use tilekit::modelfile::Modelfile;
pub mod cpu;
pub mod mlx;

pub struct RunArgs {
    pub modelfile: Modelfile,
}

pub enum Runtime {
    Mlx(MLXRuntime),
    Cpu(CPURuntime),
}

impl Runtime {
    pub async fn run(&self, run_args: RunArgs) {
        match self {
            Runtime::Mlx(runtime) => runtime.run(run_args).await,
            Runtime::Cpu(runtime) => runtime.run(run_args).await,
        }
    }

    pub async fn start_server_daemon(&self) -> Result<()> {
        match self {
            Runtime::Mlx(runtime) => runtime.start_server_daemon().await,
            Runtime::Cpu(runtime) => runtime.start_server_daemon().await,
        }
    }

    pub async fn stop_server_daemon(&self) -> Result<()> {
        match self {
            Runtime::Mlx(runtime) => runtime.stop_server_daemon().await,
            Runtime::Cpu(runtime) => runtime.stop_server_daemon().await,
        }
    }
}

#[cfg(target_os = "macos")]
pub fn build_runtime() -> Runtime {
    Runtime::Mlx(MLXRuntime::new())
}

#[cfg(not(target_os = "macos"))]
pub fn build_runtime() -> Runtime {
    Runtime::Cpu(CPURuntime::new())
}
