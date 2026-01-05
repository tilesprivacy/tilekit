use anyhow::Result;

pub struct CPURuntime {}

impl Default for CPURuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl CPURuntime {
    pub fn new() -> Self {
        CPURuntime {}
    }
    pub async fn run(&self, _run_args: super::RunArgs) {
        unimplemented!()
    }

    pub async fn start_server_daemon(&self) -> Result<()> {
        unimplemented!()
    }

    pub async fn stop_server_daemon(&self) -> Result<()> {
        unimplemented!()
    }
}
