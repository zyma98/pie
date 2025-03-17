use crate::service::{Service, ServiceError};
use crate::instance::Id as InstanceId;
use crate::runtime::ExceptionDispatcher;
use tokio::sync::oneshot;

#[derive(Debug)]
pub enum Command {
    GetRuntimeVersion { handle: oneshot::Sender<String> },

    GetInstanceId { handle: oneshot::Sender<String> },
    
    ProgramExists { program: String, handle: oneshot::Sender<bool> },
    
    LaunchInstance { program: String, handle: oneshot::Sender<InstanceId> },
    
    TerminateInstance { instance: InstanceId, handle: oneshot::Sender<bool> },
    
    GetInstanceStatus { instance: InstanceId, handle: oneshot::Sender<String> },
}

pub struct RuntimeHelper {
    runtime_version: String,
}

impl RuntimeHelper {
    pub fn new(runtime_version: &str) -> Self {
        RuntimeHelper {
            runtime_version: runtime_version.to_string(),
        }
    }
}

impl Service for RuntimeHelper {
    type Command = Command;

    async fn handle(&mut self, inst: InstanceId, cmd: Self::Command) {
        match cmd {
            Command::GetRuntimeVersion { handle } => {
                handle.send(self.runtime_version.clone()).unwrap();
            }
            Command::GetInstanceId { handle } => {
                handle.send(inst.to_string()).unwrap();
            }
        }
    }
}
