use std::future::Future;
use wasmtime::component::Resource;
use wasmtime::component::{bindgen, ResourceTable};
use wasmtime::Result;
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

bindgen!({
    path: "../spi/app/wit",
    world: "app",
    async: true,
    with: {
        "spi:lm/inference/language-model": LanguageModel,
        "spi:app/system/channel": Channel
    },
    // Interactions with `ResourceTable` can possibly trap so enable the ability
    // to return traps from generated functions.
    trappable_imports: true,
});

pub struct InstanceState {
    // These two are required basically as a standard way to enable the impl of WasiView
    pub wasi_ctx: WasiCtx,
    pub resource_table: ResourceTable,
}

impl WasiView for InstanceState {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi_ctx
    }
}

impl InstanceState {
    pub fn new() -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_stderr().inherit_network().inherit_stdout();

        InstanceState {
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
        }
    }
}

pub struct LanguageModel {
    model_id: String,
}

pub struct Channel {}

impl spi::lm::inference::Host for InstanceState {}
impl spi::lm::inference::HostLanguageModel for InstanceState {
    async fn new(&mut self, model_id: String) -> Result<Resource<LanguageModel>, wasmtime::Error> {
        let handle = LanguageModel { model_id };
        Ok(self.resource_table.push(handle)?)
    }

    async fn tokenize(
        &mut self,
        resource: Resource<LanguageModel>,
        text: String,
    ) -> Result<Vec<u32>, wasmtime::Error> {
        Ok(vec![0])
    }

    async fn detokenize(
        &mut self,
        resource: Resource<LanguageModel>,
        tokens: Vec<u32>,
    ) -> Result<String, wasmtime::Error> {
        Ok("Hello".to_string())
    }

    async fn predict(
        &mut self,
        resource: Resource<LanguageModel>,
        tokens: Vec<u32>,
    ) -> Result<u32, wasmtime::Error> {
        Ok(7)
    }

    async fn drop(&mut self, rep: Resource<LanguageModel>) -> Result<()> {
        Ok(())
    }
}
//
impl spi::app::system::Host for InstanceState {
    async fn get_version(&mut self) -> Result<String, wasmtime::Error> {
        Ok("0.1.0".to_string())
    }
}

impl spi::app::system::HostChannel for InstanceState {
    async fn new(&mut self) -> Result<Resource<Channel>, wasmtime::Error> {
        let handle = Channel {};
        Ok(self.resource_table.push(handle)?)
    }

    async fn request(
        &mut self,
        self_: Resource<Channel>,
        message: String,
    ) -> Result<String, wasmtime::Error> {
        Ok("sdsd".to_string())
    }

    async fn send(&mut self, self_: Resource<Channel>, message: String) -> Result<()> {
        Ok(())
    }

    async fn fetch(&mut self, self_: Resource<Channel>) -> Result<String, wasmtime::Error> {
        Ok("sdsd".to_string())
    }

    async fn drop(&mut self, rep: Resource<Channel>) -> Result<()> {
        Ok(())
    }
}

impl spi::lm::kvcache::Host for InstanceState {}
