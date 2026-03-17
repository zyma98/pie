use crate::forward::ForwardPass;
use crate::queues::Queue;

use inferlib_engine_bindings::inferlet::zo::evolve::set_adapter_seed;

impl Queue {
    pub(crate) fn initialize_adapter(
        &self,
        adapter_ptr: u32,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) {
        inferlib_engine_bindings::inferlet::zo::evolve::initialize_adapter(
            &self.inner,
            adapter_ptr,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        )
    }

    pub(crate) fn update_adapter(
        &self,
        adapter_ptr: u32,
        scores: &[f32],
        seeds: &[i64],
        max_sigma: f32,
    ) {
        inferlib_engine_bindings::inferlet::zo::evolve::update_adapter(
            &self.inner,
            adapter_ptr,
            scores,
            seeds,
            max_sigma,
        )
    }
}

impl ForwardPass {
    pub(crate) fn set_adapter_seed(&self, seed: i64) {
        set_adapter_seed(&self.inner, seed);
    }
}
