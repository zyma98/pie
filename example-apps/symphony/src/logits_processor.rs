pub trait LogitsProcessor {
    fn process(&mut self, token_ids: &[u32], logits: &[f32]) -> Vec<f32>;
}

pub struct Pass {}

impl Pass {
    pub fn new() -> Self {
        Self {}
    }
}

impl LogitsProcessor for Pass {
    fn process(&mut self, token_ids: &[u32], logits: &[f32]) -> Vec<f32> {
        logits.to_vec()
    }
}
