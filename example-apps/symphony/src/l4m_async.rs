use crate::l4m;

// Async wrapper for l4m::tokenize

pub async fn tokenize(text: String) -> Vec<u32> {
    tokio::task::spawn_blocking(move || l4m::tokenize(&text))
        .await
        .unwrap()
}

pub async fn detokenize(token_ids: Vec<u32>) -> String {
    tokio::task::spawn_blocking(move || l4m::detokenize(&token_ids))
        .await
        .unwrap()
}

pub async fn get_vocabs() -> Vec<Vec<u8>> {
    tokio::task::spawn_blocking(|| l4m::get_vocabs())
        .await
        .unwrap()
}

// self.stream, slice::from_ref(&next_dist), 32);
pub async fn sample_top_k(stream_id: u32, dists: Vec<u32>, k: u32) -> Vec<(Vec<u32>, Vec<f32>)> {
    tokio::task::spawn_blocking(move || l4m::sample_top_k(stream_id, &dists, k))
        .await
        .unwrap()
}
