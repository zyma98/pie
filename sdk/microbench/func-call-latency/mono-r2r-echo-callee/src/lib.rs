#[inline(never)]
pub fn echo(s: &str) -> String {
    std::hint::black_box(s.to_owned())
}
