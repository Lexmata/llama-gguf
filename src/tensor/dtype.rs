#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32, F16, BF16,
    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
    Q2K, Q3K, Q4K, Q5K, Q6K,
    IQ2XXS, IQ2XS, IQ3XXS, IQ3S, IQ4XS, IQ4NL,
}
