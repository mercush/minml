use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float32,
    Int32,
}

pub const fn dtype_bytes(t: DType) -> usize {
    match t {
        DType::Float32 => 4,
        DType::Int32 => 4,
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DType::Float32 => "Float32",
            DType::Int32 => "Int32",
        })
    }
}
