// CPU backend. Plain heap-allocated bytes; allocate/h2d/d2h are memcpy.
// Kernels are loops. Lives in this crate (always built).
pub(crate) mod backend;
pub(crate) mod kernels;
pub(crate) mod random;
