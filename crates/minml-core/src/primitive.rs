use crate::array::Array;
use crate::error::Result;

// A Primitive is the op-specific node attached to a lazy Array. Its single
// job is to dispatch on the output device. eval() is sync — even WebGPU's
// kernel launches are sync (queue.submit returns immediately); only
// device->host readback is async, and that lives outside this trait.
pub trait Primitive: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn eval(&self, inputs: &[Array], output: &Array) -> Result<()>;
}
