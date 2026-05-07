use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda,
    WebGpu,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Device::Cpu => "CPU",
            Device::Cuda => "CUDA",
            Device::WebGpu => "WebGPU",
        })
    }
}

// Default device is a process-wide setting. Atomic so set_default_device
// is callable from any binding without locking.
static DEFAULT_DEVICE: AtomicU8 = AtomicU8::new(0);

fn encode(d: Device) -> u8 {
    match d {
        Device::Cpu => 0,
        Device::Cuda => 1,
        Device::WebGpu => 2,
    }
}
fn decode(v: u8) -> Device {
    match v {
        1 => Device::Cuda,
        2 => Device::WebGpu,
        _ => Device::Cpu,
    }
}

pub fn default_device() -> Device {
    decode(DEFAULT_DEVICE.load(Ordering::Relaxed))
}

pub fn set_default_device(d: Device) {
    DEFAULT_DEVICE.store(encode(d), Ordering::Relaxed);
}
