export enum Device {
  Cpu = "CPU",
  Cuda = "CUDA",
  WebGpu = "WebGPU",
}

let DEFAULT_DEVICE: Device = Device.Cpu;

export function default_device(): Device {
  return DEFAULT_DEVICE;
}

export function set_default_device(d: Device): void {
  DEFAULT_DEVICE = d;
}
