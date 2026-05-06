// minml/device.h
#pragma once

namespace minml {

enum class Device { CPU, CUDA, WebGPU };

Device default_device();
void set_default_device(Device d);

const char* device_name(Device d);

}  // namespace minml
