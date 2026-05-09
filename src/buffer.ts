import { Device } from "./device.js";

export interface Buffer {
  bytes(): number;
  device(): Device;
}
