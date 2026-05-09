// Threefry-2x32-20: counter-based, splittable PRNG (Salmon et al. SC'11).
// Pure function from (key, counter) -> uniform u32. Same hash on every
// backend so a given PRNGKey produces the same draws on CPU/WebGPU.

function rotl32(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

const R = [13, 15, 26, 6, 17, 29, 16, 24] as const;

export function threefry_2x32(
  k0: number,
  k1: number,
  ctr0: number,
  ctr1: number,
): [number, number] {
  const k2 = (0x1bd11bda ^ k0 ^ k1) >>> 0;
  let x0 = (ctr0 + k0) >>> 0;
  let x1 = (ctr1 + k1) >>> 0;

  for (let round = 0; round < 20; round++) {
    x0 = (x0 + x1) >>> 0;
    x1 = rotl32(x1, R[round % 8]);
    x1 = (x1 ^ x0) >>> 0;
    if ((round + 1) % 4 === 0) {
      const s = (round + 1) / 4;
      let ks0: number;
      let ks1: number;
      switch (s % 3) {
        case 0:
          ks0 = k0;
          ks1 = k1;
          break;
        case 1:
          ks0 = k1;
          ks1 = k2;
          break;
        default:
          ks0 = k2;
          ks1 = k0;
          break;
      }
      x0 = (x0 + ks0) >>> 0;
      x1 = (((x1 + ks1) >>> 0) + s) >>> 0;
    }
  }
  return [x0, x1];
}

export function threefry_u32(k0: number, k1: number, i: number): number {
  return threefry_2x32(k0, k1, i, 0)[0];
}

// (u >> 8) is in [0, 2^24); 2^-24 is exactly representable in f32 and the
// product is exact in f64 too, so no fround needed to match the Rust output.
export function u32_to_unit_f32(u: number): number {
  return (u >>> 8) * (1 / 16777216);
}
