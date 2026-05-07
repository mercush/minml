// Threefry-2x32-20: counter-based, splittable PRNG (Salmon et al. SC'11).
// Pure function from (key, counter) -> uniform u32. Same hash on every
// backend so a given PRNGKey produces the same draws on CPU/CUDA/WebGPU.
//
// Direct port of src/threefry.h.

#[inline]
fn rotl32(x: u32, k: u32) -> u32 {
    (x << k) | (x >> (32 - k))
}

const R: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];

#[inline]
pub fn threefry_2x32(k0: u32, k1: u32, ctr0: u32, ctr1: u32) -> (u32, u32) {
    let k2 = 0x1BD11BDA ^ k0 ^ k1;
    let mut x0 = ctr0.wrapping_add(k0);
    let mut x1 = ctr1.wrapping_add(k1);

    // 20 rounds, key injected every 4 rounds.
    for round in 0..20u32 {
        x0 = x0.wrapping_add(x1);
        x1 = rotl32(x1, R[(round as usize) % 8]);
        x1 ^= x0;
        if (round + 1) % 4 == 0 {
            let s = (round + 1) / 4;
            let (ks0, ks1) = match s % 3 {
                0 => (k0, k1),
                1 => (k1, k2),
                _ => (k2, k0),
            };
            x0 = x0.wrapping_add(ks0);
            x1 = x1.wrapping_add(ks1).wrapping_add(s);
        }
    }
    (x0, x1)
}

#[inline]
pub fn threefry_u32(k0: u32, k1: u32, i: u32) -> u32 {
    threefry_2x32(k0, k1, i, 0).0
}

#[inline]
pub fn u32_to_unit_f32(u: u32) -> f32 {
    ((u >> 8) as f32) * (1.0 / 16_777_216.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonzero_outputs() {
        // Zero key + zero counter must still produce a non-trivial output
        // (the round-key constant 0x1BD11BDA mixes in even when k0/k1=0).
        let (a, b) = threefry_2x32(0, 0, 0, 0);
        assert!(a != 0 || b != 0);
        // Smoke test: changing any input changes the output.
        let other = threefry_2x32(1, 2, 3, 4);
        assert_ne!((a, b), other);
    }

    #[test]
    fn split_is_deterministic() {
        let (a1, b1) = threefry_2x32(42, 0xCAFEBABE, 0, 0);
        let (a2, b2) = threefry_2x32(42, 0xCAFEBABE, 0, 0);
        assert_eq!((a1, b1), (a2, b2));
    }
}
