use crate::threefry::threefry_2x32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PRNGKey {
    k0: u32,
    k1: u32,
}

impl PRNGKey {
    pub const fn new(k0: u32, k1: u32) -> Self {
        Self { k0, k1 }
    }

    // Hash the seed once so seed=0 doesn't produce a degenerate key.
    pub fn from_seed(seed: u32) -> Self {
        let (a, b) = threefry_2x32(seed, 0xCAFEBABE, 0, 0);
        Self { k0: a, k1: b }
    }

    pub fn k0(&self) -> u32 {
        self.k0
    }
    pub fn k1(&self) -> u32 {
        self.k1
    }

    // n derived keys; same parent always yields the same children.
    pub fn split(&self, n: usize) -> Vec<PRNGKey> {
        (0..n)
            .map(|i| {
                let (a, b) = threefry_2x32(self.k0, self.k1, i as u32, 0xC0FFEE);
                PRNGKey { k0: a, k1: b }
            })
            .collect()
    }
}

impl Default for PRNGKey {
    fn default() -> Self {
        Self { k0: 0, k1: 0 }
    }
}
