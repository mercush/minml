import { threefry_2x32 } from "./threefry.js";

export class PRNGKey {
  readonly k0: number;
  readonly k1: number;

  constructor(k0: number, k1: number) {
    this.k0 = k0 >>> 0;
    this.k1 = k1 >>> 0;
  }

  // Hash the seed once so seed=0 doesn't produce a degenerate key.
  static from_seed(seed: number): PRNGKey {
    const [a, b] = threefry_2x32(seed >>> 0, 0xcafebabe, 0, 0);
    return new PRNGKey(a, b);
  }

  // n derived keys; same parent always yields the same children.
  split(n: number): PRNGKey[] {
    const out: PRNGKey[] = [];
    for (let i = 0; i < n; i++) {
      const [a, b] = threefry_2x32(this.k0, this.k1, i >>> 0, 0xc0ffee);
      out.push(new PRNGKey(a, b));
    }
    return out;
  }

  equals(other: PRNGKey): boolean {
    return this.k0 === other.k0 && this.k1 === other.k1;
  }
}
