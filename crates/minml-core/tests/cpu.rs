// Integration tests: round-trip the CPU backend through the lazy graph,
// verifying that examples/example.cpp's outputs reproduce in Rust.

use minml_core::{
    add, categorical_sample, dirichlet_sample, dot, gather, mul, ones, randint, slice_axis0,
    stack, vmap_apply, Array, DType, Device, PRNGKey,
};

#[test]
fn example_cpp_match() {
    // Mirrors examples/example.cpp:
    //   x = [1, 2, 3, 4]; y = [10, 20, 30, 40]
    //   add(x, y) = [11, 22, 33, 44]
    //   dot(x, y) = 1*10 + 2*20 + 3*30 + 4*40 = 300
    //   dot(x+y, x+y) = 11^2 + 22^2 + 33^2 + 44^2 = 3630.
    let x = Array::from_f32_1d(vec![1.0, 2.0, 3.0, 4.0], Device::Cpu).unwrap();
    let y = Array::from_f32_1d(vec![10.0, 20.0, 30.0, 40.0], Device::Cpu).unwrap();

    let s = add(&x, &y).unwrap();
    assert_eq!(s.tolist_sync().unwrap(), vec![11.0, 22.0, 33.0, 44.0]);

    let d = dot(&x, &y).unwrap();
    assert_eq!(d.item_sync().unwrap(), 300.0);

    // Lazy graph: add evaluated twice via shared subgraph.
    let xy = add(&x, &y).unwrap();
    let dd = dot(&xy, &xy).unwrap();
    assert_eq!(dd.item_sync().unwrap(), 11.0_f32.powi(2) + 22.0_f32.powi(2)
        + 33.0_f32.powi(2) + 44.0_f32.powi(2));
}

#[test]
fn mul_works() {
    let a = Array::from_f32_1d(vec![1.0, 2.0, 3.0], Device::Cpu).unwrap();
    let b = Array::from_f32_1d(vec![4.0, 5.0, 6.0], Device::Cpu).unwrap();
    let r = mul(&a, &b).unwrap();
    assert_eq!(r.tolist_sync().unwrap(), vec![4.0, 10.0, 18.0]);
}

#[test]
fn ones_float() {
    let r = ones(vec![5], DType::Float32, Device::Cpu);
    assert_eq!(r.tolist_sync().unwrap(), vec![1.0; 5]);
}

#[test]
fn ones_int() {
    let r = ones(vec![3], DType::Int32, Device::Cpu);
    assert_eq!(r.tolist_int_sync().unwrap(), vec![1; 3]);
}

#[test]
fn randint_in_range() {
    let r = randint(123, 456, 10, 20, vec![64], Device::Cpu);
    let v = r.tolist_int_sync().unwrap();
    assert_eq!(v.len(), 64);
    for x in &v {
        assert!(*x >= 10 && *x < 20, "out of range: {x}");
    }
    // Determinism: same key + op + shape -> same draws.
    let r2 = randint(123, 456, 10, 20, vec![64], Device::Cpu);
    assert_eq!(v, r2.tolist_int_sync().unwrap());
}

#[test]
fn gather_works() {
    let table = Array::from_f32_with_shape(
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![3, 2],
        Device::Cpu,
    )
    .unwrap();
    let idx = Array::from_i32_1d(vec![2, 0, 1], Device::Cpu).unwrap();
    let r = gather(&table, &idx).unwrap();
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.tolist_sync().unwrap(), vec![50.0, 60.0, 10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn dirichlet_sums_to_one() {
    let alpha = Array::from_f32_1d(vec![1.0, 1.0, 1.0], Device::Cpu).unwrap();
    let key = PRNGKey::from_seed(7);
    let s = dirichlet_sample(key.k0(), key.k1(), &alpha, vec![4]).unwrap();
    assert_eq!(s.shape(), &[4, 3]);
    let v = s.tolist_sync().unwrap();
    for row in v.chunks(3) {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row sum {sum}");
        for x in row {
            assert!(*x >= 0.0);
        }
    }
}

#[test]
fn categorical_in_range() {
    let probs = Array::from_f32_1d(vec![0.1, 0.7, 0.2], Device::Cpu).unwrap();
    let key = PRNGKey::from_seed(13);
    let s = categorical_sample(key.k0(), key.k1(), &probs, vec![100]).unwrap();
    let v = s.tolist_int_sync().unwrap();
    for x in &v {
        assert!(*x >= 0 && *x < 3);
    }
}

#[test]
fn prng_split_deterministic() {
    let parent = PRNGKey::from_seed(1234);
    let a = parent.split(5);
    let b = parent.split(5);
    assert_eq!(a, b);
    assert_ne!(a[0], a[1]);
}

#[test]
fn slice_and_stack_roundtrip() {
    let a = Array::from_f32_with_shape(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3, 2],
        Device::Cpu,
    )
    .unwrap();
    let parts = slice_axis0(&a).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].shape(), &[2]);
    let restacked = stack(&parts).unwrap();
    assert_eq!(restacked.shape(), &[3, 2]);
    assert_eq!(restacked.tolist_sync().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn vmap_add_per_row() {
    let xs = Array::from_f32_with_shape(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3, 2],
        Device::Cpu,
    )
    .unwrap();
    let ys = Array::from_f32_with_shape(
        vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
        vec![3, 2],
        Device::Cpu,
    )
    .unwrap();
    let mut f = |_iter: usize, args: &[Array]| -> minml_core::MinmlResult<Vec<Array>> {
        Ok(vec![add(&args[0], &args[1])?])
    };
    let out = vmap_apply(3, &[xs, ys], &[0, 0], &mut f).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(
        out[0].tolist_sync().unwrap(),
        vec![11.0, 12.0, 23.0, 24.0, 35.0, 36.0]
    );
}
