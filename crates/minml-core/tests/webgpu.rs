// WebGPU integration test. Native build only — uses pollster to drive the
// async init, then runs the same example.cpp scenario through the wgpu
// backend.

#![cfg(feature = "webgpu")]

use minml_core::{add, dot, webgpu, Array, Device};

#[test]
fn webgpu_example_cpp_match() {
    pollster::block_on(async {
        if let Err(e) = webgpu::init().await {
            // No GPU available in CI / sandbox: skip rather than fail.
            eprintln!("skipping: webgpu init failed: {e}");
            return;
        }

        let x = Array::from_f32_1d(vec![1.0, 2.0, 3.0, 4.0], Device::WebGpu).unwrap();
        let y = Array::from_f32_1d(vec![10.0, 20.0, 30.0, 40.0], Device::WebGpu).unwrap();

        let s = add(&x, &y).unwrap();
        let s_v = s.tolist().await.unwrap();
        assert_eq!(s_v, vec![11.0, 22.0, 33.0, 44.0]);

        let d = dot(&x, &y).unwrap();
        let d_v = d.item().await.unwrap();
        assert_eq!(d_v, 300.0);

        let xy = add(&x, &y).unwrap();
        let dd = dot(&xy, &xy).unwrap();
        let dd_v = dd.item().await.unwrap();
        let expected = 11.0_f32.powi(2) + 22.0_f32.powi(2) + 33.0_f32.powi(2) + 44.0_f32.powi(2);
        assert!((dd_v - expected).abs() < 1e-2, "{dd_v} != {expected}");
    });
}
