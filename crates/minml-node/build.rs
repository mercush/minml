// napi-rs's build.rs sets up the linker flags so the resulting cdylib
// loads as a Node native addon (.node). Mirrors @napi-rs/cli's expectation.
fn main() {
    napi_build::setup();
}
