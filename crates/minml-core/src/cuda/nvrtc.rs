// Tiny NVRTC FFI: just enough to JIT-compile CUDA-C source strings to PTX.
//
// cuda-oxide's cuda-core wraps the CUDA driver API but doesn't yet expose
// NVRTC, so we declare the seven functions we need directly. `build.rs`
// links `-lnvrtc`; the toolkit ships it alongside the driver.

use crate::error::{MinmlError, Result};
use std::ffi::{c_char, c_int, CStr, CString};
use std::ptr;

#[repr(C)]
struct NvrtcProgramOpaque {
    _priv: [u8; 0],
}
type NvrtcProgram = *mut NvrtcProgramOpaque;

#[allow(non_camel_case_types)]
type nvrtcResult = c_int;
const NVRTC_SUCCESS: nvrtcResult = 0;

unsafe extern "C" {
    fn nvrtcCreateProgram(
        prog: *mut NvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> nvrtcResult;
    fn nvrtcDestroyProgram(prog: *mut NvrtcProgram) -> nvrtcResult;
    fn nvrtcCompileProgram(
        prog: NvrtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> nvrtcResult;
    fn nvrtcGetPTXSize(prog: NvrtcProgram, size: *mut usize) -> nvrtcResult;
    fn nvrtcGetPTX(prog: NvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
    fn nvrtcGetProgramLogSize(prog: NvrtcProgram, size: *mut usize) -> nvrtcResult;
    fn nvrtcGetProgramLog(prog: NvrtcProgram, log: *mut c_char) -> nvrtcResult;
    fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char;
}

fn err_str(rc: nvrtcResult) -> String {
    unsafe {
        let p = nvrtcGetErrorString(rc);
        if p.is_null() {
            format!("nvrtc rc={rc}")
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

// RAII guard: makes sure the program is destroyed on every exit path.
struct ProgramGuard(NvrtcProgram);
impl Drop for ProgramGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = nvrtcDestroyProgram(&mut self.0);
            }
        }
    }
}

fn fetch_log(prog: NvrtcProgram) -> String {
    let mut size: usize = 0;
    if unsafe { nvrtcGetProgramLogSize(prog, &mut size) } != NVRTC_SUCCESS || size <= 1 {
        return String::new();
    }
    let mut buf = vec![0u8; size];
    if unsafe { nvrtcGetProgramLog(prog, buf.as_mut_ptr() as *mut c_char) } != NVRTC_SUCCESS {
        return String::new();
    }
    // The log is null-terminated; strip the trailing NUL before decoding.
    if let Some(&0) = buf.last() {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).into_owned()
}

// Compile a CUDA-C source string to PTX. `name` is shown in error logs;
// `arch` is e.g. "sm_70". Targeting a low-ish baseline lets the PTX JIT
// re-target newer GPUs at module-load time.
pub fn compile_to_ptx(src: &str, name: &str, arch: &str) -> Result<String> {
    let c_src = CString::new(src).map_err(|_| MinmlError::Other("nvrtc: src has NUL".into()))?;
    let c_name =
        CString::new(name).map_err(|_| MinmlError::Other("nvrtc: name has NUL".into()))?;

    let mut prog: NvrtcProgram = ptr::null_mut();
    let rc = unsafe {
        nvrtcCreateProgram(
            &mut prog,
            c_src.as_ptr(),
            c_name.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    };
    if rc != NVRTC_SUCCESS {
        return Err(MinmlError::Other(format!(
            "nvrtcCreateProgram: {}",
            err_str(rc)
        )));
    }
    let _guard = ProgramGuard(prog);

    let arch_opt = CString::new(format!("--gpu-architecture={arch}"))
        .map_err(|_| MinmlError::Other("nvrtc: arch has NUL".into()))?;
    let opts: [*const c_char; 1] = [arch_opt.as_ptr()];
    let rc = unsafe { nvrtcCompileProgram(prog, opts.len() as c_int, opts.as_ptr()) };
    if rc != NVRTC_SUCCESS {
        return Err(MinmlError::Other(format!(
            "nvrtcCompileProgram: {} :: {}",
            err_str(rc),
            fetch_log(prog)
        )));
    }

    let mut size: usize = 0;
    let rc = unsafe { nvrtcGetPTXSize(prog, &mut size) };
    if rc != NVRTC_SUCCESS {
        return Err(MinmlError::Other(format!(
            "nvrtcGetPTXSize: {}",
            err_str(rc)
        )));
    }
    let mut buf = vec![0u8; size];
    let rc = unsafe { nvrtcGetPTX(prog, buf.as_mut_ptr() as *mut c_char) };
    if rc != NVRTC_SUCCESS {
        return Err(MinmlError::Other(format!("nvrtcGetPTX: {}", err_str(rc))));
    }
    // PTX is null-terminated; strip the trailing NUL.
    if let Some(&0) = buf.last() {
        buf.pop();
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}
