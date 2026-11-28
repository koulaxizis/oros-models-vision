//! DeepFilterNet3 for orOS: the official libDF runtime (tract) with the
//! official DFN3 model built in, as a WebAssembly module with no imports.
//! C ABI only (no wasm-bindgen, so no JavaScript glue and no eval):
//!   dfo_alloc(n) / dfo_free(p, n)      bytes in the module's memory
//!   dfo_create(channels, atten_lim_db) a state (48 kHz, planar channels)
//!   dfo_hop(st)                        samples per frame and channel (480)
//!   dfo_delay(st)                      samples the output is late (1440)
//!   dfo_set_atten_lim(st, db)          100 or more = no limit
//!   dfo_set_pf_beta(st, beta)          post filter, 0 = off
//!   dfo_process(st, inp, out) -> lsnr  one frame: channels x hop f32 each
//!   dfo_destroy(st)
use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::{ArrayView2, ArrayViewMut2};
use std::alloc::{alloc, dealloc, Layout};

pub struct State {
    m: DfTract,
    ch: usize,
}

#[no_mangle]
pub extern "C" fn dfo_alloc(n: usize) -> *mut u8 {
    unsafe { alloc(Layout::from_size_align(n.max(1), 16).unwrap()) }
}

#[no_mangle]
pub unsafe extern "C" fn dfo_free(p: *mut u8, n: usize) {
    dealloc(p, Layout::from_size_align(n.max(1), 16).unwrap())
}

#[no_mangle]
pub extern "C" fn dfo_create(channels: usize, atten_lim_db: f32) -> *mut State {
    let ch = channels.clamp(1, 2);
    let rp = RuntimeParams::default_with_ch(ch).with_atten_lim(atten_lim_db);
    match DfTract::new(DfParams::default(), &rp) {
        Ok(m) => Box::into_raw(Box::new(State { m, ch })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn dfo_hop(st: *const State) -> usize {
    (*st).m.hop_size
}

#[no_mangle]
pub unsafe extern "C" fn dfo_delay(st: *const State) -> usize {
    let m = &(*st).m;
    m.fft_size - m.hop_size + m.lookahead * m.hop_size
}

#[no_mangle]
pub unsafe extern "C" fn dfo_set_atten_lim(st: *mut State, db: f32) {
    (*st).m.set_atten_lim(db)
}

#[no_mangle]
pub unsafe extern "C" fn dfo_set_pf_beta(st: *mut State, beta: f32) {
    (*st).m.set_pf_beta(beta)
}

#[no_mangle]
pub unsafe extern "C" fn dfo_process(st: *mut State, inp: *const f32, out: *mut f32) -> f32 {
    let s = &mut *st;
    let hop = s.m.hop_size;
    let i = ArrayView2::from_shape((s.ch, hop), std::slice::from_raw_parts(inp, s.ch * hop)).unwrap();
    let o = ArrayViewMut2::from_shape((s.ch, hop), std::slice::from_raw_parts_mut(out, s.ch * hop)).unwrap();
    s.m.process(i, o).unwrap_or(f32::NAN)
}

#[no_mangle]
pub unsafe extern "C" fn dfo_destroy(st: *mut State) {
    if !st.is_null() {
        drop(Box::from_raw(st))
    }
}

#[cfg(target_arch = "wasm32")]
fn no_entropy(_: &mut [u8]) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}
#[cfg(target_arch = "wasm32")]
getrandom::register_custom_getrandom!(no_entropy);
