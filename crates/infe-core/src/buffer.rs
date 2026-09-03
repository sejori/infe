//! Buffer types for crossing the `PyO3` boundary.
//!
//! The boundary rule (BRIEF §5.1) mandates that data crosses as `DLPack` / Arrow
//! / numpy views over pre-allocated buffers, not Python objects. These types

#![allow(unsafe_code)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::missing_panics_doc)]
//! represent raw, typed views into contiguous memory that can be constructed
//! from a `DLPack` capsule on the Python side and dereferenced on the Rust side
//! without copying.

use std::fmt;

/// Element types supported by buffer views.
///
/// These map directly to numpy/DLPack dtypes. The set is intentionally narrow:
/// inference data planes move floats (weights, activations, KV cache) and
/// integers (token ids, block ids, slot mappings, `cu_seqlens`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    /// `float16` — activations, KV cache entries on most GPUs.
    F16 = 0,
    /// `bfloat16` — activations, KV cache entries on H100+.
    BF16 = 1,
    /// `float32` — fallback / CPU-side computation.
    F32 = 2,
    /// `int32` — token ids, block ids, slot mappings.
    I32 = 3,
    /// `int64` — `cu_seqlens`, large index arrays.
    I64 = 4,
    /// `uint8` — raw byte views (e.g. packed bool masks).
    U8 = 5,
}

impl DType {
    /// Size of one element in bytes.
    #[must_use]
    pub const fn byte_len(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }

    /// `DLPack` type code for this dtype.
    #[must_use]
    pub const fn dlpack_code(self) -> u8 {
        match self {
            Self::F16 => 1,
            Self::BF16 => 5,
            Self::F32 => 2,
            Self::I32 => 3,
            Self::I64 => 4,
            Self::U8 => 0,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
        })
    }
}

/// An immutable, typed view into a contiguous buffer.
///
/// In the `PyO3` bindings this is constructed from a `DLPack` `DLTensor` capsule
/// (or a numpy array that exports the array interface). On the Rust side it is
/// a fat pointer + dtype + shape with no ownership: the Python side owns the
/// memory and the view is valid only for the duration of the step call.
///
/// # Safety
///
/// The caller must guarantee that the underlying memory is valid for the
/// lifetime of the view and that the dtype and shape are correct. The view
/// does not copy; it is a borrow.
#[derive(Clone, Copy)]
pub struct BufferView<'a> {
    /// Raw pointer to the start of the buffer.
    ptr: *const u8,
    /// Number of dimensions in the shape.
    ndim: u8,
    /// Shape of the buffer (up to 8 dims; inference tensors are shallow).
    shape: [usize; 8],
    /// Element type.
    dtype: DType,
    /// Lifetime tie to the owning buffer.
    _marker: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> BufferView<'a> {
    /// Construct a view from a raw pointer, shape, and dtype.
    ///
    /// # Safety
    ///
    /// `ptr` must point to at least `shape.iter().product::<usize>() *
    /// dtype.byte_len()` bytes of valid, initialized memory for the lifetime
    /// `'a`.
    #[must_use]
    pub unsafe fn from_raw(ptr: *const u8, shape: &[usize], dtype: DType) -> Self {
        assert!(
            shape.len() <= 8,
            "BufferView supports at most 8 dimensions, got {}",
            shape.len()
        );
        let mut s = [0usize; 8];
        s[..shape.len()].copy_from_slice(shape);
        Self {
            ptr,
            ndim: shape.len() as u8,
            shape: s,
            dtype,
            _marker: std::marker::PhantomData,
        }
    }

    /// Construct a view from a byte slice, inferring 1-D shape.
    #[must_use]
    pub fn from_bytes(bytes: &'a [u8], dtype: DType) -> Self {
        let n = bytes.len() / dtype.byte_len();
        // SAFETY: bytes is valid for 'a, shape is correct.
        unsafe { Self::from_raw(bytes.as_ptr(), &[n], dtype) }
    }

    /// Shape of the buffer (the slice that was passed in).
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim as usize]
    }

    /// Number of elements (product of shape dims).
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape[..self.ndim as usize].iter().product()
    }

    /// Total bytes spanned by this view.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.numel() * self.dtype.byte_len()
    }

    /// Element dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Raw pointer to the buffer start.
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// View as `&[f32]` if the dtype is F32.
    #[must_use]
    pub fn as_f32(&self) -> Option<&'a [f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        let n = self.numel();
        // SAFETY: dtype is F32, numel is correct, ptr is aligned and valid
        // for 'a.
        unsafe { Some(std::slice::from_raw_parts(self.ptr.cast::<f32>(), n)) }
    }

    /// View as `&[i32]` if the dtype is I32.
    #[must_use]
    pub fn as_i32(&self) -> Option<&'a [i32]> {
        if self.dtype != DType::I32 {
            return None;
        }
        let n = self.numel();
        unsafe { Some(std::slice::from_raw_parts(self.ptr.cast::<i32>(), n)) }
    }

    /// View as `&[i64]` if the dtype is I64.
    #[must_use]
    pub fn as_i64(&self) -> Option<&'a [i64]> {
        if self.dtype != DType::I64 {
            return None;
        }
        let n = self.numel();
        unsafe { Some(std::slice::from_raw_parts(self.ptr.cast::<i64>(), n)) }
    }
}

impl fmt::Debug for BufferView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferView")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape())
            .field("numel", &self.numel())
            .finish()
    }
}

/// A mutable, typed view into a contiguous buffer.
///
/// Like [`BufferView`] but allows writing. Used for output buffers that the
/// engine pre-allocates and the Rust component writes into.
#[derive(Debug)]
pub struct BufferViewMut<'a> {
    ptr: *mut u8,
    ndim: u8,
    shape: [usize; 8],
    dtype: DType,
    _marker: std::marker::PhantomData<&'a mut [u8]>,
}

impl<'a> BufferViewMut<'a> {
    /// Construct a mutable view from a raw pointer, shape, and dtype.
    ///
    /// # Safety
    ///
    /// `ptr` must point to at least `shape.iter().product::<usize>() *
    /// dtype.byte_len()` bytes of valid, writable memory for the lifetime
    /// `'a`, and must not be aliased.
    pub unsafe fn from_raw(ptr: *mut u8, shape: &[usize], dtype: DType) -> Self {
        assert!(
            shape.len() <= 8,
            "BufferViewMut supports at most 8 dimensions, got {}",
            shape.len()
        );
        let mut s = [0usize; 8];
        s[..shape.len()].copy_from_slice(shape);
        Self {
            ptr,
            ndim: shape.len() as u8,
            shape: s,
            dtype,
            _marker: std::marker::PhantomData,
        }
    }

    /// Shape of the buffer.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim as usize]
    }

    /// Number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape[..self.ndim as usize].iter().product()
    }

    /// Element dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Demote to an immutable view (cheap).
    #[must_use]
    pub fn as_view(&self) -> BufferView<'a> {
        // SAFETY: same validity as the mutable view.
        unsafe { BufferView::from_raw(self.ptr, self.shape(), self.dtype) }
    }

    /// View as `&mut [f32]` if the dtype is F32.
    #[must_use]
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        let n = self.numel();
        unsafe { Some(std::slice::from_raw_parts_mut(self.ptr.cast::<f32>(), n)) }
    }

    /// View as `&mut [i32]` if the dtype is I32.
    #[must_use]
    pub fn as_i32_mut(&mut self) -> Option<&mut [i32]> {
        if self.dtype != DType::I32 {
            return None;
        }
        let n = self.numel();
        unsafe { Some(std::slice::from_raw_parts_mut(self.ptr.cast::<i32>(), n)) }
    }

    /// View as `&mut [i64]` if the dtype is I64.
    #[must_use]
    pub fn as_i64_mut(&mut self) -> Option<&mut [i64]> {
        if self.dtype != DType::I64 {
            return None;
        }
        let n = self.numel();
        unsafe { Some(std::slice::from_raw_parts_mut(self.ptr.cast::<i64>(), n)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_byte_lengths() {
        assert_eq!(DType::F16.byte_len(), 2);
        assert_eq!(DType::BF16.byte_len(), 2);
        assert_eq!(DType::F32.byte_len(), 4);
        assert_eq!(DType::I32.byte_len(), 4);
        assert_eq!(DType::I64.byte_len(), 8);
        assert_eq!(DType::U8.byte_len(), 1);
    }

    #[test]
    fn dtype_dlpack_codes() {
        // DLPack DLDeviceType / dtype codes (dlpack.h)
        assert_eq!(DType::U8.dlpack_code(), 0);
        assert_eq!(DType::F16.dlpack_code(), 1);
        assert_eq!(DType::F32.dlpack_code(), 2);
        assert_eq!(DType::I32.dlpack_code(), 3);
        assert_eq!(DType::I64.dlpack_code(), 4);
        assert_eq!(DType::BF16.dlpack_code(), 5);
    }

    #[test]
    fn buffer_view_from_f32_slice() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        let view = BufferView::from_bytes(bytes, DType::F32);
        assert_eq!(view.dtype(), DType::F32);
        assert_eq!(view.shape(), &[4]);
        assert_eq!(view.numel(), 4);
        assert_eq!(view.byte_len(), 16);
        assert_eq!(view.as_f32(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn buffer_view_from_i32_slice() {
        let data: Vec<i32> = vec![10, 20, 30];
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        let view = BufferView::from_bytes(bytes, DType::I32);
        assert_eq!(view.as_i32(), Some(&[10, 20, 30][..]));
    }

    #[test]
    fn buffer_view_dtype_mismatch_returns_none() {
        let data: Vec<f32> = vec![1.0, 2.0];
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        let view = BufferView::from_bytes(bytes, DType::F32);
        assert_eq!(view.as_i32(), None);
        assert_eq!(view.as_i64(), None);
    }

    #[test]
    fn buffer_view_multidim() {
        let data: Vec<f32> = vec![0.0; 12]; // 3x4
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        // SAFETY: 12 elements of f32, contiguous.
        let view = unsafe { BufferView::from_raw(bytes.as_ptr(), &[3, 4], DType::F32) };
        assert_eq!(view.shape(), &[3, 4]);
        assert_eq!(view.numel(), 12);
        assert_eq!(view.byte_len(), 48);
    }

    #[test]
    fn buffer_view_mut_f32() {
        let mut data: Vec<f32> = vec![0.0; 4];
        let mut view =
            unsafe { BufferViewMut::from_raw(data.as_mut_ptr().cast::<u8>(), &[4], DType::F32) };
        {
            let slice = view.as_f32_mut().unwrap();
            slice[0] = 42.0;
            slice[3] = 99.0;
        }
        assert_eq!(data, vec![42.0, 0.0, 0.0, 99.0]);
    }

    #[test]
    fn buffer_view_mut_demote_to_view() {
        let mut data: Vec<i64> = vec![1, 2, 3];
        let view_mut =
            unsafe { BufferViewMut::from_raw(data.as_mut_ptr().cast::<u8>(), &[3], DType::I64) };
        let view = view_mut.as_view();
        assert_eq!(view.as_i64(), Some(&[1i64, 2, 3][..]));
    }
}
