//! Shopping Sessions pipeline. rules.md is the spec; each phase names the rule it implements.
//!
//! One shuffle by person: every ID is resolved to its person while its line is parsed.
//!   0. links  -> lock-free table of every linked ID (parallel insert + union-find); person = root slot index,
//!                name = byte-wise smallest member. IDs first seen in events/clicks become their own person.
//!   1. events -> parse in parallel; a valid event gets its person from the table at once and goes, 40 bytes,
//!                to its event_id bucket (invalid ones too: they still win dedup). Clicks go, 24 bytes, straight
//!                to their person bucket.
//!   2. per event_id bucket: dedup (smallest ingest_ts wins, exact on the ID) -> 16 byte records to person buckets
//!   3. per person bucket: counting sort by person, sort by ts, sessionize, credit, assist, aggregate
//!   4. write Parquet

use memmap2::Mmap;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::time::Instant;

type Bytes = &'static [u8];

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;
const CHUNK: usize = 4 << 20;
const EBITS: u32 = 9;
const EBK: usize = 1 << EBITS; // event_id buckets
const NB: usize = 256; // person buckets: pid & 255

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags))
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

// ---------------------------------------------------------------- hashing

#[inline]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}

#[inline]
fn hash_bytes(b: &[u8]) -> u64 {
    const K: u64 = 0x9E3779B97F4A7C15;
    let mut h = K ^ (b.len() as u64);
    let mut chunks = b.chunks_exact(8);
    for c in &mut chunks {
        h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(K).rotate_left(29);
    }
    let r = chunks.remainder();
    if !r.is_empty() {
        let mut buf = [0u8; 8];
        buf[..r.len()].copy_from_slice(r);
        h = (h ^ u64::from_le_bytes(buf)).wrapping_mul(K).rotate_left(29);
    }
    mix(h)
}

/// The `len` (<= 16) bytes at `p`, zero padded to 16; reads 16 bytes.
#[inline(always)]
unsafe fn load16(p: *const u8, len: usize) -> u128 {
    let v = (p as *const u128).read_unaligned();
    if len == 0 {
        0
    } else {
        v & (u128::MAX >> (128 - 8 * len as u32))
    }
}

/// `hash_bytes` of a key of `len` (<= 16) bytes given as `load16` of it.
#[inline(always)]
fn hash16(v: u128, len: usize) -> u64 {
    const K: u64 = 0x9E3779B97F4A7C15;
    let round = |h: u64, w: u64| (h ^ w).wrapping_mul(K).rotate_left(29);
    let h0 = K ^ len as u64;
    let h1 = round(h0, v as u64);
    let h2 = round(h1, (v >> 64) as u64);
    mix(if len > 8 { h2 } else if len > 0 { h1 } else { h0 })
}

/// Hasher for byte-string keys: one call to `write` carries the whole key.
#[derive(Default)]
struct BytesHasher(u64);
impl Hasher for BytesHasher {
    fn write(&mut self, b: &[u8]) {
        self.0 = self.0.rotate_left(5) ^ hash_bytes(b);
    }
    fn write_usize(&mut self, _: usize) {}
    fn finish(&self) -> u64 {
        self.0
    }
}
type BytesMap<V> = HashMap<Bytes, V, BuildHasherDefault<BytesHasher>>;

/// An ID of up to 15 bytes stored inline (zero padded, length in the last byte), else a pointer to it.
/// Inline IDs compare as one 128-bit value: no pointer chase into the input files.
#[derive(Clone, Copy)]
struct SStr([u8; 16]);
impl SStr {
    const LONG: u8 = 0xFF;
    #[inline]
    fn new(b: Bytes) -> SStr {
        let mut x = [0u8; 16];
        if b.len() <= 15 {
            x[..b.len()].copy_from_slice(b);
            x[15] = b.len() as u8;
        } else {
            x[..8].copy_from_slice(&(b.as_ptr() as u64).to_le_bytes());
            x[8..16].copy_from_slice(&(b.len() as u64).to_le_bytes());
            x[15] = SStr::LONG; // lengths stay far below 2^56
        }
        SStr(x)
    }
    #[inline]
    fn get(&self) -> &[u8] {
        if self.0[15] != SStr::LONG {
            &self.0[..self.0[15] as usize]
        } else {
            let p = u64::from_le_bytes(self.0[..8].try_into().unwrap()) as *const u8;
            let mut l = [0u8; 8];
            l[..7].copy_from_slice(&self.0[8..15]);
            unsafe { std::slice::from_raw_parts(p, u64::from_le_bytes(l) as usize) }
        }
    }
}
impl PartialEq for SStr {
    #[inline]
    fn eq(&self, o: &Self) -> bool {
        if self.0[15] != SStr::LONG && o.0[15] != SStr::LONG {
            u128::from_ne_bytes(self.0) == u128::from_ne_bytes(o.0)
        } else {
            self.get() == o.get()
        }
    }
}


impl SStr {
    /// Same as `new` for a field of a file that ends at address `end`: one 16 byte load when it fits.
    #[inline(always)]
    fn field(b: Bytes, end: usize) -> SStr {
        let l = b.len();
        if l <= 15 && b.as_ptr() as usize + 16 <= end {
            let w = unsafe { (b.as_ptr() as *const [u8; 16]).read_unaligned() };
            let x = (u128::from_le_bytes(w) & ((1u128 << (8 * l)) - 1)) | ((l as u128) << 120);
            return SStr(x.to_le_bytes());
        }
        SStr::new(b)
    }
    /// `hash_bytes` of the ID.
    #[inline(always)]
    fn hash(&self) -> u64 {
        if self.0[15] != SStr::LONG {
            let len = self.0[15] as usize;
            hash16(u128::from_le_bytes(self.0) & (u128::MAX >> 8), len)
        } else {
            hash_bytes(self.get())
        }
    }
}

// ---------------------------------------------------------------- CSV dialect

/// A field is either unquoted with no `"` in it, or `"` + no `"` + `"`.
#[inline]
fn unquote(f: &[u8]) -> Option<&[u8]> {
    match f.first() {
        Some(b'"') => {
            if f.len() >= 2 && f[f.len() - 1] == b'"' && !f[1..f.len() - 1].contains(&b'"') {
                Some(&f[1..f.len() - 1])
            } else {
                None
            }
        }
        _ => {
            if f.contains(&b'"') {
                None
            } else {
                Some(f)
            }
        }
    }
}

/// Exactly N comma-separated fields, each well quoted; otherwise the line is malformed.
#[inline]
fn split<const N: usize>(line: &'static [u8]) -> Option<[Bytes; N]> {
    let mut out: [Bytes; N] = [&[]; N];
    let mut n = 0;
    let mut start = 0;
    for i in 0..=line.len() {
        if i == line.len() || line[i] == b',' {
            if n == N {
                return None;
            }
            out[n] = unquote(&line[start..i])?;
            n += 1;
            start = i + 1;
        }
    }
    if n == N {
        Some(out)
    } else {
        None
    }
}

/// Bit k set when b[k] is `,`, `"` or `\n`, for the first min(64, len) bytes.
#[inline(always)]
fn special_mask(b: &[u8]) -> u64 {
    if b.len() < 64 {
        let mut m = 0u64;
        for (k, &c) in b.iter().enumerate() {
            m |= ((c == b',' || c == b'"' || c == b'\n') as u64) << k;
        }
        return m;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let p = b.as_ptr();
        let (cm, qt, nl) = (vdupq_n_u8(b','), vdupq_n_u8(b'"'), vdupq_n_u8(b'\n'));
        let hit = |v: uint8x16_t| vorrq_u8(vorrq_u8(vceqq_u8(v, cm), vceqq_u8(v, qt)), vceqq_u8(v, nl));
        let bits: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bits = vld1q_u8(bits.as_ptr());
        let t0 = vandq_u8(hit(vld1q_u8(p)), bits);
        let t1 = vandq_u8(hit(vld1q_u8(p.add(16))), bits);
        let t2 = vandq_u8(hit(vld1q_u8(p.add(32))), bits);
        let t3 = vandq_u8(hit(vld1q_u8(p.add(48))), bits);
        let s0 = vpaddq_u8(t0, t1);
        let s1 = vpaddq_u8(t2, t3);
        let s = vpaddq_u8(s0, s1);
        let s = vpaddq_u8(s, s);
        return vgetq_lane_u64(vreinterpretq_u64_u8(s), 0);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let p = b.as_ptr() as *const __m256i;
        let (cm, qt, nl) = (_mm256_set1_epi8(b',' as i8), _mm256_set1_epi8(b'"' as i8), _mm256_set1_epi8(b'\n' as i8));
        let hit = |v: __m256i| {
            _mm256_movemask_epi8(_mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(v, cm), _mm256_cmpeq_epi8(v, qt)), _mm256_cmpeq_epi8(v, nl))) as u32 as u64
        };
        return hit(_mm256_loadu_si256(p)) | (hit(_mm256_loadu_si256(p.add(1))) << 32);
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    unsafe {
        use std::arch::x86_64::*;
        let p = b.as_ptr() as *const __m128i;
        let (cm, qt, nl) = (_mm_set1_epi8(b',' as i8), _mm_set1_epi8(b'"' as i8), _mm_set1_epi8(b'\n' as i8));
        let hit = |v: __m128i| {
            _mm_movemask_epi8(_mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(v, cm), _mm_cmpeq_epi8(v, qt)), _mm_cmpeq_epi8(v, nl))) as u32 as u64
        };
        return hit(_mm_loadu_si128(p)) | (hit(_mm_loadu_si128(p.add(1))) << 16) | (hit(_mm_loadu_si128(p.add(2))) << 32) | (hit(_mm_loadu_si128(p.add(3))) << 48);
    }
    #[allow(unreachable_code)]
    {
        let mut m = 0u64;
        for (k, &c) in b[..64].iter().enumerate() {
            m |= ((c == b',' || c == b'"' || c == b'\n') as u64) << k;
        }
        m
    }
}

/// [commas, quotes, newlines]: bit k set when p[k] is that byte, for the 64 bytes at `p` (all must be readable).
#[inline(always)]
unsafe fn masks64(p: *const u8) -> [u64; 3] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        let (a, b) = (_mm256_loadu_si256(p as *const __m256i), _mm256_loadu_si256(p.add(32) as *const __m256i));
        let m = |c: u8| {
            let k = _mm256_set1_epi8(c as i8);
            (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, k)) as u32 as u64) | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, k)) as u32 as u64) << 32)
        };
        return [m(b','), m(b'"'), m(b'\n')];
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        use std::arch::x86_64::*;
        let v = [0, 16, 32, 48].map(|o| _mm_loadu_si128(p.add(o) as *const __m128i));
        let m = |c: u8| {
            let k = _mm_set1_epi8(c as i8);
            (0..4).fold(0u64, |acc, i| acc | (_mm_movemask_epi8(_mm_cmpeq_epi8(v[i], k)) as u32 as u64) << (16 * i))
        };
        return [m(b','), m(b'"'), m(b'\n')];
    }
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let v = [vld1q_u8(p), vld1q_u8(p.add(16)), vld1q_u8(p.add(32)), vld1q_u8(p.add(48))];
        let bits: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bits = vld1q_u8(bits.as_ptr());
        let m = |c: u8| {
            let k = vdupq_n_u8(c);
            let t = |x: uint8x16_t| vandq_u8(vceqq_u8(x, k), bits);
            let s = vpaddq_u8(vpaddq_u8(t(v[0]), t(v[1])), vpaddq_u8(t(v[2]), t(v[3])));
            vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(s, s)), 0)
        };
        return [m(b','), m(b'"'), m(b'\n')];
    }
    #[allow(unreachable_code)]
    {
        let mut r = [0u64; 3];
        for k in 0..64 {
            let c = *p.add(k);
            r[0] |= ((c == b',') as u64) << k;
            r[1] |= ((c == b'"') as u64) << k;
            r[2] |= ((c == b'\n') as u64) << k;
        }
        r
    }
}

/// Calls `f` with the N unquoted fields of every well-formed line that starts in [start, end)
/// of `file`, skipping the header line. Same dialect as `split`, in one SIMD pass over the chunk.
#[inline]
#[allow(unused_assignments)]
fn for_records<const N: usize>(file: Bytes, start: usize, end: usize, mut f: impl FnMut([Bytes; N])) {
    let len = file.len();
    let first = if start == 0 { 0 } else { start - 1 };
    let Some(nl) = memchr::memchr(b'\n', &file[first..]) else { return };
    let mut line = first + nl + 1; // first line start at or after `start` (the header when start == 0)
    if line >= end || line >= len {
        return;
    }
    let mut out: [Bytes; N] = [&[]; N];
    let mut n = 0usize; // fields closed so far
    let mut fs = line; // field start
    let mut q = 0u32; // quotes in this field
    let mut qlast = 0usize;
    let mut qfirst = 0usize;
    let mut bad = false;
    // closes the field [fs, i): returns false when malformed
    macro_rules! close {
        ($i:expr) => {{
            let i = $i;
            if n == N {
                bad = true;
            } else if q == 0 {
                out[n] = &file[fs..i];
            } else if q == 2 && qfirst == fs && qlast + 1 == i {
                out[n] = &file[fs + 1..i - 1];
            } else {
                bad = true;
            }
            n += 1;
            q = 0;
            fs = i + 1;
        }};
    }
    let mut p = line;
    while p < len {
        let mut m = special_mask(&file[p..]);
        while m != 0 {
        let i = p + m.trailing_zeros() as usize;
        m &= m - 1;
        match file[i] {
            b'"' => {
                if q == 0 {
                    qfirst = i;
                }
                q += 1;
                qlast = i;
            }
            b',' => close!(i),
            _ => {
                close!(i);
                if !bad && n == N {
                    f(out);
                }
                n = 0;
                bad = false;
                line = i + 1;
                if line >= end {
                    return;
                }
            }
        }
        }
        p += 64;
    }
    // last line without a trailing newline
    if line < len {
        close!(len);
        if !bad && n == N {
            f(out);
        }
    }
}

/// Calls `f` for every line that starts in [start, end) of `file`, skipping the header line.
#[inline]
fn for_lines(file: Bytes, start: usize, end: usize, mut f: impl FnMut(Bytes)) {
    let len = file.len();
    let mut p = if start == 0 {
        match file.iter().position(|&c| c == b'\n') {
            Some(i) => i + 1,
            None => return,
        }
    } else {
        match file[start - 1..].iter().position(|&c| c == b'\n') {
            Some(i) => start - 1 + i + 1,
            None => return,
        }
    };
    while p < end && p < len {
        let e = match file[p..].iter().position(|&c| c == b'\n') {
            Some(i) => p + i,
            None => len,
        };
        f(&file[p..e]);
        p = e + 1;
    }
}

// ---------------------------------------------------------------- timestamps




fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// `YYYY-MM-DDTHH:MM:SS` (the first 19 bytes) as epoch ms, calendar-checked.
#[inline]
fn civil(b: &[u8]) -> Option<i64> {
    let b: &[u8; 19] = b[..19].try_into().ok()?;
    let d = |i: usize| b[i].wrapping_sub(b'0') as u32;
    // every digit position must hold 0-9 and every separator must match
    let mut bad = (b[4] ^ b'-') | (b[7] ^ b'-') | (b[10] ^ b'T') | (b[13] ^ b':') | (b[16] ^ b':');
    for i in [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18] {
        bad |= (d(i) > 9) as u8;
    }
    if bad != 0 {
        return None;
    }
    let y = d(0) * 1000 + d(1) * 100 + d(2) * 10 + d(3);
    let mo = d(5) * 10 + d(6);
    let dd = d(8) * 10 + d(9);
    let h = d(11) * 10 + d(12);
    let mi = d(14) * 10 + d(15);
    let sec = d(17) * 10 + d(18);
    const DIM: [u32; 13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if mo.wrapping_sub(1) > 11 || dd == 0 || h > 23 || mi > 59 || sec > 59 {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    if dd > DIM[mo as usize] + (mo == 2 && leap) as u32 {
        return None;
    }
    let days = days_from_civil(y as i64, mo as i64, dd as i64);
    Some(days * DAY + (h * 3_600_000 + mi * 60_000 + sec * 1000) as i64)
}

/// F1 `...Z`, F2 `...+HH:MM`, F3 digits = epoch ms. None = bad timestamp.
#[inline]
fn parse_ts(b: &[u8]) -> Option<i64> {
    match b.len() {
        20 if b[19] == b'Z' => civil(b),
        25 if (b[19] == b'+' || b[19] == b'-') && b[22] == b':' => {
            let (h1, h2, m1, m2) = (b[20].wrapping_sub(b'0'), b[21].wrapping_sub(b'0'), b[23].wrapping_sub(b'0'), b[24].wrapping_sub(b'0'));
            if h1 > 9 || h2 > 9 || m1 > 9 || m2 > 9 {
                return None;
            }
            let (oh, om) = ((h1 * 10 + h2) as i64, (m1 * 10 + m2) as i64);
            if oh > 23 || om > 59 {
                return None;
            }
            let off = (oh * 60 + om) * 60_000;
            let t = civil(b)?;
            Some(if b[19] == b'+' { t - off } else { t + off })
        }
        0 => None,
        _ => {
            // F3: digits only (an all-digit value never has `Z`, `+` or `-` at index 19)
            let mut v = 0i64;
            for &c in b {
                let x = c.wrapping_sub(b'0');
                if x > 9 {
                    return None;
                }
                v = v.checked_mul(10)?.checked_add(x as i64)?;
            }
            Some(v)
        }
    }
}

/// Digits only, 1..=10,000,000.
#[inline]
fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() {
        return None;
    }
    let mut v = 0u64;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (c - b'0') as u64;
        if v > 10_000_000 {
            return None;
        }
    }
    if v == 0 {
        None
    } else {
        Some(v as u32)
    }
}

// Fast paths of `civil` / `parse_ts` for a field inside a line with 16+ readable bytes past its end (SWAR, no copies).

const HI: u64 = 0x8080_8080_8080_8080;

/// `civil` on the 19 bytes at `p` (reads 20).
#[inline(always)]
unsafe fn civil_at(p: *const u8) -> Option<i64> {
    // x ^ pattern: a digit becomes 0..=9, a matching separator 0. Per byte, adding 0x76 (digit) or 0x7F (separator)
    // sets the high bit exactly when the byte is out of range; a carry out of a byte only follows a byte whose own
    // high bit is set, so it can never hide a bad byte.
    let a = (p as *const u64).read_unaligned() ^ u64::from_le_bytes(*b"0000-00-");
    let b = (p.add(8) as *const u64).read_unaligned() ^ u64::from_le_bytes(*b"00T00:00");
    let c = ((p.add(16) as *const u32).read_unaligned() as u64 & 0xFF_FFFF) ^ u64::from_le_bytes(*b":00\0\0\0\0\0");
    const AA: u64 = u64::from_le_bytes([0x76, 0x76, 0x76, 0x76, 0x7F, 0x76, 0x76, 0x7F]);
    const AB: u64 = u64::from_le_bytes([0x76, 0x76, 0x7F, 0x76, 0x76, 0x7F, 0x76, 0x76]);
    const AC: u64 = u64::from_le_bytes([0x7F, 0x76, 0x76, 0, 0, 0, 0, 0]);
    if (a.wrapping_add(AA) | a | b.wrapping_add(AB) | b | c.wrapping_add(AC) | c) & HI != 0 {
        return None;
    }
    // byte k of x * 10 + (x >> 8) = 10 * digit k + digit k+1
    let pa = a * 10 + (a >> 8);
    let pb = b * 10 + (b >> 8);
    let y = (pa & 0xFF) * 100 + ((pa >> 16) & 0xFF);
    let mo = (pa >> 40) & 0xFF;
    let dd = pb & 0xFF;
    let h = (pb >> 24) & 0xFF;
    let mi = (pb >> 48) & 0xFF;
    let sec = ((c >> 8) & 0xFF) * 10 + ((c >> 16) & 0xFF);
    const DIM: [u64; 13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if (mo.wrapping_sub(1) > 11) | (dd == 0) | (h > 23) | (mi > 59) | (sec > 59) {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    if dd > DIM[mo as usize] + ((mo == 2) & leap) as u64 {
        return None;
    }
    let days = days_from_civil(y as i64, mo as i64, dd as i64);
    Some(days * DAY + (h * 3_600_000 + mi * 60_000 + sec * 1000) as i64)
}

/// 8 digit values (byte 0 the most significant) as a number.
#[inline(always)]
fn swar8(d: u64) -> u64 {
    let d = d * 10 + (d >> 8);
    ((d & 0x0000_00FF_0000_00FF).wrapping_mul(100 + (1_000_000 << 32))).wrapping_add(((d >> 16) & 0x0000_00FF_0000_00FF).wrapping_mul(1 + (10_000 << 32))) >> 32
}

/// `parse_ts` on the `len` bytes at `p`.
#[inline(always)]
unsafe fn ts_at(p: *const u8, len: usize) -> Option<i64> {
    let at = |i: usize| *p.add(i);
    if len == 20 && at(19) == b'Z' {
        return civil_at(p);
    }
    if len == 25 && (at(19) == b'+' || at(19) == b'-') && at(22) == b':' {
        let (h1, h2, m1, m2) = (at(20).wrapping_sub(b'0'), at(21).wrapping_sub(b'0'), at(23).wrapping_sub(b'0'), at(24).wrapping_sub(b'0'));
        if h1 > 9 || h2 > 9 || m1 > 9 || m2 > 9 {
            return None;
        }
        let (oh, om) = ((h1 * 10 + h2) as i64, (m1 * 10 + m2) as i64);
        if oh > 23 || om > 59 {
            return None;
        }
        let off = (oh * 60 + om) * 60_000;
        let t = civil_at(p)?;
        return Some(if at(19) == b'+' { t - off } else { t + off });
    }
    if len == 0 || len > 16 {
        return parse_ts(std::slice::from_raw_parts(p, len));
    }
    // F3 of 1..=16 bytes: right-align into 16 bytes behind leading '0's, check all digits, convert
    let sh = 8 * (16 - len) as u32;
    let zeros = u128::from_le_bytes([b'0'; 16]);
    let v = ((p as *const u128).read_unaligned() << sh) | (zeros & ((1u128 << sh) - 1));
    let d = v ^ zeros;
    let (lo, hi) = (d as u64, (d >> 64) as u64);
    if (lo.wrapping_add(0x7676_7676_7676_7676) | lo | hi.wrapping_add(0x7676_7676_7676_7676) | hi) & HI != 0 {
        return None;
    }
    Some((swar8(lo) * 100_000_000 + swar8(hi)) as i64)
}

// ---------------------------------------------------------------- input files

fn map_file(p: &Path) -> Bytes {
    let f = std::fs::File::open(p).unwrap_or_else(|e| die(&format!("open {}: {e}", p.display())));
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if len == 0 {
        return &[];
    }
    let m = unsafe { Mmap::map(&f) }.unwrap_or_else(|e| die(&format!("mmap {}: {e}", p.display())));
    let m: &'static Mmap = Box::leak(Box::new(m));
    &m[..]
}

fn csv_files(dir: &Path) -> Vec<Bytes> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "csv").unwrap_or(false) && p.is_file())
            .collect(),
        Err(e) => die(&format!("read_dir {}: {e}", dir.display())),
    };
    paths.sort();
    paths.iter().map(|p| map_file(p)).collect()
}

/// (file, start, end) pieces of about CHUNK bytes, largest files first.
fn chunks(files: &[Bytes]) -> Vec<(Bytes, usize, usize)> {
    chunks_of(files, CHUNK)
}

fn chunks_of(files: &[Bytes], size: usize) -> Vec<(Bytes, usize, usize)> {
    let mut out = Vec::new();
    for &f in files {
        let mut s = 0;
        while s < f.len() {
            let e = (s + size).min(f.len());
            out.push((f, s, e));
            s = e;
        }
    }
    out
}

/// (user + sys CPU seconds, instructions retired) of this process so far; instructions only on macOS
fn usage() -> (f64, u64) {
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) } != 0 {
        return (0.0, 0);
    }
    let t = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 * 1e-6;
    #[cfg(target_os = "macos")]
    let ins = unsafe {
        let mut ri: libc::rusage_info_v4 = std::mem::zeroed();
        let p = &mut ri as *mut libc::rusage_info_v4 as *mut libc::rusage_info_t;
        if libc::proc_pid_rusage(libc::getpid(), libc::RUSAGE_INFO_V4, p) == 0 {
            ri.ri_instructions
        } else {
            0
        }
    };
    #[cfg(not(target_os = "macos"))]
    let ins = 0;
    (t(u.ru_utime) + t(u.ru_stime), ins)
}

fn die(msg: &str) -> ! {
    eprintln!("fatal: {msg}");
    std::process::exit(1)
}

/// Runs `work(state, i)` for i in 0..n across `threads` threads; returns each thread's state.
fn par<S: Send>(threads: usize, n: usize, init: impl Fn() -> S + Sync, work: impl Fn(&mut S, usize) + Sync) -> Vec<S> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut s = init();
                    loop {
                        let i = next.fetch_add(1, Relaxed);
                        if i >= n {
                            break;
                        }
                        work(&mut s, i);
                    }
                    s
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Parses a small clean file (products.csv / campaigns.csv) into its 2-field rows.
fn small_table(path: &Path) -> Vec<(Bytes, Bytes)> {
    let f = map_file(path);
    let mut rows = Vec::new();
    for_lines(f, 0, f.len(), |l| {
        if let Some([a, b]) = split::<2>(l) {
            rows.push((a, b));
        }
    });
    rows
}

fn manifest_ts(text: &str, key: &str) -> i64 {
    let at = text.find(&format!("\"{key}\"")).unwrap_or_else(|| die(&format!("manifest: no {key}")));
    let rest = &text[at + key.len() + 2..];
    let q1 = rest.find('"').unwrap_or_else(|| die("manifest: bad value")) + 1;
    let q2 = rest[q1..].find('"').unwrap_or_else(|| die("manifest: bad value")) + q1;
    parse_ts(rest[q1..q2].as_bytes()).unwrap_or_else(|| die(&format!("manifest: bad {key}")))
}

// ---------------------------------------------------------------- records

const VIEW: u8 = 0;
const CART: u8 = 1;
const PURCHASE: u8 = 2;

/// A byte slice in 8 bytes: address in the low 48 bits, length in the high 16.
#[derive(Clone, Copy)]
struct Packed(u64);
impl Packed {
    #[inline]
    fn new(b: Bytes) -> Packed {
        let p = b.as_ptr() as u64;
        if p >> 48 != 0 || b.len() >= 1 << 16 {
            die("field too long or address above 48 bits");
        }
        Packed(p | (b.len() as u64) << 48)
    }
    #[inline]
    fn get(self) -> Bytes {
        unsafe { std::slice::from_raw_parts((self.0 & ((1 << 48) - 1)) as *const u8, (self.0 >> 48) as usize) }
    }
}

/// One parsed (well-formed, non-empty event_id) event line.
#[derive(Clone, Copy)]
struct RawEvent {
    h: u64,
    ingest: i64,
    ts: i64,
    eid: Packed,
    uid: Packed,
    price: u32,
    cat: u16,
    kind: u8,
    keep: bool, // valid and not late
}

struct RawClick {
    uid: Bytes,
    cid: Bytes,
    ts: i64,
    camp: u16,
}

const KEEP: u32 = 1 << 26;

/// An event line on its way to dedup, in its event_id bucket. `pk` = price | kind << 24 | KEEP.
#[derive(Clone, Copy)]
struct R1 {
    eid: SStr,
    ingest: i64,
    ts: u32, // ms after ws - 7 days
    pid: u32,
    pk: u32,
    cat: u32,
}

/// A surviving event in its person bucket.
#[derive(Clone, Copy)]
struct Rec {
    pid: u32,
    ts: u32,
    pk: u32,
    cat: u32,
}

/// A valid click in its person bucket.
#[derive(Clone, Copy)]
struct CRec {
    pid: u32,
    ts: u32,
    camp: u32,
    cid: Packed,
}

#[derive(Clone, Copy, Default)]
struct EvS {
    ts: u32,
    pk: u32,
    cat: u32,
}

#[derive(Clone, Copy)]
struct CkS {
    ts: u32,
    camp: u32,
    cid: Packed,
}

struct UserRow {
    name: SStr,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: Option<u16>,
}

/// Small read-only map from an ID to a u16 (products -> category, campaigns -> index); open addressing on
/// inline IDs, no hashing of bytes and no pointer chase.
struct IdMap {
    slots: Vec<(SStr, u16)>,
    mask: usize,
}
impl IdMap {
    const EMPTY: u16 = u16::MAX;
    fn build(items: impl Iterator<Item = (SStr, u16)>, n: usize) -> IdMap {
        let cap = (2 * n + 2).next_power_of_two().max(16);
        let mut m = IdMap { slots: vec![(SStr([0; 16]), IdMap::EMPTY); cap], mask: cap - 1 };
        for (k, v) in items {
            let mut i = (k.hash() >> 32) as usize & m.mask;
            while m.slots[i].1 != IdMap::EMPTY {
                i = (i + 1) & m.mask;
            }
            m.slots[i] = (k, v);
        }
        m
    }
    #[inline(always)]
    fn get(&self, k: SStr) -> Option<u16> {
        let mut i = (k.hash() >> 32) as usize & self.mask;
        loop {
            let (key, v) = &self.slots[i];
            if *v == IdMap::EMPTY {
                return None;
            }
            if *key == k {
                return Some(*v);
            }
            i = (i + 1) & self.mask;
        }
    }
}

struct Ctx {
    ws: i64,
    we: i64,
    base: i64, // ws - 7 days: every kept ts is in [base, base + 2^32)
    day0: i64,
    ndays: usize,
    ncamp: usize, // organic is index ncamp
    products: IdMap,
    campaigns: IdMap,
    unknown_cat: u16,
    cat_rank: Vec<u32>,
}

// ---------------------------------------------------------------- buckets

/// `n` append-only lists in one block of `n * cap` items, so the kernel backs them with a few huge pages instead
/// of a page fault per 4 KiB per list. A list that outgrows its share continues in `more`.
struct Buckets<T> {
    base: *mut T,
    cap: usize,
    len: Vec<usize>,
    more: Vec<Vec<T>>,
}
unsafe impl<T: Send> Send for Buckets<T> {}
unsafe impl<T: Sync> Sync for Buckets<T> {}

impl<T: Copy> Buckets<T> {
    fn new(n: usize, cap: usize) -> Buckets<T> {
        let bytes = n * cap * std::mem::size_of::<T>();
        let mut base = std::ptr::null_mut();
        if n * cap > 0 {
            let p = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0) };
            if p == libc::MAP_FAILED {
                die("mmap of a bucket block failed");
            }
            base = p as *mut T;
        }
        Buckets { base, cap, len: vec![0; n], more: (0..n).map(|_| Vec::new()).collect() }
    }
    /// Lists of about `total / (threads * n)` items each (+25% for uneven shares).
    fn sized(n: usize, total: usize, threads: usize) -> Buckets<T> {
        Buckets::new(n, total / (threads * n).max(1) * 5 / 4 + 16)
    }
    #[inline(always)]
    fn push(&mut self, b: usize, x: T) {
        let l = self.len[b];
        if l < self.cap {
            unsafe { self.base.add(b * self.cap + l).write(x) };
            self.len[b] = l + 1;
        } else {
            self.more[b].push(x);
        }
    }
    fn len_of(&self, b: usize) -> usize {
        self.len[b] + self.more[b].len()
    }
    /// List `b` as two slices, in order.
    #[inline(always)]
    fn parts(&self, b: usize) -> [&[T]; 2] {
        let a = if self.len[b] == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(self.base.add(b * self.cap), self.len[b]) } };
        [a, &self.more[b]]
    }
    #[inline(always)]
    fn get(&self, b: usize, i: usize) -> &T {
        let [a, m] = self.parts(b);
        if i < a.len() {
            &a[i]
        } else {
            &m[i - a.len()]
        }
    }
    fn iter(&self, b: usize) -> impl Iterator<Item = &T> {
        let [a, m] = self.parts(b);
        a.iter().chain(m.iter())
    }
}

// ---------------------------------------------------------------- person table

const NONE: u32 = u32::MAX;
const GRAB: usize = 4096;

fn map_zeroed(bytes: usize) -> *mut u8 {
    let p = unsafe { libc::mmap(std::ptr::null_mut(), bytes.max(4096), libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0) };
    if p == libc::MAP_FAILED {
        die("mmap of the person table failed");
    }
    p as *mut u8
}

impl SStr {
    /// A second 32 bit hash of the ID, independent of `hash`.
    #[inline(always)]
    fn hash2(&self) -> u32 {
        if self.0[15] != SStr::LONG {
            let lo = u64::from_le_bytes(self.0[..8].try_into().unwrap());
            let hi = u64::from_le_bytes(self.0[8..].try_into().unwrap());
            (mum(mum(hi ^ 0xA0761D6478BD642F, lo ^ 0xE7037ED1A0B428DB), 0x8EBC6AF09C88C6E3) >> 32) as u32
        } else {
            let mut h = 0xD6E8FEB86659FD93u64;
            for &c in self.get() {
                h = (h ^ c as u64).wrapping_mul(0x100000001B3).rotate_left(23);
            }
            (mum(h, 0x8EBC6AF09C88C6E3) >> 32) as u32
        }
    }
}

#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let p = (a as u128) * (b as u128);
    (p as u64) ^ ((p >> 64) as u64)
}

/// The part of a table slot every lookup reads: `h` is 0 (empty), 1 (being filled) or the ID's hash (>= 2);
/// `w` is a second 32 bit hash of the ID << 32 | union-find parent, after `finalize` the person (root slot index).
/// An ID is identified by the 96 bits of both hashes: a collision of two different IDs has probability ~1e-16.
#[repr(C, align(16))]
struct Slot {
    h: AtomicU64,
    w: AtomicU64,
}

/// Lock-free open-addressing table of IDs. Person = slot index of the union-find root. The ID bytes and the
/// smallest ID of a person live in arrays of their own that lookups never touch.
struct Table {
    slots: *mut Slot,
    keys: *mut SStr,
    best: *mut AtomicU32, // on a root: 1 + slot of the smallest ID of its person; 0 = the root's own
    cap: usize,
    shift: u32,
    used: AtomicUsize,
    limit: usize,
    overflow: AtomicBool,
}
unsafe impl Sync for Table {}
unsafe impl Send for Table {}

impl Table {
    fn new(cap: usize) -> Table {
        assert!(cap.is_power_of_two() && cap >= 1024 && cap < (1 << 31));
        Table {
            slots: map_zeroed(cap * 16) as *mut Slot,
            keys: map_zeroed(cap * 16) as *mut SStr,
            best: map_zeroed(cap * 4) as *mut AtomicU32,
            cap,
            shift: 64 - cap.trailing_zeros(),
            used: AtomicUsize::new(0),
            limit: cap / 5 * 4,
            overflow: AtomicBool::new(false),
        }
    }
    /// Gives the memory back; the table is leaked (it lives to the end of the run) so `P1` can borrow it for good.
    fn release(&self) {
        unsafe {
            libc::munmap(self.slots as *mut libc::c_void, self.cap * 16);
            libc::munmap(self.keys as *mut libc::c_void, self.cap * 16);
            libc::munmap(self.best as *mut libc::c_void, self.cap * 4);
        }
    }
    #[inline(always)]
    fn slot(&self, i: usize) -> &Slot {
        debug_assert!(i < self.cap);
        unsafe { &*self.slots.add(i) }
    }
    #[inline(always)]
    fn key(&self, i: u32) -> SStr {
        unsafe { *self.keys.add(i as usize) }
    }
    #[inline(always)]
    fn best_of(&self, i: u32) -> &AtomicU32 {
        unsafe { &*self.best.add(i as usize) }
    }
    #[inline(always)]
    fn prefetch(&self, h: u64) {
        prefetch(unsafe { self.slots.add((h >> self.shift) as usize) });
    }
    #[cold]
    fn grab(&self, credits: &mut usize) -> bool {
        if self.overflow.load(Relaxed) {
            return false;
        }
        let u = self.used.fetch_add(GRAB, Relaxed);
        if u + GRAB > self.limit {
            self.overflow.store(true, Relaxed);
            return false;
        }
        *credits = GRAB;
        true
    }
    /// The slot of `key` (hash `h`, already >= 2), inserting it as a new person when absent; with LOG a new slot
    /// is also appended to `log`. NONE when the table is out of room (the caller must give up: see `overflow`).
    #[inline]
    fn slot_of<const LOG: bool>(&self, key: SStr, h: u64, credits: &mut usize, log: &mut Vec<u32>) -> u32 {
        let h2 = key.hash2() as u64;
        let mask = self.cap - 1;
        let mut i = (h >> self.shift) as usize;
        loop {
            let s = self.slot(i);
            let mut sh = s.h.load(Acquire);
            if sh == 0 {
                if *credits == 0 && !self.grab(credits) {
                    return NONE;
                }
                match s.h.compare_exchange(0, 1, Acquire, Acquire) {
                    Ok(_) => {
                        *credits -= 1;
                        unsafe { self.keys.add(i).write(key) };
                        s.w.store(h2 << 32 | i as u64, Relaxed);
                        s.h.store(h, Release);
                        if LOG {
                            log.push(i as u32);
                        }
                        return i as u32;
                    }
                    Err(cur) => sh = cur,
                }
            }
            while sh == 1 {
                std::hint::spin_loop();
                sh = s.h.load(Acquire);
            }
            if sh == h && s.w.load(Relaxed) >> 32 == h2 {
                return i as u32;
            }
            i = (i + 1) & mask;
        }
    }
    #[inline(always)]
    fn parent(&self, i: u32) -> u32 {
        self.slot(i as usize).w.load(Acquire) as u32
    }
    /// The person of slot `i`: its parent once the unions are done.
    #[inline(always)]
    fn pid(&self, i: u32) -> u32 {
        self.parent(i)
    }
    /// parent[x] = new if it is still `old`.
    #[inline]
    fn set_parent(&self, x: u32, old: u32, new: u32) -> bool {
        let w = &self.slot(x as usize).w;
        let cur = w.load(Acquire);
        if cur as u32 != old {
            return false;
        }
        w.compare_exchange(cur, (cur & !0xFFFF_FFFF) | new as u64, Release, Relaxed).is_ok()
    }
    fn find(&self, mut x: u32) -> u32 {
        loop {
            let p = self.parent(x);
            if p == x {
                return x;
            }
            let gp = self.parent(p);
            if gp != p {
                self.set_parent(x, p, gp);
            }
            x = gp;
        }
    }
    fn union(&self, a: u32, b: u32) {
        loop {
            let (ra, rb) = (self.find(a), self.find(b));
            if ra == rb {
                return;
            }
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            if self.set_parent(hi, hi, lo) {
                return;
            }
        }
    }
    /// Once all unions are done: point each slot of `log` at its root and pick each root's smallest ID.
    fn finalize(&self, log: &[u32]) {
        for &i in log {
            let r = self.find(i);
            let w = &self.slot(i as usize).w;
            w.store((w.load(Relaxed) & !0xFFFF_FFFF) | r as u64, Relaxed);
            let mine = self.key(i);
            let rb = self.best_of(r);
            loop {
                let cur = rb.load(Acquire);
                if cur == i + 1 {
                    break;
                }
                let curkey = if cur == 0 { self.key(r) } else { self.key(cur - 1) };
                if curkey.get() <= mine.get() {
                    break;
                }
                if rb.compare_exchange(cur, i + 1, AcqRel, Acquire).is_ok() {
                    break;
                }
            }
        }
    }
    #[inline(always)]
    fn prefetch_name(&self, pid: u32) {
        prefetch(unsafe { self.best.add(pid as usize) });
    }
    #[inline(always)]
    fn prefetch_key(&self, pid: u32) {
        let b = self.best_of(pid).load(Relaxed);
        prefetch(unsafe { self.keys.add(if b == 0 { pid } else { b - 1 } as usize) });
    }
    fn name(&self, pid: u32) -> SStr {
        let b = self.best_of(pid).load(Relaxed);
        self.key(if b == 0 { pid } else { b - 1 })
    }
}

// ---------------------------------------------------------------- parsing

fn parse_click(ctx: &Ctx, [cid, uid, camp, ts]: [Bytes; 4], end: usize) -> Option<RawClick> {
    if cid.is_empty() || uid.is_empty() {
        return None;
    }
    let t = parse_ts(ts)?;
    if t < ctx.ws - WEEK || t >= ctx.we {
        return None;
    }
    let camp = ctx.campaigns.get(SStr::field(camp, end))?;
    Some(RawClick { uid, cid, ts: t, camp })
}

// ---------------------------------------------------------------- phase 0: links

const BATCH: usize = 16;

struct LinkSt {
    pend: Vec<(SStr, u64, SStr, u64)>,
    credits: usize,
    log: Vec<u32>, // slots this thread created
}

fn flush_links(tbl: &Table, st: &mut LinkSt) {
    let mut pend = std::mem::take(&mut st.pend);
    for &(a, ha, b, hb) in &pend {
        let ia = tbl.slot_of::<true>(a, ha, &mut st.credits, &mut st.log);
        let ib = tbl.slot_of::<true>(b, hb, &mut st.credits, &mut st.log);
        if ia != NONE && ib != NONE {
            tbl.union(ia, ib);
        }
    }
    pend.clear();
    st.pend = pend;
}

/// Every ID of a link line goes into the table; the two IDs of a line are unioned.
fn build_links(tbl: &Table, files: &[Bytes], threads: usize) {
    let ch = chunks(files);
    let sts = par(
        threads,
        ch.len(),
        || LinkSt { pend: Vec::with_capacity(BATCH), credits: 0, log: Vec::new() },
        |st, i| {
            let (f, s, e) = ch[i];
            let end = f.as_ptr() as usize + f.len();
            for_records::<2>(f, s, e, |[a, b]| {
                if a.is_empty() || b.is_empty() {
                    return;
                }
                let (ka, kb) = (SStr::field(a, end), SStr::field(b, end));
                let (ha, hb) = (ka.hash() | 2, kb.hash() | 2);
                tbl.prefetch(ha);
                tbl.prefetch(hb);
                st.pend.push((ka, ha, kb, hb));
                if st.pend.len() == BATCH {
                    flush_links(tbl, st);
                }
            });
            flush_links(tbl, st);
        },
    );
    const PART: usize = 1 << 15;
    let logs: Vec<Vec<u32>> = sts.into_iter().map(|s| s.log).collect();
    let tasks: Vec<(usize, usize, usize)> = logs.iter().enumerate().flat_map(|(l, v)| (0..v.len()).step_by(PART).map(move |a| (l, a, (a + PART).min(v.len())))).collect();
    par(threads, tasks.len(), || (), |_, i| {
        let (l, a, b) = tasks[i];
        tbl.finalize(&logs[l][a..b]);
    });
}

// ---------------------------------------------------------------- phase 1: parse

struct PendE {
    eid: SStr,
    eh: u64,
    uid: SStr,
    uh: u64,
    ingest: i64,
    ts: u32,
    pk: u32,
    cat: u32,
}

struct PendC {
    uid: SStr,
    uh: u64,
    ts: u32,
    camp: u32,
    cid: Packed,
}

/// One parse thread's output.
struct P1<'a> {
    tbl: &'a Table,
    base: i64,
    eb: Buckets<R1>,
    ck: Buckets<CRec>,
    pe: Vec<PendE>,
    pc: Vec<PendC>,
    credits: usize,
    nolog: Vec<u32>,
}

impl P1<'_> {
    fn new<'a>(tbl: &'a Table, base: i64, eb: Buckets<R1>, ck: Buckets<CRec>) -> P1<'a> {
        P1 { tbl, base, eb, ck, pe: Vec::with_capacity(BATCH), pc: Vec::with_capacity(BATCH), credits: 0, nolog: Vec::new() }
    }

    /// A valid, non-late event: its person is looked up in a batch (the slots are prefetched meanwhile), then it
    /// goes to its event_id bucket.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn keep(&mut self, eh: u64, eid: SStr, uh: u64, uid: SStr, ingest: i64, ts: i64, price: u32, cat: u16, kind: u8) {
        let uh = uh | 2;
        self.tbl.prefetch(uh);
        let pk = price | (kind as u32) << 24 | KEEP;
        self.pe.push(PendE { eid, eh, uid, uh, ingest, ts: (ts - self.base) as u32, pk, cat: cat as u32 });
        if self.pe.len() == BATCH {
            self.flush_events();
        }
    }

    /// Any other parsed event: it still takes part in dedup.
    #[inline(always)]
    fn reject(&mut self, eh: u64, eid: SStr, ingest: i64) {
        self.eb.push((eh >> (64 - EBITS)) as usize, R1 { eid, ingest, ts: 0, pid: 0, pk: 0, cat: 0 });
    }

    #[inline(always)]
    fn add(&mut self, r: RawEvent) {
        let eid = SStr::new(r.eid.get());
        if r.keep {
            let u = r.uid.get();
            self.keep(r.h, eid, hash_bytes(u), SStr::new(u), r.ingest, r.ts, r.price, r.cat, r.kind);
        } else {
            self.reject(r.h, eid, r.ingest);
        }
    }

    fn flush_events(&mut self) {
        let mut pe = std::mem::take(&mut self.pe);
        for p in pe.drain(..) {
            let i = self.tbl.slot_of::<false>(p.uid, p.uh, &mut self.credits, &mut self.nolog);
            if i == NONE {
                continue; // table full: the whole attempt is thrown away
            }
            let pid = self.tbl.pid(i);
            self.eb.push((p.eh >> (64 - EBITS)) as usize, R1 { eid: p.eid, ingest: p.ingest, ts: p.ts, pid, pk: p.pk, cat: p.cat });
        }
        self.pe = pe;
    }

    fn flush_clicks(&mut self) {
        let mut pc = std::mem::take(&mut self.pc);
        for p in pc.drain(..) {
            let i = self.tbl.slot_of::<false>(p.uid, p.uh, &mut self.credits, &mut self.nolog);
            if i == NONE {
                continue;
            }
            let pid = self.tbl.pid(i);
            self.ck.push((pid as usize) & (NB - 1), CRec { pid, ts: p.ts, camp: p.camp, cid: p.cid });
        }
        self.pc = pc;
    }
}

fn parse_clicks(ctx: &Ctx, st: &mut P1, f: Bytes, s: usize, e: usize) {
    let end = f.as_ptr() as usize + f.len();
    let base = ctx.base;
    for_records::<4>(f, s, e, |fields| {
        if let Some(c) = parse_click(ctx, fields, end) {
            let uid = SStr::field(c.uid, end);
            let uh = uid.hash() | 2;
            st.tbl.prefetch(uh);
            st.pc.push(PendC { uid, uh, ts: (c.ts - base) as u32, camp: c.camp as u32, cid: Packed::new(c.cid) });
            if st.pc.len() == BATCH {
                st.flush_clicks();
            }
        }
    });
    st.flush_clicks();
}

// ---------------------------------------------------------------- phase 2: dedup

struct P2 {
    out: Buckets<Rec>,
    tab: Vec<u64>,
    hs: Vec<u64>,
    dead: Vec<bool>,
    offs: Vec<usize>,
}

/// Dedups event_id bucket `b` across all parse threads (rules, Events step 2: the smallest ingest_ts wins).
/// A slot holds a 24 bit tag of the ID's hash and the location (thread << 32 | index) + 1 of the copy that leads.
fn dedup_bucket(p1: &[P1<'_>], b: usize, st: &mut P2) {
    st.offs.clear();
    let mut n = 0;
    for t in p1 {
        st.offs.push(n);
        n += t.eb.len_of(b);
    }
    if n == 0 {
        return;
    }
    let cap = (n + n / 2 + 1).next_power_of_two().max(16);
    let mask = cap - 1;
    st.tab.clear();
    st.tab.resize(cap, 0);
    st.dead.clear();
    st.dead.resize(n, false);
    st.hs.clear();
    for t in p1 {
        st.hs.extend(t.eb.iter(b).map(|r| r.eid.hash()));
    }
    const AHEAD: usize = 8;
    for g in 0..n.min(AHEAD) {
        prefetch(&st.tab[st.hs[g] as usize & mask]);
    }
    let mut g = 0;
    for (t, th) in p1.iter().enumerate() {
        for (i, r) in th.eb.iter(b).enumerate() {
            if g + AHEAD < n {
                prefetch(&st.tab[st.hs[g + AHEAD] as usize & mask]);
            }
            let h = st.hs[g];
            let tag = (h >> 31) & 0xFF_FFFF;
            let loc = ((t as u64) << 32 | i as u64) + 1;
            let mut s = h as usize & mask;
            loop {
                let x = st.tab[s];
                if x == 0 {
                    st.tab[s] = tag << 40 | loc;
                    break;
                }
                if x >> 40 == tag {
                    let wl = (x & ((1 << 40) - 1)) - 1;
                    let (wt, wi) = ((wl >> 32) as usize, (wl & 0xFFFF_FFFF) as usize);
                    let w = p1[wt].eb.get(b, wi);
                    if w.eid == r.eid {
                        if r.ingest < w.ingest {
                            st.dead[st.offs[wt] + wi] = true;
                            st.tab[s] = tag << 40 | loc;
                        } else {
                            st.dead[g] = true;
                        }
                        break;
                    }
                }
                s = (s + 1) & mask;
            }
            g += 1;
        }
    }
    g = 0;
    for th in p1 {
        for r in th.eb.iter(b) {
            if !st.dead[g] && r.pk & KEEP != 0 {
                st.out.push(r.pid as usize & (NB - 1), Rec { pid: r.pid, ts: r.ts, pk: r.pk, cat: r.cat });
            }
            g += 1;
        }
    }
}

// ---------------------------------------------------------------- phase 3: persons

struct Agg {
    rows: Vec<UserRow>,
    cred: Vec<[i64; 3]>,
    assist: Vec<i64>,
    stamp: Vec<u32>,
    iv: Vec<(i64, i64)>,
    cnt_e: Vec<u32>,
    cnt_c: Vec<u32>,
    evs: Vec<EvS>,
    cks: Vec<CkS>,
    plist: Vec<(u32, u32, u32, u32, u32)>,
    sessions: Vec<Sess>,
    day_groups: Vec<(usize, usize)>,
    cats: Vec<(u16, i64)>,
    touched: Vec<usize>,
}

struct Sess {
    start: i64,
    d: usize,
    purchased: bool,
    revenue: i64,
}

/// Person bucket `b`: counting sort of its events and clicks by person, then each person on its own.
fn bucket_work(ctx: &Ctx, tbl: &Table, ev_src: &[&Buckets<Rec>], ck_src: &[&Buckets<CRec>], b: usize, agg: &mut Agg) {
    let nloc = (tbl.cap >> 8) + 1;
    let n_e: usize = ev_src.iter().map(|t| t.len_of(b)).sum();
    if n_e == 0 {
        return;
    }
    let n_c: usize = ck_src.iter().map(|t| t.len_of(b)).sum();

    agg.cnt_e.clear();
    agg.cnt_e.resize(nloc, 0);
    for t in ev_src {
        for r in t.iter(b) {
            agg.cnt_e[(r.pid >> 8) as usize] += 1;
        }
    }
    let mut sum = 0;
    for c in agg.cnt_e.iter_mut() {
        let x = *c;
        *c = sum;
        sum += x;
    }
    let mut evs = std::mem::take(&mut agg.evs);
    evs.clear();
    evs.resize(n_e, EvS::default());
    for t in ev_src {
        for r in t.iter(b) {
            let c = &mut agg.cnt_e[(r.pid >> 8) as usize];
            evs[*c as usize] = EvS { ts: r.ts, pk: r.pk, cat: r.cat };
            *c += 1;
        }
    }

    agg.cnt_c.clear();
    agg.cnt_c.resize(nloc, 0);
    for t in ck_src {
        for r in t.iter(b) {
            agg.cnt_c[(r.pid >> 8) as usize] += 1;
        }
    }
    let mut sum = 0;
    for c in agg.cnt_c.iter_mut() {
        let x = *c;
        *c = sum;
        sum += x;
    }
    let mut cks = std::mem::take(&mut agg.cks);
    cks.clear();
    cks.resize(n_c, CkS { ts: 0, camp: 0, cid: Packed(0) });
    for t in ck_src {
        for r in t.iter(b) {
            let c = &mut agg.cnt_c[(r.pid >> 8) as usize];
            cks[*c as usize] = CkS { ts: r.ts, camp: r.camp, cid: r.cid };
            *c += 1;
        }
    }

    // cnt[li] is now the end of person li's run; its start is the end of li - 1
    let mut plist = std::mem::take(&mut agg.plist);
    plist.clear();
    let (mut e0, mut c0) = (0usize, 0usize);
    for li in 0..nloc {
        let (e1, c1) = (agg.cnt_e[li] as usize, agg.cnt_c[li] as usize);
        if e1 != e0 {
            plist.push((((li << 8) | b) as u32, e0 as u32, e1 as u32, c0 as u32, c1 as u32));
        }
        e0 = e1;
        c0 = c1;
    }
    for j in 0..plist.len() {
        if j + 12 < plist.len() {
            tbl.prefetch_name(plist[j + 12].0);
        }
        if j + 6 < plist.len() {
            tbl.prefetch_key(plist[j + 6].0);
        }
        let (pid, es, e1, cs, c1) = plist[j];
        let ev = &mut evs[es as usize..e1 as usize];
        if ev.len() > 1 {
            ev.sort_unstable_by_key(|e| e.ts);
        }
        let ck = &mut cks[cs as usize..c1 as usize];
        if ck.len() > 1 {
            ck.sort_unstable_by_key(|c| c.ts);
        }
        person_work(ctx, pid, tbl.name(pid), ev, ck, agg);
    }
    agg.plist = plist;
    agg.evs = evs;
    agg.cks = cks;
}

fn flush_assist(assist: &mut [i64], width: usize, k: usize, lo: i64, hi: i64, sessions: &[Sess], day_groups: &[(usize, usize)]) {
    let a = sessions.partition_point(|s| s.start < lo);
    let b = sessions.partition_point(|s| s.start <= hi);
    if a >= b {
        return;
    }
    for (g, &(d, first)) in day_groups.iter().enumerate() {
        let end = day_groups.get(g + 1).map(|x| x.1).unwrap_or(sessions.len());
        let n = end.min(b) as i64 - first.max(a) as i64;
        if n > 0 {
            assist[k * width + d] += n;
        }
    }
}

/// One person: sessions, user_daily rows, click credit and assists.
fn person_work(ctx: &Ctx, pid: u32, name: SStr, ev: &[EvS], cs: &[CkS], agg: &mut Agg) {
    let base = ctx.base;
    let width = ctx.ndays;
    let organic = ctx.ncamp;
    let Agg { rows, cred, assist, stamp, iv, sessions, day_groups, cats, touched, .. } = agg;

    // sessions + user_daily rows
    sessions.clear();
    day_groups.clear();
    let mut row: Option<UserRow> = None;
    let mut prev = i64::MIN;
    for e in ev {
        let ts = e.ts as i64 + base;
        if prev == i64::MIN || ts - prev > GAP {
            let d = ((ts.div_euclid(DAY)) - ctx.day0) as usize;
            sessions.push(Sess { start: ts, d, purchased: false, revenue: 0 });
            let day = (ctx.day0 + d as i64) as i32;
            if row.as_ref().map(|r| r.day != day).unwrap_or(true) {
                if let Some(r) = row.take() {
                    rows.push(finish_row(r, cats, &ctx.cat_rank));
                }
                row = Some(UserRow { name, day, sessions: 0, events: 0, revenue: 0, top: None });
                day_groups.push((d, sessions.len() - 1));
            }
            row.as_mut().unwrap().sessions += 1;
        }
        prev = ts;
        let r = row.as_mut().unwrap();
        r.events += 1;
        if (e.pk >> 24) & 3 == PURCHASE as u32 {
            let price = (e.pk & 0xFF_FFFF) as i64;
            let s = sessions.last_mut().unwrap();
            s.purchased = true;
            s.revenue += price;
            r.revenue += price;
            let cat = e.cat as u16;
            match cats.iter_mut().find(|c| c.0 == cat) {
                Some(c) => c.1 += price,
                None => cats.push((cat, price)),
            }
        }
    }
    if let Some(r) = row.take() {
        rows.push(finish_row(r, cats, &ctx.cat_rank));
    }

    let cts = |c: &CkS| c.ts as i64 + base;

    // credit: latest click with start - 7d <= ts <= start; tie -> smallest click_id
    let mut q = 0;
    for s in sessions.iter() {
        while q < cs.len() && cts(&cs[q]) <= s.start {
            q += 1;
        }
        let mut camp = organic;
        if q > 0 {
            let t = cts(&cs[q - 1]);
            if s.start - t <= WEEK {
                let mut best = &cs[q - 1];
                let mut k = q - 1;
                while k > 0 && cts(&cs[k - 1]) == t {
                    k -= 1;
                    if cs[k].cid.get() < best.cid.get() {
                        best = &cs[k];
                    }
                }
                camp = best.camp as usize;
            }
        }
        let a = &mut cred[camp * width + s.d];
        a[0] += 1;
        a[1] += s.purchased as i64;
        a[2] += s.revenue;
    }

    // assists: a click at t assists sessions starting in [t, t + 7d]; merge per campaign
    touched.clear();
    for c in cs {
        let k = c.camp as usize;
        let t = cts(c);
        if stamp[k] != pid {
            stamp[k] = pid;
            iv[k] = (t, t + WEEK);
            touched.push(k);
        } else if t <= iv[k].1 {
            iv[k].1 = t + WEEK;
        } else {
            let (lo, hi) = iv[k];
            flush_assist(assist, width, k, lo, hi, sessions, day_groups);
            iv[k] = (t, t + WEEK);
        }
    }
    for &k in touched.iter() {
        let (lo, hi) = iv[k];
        flush_assist(assist, width, k, lo, hi, sessions, day_groups);
        stamp[k] = u32::MAX;
    }
}

#[inline]
fn parse_event(ctx: &Ctx, [eid, uid, ts, ingest, etype, pid, price]: [Bytes; 7]) -> Option<RawEvent> {
    if eid.is_empty() {
        return None;
    }
    let ingest = parse_ts(ingest)?; // always valid F1 per rules.md
    let mut r = RawEvent { h: hash_bytes(eid), ingest, ts: 0, eid: Packed::new(eid), uid: Packed::new(uid), price: 0, cat: 0, kind: 0, keep: false };
    if uid.is_empty() || pid.is_empty() {
        return Some(r);
    }
    let Some(t) = parse_ts(ts) else { return Some(r) };
    if t < ctx.ws || t >= ctx.we {
        return Some(r);
    }
    r.kind = match etype {
        b"view" => VIEW,
        b"cart" => CART,
        b"purchase" => PURCHASE,
        _ => return Some(r),
    };
    if r.kind == PURCHASE {
        let Some(p) = parse_price(price) else { return Some(r) };
        r.price = p;
        r.cat = ctx.products.get(SStr::new(pid)).unwrap_or(ctx.unknown_cat);
    }
    if ingest - t > DAY {
        return Some(r);
    }
    r.ts = t;
    r.keep = true;
    Some(r)
}

/// Parses the event lines that start in [start, end) of `file` (header skipped) into `st`. A line with its `\n`
/// in its first 128 bytes and 144 readable bytes takes `fast_event`; any other goes through `split` + `parse_event`.
fn parse_events(ctx: &Ctx, file: Bytes, start: usize, end: usize, st: &mut P1) {
    let len = file.len();
    let first = start.saturating_sub(1);
    let Some(nl) = memchr::memchr(b'\n', &file[first..]) else { return };
    let mut p = first + nl + 1;
    let fast_end = len.saturating_sub(144);
    while p < end && p < len {
        if p < fast_end {
            // SAFETY: p + 144 <= len
            unsafe {
                let line = file.as_ptr().add(p);
                let [c0, q0, n0] = masks64(line);
                let [c1, q1, n1] = masks64(line.add(64));
                let n = n0 as u128 | (n1 as u128) << 64;
                if n != 0 {
                    let e = n.trailing_zeros() as usize;
                    let lm = (1u128 << e) - 1;
                    fast_event(ctx, line, e, (c0 as u128 | (c1 as u128) << 64) & lm, (q0 as u128 | (q1 as u128) << 64) & lm, st);
                    p += e + 1;
                    continue;
                }
            }
        }
        let e = memchr::memchr(b'\n', &file[p..]).map_or(len, |i| p + i);
        if let Some(r) = split::<7>(&file[p..e]).and_then(|f| parse_event(ctx, f)) {
            st.add(r);
        }
        p = e + 1;
    }
    st.flush_events();
}

/// `split::<7>` + `parse_event` + `P1::add` for the line of `e` (< 128) bytes at `line`, given its comma and
/// quote bits `c` and `q`; line[e] is its `\n` and line[..144] is readable.
#[inline(always)]
unsafe fn fast_event(ctx: &Ctx, line: *const u8, e: usize, c: u128, q: u128, st: &mut P1) {
    if c.count_ones() != 6 {
        return;
    }
    // Well quoted <=> every quote sits at a field's first or last byte (s, ends), none is both (a lone `"`), and
    // per field a first-byte quote (x) comes with a last-byte quote (y, moved onto the boundary after it). Then
    // y - x borrows across exactly the quoted fields' bytes, so a field with only one of the two leaves a bit on
    // the boundary that ends it (or the first such field does).
    let b = c | (1u128 << e);
    let s = (b << 1) | 1;
    let ends = b >> 1;
    let x = q & s;
    let y = (q << 1) & b;
    if (q & !(s | ends)) | (x & ends) | (y.wrapping_sub(x) & b) != 0 {
        return;
    }
    let mut fs = [0usize; 7];
    let mut fe = [0usize; 7];
    let mut cc = c;
    let mut at = 0;
    for i in 0..6 {
        let k = cc.trailing_zeros() as usize;
        cc &= cc - 1;
        fs[i] = at;
        fe[i] = k;
        at = k + 1;
    }
    fs[6] = at;
    fe[6] = e;
    for i in 0..7 {
        // a quoted field starts with `"` (an empty unquoted field starts on its `,` or `\n`)
        let qd = (*line.add(fs[i]) == b'"') as usize;
        fs[i] += qd;
        fe[i] -= qd;
    }
    let f = |i: usize| (line.add(fs[i]), fe[i] - fs[i]);
    let (ep, el) = f(0);
    if el == 0 {
        return;
    }
    let (ip, il) = f(3);
    let Some(ingest) = ts_at(ip, il) else { return };
    let (h, eid) = if el <= 15 {
        let v = load16(ep, el);
        (hash16(v, el), SStr((v | (el as u128) << 120).to_le_bytes()))
    } else {
        let s = std::slice::from_raw_parts(ep, el);
        (hash_bytes(s), SStr::new(s))
    };
    let ((up, ul), (pp, pl)) = (f(1), f(5));
    let (tp, tl) = f(2);
    let ts = match ts_at(tp, tl) {
        Some(t) if ul != 0 && pl != 0 && t >= ctx.ws && t < ctx.we && ingest - t <= DAY => t,
        _ => return st.reject(h, eid, ingest),
    };
    let (kp, kl) = f(4);
    let w = (kp as *const u64).read_unaligned();
    let kind = match kl {
        4 if w as u32 == u32::from_le_bytes(*b"view") => VIEW,
        4 if w as u32 == u32::from_le_bytes(*b"cart") => CART,
        8 if w == u64::from_le_bytes(*b"purchase") => PURCHASE,
        _ => return st.reject(h, eid, ingest),
    };
    let (mut price, mut cat) = (0, 0);
    if kind == PURCHASE {
        let (rp, rl) = f(6);
        let Some(p) = parse_price(std::slice::from_raw_parts(rp, rl)) else { return st.reject(h, eid, ingest) };
        price = p;
        let key = if pl <= 15 { SStr((load16(pp, pl) | (pl as u128) << 120).to_le_bytes()) } else { SStr::new(std::slice::from_raw_parts(pp, pl)) };
        cat = ctx.products.get(key).unwrap_or(ctx.unknown_cat);
    }
    let (uh, uid) = if ul <= 15 {
        let v = load16(up, ul);
        (hash16(v, ul), SStr((v | (ul as u128) << 120).to_le_bytes()))
    } else {
        let u = std::slice::from_raw_parts(up, ul);
        (hash_bytes(u), SStr::new(u))
    };
    st.keep(h, eid, uh, uid, ingest, ts, price, cat, kind);
}



// ---------------------------------------------------------------- main

fn main() {
    let t0 = Instant::now();
    let data = PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let threads: usize = std::env::var("NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8));
    if threads > 200 {
        die("too many threads");
    }
    let usage0 = std::cell::Cell::new(usage());
    let lap = |what: &str| {
        let (c, ins) = usage();
        let (c0, i0) = usage0.get();
        eprintln!("[{:7.3}s wall, {:6.3}s cpu, {:6.2}G ins] {what}", t0.elapsed().as_secs_f64(), c - c0, (ins - i0) as f64 / 1e9);
        usage0.set((c, ins));
    };

    // manifest
    let manifest = std::fs::read_to_string(data.join("manifest.json")).unwrap_or_else(|e| die(&format!("manifest.json: {e}")));
    let ws = manifest_ts(&manifest, "window_start");
    let we = manifest_ts(&manifest, "window_end");
    if we <= ws {
        die("manifest: empty window");
    }
    if we - ws + WEEK >= 1 << 32 {
        die("manifest: window too long");
    }
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1) as usize;

    // products -> categories; "unknown" is its own category, never one of the file's
    let mut cat_names: Vec<Bytes> = Vec::new();
    let mut cat_idx: BytesMap<u16> = BytesMap::default();
    let mut products: BytesMap<u16> = BytesMap::default();
    for (p, c) in small_table(&data.join("products.csv")) {
        let n = cat_names.len() as u16;
        let ci = *cat_idx.entry(c).or_insert_with(|| {
            cat_names.push(c);
            n
        });
        products.insert(p, ci);
    }
    let unknown_cat = cat_names.len() as u16;
    cat_names.push(b"unknown");
    let mut order: Vec<usize> = (0..cat_names.len()).collect();
    order.sort_by(|&a, &b| cat_names[a].cmp(cat_names[b]));
    let mut cat_rank = vec![0u32; cat_names.len()];
    for (r, &i) in order.iter().enumerate() {
        cat_rank[i] = r as u32;
    }

    // campaigns
    let mut camp_ids: Vec<Bytes> = Vec::new();
    let mut camp_channel: Vec<Bytes> = Vec::new();
    let mut campaigns: BytesMap<u16> = BytesMap::default();
    for (c, ch) in small_table(&data.join("campaigns.csv")) {
        if campaigns.contains_key(c) {
            continue;
        }
        campaigns.insert(c, camp_ids.len() as u16);
        camp_ids.push(c);
        camp_channel.push(ch);
    }
    let ncamp = camp_ids.len();
    lap("small tables");

    let products = IdMap::build(products.iter().map(|(k, v)| (SStr::new(*k), *v)), products.len());
    let campaigns = IdMap::build(campaigns.iter().map(|(k, v)| (SStr::new(*k), *v)), campaigns.len());
    let ctx = Ctx { ws, we, base: ws - WEEK, day0, ndays, ncamp, products, campaigns, unknown_cat, cat_rank };

    let link_files = csv_files(&data.join("links"));
    let ev_files = csv_files(&data.join("events"));
    let ck_files = csv_files(&data.join("clicks"));
    let ev_chunks = chunks(&ev_files);
    let ck_chunks = chunks(&ck_files);
    let ev_est = ev_files.iter().map(|f| f.len()).sum::<usize>() / 60; // lines are ~75 bytes
    let ck_est = ck_files.iter().map(|f| f.len()).sum::<usize>() / 40; // lines are ~50 bytes
    let link_lines: usize = {
        let ch = chunks(&link_files);
        par(threads, ch.len(), || 0usize, |n, i| {
            let (f, s, e) = ch[i];
            *n += memchr::memchr_iter(b'\n', &f[s..e]).count();
        })
        .iter()
        .sum()
    };
    // distinct IDs: about one in 13 event lines (a retry with 4x the room follows if the table fills up)
    let _ = (link_lines, ck_est);
    let mut cap = (ev_est * 4 / 5 / 8 + 1024).next_power_of_two();

    // 0 + 1. links, then events and clicks: every ID resolves to its person while its line is parsed
    let (tbl, p1) = loop {
        let tbl: &'static Table = Box::leak(Box::new(Table::new(cap)));
        build_links(tbl, &link_files, threads);
        lap(&format!("links joined ({link_lines} lines, table {cap} slots)"));
        let ne = ev_chunks.len();
        let p1 = par(
            threads,
            ne + ck_chunks.len(),
            || P1::new(tbl, ctx.base, Buckets::sized(EBK, ev_est, threads), Buckets::sized(NB, ck_est, threads)),
            |st, i| {
                if i < ne {
                    let (f, s, e) = ev_chunks[i];
                    parse_events(&ctx, f, s, e, st);
                } else {
                    let (f, s, e) = ck_chunks[i - ne];
                    parse_clicks(&ctx, st, f, s, e);
                }
            },
        );
        if tbl.overflow.load(Relaxed) {
            eprintln!("person table too small ({cap} slots): retrying with {}", cap * 4);
            drop(p1);
            tbl.release();
            cap *= 4;
            continue;
        }
        break (tbl, p1);
    };
    lap(&format!("events and clicks parsed, {} persons known", tbl.used.load(Relaxed)));

    // 2. dedup per event_id bucket
    let n_r1: usize = p1.iter().map(|t| (0..EBK).map(|b| t.eb.len_of(b)).sum::<usize>()).sum();
    let p2 = par(
        threads,
        EBK,
        || P2 { out: Buckets::sized(NB, n_r1, threads), tab: Vec::new(), hs: Vec::new(), dead: Vec::new(), offs: Vec::new() },
        |st, b| dedup_bucket(&p1, b, st),
    );
    lap("deduplicated");

    // 3. per person bucket
    let width = ctx.ndays;
    let ev_src: Vec<&Buckets<Rec>> = p2.iter().map(|s| &s.out).collect();
    let ck_src: Vec<&Buckets<CRec>> = p1.iter().map(|s| &s.ck).collect();
    let results = par(
        threads,
        NB,
        || Agg {
            rows: Vec::new(),
            cred: vec![[0i64; 3]; (ncamp + 1) * width],
            assist: vec![0i64; (ncamp + 1) * width],
            stamp: vec![u32::MAX; ncamp + 1],
            iv: vec![(0i64, 0i64); ncamp + 1],
            cnt_e: Vec::new(),
            cnt_c: Vec::new(),
            evs: Vec::new(),
            cks: Vec::new(),
            plist: Vec::new(),
            sessions: Vec::new(),
            day_groups: Vec::new(),
            cats: Vec::new(),
            touched: Vec::new(),
        },
        |agg, b| bucket_work(&ctx, tbl, &ev_src, &ck_src, b, agg),
    );
    lap("sessions aggregated");

    // 4. outputs
    let mut cred = vec![[0i64; 3]; (ncamp + 1) * width];
    let mut assist = vec![0i64; (ncamp + 1) * width];
    for r in &results {
        for (a, b) in cred.iter_mut().zip(&r.cred) {
            a[0] += b[0];
            a[1] += b[1];
            a[2] += b[2];
        }
        for (a, b) in assist.iter_mut().zip(&r.assist) {
            *a += b;
        }
    }
    let mut camp_rows = Vec::new();
    for k in 0..=ncamp {
        for d in 0..width {
            let c = cred[k * width + d];
            let a = assist[k * width + d];
            if c[0] > 0 || a > 0 {
                let (id, ch): (Bytes, Bytes) = if k == ncamp { (b"organic", b"organic") } else { (camp_ids[k], camp_channel[k]) };
                camp_rows.push((id, ch, (day0 + d as i64) as i32, c[0], c[1], c[2], a));
            }
        }
    }
    let user_parts: Vec<&[UserRow]> = results.iter().map(|r| r.rows.as_slice()).collect();
    write_outputs(&out, &user_parts, &camp_rows, &cat_names, threads);
    lap(&format!("written: {} user rows, {} campaign rows", user_parts.iter().map(|p| p.len()).sum::<usize>(), camp_rows.len()));
    // everything is on disk: no destructors, the memory is released in parallel
    fast_exit(threads)
}

fn finish_row(mut r: UserRow, cats: &mut Vec<(u16, i64)>, rank: &[u32]) -> UserRow {
    r.top = cats
        .iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(rank[b.0 as usize].cmp(&rank[a.0 as usize])))
        .map(|c| c.0);
    cats.clear();
    r
}

// ---------------------------------------------------------------- Parquet
//
// Hand-written writer: one row group per file, one PLAIN page per column, no compression, no statistics.
// The footer is Thrift compact protocol (FileMetaData).

type CampRow = (Bytes, Bytes, i32, i64, i64, i64, i64);

/// Thrift compact protocol writer for the few structs a Parquet footer needs.
struct Thrift {
    b: Vec<u8>,
    last: i16,
    stack: Vec<i16>,
}

impl Thrift {
    const I32: u8 = 5;
    const I64: u8 = 6;
    const BINARY: u8 = 8;
    const LIST: u8 = 9;
    const STRUCT: u8 = 12;

    fn new() -> Thrift {
        Thrift { b: Vec::new(), last: 0, stack: Vec::new() }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.b.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.b.push(v as u8);
    }
    fn field(&mut self, id: i16, ty: u8) {
        let d = id - self.last;
        if (1..=15).contains(&d) {
            self.b.push((d as u8) << 4 | ty);
        } else {
            self.b.push(ty);
            self.varint(((id << 1) ^ (id >> 15)) as u16 as u64);
        }
        self.last = id;
    }
    fn i32(&mut self, id: i16, v: i32) {
        self.field(id, Self::I32);
        self.varint(((v << 1) ^ (v >> 31)) as u32 as u64);
    }
    fn i64(&mut self, id: i16, v: i64) {
        self.field(id, Self::I64);
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }
    fn str(&mut self, id: i16, s: &str) {
        self.field(id, Self::BINARY);
        self.varint(s.len() as u64);
        self.b.extend_from_slice(s.as_bytes());
    }
    /// List header; the elements follow (i32s via `elem_i32`, strings via `elem_str`, structs via `begin`..`end`).
    fn list(&mut self, id: i16, elem: u8, n: usize) {
        self.field(id, Self::LIST);
        if n < 15 {
            self.b.push((n as u8) << 4 | elem);
        } else {
            self.b.push(0xF0 | elem);
            self.varint(n as u64);
        }
    }
    fn elem_i32(&mut self, v: i32) {
        self.varint(((v << 1) ^ (v >> 31)) as u32 as u64);
    }
    fn elem_str(&mut self, s: &str) {
        self.varint(s.len() as u64);
        self.b.extend_from_slice(s.as_bytes());
    }
    /// Starts a struct as field `id` of the current struct.
    fn sub(&mut self, id: i16) {
        self.field(id, Self::STRUCT);
        self.begin();
    }
    /// Starts a struct (list element, or the top level).
    fn begin(&mut self) {
        self.stack.push(self.last);
        self.last = 0;
    }
    fn end(&mut self) {
        self.b.push(0);
        self.last = self.stack.pop().unwrap();
    }
}

const T_INT32: i32 = 1;
const T_INT64: i32 = 2;
const T_BYTES: i32 = 6;
const CT_UTF8: i32 = 0;
const CT_DATE: i32 = 6;

struct ColSpec {
    name: &'static str,
    ty: i32,
    optional: bool,
    converted: Option<i32>,
}

const fn col(name: &'static str, ty: i32, converted: Option<i32>) -> ColSpec {
    ColSpec { name, ty, optional: false, converted }
}

const USER_COLS: [ColSpec; 6] = [
    col("user_id", T_BYTES, Some(CT_UTF8)),
    col("day", T_INT32, Some(CT_DATE)),
    col("sessions", T_INT64, None),
    col("events", T_INT64, None),
    col("revenue_cents", T_INT64, None),
    ColSpec { name: "top_category", ty: T_BYTES, optional: true, converted: Some(CT_UTF8) },
];

const CAMP_COLS: [ColSpec; 7] = [
    col("campaign_id", T_BYTES, Some(CT_UTF8)),
    col("channel", T_BYTES, Some(CT_UTF8)),
    col("day", T_INT32, Some(CT_DATE)),
    col("sessions", T_INT64, None),
    col("purchasing_sessions", T_INT64, None),
    col("revenue_cents", T_INT64, None),
    col("assisted_sessions", T_INT64, None),
];

/// Writes `path` with one row group of `n` rows. `fill(i, buf)` appends column i's PLAIN page body to `buf`
/// (for the optional column: the definition levels first, as a length-prefixed RLE/bit-packed run).
fn write_parquet(path: &Path, cols: &[ColSpec], n: usize, mut fill: impl FnMut(usize, &mut Vec<u8>)) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::with_capacity(1 << 16, std::fs::File::create(path)?);
    f.write_all(b"PAR1")?;
    let mut off = 4i64;
    let mut body = Vec::new();
    let mut chunks = Vec::with_capacity(cols.len()); // (offset, size)
    for i in 0..cols.len() {
        body.clear();
        fill(i, &mut body);
        let size = i32::try_from(body.len()).map_err(|_| std::io::Error::other("page over 2 GiB"))?;
        let mut h = Thrift::new();
        h.begin();
        h.i32(1, 0); // DATA_PAGE
        h.i32(2, size);
        h.i32(3, size);
        h.sub(5);
        h.i32(1, n as i32); // num_values
        h.i32(2, 0); // PLAIN
        h.i32(3, 3); // definition levels: RLE
        h.i32(4, 3); // repetition levels: RLE
        h.end();
        h.end();
        f.write_all(&h.b)?;
        f.write_all(&body)?;
        let total = (h.b.len() + body.len()) as i64;
        chunks.push((off, total));
        off += total;
    }
    let mut t = Thrift::new();
    t.begin();
    t.i32(1, 1); // version
    t.list(2, Thrift::STRUCT, cols.len() + 1);
    t.begin();
    t.str(4, "schema");
    t.i32(5, cols.len() as i32);
    t.end();
    for c in cols {
        t.begin();
        t.i32(1, c.ty);
        t.i32(3, c.optional as i32); // REQUIRED 0, OPTIONAL 1
        t.str(4, c.name);
        if let Some(ct) = c.converted {
            t.i32(6, ct);
            t.sub(10); // LogicalType: STRING = 1, DATE = 6
            t.sub(if ct == CT_UTF8 { 1 } else { 6 });
            t.end();
            t.end();
        }
        t.end();
    }
    t.i64(3, n as i64);
    t.list(4, Thrift::STRUCT, 1);
    t.begin();
    t.list(1, Thrift::STRUCT, cols.len());
    for (c, &(o, size)) in cols.iter().zip(&chunks) {
        t.begin();
        t.i64(2, o); // file_offset
        t.sub(3); // ColumnMetaData
        t.i32(1, c.ty);
        t.list(2, Thrift::I32, 2);
        t.elem_i32(0); // PLAIN
        t.elem_i32(3); // RLE
        t.list(3, Thrift::BINARY, 1);
        t.elem_str(c.name);
        t.i32(4, 0); // UNCOMPRESSED
        t.i64(5, n as i64);
        t.i64(6, size);
        t.i64(7, size);
        t.i64(9, o); // data_page_offset
        t.end();
        t.end();
    }
    t.i64(2, off - 4); // total_byte_size
    t.i64(3, n as i64);
    t.end();
    t.str(6, "sessions");
    t.end();
    f.write_all(&t.b)?;
    f.write_all(&(t.b.len() as u32).to_le_bytes())?;
    f.write_all(b"PAR1")?;
    f.flush()
}

fn plain_bytes<'a>(buf: &mut Vec<u8>, vals: impl Iterator<Item = &'a [u8]>) {
    for v in vals {
        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
        buf.extend_from_slice(v);
    }
}

fn plain_i64(buf: &mut Vec<u8>, vals: impl Iterator<Item = i64>) {
    for v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_outputs(out: &Path, users: &[&[UserRow]], camps: &[CampRow], cat_names: &[Bytes], threads: usize) {
    let ud = out.join("user_daily");
    let cd = out.join("campaign_daily");
    for d in [&ud, &cd] {
        std::fs::create_dir_all(d).unwrap_or_else(|e| die(&format!("mkdir {}: {e}", d.display())));
    }
    // `threads` files of about equal size, each a list of row slices
    let total: usize = users.iter().map(|p| p.len()).sum();
    let per = total.div_ceil(threads).max(1);
    let mut files: Vec<Vec<&[UserRow]>> = vec![Vec::new()];
    let mut room = per;
    for &p in users {
        let mut p = p;
        while !p.is_empty() {
            if room == 0 {
                files.push(Vec::new());
                room = per;
            }
            let (a, b) = p.split_at(room.min(p.len()));
            files.last_mut().unwrap().push(a);
            room -= a.len();
            p = b;
        }
    }
    std::thread::scope(|sc| {
        for (gi, g) in files.iter().enumerate() {
            let ud = &ud;
            sc.spawn(move || {
                write_users(&ud.join(format!("part-{gi}.parquet")), g, cat_names)
                    .unwrap_or_else(|e| die(&format!("write user_daily: {e}")))
            });
        }
        write_camps(&cd.join("part-0.parquet"), camps).unwrap_or_else(|e| die(&format!("write campaign_daily: {e}")));
    });
}

fn write_users(path: &Path, segs: &[&[UserRow]], cat_names: &[Bytes]) -> std::io::Result<()> {
    let n: usize = segs.iter().map(|s| s.len()).sum();
    let rows = || segs.iter().flat_map(|s| s.iter());
    write_parquet(path, &USER_COLS, n, |c, b| match c {
        0 => {
            b.reserve(n * 24);
            plain_bytes(b, rows().map(|r| r.name.get()))
        }
        1 => {
            b.reserve(n * 4);
            rows().for_each(|r| b.extend_from_slice(&r.day.to_le_bytes()))
        }
        2 => plain_i64(b, rows().map(|r| r.sessions)),
        3 => plain_i64(b, rows().map(|r| r.events)),
        4 => plain_i64(b, rows().map(|r| r.revenue)),
        _ => {
            // definition levels: one bit-packed run of ceil(n / 8) groups of 8 bits (1 = present)
            let groups = n.div_ceil(8);
            let mut run = Vec::with_capacity(groups + 5);
            let mut v = ((groups as u64) << 1) | 1;
            while v >= 0x80 {
                run.push(v as u8 | 0x80);
                v >>= 7;
            }
            run.push(v as u8);
            let at = run.len();
            run.resize(at + groups, 0);
            for (i, r) in rows().enumerate() {
                run[at + i / 8] |= (r.top.is_some() as u8) << (i % 8);
            }
            b.extend_from_slice(&(run.len() as u32).to_le_bytes());
            b.extend_from_slice(&run);
            plain_bytes(b, rows().filter_map(|r| r.top.map(|t| cat_names[t as usize])))
        }
    })
}

fn write_camps(path: &Path, rows: &[CampRow]) -> std::io::Result<()> {
    write_parquet(path, &CAMP_COLS, rows.len(), |c, b| match c {
        0 => plain_bytes(b, rows.iter().map(|r| r.0)),
        1 => plain_bytes(b, rows.iter().map(|r| r.1)),
        2 => rows.iter().for_each(|r| b.extend_from_slice(&r.2.to_le_bytes())),
        3 => plain_i64(b, rows.iter().map(|r| r.3)),
        4 => plain_i64(b, rows.iter().map(|r| r.4)),
        5 => plain_i64(b, rows.iter().map(|r| r.5)),
        _ => plain_i64(b, rows.iter().map(|r| r.6)),
    })
}

// ---------------------------------------------------------------- exit

/// Ends the process once the outputs are on disk. Left to the kernel, freeing the GBs of heap and unmapping the
/// input files runs on one thread (about 0.1 s); here every thread zaps a share first with MADV_DONTNEED.
/// Nothing may be freed or dropped after the first zap: the heap, malloc state included, reads as zeros from then on.
/// Never zapped: the threads' stacks (their TCBs sit in them), this thread's TLS block, every region under 1 MiB
/// (libc state), `[heap]` and the other named regions. Any surprise before the zap falls back to a plain exit.
#[cfg(target_os = "linux")]
fn fast_exit(threads: usize) -> ! {
    use std::sync::atomic::{AtomicUsize, Ordering::{Acquire, Release}};
    const MAXP: usize = 2048;
    const PIECE: usize = 16 << 20;
    const MARGIN: usize = 64 << 10;
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else { std::process::exit(0) };
    // shared state lives on this thread's stack: the heap is about to be zeroed
    let reg = AtomicUsize::new(0);
    let go = AtomicBool::new(false);
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let np = AtomicUsize::new(0);
    let keep: [[AtomicUsize; 2]; 32] = std::array::from_fn(|_| [AtomicUsize::new(0), AtomicUsize::new(0)]);
    let pieces: [[AtomicUsize; 2]; MAXP] = std::array::from_fn(|_| [AtomicUsize::new(0), AtomicUsize::new(0)]);
    std::thread::scope(|sc| {
        let mut n = 0;
        while n < threads.clamp(1, 32) {
            let (reg, go, next, done, np, keep, pieces) = (&reg, &go, &next, &done, &np, &keep, &pieces);
            let k = n;
            let spawned = std::thread::Builder::new().spawn_scoped(sc, move || {
                // publish this thread's stack: never zap it
                let (mut lo, mut hi) = (0usize, 0usize);
                unsafe {
                    let mut attr: libc::pthread_attr_t = std::mem::zeroed();
                    if libc::pthread_getattr_np(libc::pthread_self(), &mut attr) == 0 {
                        let mut addr = std::ptr::null_mut();
                        let mut size = 0usize;
                        libc::pthread_attr_getstack(&attr, &mut addr, &mut size);
                        lo = addr as usize;
                        hi = lo + size;
                        libc::pthread_attr_destroy(&mut attr);
                    }
                }
                keep[k][0].store(lo.saturating_sub(MARGIN), Relaxed);
                keep[k][1].store(hi + MARGIN, Relaxed);
                reg.fetch_add(1, Release);
                while !go.load(Acquire) {
                    unsafe { libc::sched_yield() };
                }
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= np.load(Relaxed) {
                        break;
                    }
                    unsafe { libc::madvise(pieces[i][0].load(Relaxed) as *mut libc::c_void, pieces[i][1].load(Relaxed), libc::MADV_DONTNEED) };
                    done.fetch_add(1, Release);
                }
                // never return: thread exit would free into the zeroed heap
                loop {
                    unsafe { libc::sched_yield() };
                }
            });
            if spawned.is_err() {
                break;
            }
            n += 1;
        }
        if n == 0 {
            std::process::exit(0);
        }
        while reg.load(Acquire) < n {
            std::thread::yield_now();
        }
        // a thread that could not read its stack bounds publishes (0, MARGIN): nothing is zapped then
        if (0..n).any(|k| keep[k][1].load(Relaxed) <= MARGIN) {
            std::process::exit(0);
        }
        let me = unsafe { libc::pthread_self() } as usize;
        let mut excl: Vec<(usize, usize)> = (0..n).map(|k| (keep[k][0].load(Relaxed), keep[k][1].load(Relaxed))).collect();
        excl.push((me.saturating_sub(MARGIN), me + MARGIN));
        let mut cnt = 0usize;
        for line in maps.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            // anonymous heap (no name), or an input file mapped read-only and shared
            let anon = f.len() == 5 && f[1] == "rw-p";
            let file = f.len() == 6 && f[1] == "r--s" && f[5].starts_with('/') && !f[5].starts_with("/dev/");
            if !anon && !file {
                continue;
            }
            let Some((a, b)) = f[0].split_once('-') else { continue };
            let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16)) else { continue };
            let mut segs = vec![(a, b)];
            for &(x, y) in &excl {
                let mut out = Vec::new();
                for (s, e) in segs {
                    if y <= s || x >= e {
                        out.push((s, e));
                    } else {
                        if x > s {
                            out.push((s, x & !4095));
                        }
                        if y < e {
                            out.push(((y + 4095) & !4095, e));
                        }
                    }
                }
                segs = out;
            }
            for (s, e) in segs {
                if e <= s || e - s < (1 << 20) {
                    continue;
                }
                let mut p = s;
                while p < e && cnt < MAXP {
                    let l = PIECE.min(e - p);
                    pieces[cnt][0].store(p, Relaxed);
                    pieces[cnt][1].store(l, Relaxed);
                    cnt += 1;
                    p += l;
                }
            }
        }
        np.store(cnt, Relaxed);
        go.store(true, Release);
        while done.load(Acquire) < cnt {
            std::thread::yield_now();
        }
        unsafe { libc::_exit(0) }
    });
    unreachable!()
}

#[cfg(not(target_os = "linux"))]
fn fast_exit(_threads: usize) -> ! {
    std::process::exit(0)
}
#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn pick<'a>(&mut self, v: &[&'a str]) -> &'a str {
            v[self.below(v.len())]
        }
    }

    fn ts_value(r: &mut Rng) -> String {
        let good = ["2028-02-20T00:00:00Z", "2028-02-29T23:59:59Z", "2028-03-01T12:30:00+05:30", "2028-02-21T01:00:00-00:00", "2028-02-25T10:00:00-10:00"];
        let bad = ["", "2027-02-29T00:00:00Z", "2028-02-30T00:00:00Z", "2028-13-01T00:00:00Z", "2028-02-21T24:00:00Z", "2028-02-21T23:60:00Z", "2028-02-21T23:59:60Z", "2028-02-21t00:00:00Z", "2028-02-21T00:00:00z", "2028-02-21 00:00:00Z", "2028-02-21T00:00:00", "2028-02-21T00:00:00.5Z", "2028-02-21T00:00:00+24:00", "2028-02-21T00:00:00+05:60", "2028-02-21T00:00:00+0500", "1.79e12", "-5", "1835000000000.0", "00000000000000000000001835000000000", "99999999999999999999", "2028-00-10T00:00:00Z", "2028-02-00T00:00:00Z", "0000-01-01T00:00:00Z", "9999-12-31T23:59:59Z"];
        match r.below(6) {
            0 => r.pick(&good).to_string(),
            1 => r.pick(&bad).to_string(),
            2 => {
                // F3 of any length, sometimes with leading zeros or a stray byte
                let ms = 1835049600000u64 + r.below(20 * 86_400_000) as u64;
                let mut s = format!("{}{}", "0".repeat(r.below(4)), ms);
                if r.below(4) == 0 {
                    s.truncate(1 + r.below(s.len()));
                }
                if r.below(8) == 0 {
                    let k = r.below(s.len());
                    s.replace_range(k..k + 1, r.pick(&["a", "/", ":", " ", "٢", "\r"]));
                }
                s
            }
            _ => {
                // F1 / F2 near the window, sometimes with one byte changed
                let (y, mo, d, h, mi, s) = (2028, 2 + r.below(2), 1 + r.below(31), r.below(25), r.below(61), r.below(61));
                let mut v = format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}");
                v += &match r.below(3) {
                    0 => "Z".to_string(),
                    1 => format!("{}{:02}:{:02}", if r.below(2) == 0 { '+' } else { '-' }, r.below(25), r.below(61)),
                    _ => r.pick(&["Z\r", "", "+05:30\r", "z"]).to_string(),
                };
                if r.below(10) == 0 {
                    let k = r.below(v.len());
                    v.replace_range(k..k + 1, r.pick(&["0", "9", ":", "-", "T", "x", "/", "\u{7f}"]));
                }
                v
            }
        }
    }

    fn line(r: &mut Rng) -> String {
        if r.below(200) == 0 {
            return String::new();
        }
        let id = |r: &mut Rng, c: char| -> String {
            match r.below(12) {
                0 => String::new(),
                1 => format!("{c}{}", "x".repeat(r.below(40))),
                2 => format!(" {c}{}", r.below(1000)),
                _ => format!("{c}{:x}", r.next() % (1u64 << (4 * (4 + r.below(8))))),
            }
        };
        let dirty = r.below(3) == 0;
        let mut f: Vec<String> = vec![
            id(r, 'e'),
            id(r, 'u'),
            ts_value(r),
            if r.below(10) == 0 { ts_value(r) } else { "2028-02-22T00:00:00Z".to_string() },
            r.pick(&["view", "cart", "purchase", "view", "cart", "purchase", "Purchase", "cart ", "", "views", "purchas"]).to_string(),
            r.pick(&["p1", "p2", "p3", "P1", "", "p9999", "p2\r"]).to_string(),
            r.pick(&["", "100", "0250", "0", "10000000", "10000001", "-5", "1.5", "12\r", "٢٥٠", "00000000000000000000000000001"]).to_string(),
        ];
        if !dirty {
            let ms = 1835049600000u64 + r.below(14 * 86_400_000) as u64;
            f[2] = r.pick(&[&ms.to_string()[..], "2028-02-21T05:00:00Z", "2028-03-01T12:30:00+05:30", "2028-02-29T23:59:59Z"]).to_string();
            f[4] = r.pick(&["view", "cart", "purchase"]).to_string();
            f[5] = r.pick(&["p1", "p2", "p3"]).to_string();
        }
        if r.below(3) == 0 {
            f[2] = r.pick(&["2028-02-21T05:00:00Z", "2028-02-20T00:00:00Z"]).to_string();
            f[3] = r.pick(&["2028-02-22T05:00:00Z", "2028-02-22T05:00:01Z", "2028-02-21T04:00:00Z", "2028-03-10T00:00:00Z"]).to_string();
        }
        for v in f.iter_mut() {
            *v = match if dirty { r.below(14) } else { [0, 1, 2, 3, 8, 9, 10, 11, 12, 13][r.below(10)] } {
                0..=3 => format!("\"{v}\""),
                4 => format!("\"{v}"),
                5 => format!("{v}\""),
                6 if !v.is_empty() => {
                    let k = r.below(v.len() + 1);
                    let mut w = v.clone();
                    if w.is_char_boundary(k) {
                        w.insert(k, '"');
                    }
                    w
                }
                7 => r.pick(&["\"", "\"\"", "\"\"\"", "\"\"\"\"", "\"a\"b\"", "a\"\"b"]).to_string(),
                _ => v.clone(),
            };
        }
        match if dirty { r.below(40) } else { 99 } {
            0 => {
                f.pop();
            }
            1 => f.push(String::new()),
            2 => f.insert(r.below(7), "x".to_string()),
            3 => f[r.below(7)] = "x".repeat(100 + r.below(60)),
            _ => {}
        }
        f.join(",")
    }

    fn ctx() -> Ctx {
        let products = IdMap::build([(SStr::new(b"p1"), 0u16), (SStr::new(b"p2"), 1u16)].into_iter(), 2);
        Ctx {
            ws: parse_ts(b"2028-02-20T00:00:00Z").unwrap(),
            we: parse_ts(b"2028-03-05T00:00:00Z").unwrap(),
            base: parse_ts(b"2028-02-20T00:00:00Z").unwrap() - WEEK,
            day0: 0,
            ndays: 14,
            ncamp: 0,
            products,
            campaigns: IdMap::build(std::iter::empty(), 0),
            unknown_cat: 2,
            cat_rank: vec![0, 1, 2],
        }
    }

    fn fresh(t: &Table) -> P1<'_> {
        P1::new(t, ctx().base, Buckets::new(EBK, 64), Buckets::new(NB, 64))
    }

    type Flat = Vec<(Vec<u8>, i64, u32, u32, u32, u32)>;
    fn flat(st: &P1) -> Flat {
        (0..EBK).flat_map(|b| st.eb.iter(b)).map(|r| (r.eid.get().to_vec(), r.ingest, r.ts, r.pid, r.pk, r.cat)).collect()
    }

    #[test]
    fn fast_parse_matches_reference() {
        let c = ctx();
        let mut r = Rng(0x1234_5678_9abc_def1);
        let mut kept = 0;
        for round in 0..300 {
            let mut text = String::from("event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
            for _ in 0..r.below(400) {
                text += &line(&mut r);
                text.push('\n');
            }
            if round % 3 == 0 {
                text += &line(&mut r); // no trailing newline
            }
            let file: Bytes = Box::leak(text.into_bytes().into_boxed_slice());
            // the same random chunking for both
            let mut cuts = vec![0, file.len()];
            for _ in 0..r.below(4) {
                cuts.push(r.below(file.len() + 1));
            }
            cuts.sort();
            let (ta, tb) = (Table::new(1 << 16), Table::new(1 << 16));
            let (mut a, mut b) = (fresh(&ta), fresh(&tb));
            for w in cuts.windows(2) {
                parse_events(&c, file, w[0], w[1], &mut a);
                for_records::<7>(file, w[0], w[1], |f| {
                    if let Some(e) = parse_event(&c, f) {
                        b.add(e);
                    }
                });
                b.flush_events();
            }
            let (fa, fb) = (flat(&a), flat(&b));
            kept += fb.iter().filter(|r| r.4 & KEEP != 0).count();
            assert!(fa == fb, "round {round}: fast parse differs\n{}", std::str::from_utf8(file).unwrap());
        }
        assert!(kept > 1000, "too few kept events ({kept}) to mean anything");
    }

    #[test]
    fn ts_at_matches_parse_ts() {
        let mut r = Rng(0xfeed_beef_1357_9bdf);
        for _ in 0..200_000 {
            let s = ts_value(&mut r);
            let mut buf = s.clone().into_bytes();
            buf.extend_from_slice(&[b'7'; 32]);
            assert_eq!(unsafe { ts_at(buf.as_ptr(), s.len()) }, parse_ts(s.as_bytes()), "{s:?}");
        }
    }

    #[test]
    fn hash16_matches_hash_bytes() {
        let mut r = Rng(42);
        for len in 0..=16 {
            for _ in 0..1000 {
                let buf: Vec<u8> = (0..32).map(|_| r.next() as u8).collect();
                assert_eq!(hash16(unsafe { load16(buf.as_ptr(), len) }, len), hash_bytes(&buf[..len]));
            }
        }
    }
}

