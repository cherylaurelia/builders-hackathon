//! Shopping Sessions pipeline. rules.md is the spec; each phase names the rule it implements.
//!
//! Phases:
//!   1. links  -> union-find -> map every linked ID to its person name (byte-wise smallest ID)
//!   2. events -> parse in parallel, bucketed by hash(event_id)
//!   3. per event bucket: dedup (smallest ingest_ts) -> keep valid, non-late -> bucket by hash(person)
//!   4. clicks -> parse + validate in parallel, bucketed by hash(person) (copies are identical: no dedup needed)
//!   5. per person bucket: sessionize, credit, assist, aggregate
//!   6. write Parquet

use memmap2::Mmap;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

type Bytes = &'static [u8];

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;
const CHUNK: usize = 4 << 20;
const EB: usize = 256; // event_id buckets
const PB: usize = 256; // person buckets
const UB: usize = 256; // user_id buckets: each holds a cache-sized slice of the link table

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

/// Key whose hash was computed earlier; equality is still exact on the bytes.
#[derive(Clone, Copy)]
struct PreHashed {
    h: u64,
    s: SStr,
}
impl PartialEq for PreHashed {
    fn eq(&self, o: &Self) -> bool {
        self.h == o.h && self.s == o.s
    }
}
impl Eq for PreHashed {}
impl Hash for PreHashed {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.h)
    }
}
#[derive(Default)]
struct IdHasher(u64);
impl Hasher for IdHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!()
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
    fn finish(&self) -> u64 {
        self.0
    }
}
type PreMap<V> = HashMap<PreHashed, V, BuildHasherDefault<IdHasher>>;

/// The person bucket an unlinked ID would have: such events and clicks never move in phase 4b.
#[inline]
fn uid_bucket(h: u64) -> usize {
    const _: () = assert!(UB == PB);
    person_bucket(h)
}

#[inline]
fn person_bucket(h: u64) -> usize {
    // middle bits: hashbrown uses the low bits for the slot and the top 7 for tags
    ((h >> 40) as usize) & (PB - 1)
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




const fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Days since the epoch of the 1st of each month from January 1900 (index (y - 1900) * 12 + m - 1) to January 2156:
/// a month's length is the gap to the next entry.
static MONTH_START: [i32; 3073] = {
    let mut t = [0i32; 3073];
    let mut i = 0;
    while i < t.len() {
        t[i] = days_from_civil(1900 + (i / 12) as i64, (i % 12) as i64 + 1, 1) as i32;
        i += 1;
    }
    t
};

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
    if (mo.wrapping_sub(1) > 11) | (dd == 0) | (h > 23) | (mi > 59) | (sec > 59) {
        return None;
    }
    let days = if (1900..2156).contains(&y) {
        let i = (y as usize - 1900) * 12 + mo as usize - 1;
        let (first, next) = (MONTH_START[i] as i64, MONTH_START[i + 1] as i64);
        if dd as i64 > next - first {
            return None;
        }
        first + dd as i64 - 1
    } else {
        const DIM: [u64; 13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        if dd > DIM[mo as usize] + ((mo == 2) & leap) as u64 {
            return None;
        }
        days_from_civil(y as i64, mo as i64, dd as i64)
    };
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

/// `n` empty vectors, each with room for about `total / (threads * n)` items (+25% for uneven shares).
fn sized<T>(n: usize, total: usize, threads: usize) -> Vec<Vec<T>> {
    // a multiple of 5 keeps a Vec<UEvent> buffer a whole number of PEvents (to_pevents); doubling preserves it
    let each = (total / (threads * n).max(1) * 5 / 4 + 16).next_multiple_of(5);
    (0..n).map(|_| Vec::with_capacity(each)).collect()
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

/// A valid, non-late event in its user_id bucket, until dedup and person resolution.
struct UEvent {
    uh: u64,
    uid: SStr,
    eid: Packed,
    ingest: i64,
    ts: i64,
    price: u32,
    cat: u16,
    kind: u8,
    dropped: AtomicBool, // a copy of an event_id whose kept copy is another one
}

/// hash(event_id) and where its line went: `LOC_BAD` set = the thread's invalid list, else a user_id bucket.
struct ERec {
    h: u64,
    loc: u64,
}
const LOC_BAD: u64 = 1 << 63;
const LOC_T: u32 = 56; // thread, 7 bits
const LOC_B: u32 = 40; // user_id bucket, 16 bits; the index is the low 40 bits

/// One parse thread's output.
struct EvParse {
    t: usize,
    ub: Vec<Vec<UEvent>>,
    bad: Vec<(Packed, i64)>, // (event_id, ingest_ts) of invalid or late lines: they still win dedup
    eb: Vec<Vec<ERec>>,
    ck: Vec<Vec<PClick>>, // this thread's valid clicks, by user_id bucket
}

/// A surviving event, routed to its person's bucket.
#[derive(Clone, Copy)]
struct PEvent {
    nh: u64,
    name: SStr,
    ts: i64,
    price: u32,
    cat: u16,
    kind: u8,
}

/// The PEvents `f` makes of the events of `v`, written over `v`'s own buffer when its byte size is a whole
/// number of PEvents (a PEvent is smaller, so the write position never passes the read position).
fn to_pevents(v: Vec<UEvent>, mut f: impl FnMut(&UEvent) -> Option<PEvent>) -> Vec<PEvent> {
    use std::mem::{align_of, size_of};
    const _: () = assert!(size_of::<PEvent>() <= size_of::<UEvent>() && align_of::<PEvent>() == align_of::<UEvent>());
    let bytes = v.capacity() * size_of::<UEvent>();
    if bytes % size_of::<PEvent>() != 0 {
        return v.iter().filter_map(f).collect();
    }
    let mut v = std::mem::ManuallyDrop::new(v);
    let (src, len) = (v.as_mut_ptr(), v.len());
    let dst = src as *mut PEvent;
    let mut k = 0;
    for i in 0..len {
        // event i is read whole before PEvent k <= i is written, and PEvent k ends before event i + 1 starts
        let e = unsafe { std::ptr::read(src.add(i)) };
        if let Some(p) = f(&e) {
            unsafe { std::ptr::write(dst.add(k), p) };
            k += 1;
        }
    }
    unsafe { Vec::from_raw_parts(dst, k, bytes / size_of::<PEvent>()) }
}

/// A valid click, routed to its person's bucket: 40 bytes.
#[derive(Clone, Copy)]
struct PClick {
    nh: u64,
    name: SStr,
    cid: Packed,
    ts: u32, // ms since ws - WEEK: a click ts is in [ws - WEEK, we)
    camp: u16,
}

struct UserRow {
    name: SStr,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: Option<u16>,
}

struct Ctx {
    ws: i64,
    we: i64,
    day0: i64,
    ndays: usize,
    ncamp: usize, // organic is index ncamp
    products: BytesMap<u16>,
    campaigns: BytesMap<u16>,
    unknown_cat: u16,
    cat_rank: Vec<u32>,
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
    let usage0 = std::cell::Cell::new(usage());
    let lap = |what: &str| {
        let (c, ins) = usage();
        let (c0, i0) = usage0.get();
        // one write(2) per line: eprintln! issues one per formatted piece
        let line = format!("[{:7.3}s wall, {:6.3}s cpu, {:6.2}G ins] {what}\n", t0.elapsed().as_secs_f64(), c - c0, (ins - i0) as f64 / 1e9);
        let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
        usage0.set((c, ins));
    };

    // manifest
    let manifest = std::fs::read_to_string(data.join("manifest.json")).unwrap_or_else(|e| die(&format!("manifest.json: {e}")));
    let ws = manifest_ts(&manifest, "window_start");
    let we = manifest_ts(&manifest, "window_end");
    if we <= ws {
        die("manifest: empty window");
    }
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1) as usize;

    // products -> categories and campaigns, on a side thread while the links are built (7 ms of serial start-up)
    let tables = {
        let data = data.clone();
        std::thread::spawn(move || {
    // "unknown" is its own category, never one of the file's
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
    (cat_names, products, unknown_cat, cat_rank, camp_ids, camp_channel, campaigns)
        })
    };

    // 1. links, on every thread before anything else: small, and a side thread would trail the event parse
    let links = build_links(&csv_files(&data.join("links")), threads);
    lap(&format!("links built: {} linked ids", links.iter().map(|m| m.len()).sum::<usize>()));
    let (cat_names, products, unknown_cat, cat_rank, camp_ids, camp_channel, campaigns) = tables.join().unwrap_or_else(|_| die("tables thread panicked"));
    let ncamp = camp_ids.len();
    lap("small tables");

    let ctx = Ctx { ws, we, day0, ndays, ncamp, products, campaigns, unknown_cat, cat_rank };

    // 2. events: parse. A valid, non-late event goes to its user_id bucket; an invalid one to a side list.
    //    Every event also leaves a (hash(event_id), location) record in its event_id bucket.
    //    The click chunks are the last tasks of the same par: nothing about clicks waits for dedup, and the
    //    small last tasks even out the threads' finishing times.
    let ev_files = csv_files(&data.join("events"));
    let ev_chunks = chunks(&ev_files);
    let ev_est = ev_files.iter().map(|f| f.len()).sum::<usize>() / 60; // lines are ~75 bytes
    let ck_files = csv_files(&data.join("clicks"));
    let ck_chunks = chunks(&ck_files);
    let ck_est = ck_files.iter().map(|f| f.len()).sum::<usize>() / 40; // lines are ~50 bytes
    let tid = AtomicUsize::new(0);
    let mut parsed = par(
        threads,
        ev_chunks.len() + ck_chunks.len(),
        || EvParse {
            t: tid.fetch_add(1, Relaxed),
            ub: sized::<UEvent>(UB, ev_est, threads),
            bad: Vec::new(),
            eb: sized::<ERec>(EB, ev_est, threads),
            ck: sized::<PClick>(UB, ck_est, threads),
        },
        |st, i| {
            if i < ev_chunks.len() {
                let (f, s, e) = ev_chunks[i];
                parse_events(&ctx, f, s, e, st);
            } else {
                let (f, s, e) = ck_chunks[i - ev_chunks.len()];
                parse_clicks(&ctx, f, s, e, &mut st.ck);
            }
        },
    );
    parsed.sort_by_key(|p| p.t); // locations name threads by t
    if parsed.len() > 127 {
        die("too many threads");
    }
    let clicks: Vec<Vec<Vec<PClick>>> = parsed.iter_mut().map(|p| std::mem::take(&mut p.ck)).collect();
    lap("events and clicks parsed");

    // 3. duplicates: per event_id bucket, find hashes seen more than once (2-4% of events). Among the copies of
    //    one event_id the smallest ingest_ts is kept, so every other valid copy is marked dropped.
    let rec = |loc: u64| -> (i64, Bytes, Option<&UEvent>) {
        let t = &parsed[((loc >> LOC_T) & 0x7f) as usize];
        let i = (loc & ((1 << LOC_B) - 1)) as usize;
        if loc & LOC_BAD != 0 {
            let (eid, ingest) = t.bad[i];
            (ingest, eid.get(), None)
        } else {
            let e = &t.ub[((loc >> LOC_B) & 0xffff) as usize][i];
            (e.ingest, e.eid.get(), Some(e))
        }
    };
    let n_dups: usize = par(
        threads,
        EB,
        || (Vec::<(u64, u64)>::new(), Vec::<(u64, u64)>::new(), Vec::<(i64, Bytes, Option<&UEvent>)>::new(), 0usize),
        |(table, dups, grp, nd), b| {
            // key = hash | 1 so 0 marks an empty slot; hashes that differ only in bit 0 just meet in a group below
            let n: usize = parsed.iter().map(|t| t.eb[b].len()).sum();
            let cap = (n * 2).next_power_of_two().max(16);
            let mask = cap - 1;
            table.clear();
            table.resize(cap, (0, 0));
            dups.clear();
            for t in &parsed {
                for r in &t.eb[b] {
                    let k = r.h | 1;
                    let mut s = (k as usize >> 1) & mask;
                    loop {
                        let (x, first) = table[s];
                        if x == 0 {
                            table[s] = (k, r.loc);
                            break;
                        }
                        if x == k {
                            if first != u64::MAX {
                                dups.push((k, first));
                                table[s].1 = u64::MAX;
                            }
                            dups.push((k, r.loc));
                            break;
                        }
                        s = (s + 1) & mask;
                    }
                }
            }
            *nd += dups.len();
            dups.sort_unstable();
            let mut i = 0;
            while i < dups.len() {
                let mut j = i + 1;
                while j < dups.len() && dups[j].0 == dups[i].0 {
                    j += 1;
                }
                grp.clear();
                grp.extend(dups[i..j].iter().map(|&(_, loc)| rec(loc)));
                grp.sort_unstable_by_key(|g| g.0);
                // ordered by ingest: the first copy of each distinct event_id is its kept copy
                for k in 0..grp.len() {
                    if grp[..k].iter().any(|g| g.1 == grp[k].1) {
                        if let Some(e) = grp[k].2 {
                            e.dropped.store(true, Relaxed);
                        }
                    }
                }
                i = j;
            }
        },
    )
    .iter()
    .map(|x| x.3)
    .sum();
    lap(&format!("duplicates marked: {n_dups} copies share a hash"));

    // 4b. per user_id bucket, which is also the person bucket of an unlinked ID: user_id -> person through that
    //     bucket's slice of the link table. Events and clicks stay where they are, rewritten in place, unless
    //     their person's name hashes to another bucket: only those are copied out.
    let n_ev: usize = parsed.iter().flat_map(|t| t.ub.iter()).map(|v| v.len()).sum();
    let n_ck: usize = clicks.iter().flatten().map(|v| v.len()).sum();
    let nt = parsed.len();
    let mut ev_in: Vec<Vec<Vec<UEvent>>> = (0..UB).map(|_| Vec::with_capacity(nt)).collect();
    for t in &mut parsed {
        for (u, v) in t.ub.drain(..).enumerate() {
            ev_in[u].push(v);
        }
    }
    let nc = clicks.len();
    let mut ck_in: Vec<Vec<Vec<PClick>>> = (0..UB).map(|_| Vec::with_capacity(nc)).collect();
    for t in clicks {
        for (u, v) in t.into_iter().enumerate() {
            ck_in[u].push(v);
        }
    }
    let work: Vec<std::sync::Mutex<(Vec<Vec<UEvent>>, Vec<Vec<PClick>>)>> =
        ev_in.into_iter().zip(ck_in).map(std::sync::Mutex::new).collect();
    let resolved = par(
        threads,
        UB,
        || (Vec::new(), sized::<PEvent>(PB, n_ev / 2, threads), sized::<PClick>(PB, n_ck / 2, threads)),
        |(stay, oe, oc), u| {
            let (evs, mut cks) = std::mem::take(&mut *work[u].lock().unwrap_or_else(|_| die("resolve poisoned")));
            let map = &links[u];
            let person = |h: u64, s: SStr| map.get(&PreHashed { h, s }).copied().unwrap_or((s, h));
            let evs: Vec<Vec<PEvent>> = evs
                .into_iter()
                .map(|v| {
                    to_pevents(v, |e| {
                        if e.dropped.load(Relaxed) {
                            return None;
                        }
                        let (name, nh) = person(e.uh, e.uid);
                        let pe = PEvent { nh, name, ts: e.ts, price: e.price, cat: e.cat, kind: e.kind };
                        let p = person_bucket(nh);
                        if p == u {
                            return Some(pe);
                        }
                        oe[p].push(pe);
                        None
                    })
                })
                .collect();
            for v in &mut cks {
                v.retain_mut(|c| {
                    let (name, nh) = person(c.nh, c.name);
                    let p = person_bucket(nh);
                    if p == u {
                        c.nh = nh;
                        c.name = name;
                        return true;
                    }
                    oc[p].push(PClick { nh, name, ..*c });
                    false
                });
            }
            stay.push((u, evs, cks));
        },
    );
    drop(parsed);
    drop(work);
    // sources for the sessions phase: what stayed, per parse thread, then what moved, per resolve thread
    let mut stay_e: Vec<Vec<Vec<PEvent>>> = (0..nt).map(|_| (0..PB).map(|_| Vec::new()).collect()).collect();
    let mut stay_c: Vec<Vec<Vec<PClick>>> = (0..nc).map(|_| (0..PB).map(|_| Vec::new()).collect()).collect();
    let mut routed = Vec::new();
    let mut clicks = Vec::new();
    for (stay, oe, oc) in resolved {
        for (u, evs, cks) in stay {
            for (t, v) in evs.into_iter().enumerate() {
                stay_e[t][u] = v;
            }
            for (t, v) in cks.into_iter().enumerate() {
                stay_c[t][u] = v;
            }
        }
        routed.push(oe);
        clicks.push(oc);
    }
    let routed: Vec<Vec<Vec<PEvent>>> = stay_e.into_iter().chain(routed).collect();
    let clicks: Vec<Vec<Vec<PClick>>> = stay_c.into_iter().chain(clicks).collect();
    lap("user ids resolved");

    // 5. per person bucket
    let width = ctx.ndays;
    let results = par(
        threads,
        PB,
        || Agg::new(ncamp, width),
        |agg, p| person_bucket_work(&ctx, &routed, &clicks, p, agg),
    );
    drop(routed);
    drop(clicks);
    lap("sessions aggregated");

    // 6. outputs
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
    if n_ev >= i32::MAX as usize {
        die("too many events for 32-bit session and event counts");
    }
    write_outputs(&out, &user_parts, &camp_rows, &cat_names, threads);
    lap(&format!("written: {} user rows, {} campaign rows", user_parts.iter().map(|p| p.len()).sum::<usize>(), camp_rows.len()));
    // everything is on disk: no destructors, the memory is released in parallel
    fast_exit(threads)
}

/// Per user_id bucket: every linked ID -> (its person's name, hash(name)). All stages run on `threads` threads.
/// A node is (bucket, index in that bucket) packed as `bucket << 24 | index`; union-find runs on dense
/// numbers `base[bucket] + index`, always linking the larger root under the smaller.
fn build_links(files: &[Bytes], threads: usize) -> Vec<PreMap<(SStr, u64)>> {
    const IX: u32 = 24;
    // A. parse: each ID goes to its bucket's list as (hash, ID); each line leaves two endpoints, (bucket, position)
    struct LParse {
        ends: Vec<u32>,
        occ: Vec<Vec<(u64, SStr)>>,
    }
    let ch = chunks_of(files, 1 << 18);
    let est = files.iter().map(|f| f.len()).sum::<usize>() / 10; // lines are ~20 bytes, two IDs each
    let parts = par(
        threads,
        ch.len(),
        || LParse { ends: Vec::new(), occ: sized(UB, est, threads) },
        |st, i| {
            let (f, s, e) = ch[i];
            for_records::<2>(f, s, e, |[a, b]| {
                if a.is_empty() || b.is_empty() {
                    return;
                }
                for id in [a, b] {
                    let h = hash_bytes(id);
                    let u = uid_bucket(h);
                    if st.occ[u].len() >= 1 << IX {
                        die("too many link lines");
                    }
                    st.ends.push(((u as u32) << IX) | st.occ[u].len() as u32);
                    st.occ[u].push((h, SStr::new(id)));
                }
            });
        },
    );

    // B. per bucket: number the distinct IDs (exact on the bytes). `local` holds the number of every listed ID,
    //    thread lists in order, so endpoint (u, j) of thread t is local[u][off[u][t] + j]
    let interned = par(
        threads,
        UB,
        Vec::<(usize, PreMap<(SStr, u64)>, Vec<(SStr, u64)>, Vec<u32>)>::new,
        |out, u| {
            let n: usize = parts.iter().map(|p| p.occ[u].len()).sum();
            let mut map: PreMap<(SStr, u64)> = PreMap::with_capacity_and_hasher(n, Default::default());
            let mut ids: Vec<(SStr, u64)> = Vec::with_capacity(n);
            let mut local: Vec<u32> = Vec::with_capacity(n);
            for p in &parts {
                for &(h, s) in &p.occ[u] {
                    let li = map
                        .entry(PreHashed { h, s })
                        .or_insert_with(|| {
                            ids.push((s, h));
                            (s, (ids.len() - 1) as u64)
                        })
                        .1 as u32;
                    local.push(li);
                }
            }
            if ids.len() >= 1 << IX {
                die("too many linked IDs in one bucket");
            }
            out.push((u, map, ids, local));
        },
    );
    let mut by_bucket: Vec<Option<(PreMap<(SStr, u64)>, Vec<(SStr, u64)>, Vec<u32>)>> = (0..UB).map(|_| None).collect();
    for (u, map, ids, local) in interned.into_iter().flatten() {
        by_bucket[u] = Some((map, ids, local));
    }
    let (mut maps, mut ids, mut local) = (Vec::with_capacity(UB), Vec::with_capacity(UB), Vec::with_capacity(UB));
    for (m, i, l) in by_bucket.into_iter().map(|x| x.unwrap_or_default()) {
        maps.push(std::sync::Mutex::new(m));
        ids.push(i);
        local.push(l);
    }
    let nt = parts.len();
    let mut base = vec![0u32; UB + 1];
    let mut off = vec![0u32; UB * nt];
    for u in 0..UB {
        base[u + 1] = base[u] + ids[u].len() as u32;
        let mut o = 0;
        for (t, p) in parts.iter().enumerate() {
            off[u * nt + t] = o;
            o += p.occ[u].len() as u32;
        }
    }
    let n = base[UB] as usize;
    let dense = |t: usize, e: u32| {
        let u = (e >> IX) as usize;
        base[u] + local[u][(off[u * nt + t] + (e & ((1 << IX) - 1))) as usize]
    };

    // C. join: lock-free union-find; a parent is always smaller than its child, so links only go downhill
    let parent: Vec<AtomicU32> = (0..n as u32).map(AtomicU32::new).collect();
    let find = |mut x: u32| -> u32 {
        loop {
            let p = parent[x as usize].load(Relaxed);
            if p == x {
                return x;
            }
            let g = parent[p as usize].load(Relaxed);
            if g != p {
                let _ = parent[x as usize].compare_exchange_weak(p, g, Relaxed, Relaxed);
            }
            x = g;
        }
    };
    let jobs: Vec<(usize, usize)> = parts
        .iter()
        .enumerate()
        .flat_map(|(t, p)| (0..p.ends.len()).step_by(1 << 16).map(move |s| (t, s)))
        .collect();
    par(
        threads,
        jobs.len(),
        || (),
        |_, j| {
            let (t, s) = jobs[j];
            let ends = &parts[t].ends;
            for k in (s..(s + (1 << 16)).min(ends.len())).step_by(2) {
                let (mut x, mut y) = (dense(t, ends[k]), dense(t, ends[k + 1]));
                loop {
                    x = find(x);
                    y = find(y);
                    if x == y {
                        break;
                    }
                    let (lo, hi) = (x.min(y), x.max(y));
                    if parent[hi as usize].compare_exchange(hi, lo, Relaxed, Relaxed).is_ok() {
                        break;
                    }
                }
            }
        },
    );

    // D. name each person by the byte-wise smallest of their IDs
    let best: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(u32::MAX)).collect();
    let id_of = |p: u32| ids[(p >> IX) as usize][(p & ((1 << IX) - 1)) as usize];
    par(
        threads,
        UB,
        || (),
        |_, u| {
            for li in 0..ids[u].len() as u32 {
                let me = ((u as u32) << IX) | li;
                let r = find(base[u] + li) as usize;
                let mine = id_of(me).0;
                let mut cur = best[r].load(Relaxed);
                while cur == u32::MAX || mine.get() < id_of(cur).0.get() {
                    match best[r].compare_exchange_weak(cur, me, Relaxed, Relaxed) {
                        Ok(_) => break,
                        Err(v) => cur = v,
                    }
                }
            }
        },
    );

    // E. every linked ID -> its person's name
    par(
        threads,
        UB,
        || (),
        |_, u| {
            let mut map = maps[u].lock().unwrap_or_else(|_| die("link map poisoned"));
            for v in map.values_mut() {
                let r = find(base[u] + v.1 as u32) as usize;
                *v = id_of(best[r].load(Relaxed));
            }
        },
    );
    maps.into_iter().map(|m| m.into_inner().unwrap_or_else(|_| die("link map poisoned"))).collect()
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
        r.cat = ctx.products.get(pid).copied().unwrap_or(ctx.unknown_cat);
    }
    if ingest - t > DAY {
        return Some(r);
    }
    r.ts = t;
    r.keep = true;
    Some(r)
}

/// `v.push(x)`, then a prefetch-for-write of where `v` writes a few pushes later. A parse thread appends to 512
/// Vecs, too many streams for the hardware prefetcher: each new cache line would stall on its fetch otherwise.
#[inline(always)]
fn push_ahead<T>(v: &mut Vec<T>, x: T) {
    v.push(x);
    let next = (v.as_ptr() as *const u8).wrapping_add(v.len() * std::mem::size_of::<T>() + 256);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_ET0 }>(next as *const i8)
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = next;
}

impl EvParse {
    /// A valid, non-late event: into its user_id bucket, plus its (hash(event_id), location) record.
    #[inline(always)]
    fn keep(&mut self, h: u64, e: UEvent) {
        let u = uid_bucket(e.uh);
        let v = &mut self.ub[u];
        let loc = ((self.t as u64) << LOC_T) | ((u as u64) << LOC_B) | v.len() as u64;
        push_ahead(v, e);
        push_ahead(&mut self.eb[(h >> 56) as usize % EB], ERec { h, loc });
    }

    /// Any other parsed event: it still takes part in dedup.
    #[inline(always)]
    fn reject(&mut self, h: u64, eid: Packed, ingest: i64) {
        let loc = LOC_BAD | ((self.t as u64) << LOC_T) | self.bad.len() as u64;
        self.bad.push((eid, ingest));
        push_ahead(&mut self.eb[(h >> 56) as usize % EB], ERec { h, loc });
    }

    #[inline(always)]
    fn add(&mut self, r: RawEvent) {
        if r.keep {
            let u = r.uid.get();
            let e = UEvent { uh: hash_bytes(u), uid: SStr::new(u), eid: r.eid, ingest: r.ingest, ts: r.ts, price: r.price, cat: r.cat, kind: r.kind, dropped: AtomicBool::new(false) };
            self.keep(r.h, e);
        } else {
            self.reject(r.h, r.eid, r.ingest);
        }
    }
}

/// Parses the event lines that start in [start, end) of `file` (header skipped) into `st`. A line with its `\n`
/// in its first 128 bytes and 144 readable bytes takes `fast_event`; any other goes through `split` + `parse_event`.
fn parse_events(ctx: &Ctx, file: Bytes, start: usize, end: usize, st: &mut EvParse) {
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
}

/// `split::<7>` + `parse_event` + `EvParse::add` for the line of `e` (< 128) bytes at `line`, given its comma and
/// quote bits `c` and `q`; line[e] is its `\n` and line[..144] is readable.
#[inline(always)]
unsafe fn fast_event(ctx: &Ctx, line: *const u8, e: usize, c: u128, q: u128, st: &mut EvParse) {
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
    let h = if el <= 16 { hash16(load16(ep, el), el) } else { hash_bytes(std::slice::from_raw_parts(ep, el)) };
    let eid = Packed::new(std::slice::from_raw_parts(ep, el));
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
        cat = ctx.products.get(std::slice::from_raw_parts(pp, pl)).copied().unwrap_or(ctx.unknown_cat);
    }
    let (uh, uid) = if ul <= 15 {
        let v = load16(up, ul);
        (hash16(v, ul), SStr((v | (ul as u128) << 120).to_le_bytes()))
    } else {
        let u = std::slice::from_raw_parts(up, ul);
        (hash_bytes(u), SStr::new(u))
    };
    st.keep(h, UEvent { uh, uid, eid, ingest, ts, price, cat, kind, dropped: AtomicBool::new(false) });
}

#[inline]
fn parse_click(ctx: &Ctx, [cid, uid, camp, ts]: [Bytes; 4]) -> Option<PClick> {
    if cid.is_empty() || uid.is_empty() {
        return None;
    }
    let t = parse_ts(ts)?;
    if t < ctx.ws - WEEK || t >= ctx.we {
        return None;
    }
    let camp = *ctx.campaigns.get(camp)?;
    // routed by user_id for now; phase 4b swaps in the person
    Some(PClick { nh: hash_bytes(uid), name: SStr::new(uid), cid: Packed::new(cid), ts: (t - (ctx.ws - WEEK)) as u32, camp })
}

/// Parses the click lines that start in [start, end) of `file` (header skipped) into `out`, bucketed by user_id.
/// Same split as `parse_events`: a line with its `\n` in its first 128 bytes and 144 readable bytes takes
/// `fast_click`, any other goes through `split` + `parse_click`.
fn parse_clicks(ctx: &Ctx, file: Bytes, start: usize, end: usize, out: &mut [Vec<PClick>]) {
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
                    fast_click(ctx, line, e, (c0 as u128 | (c1 as u128) << 64) & lm, (q0 as u128 | (q1 as u128) << 64) & lm, out);
                    p += e + 1;
                    continue;
                }
            }
        }
        let e = memchr::memchr(b'\n', &file[p..]).map_or(len, |i| p + i);
        if let Some(c) = split::<4>(&file[p..e]).and_then(|f| parse_click(ctx, f)) {
            out[uid_bucket(c.nh)].push(c);
        }
        p = e + 1;
    }
}

/// `split::<4>` + `parse_click` for the line of `e` (< 128) bytes at `line`, given its comma and quote bits `c` and
/// `q`; line[e] is its `\n` and line[..144] is readable. The line is in the file, which lives for the whole run.
#[inline(always)]
unsafe fn fast_click(ctx: &Ctx, line: *const u8, e: usize, c: u128, q: u128, out: &mut [Vec<PClick>]) {
    if c.count_ones() != 3 {
        return;
    }
    // well quoted: see `fast_event`
    let b = c | (1u128 << e);
    let s = (b << 1) | 1;
    let ends = b >> 1;
    let x = q & s;
    let y = (q << 1) & b;
    if (q & !(s | ends)) | (x & ends) | (y.wrapping_sub(x) & b) != 0 {
        return;
    }
    let mut fs = [0usize; 4];
    let mut fe = [0usize; 4];
    let mut cc = c;
    let mut at = 0;
    for i in 0..3 {
        let k = cc.trailing_zeros() as usize;
        cc &= cc - 1;
        fs[i] = at;
        fe[i] = k;
        at = k + 1;
    }
    fs[3] = at;
    fe[3] = e;
    for i in 0..4 {
        // a quoted field starts with `"` (an empty unquoted field starts on its `,` or `\n`)
        let qd = (*line.add(fs[i]) == b'"') as usize;
        fs[i] += qd;
        fe[i] -= qd;
    }
    let f = |i: usize| (line.add(fs[i]), fe[i] - fs[i]);
    let ((cp, cl), (up, ul), (kp, kl), (tp, tl)) = (f(0), f(1), f(2), f(3));
    if cl == 0 || ul == 0 {
        return;
    }
    let Some(t) = ts_at(tp, tl) else { return };
    if t < ctx.ws - WEEK || t >= ctx.we {
        return;
    }
    let Some(&camp) = ctx.campaigns.get(std::slice::from_raw_parts(kp, kl)) else { return };
    let (nh, name) = if ul <= 15 {
        let v = load16(up, ul);
        (hash16(v, ul), SStr((v | (ul as u128) << 120).to_le_bytes()))
    } else {
        let u = std::slice::from_raw_parts(up, ul);
        (hash_bytes(u), SStr::new(u))
    };
    let cid = Packed::new(std::slice::from_raw_parts(cp, cl));
    out[uid_bucket(nh)].push(PClick { nh, name, cid, ts: (t - (ctx.ws - WEEK)) as u32, camp });
}

struct Agg {
    rows: Vec<UserRow>,
    cred: Vec<[i64; 3]>,
    assist: Vec<i64>,
    stamp: Vec<u32>,
    iv: Vec<(i64, i64)>,
    seen: Vec<u32>,
    sctr: u32,
    // per-bucket scratch, reused across buckets
    pslots: Vec<u64>,
    pnames: Vec<SStr>,
    pnh: Vec<u64>,
    pcnt: Vec<u32>,
    eslots: Vec<ELine>,
    eheads: Vec<u32>,
    purs: Vec<Pur>,
    grp: Vec<Grp>,
    gstart: Vec<u32>,
    gpos: Vec<u32>,
    ctmp: Vec<CTmp>,
    cl: Vec<Click>,
    cstart: Vec<u32>,
    evs: Vec<(u32, u32)>,
    pbuf: Vec<PBuf>,
}

impl Agg {
    fn new(ncamp: usize, width: usize) -> Agg {
        Agg {
            rows: Vec::new(),
            cred: vec![[0i64; 3]; (ncamp + 1) * width],
            assist: vec![0i64; (ncamp + 1) * width],
            stamp: vec![u32::MAX; ncamp + 1],
            iv: vec![(0i64, 0i64); ncamp + 1],
            seen: vec![0u32; ncamp + 1],
            sctr: 0,
            pslots: Vec::new(),
            pnames: Vec::new(),
            pnh: Vec::new(),
            pcnt: Vec::new(),
            eslots: Vec::new(),
            eheads: Vec::new(),
            purs: Vec::new(),
            grp: Vec::new(),
            gstart: Vec::new(),
            gpos: Vec::new(),
            ctmp: Vec::new(),
            cl: Vec::new(),
            cstart: Vec::new(),
            evs: Vec::new(),
            pbuf: Vec::new(),
        }
    }
}

/// All of one person's events in one 30-minute bin of the window (`ts - base` divided by GAP), keyed by
/// `(person id << 12 | bin) + 1` (0 = empty slot). Two events in a bin are never a session break; two bins apart
/// always are; adjacent bins break exactly when `min` of the later minus `max` of the earlier exceeds GAP.
/// Four slots per cache line, one array per field so one 16-byte compare tests all four keys. A key lives in the
/// first line, from its home line on, that holds it or has a free slot. A free slot is (key 0, min MAX, max 0,
/// cnt 0), so a new entry and an existing one take the same branch-free update.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct ELine {
    key: [u32; 4],
    min: [u32; 4], // ts - base, ms
    max: [u32; 4],
    cnt: [u32; 4],
}

const EMPTY_LINE: ELine = ELine { key: [0; 4], min: [u32::MAX; 4], max: [0; 4], cnt: [0; 4] };

#[inline(always)]
fn eline_of(key: u32, lbits: u32) -> usize {
    ((key as u64).wrapping_mul(FIB) >> (64 - lbits)) as usize
}

/// Bit i set when slot i of `l` holds `key`.
#[inline(always)]
fn eline_match(l: &ELine, key: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let v = _mm_load_si128(l.key.as_ptr() as *const __m128i);
        _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(v, _mm_set1_epi32(key as i32)))) as u32
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (l.key[0] == key) as u32 | ((l.key[1] == key) as u32) << 1 | ((l.key[2] == key) as u32) << 2 | ((l.key[3] == key) as u32) << 3
    }
}

/// Slot (line * 4 + i) of `key`: where it is, else the first free one.
#[inline(always)]
fn eslot_of(lines: &[ELine], key: u32, lbits: u32) -> usize {
    let mask = lines.len() - 1;
    let mut l = eline_of(key, lbits);
    loop {
        let line = &lines[l];
        let hit = eline_match(line, key);
        let m = if hit != 0 { hit } else { eline_match(line, 0) };
        if m != 0 {
            return l * 4 + m.trailing_zeros() as usize;
        }
        l = (l + 1) & mask;
    }
}

/// An entry once grouped by person; `head` is its first purchase in `Agg::purs`, NIL if none.
#[derive(Clone, Copy)]
struct Grp {
    min: u32,
    max: u32,
    cnt: u32,
    head: u32,
}

/// A purchase waiting for its (person, bin) entry.
struct PBuf {
    key: u32,
    price: u32,
    cat: u16,
}

struct Pur {
    next: u32,
    price: u32,
    cat: u16,
}

/// A click of a known person of this bucket, before grouping by person.
struct CTmp {
    pc: u64, // person id << 16 | campaign
    cid: Packed,
    ts: u32,
}

/// A click in its person's run, sorted by ts.
#[derive(Clone, Copy)]
struct Click {
    cid: Packed,
    ts: u32, // ts - base, ms
    camp: u16,
}

const NIL: u32 = 0; // end of a purchase chain; chain links and heads are purchase index + 1
const FIB: u64 = 0x9E37_79B9_7F4A_7C15;

struct Sess {
    start: i64,
    d: usize,
    purchased: bool,
    revenue: i64,
}

#[inline(always)]
fn same_name(a: &SStr, b: &SStr) -> bool {
    if b.0[15] != SStr::LONG {
        u128::from_ne_bytes(a.0) == u128::from_ne_bytes(b.0)
    } else {
        a == b
    }
}

/// The person table: open addressing on the hash the router computed, slot = tag << 32 | (person id + 1).
fn person_find(pslots: &[u64], pnames: &[SStr], nh: u64, name: &SStr) -> Option<u32> {
    let mask = pslots.len() - 1;
    let tag = nh as u32;
    let mut s = (nh >> (64 - pslots.len().trailing_zeros())) as usize;
    loop {
        let v = pslots[s];
        if v == 0 {
            return None;
        }
        if (v >> 32) as u32 == tag {
            let id = v as u32 - 1;
            if same_name(&pnames[id as usize], name) {
                return Some(id);
            }
        }
        s = (s + 1) & mask;
    }
}

fn person_work_check(ctx: &Ctx) {
    // ts - base must fit u32 (about 49 days); the generator's windows are 14 days
    if ctx.we - (ctx.ws - WEEK) >= 1 << 32 {
        die("window longer than 42 days");
    }
}

fn person_bucket_work(ctx: &Ctx, routed: &[Vec<Vec<PEvent>>], clicks: &[Vec<Vec<PClick>>], p: usize, agg: &mut Agg) {
    person_work_check(ctx);
    let base = ctx.ws - WEEK;
    let width = ctx.ndays;
    let organic = ctx.ncamp;
    let n_ev: usize = routed.iter().map(|t| t[p].len()).sum();

    // 1a. events -> person ids (person table); one (key, ts) per event, purchases on the side
    let mut pcap = (n_ev / 10).next_power_of_two().max(256);
    let mut pslots = std::mem::take(&mut agg.pslots);
    pslots.clear();
    pslots.resize(pcap, 0);
    let mut pnames = std::mem::take(&mut agg.pnames);
    pnames.clear();
    let mut pnh = std::mem::take(&mut agg.pnh);
    pnh.clear();
    let mut evs = std::mem::take(&mut agg.evs);
    evs.clear();
    let mut pbuf = std::mem::take(&mut agg.pbuf);
    pbuf.clear();
    for t in routed {
        for e in &t[p] {
            let tsr = (e.ts - base) as u32;
            let bin = tsr / GAP as u32;
            let pmask = pcap - 1;
            let tag = e.nh as u32;
            let mut s = (e.nh >> (64 - pcap.trailing_zeros())) as usize;
            let pid = loop {
                let v = pslots[s];
                if v == 0 {
                    let id = pnames.len() as u32;
                    if id >= (1 << 20) - 1 {
                        die("person bucket holds over a million people");
                    }
                    pnames.push(e.name);
                    pnh.push(e.nh);
                    pslots[s] = (tag as u64) << 32 | (id as u64 + 1);
                    if pnames.len() * 8 > pcap * 5 {
                        // grow: re-place every person
                        pcap *= 2;
                        pslots.clear();
                        pslots.resize(pcap, 0);
                        let pmask = pcap - 1;
                        let sh = 64 - pcap.trailing_zeros();
                        for (i, &h) in pnh.iter().enumerate() {
                            let mut s = (h >> sh) as usize;
                            while pslots[s] != 0 {
                                s = (s + 1) & pmask;
                            }
                            pslots[s] = ((h as u32) as u64) << 32 | (i as u64 + 1);
                        }
                    }
                    break id;
                }
                if (v >> 32) as u32 == tag {
                    let id = v as u32 - 1;
                    if same_name(&pnames[id as usize], &e.name) {
                        break id;
                    }
                }
                s = (s + 1) & pmask;
            };
            let key = (pid << 12 | bin) + 1;
            evs.push((key, tsr));
            if e.kind == PURCHASE {
                pbuf.push(PBuf { key, price: e.price, cat: e.cat });
            }
        }
    }
    let np = pnames.len();
    agg.pslots = pslots;
    agg.pnames = pnames;
    agg.pnh = pnh;

    // 1b. (person, bin) entry table. A slot starts as (key 0, min MAX, max 0, cnt 0), so a new entry and an existing
    //     one take the same branch-free update.
    agg.pcnt.clear();
    agg.pcnt.resize(np, 0);
    let mut ecap = (n_ev * 2 / 5).next_power_of_two().max(256); // slots
    let mut elines = std::mem::take(&mut agg.eslots);
    elines.clear();
    elines.resize(ecap / 4, EMPTY_LINE);
    let mut eheads = std::mem::take(&mut agg.eheads);
    eheads.clear();
    eheads.resize(ecap, NIL);
    let mut ne = 0usize;
    for i in 0..evs.len() {
        let lbits = ecap.trailing_zeros() - 2;
        #[cfg(target_arch = "x86_64")]
        if let Some(&(k2, _)) = evs.get(i + 12) {
            unsafe { std::arch::x86_64::_mm_prefetch(elines.as_ptr().add(eline_of(k2, lbits)) as *const i8, std::arch::x86_64::_MM_HINT_T0) };
        }
        let (key, tsr) = evs[i];
        let at = eslot_of(&elines, key, lbits);
        let (l, i) = (&mut elines[at >> 2], at & 3);
        let new = (l.cnt[i] == 0) as u32;
        l.key[i] = key;
        l.cnt[i] += 1;
        l.min[i] = l.min[i].min(tsr);
        l.max[i] = l.max[i].max(tsr);
        agg.pcnt[((key - 1) >> 12) as usize] += new;
        ne += new as usize;
        if ne * 8 > ecap * 5 {
            // grow: re-place every entry
            let old = std::mem::take(&mut elines);
            let old_heads = std::mem::take(&mut eheads);
            ecap *= 2;
            elines.resize(ecap / 4, EMPTY_LINE);
            eheads.resize(ecap, NIL);
            let lbits = ecap.trailing_zeros() - 2;
            for (li, o) in old.iter().enumerate() {
                for i in 0..4 {
                    if o.key[i] != 0 {
                        let at = eslot_of(&elines, o.key[i], lbits);
                        let l = &mut elines[at >> 2];
                        l.key[at & 3] = o.key[i];
                        l.min[at & 3] = o.min[i];
                        l.max[at & 3] = o.max[i];
                        l.cnt[at & 3] = o.cnt[i];
                        eheads[at] = old_heads[li * 4 + i];
                    }
                }
            }
        }
    }
    agg.purs.clear();
    let lbits = ecap.trailing_zeros() - 2;
    for pb in &pbuf {
        let at = eslot_of(&elines, pb.key, lbits);
        agg.purs.push(Pur { next: eheads[at], price: pb.price, cat: pb.cat });
        eheads[at] = agg.purs.len() as u32;
    }
    agg.eslots = elines;
    agg.eheads = eheads;
    agg.evs = evs;
    agg.pbuf = pbuf;

    // 2. group entries by person (counting sort on the per-person entry counts), each run sorted by bin
    agg.gstart.clear();
    agg.gpos.clear();
    let mut acc = 0u32;
    for &c in &agg.pcnt {
        agg.gstart.push(acc);
        agg.gpos.push(acc);
        acc += c;
    }
    agg.gstart.push(acc);
    agg.grp.clear();
    agg.grp.resize(ne, Grp { min: 0, max: 0, cnt: 0, head: NIL });
    for (li, l) in agg.eslots.iter().enumerate() {
        for i in 0..4 {
            if l.key[i] != 0 {
                let pid = ((l.key[i] - 1) >> 12) as usize;
                let at = agg.gpos[pid] as usize;
                agg.gpos[pid] += 1;
                agg.grp[at] = Grp { min: l.min[i], max: l.max[i], cnt: l.cnt[i], head: agg.eheads[li * 4 + i] };
            }
        }
    }
    for pid in 0..np {
        let g = &mut agg.grp[agg.gstart[pid] as usize..agg.gstart[pid + 1] as usize];
        if g.len() <= 24 {
            for i in 1..g.len() {
                let x = g[i];
                let mut j = i;
                while j > 0 && g[j - 1].min > x.min {
                    g[j] = g[j - 1];
                    j -= 1;
                }
                g[j] = x;
            }
        } else {
            g.sort_unstable_by_key(|x| x.min);
        }
    }

    // 3. clicks of known people, grouped by person (counting sort) and sorted by ts within a person
    agg.ctmp.clear();
    agg.cstart.clear();
    agg.cstart.resize(np + 2, 0);
    for t in clicks {
        for c in &t[p] {
            if let Some(id) = person_find(&agg.pslots, &agg.pnames, c.nh, &c.name) {
                agg.ctmp.push(CTmp { pc: (id as u64) << 16 | c.camp as u64, cid: c.cid, ts: c.ts });
                agg.cstart[id as usize + 2] += 1;
            }
        }
    }
    for i in 2..np + 2 {
        agg.cstart[i] += agg.cstart[i - 1];
    }
    // cstart[id + 1] is now the start of person id's run; scattering advances it to the run's end
    agg.cl.clear();
    agg.cl.resize(agg.ctmp.len(), Click { cid: Packed(0), ts: 0, camp: 0 });
    for c in &agg.ctmp {
        let id = (c.pc >> 16) as usize;
        let at = agg.cstart[id + 1] as usize;
        agg.cstart[id + 1] += 1;
        agg.cl[at] = Click { cid: c.cid, ts: c.ts, camp: c.pc as u16 };
    }
    for pid in 0..np {
        let g = &mut agg.cl[agg.cstart[pid] as usize..agg.cstart[pid + 1] as usize];
        if g.len() <= 24 {
            for i in 1..g.len() {
                let x = g[i];
                let mut j = i;
                while j > 0 && g[j - 1].ts > x.ts {
                    g[j] = g[j - 1];
                    j -= 1;
                }
                g[j] = x;
            }
        } else {
            g.sort_unstable_by_key(|x| x.ts);
        }
    }

    // 4. per person: sessions + user_daily rows, credit, assists
    let mut sessions: Vec<Sess> = Vec::new();
    let mut day_groups: Vec<(usize, usize)> = Vec::new(); // (d, first session index)
    let mut cats: Vec<(u16, i64)> = Vec::new();
    let mut touched: Vec<usize> = Vec::new();
    for pid in 0..np {
        let person = pid as u32;
        let name = agg.pnames[pid];
        let gs = &agg.grp[agg.gstart[pid] as usize..agg.gstart[pid + 1] as usize];
        let cs = &agg.cl[agg.cstart[pid] as usize..agg.cstart[pid + 1] as usize];

        // sessions + user_daily rows
        sessions.clear();
        day_groups.clear();
        let mut row: Option<UserRow> = None;
        let mut prev_max = i64::MIN / 2;
        for en in gs {
            let mn = en.min as i64 + base;
            if mn - prev_max > GAP {
                let d = ((mn.div_euclid(DAY)) - ctx.day0) as usize;
                sessions.push(Sess { start: mn, d, purchased: false, revenue: 0 });
                let day = (ctx.day0 + d as i64) as i32;
                if row.as_ref().map(|r| r.day != day).unwrap_or(true) {
                    if let Some(r) = row.take() {
                        agg.rows.push(finish_row(r, &mut cats, &ctx.cat_rank));
                    }
                    row = Some(UserRow { name, day, sessions: 0, events: 0, revenue: 0, top: None });
                    day_groups.push((d, sessions.len() - 1));
                }
                row.as_mut().unwrap().sessions += 1;
            }
            prev_max = prev_max.max(en.max as i64 + base);
            let r = row.as_mut().unwrap();
            r.events += en.cnt as i64;
            let mut h = en.head;
            while h != NIL {
                let pu = &agg.purs[h as usize - 1];
                let s = sessions.last_mut().unwrap();
                s.purchased = true;
                s.revenue += pu.price as i64;
                r.revenue += pu.price as i64;
                match cats.iter_mut().find(|c| c.0 == pu.cat) {
                    Some(c) => c.1 += pu.price as i64,
                    None => cats.push((pu.cat, pu.price as i64)),
                }
                h = pu.next;
            }
        }
        if let Some(r) = row.take() {
            agg.rows.push(finish_row(r, &mut cats, &ctx.cat_rank));
        }

        let cts = |c: &Click| c.ts as i64 + base;
        let (nc, ns) = (cs.len(), sessions.len());
        if nc == 0 {
            for s in &sessions {
                let a = &mut agg.cred[organic * width + s.d];
                a[0] += 1;
                a[1] += s.purchased as i64;
                a[2] += s.revenue;
            }
        } else if nc * ns <= 2048 {
            // few clicks or few sessions: per session, look at the clicks of its 7-day window [start - 7d, start]
            // (a contiguous run of cs) for the credit and for the distinct assisting campaigns
            let (mut lo, mut hi) = (0, 0);
            for s in &sessions {
                while hi < nc && cts(&cs[hi]) <= s.start {
                    hi += 1;
                }
                while lo < hi && s.start - cts(&cs[lo]) > WEEK {
                    lo += 1;
                }
                let mut camp = organic;
                if hi > lo {
                    // credit: latest click, tie -> smallest click_id
                    let t = cts(&cs[hi - 1]);
                    let mut best = &cs[hi - 1];
                    let mut k = hi - 1;
                    while k > lo && cts(&cs[k - 1]) == t {
                        k -= 1;
                        if cs[k].cid.get() < best.cid.get() {
                            best = &cs[k];
                        }
                    }
                    camp = best.camp as usize;
                    // assists: each distinct campaign of the window once
                    if agg.sctr == u32::MAX {
                        agg.seen.fill(0);
                        agg.sctr = 0;
                    }
                    agg.sctr += 1;
                    for c in &cs[lo..hi] {
                        let kk = c.camp as usize;
                        if agg.seen[kk] != agg.sctr {
                            agg.seen[kk] = agg.sctr;
                            agg.assist[kk * width + s.d] += 1;
                        }
                    }
                }
                let a = &mut agg.cred[camp * width + s.d];
                a[0] += 1;
                a[1] += s.purchased as i64;
                a[2] += s.revenue;
            }
        } else {
            // credit: latest click with start - 7d <= ts <= start; tie -> smallest click_id
            let mut q = 0;
            for s in &sessions {
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
                let a = &mut agg.cred[camp * width + s.d];
                a[0] += 1;
                a[1] += s.purchased as i64;
                a[2] += s.revenue;
            }

            // assists: a click at t assists sessions starting in [t, t + 7d]; merge per campaign
            let flush = |assist: &mut [i64], k: usize, lo: i64, hi: i64, sessions: &[Sess], day_groups: &[(usize, usize)]| {
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
            };
            touched.clear();
            for c in cs {
                let k = c.camp as usize;
                let t = cts(c);
                if agg.stamp[k] != person {
                    agg.stamp[k] = person;
                    agg.iv[k] = (t, t + WEEK);
                    touched.push(k);
                } else if t <= agg.iv[k].1 {
                    agg.iv[k].1 = t + WEEK;
                } else {
                    let (lo, hi) = agg.iv[k];
                    flush(&mut agg.assist, k, lo, hi, &sessions, &day_groups);
                    agg.iv[k] = (t, t + WEEK);
                }
            }
            for &k in &touched {
                let (lo, hi) = agg.iv[k];
                flush(&mut agg.assist, k, lo, hi, &sessions, &day_groups);
                agg.stamp[k] = u32::MAX;
            }
        }
    }
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
    dict: bool, // dictionary encoded when the file has rows
}

const fn col(name: &'static str, ty: i32, converted: Option<i32>) -> ColSpec {
    ColSpec { name, ty, optional: false, converted, dict: false }
}

const USER_COLS: [ColSpec; 6] = [
    ColSpec { dict: true, ..col("user_id", T_BYTES, Some(CT_UTF8)) },
    col("day", T_INT32, Some(CT_DATE)),
    col("sessions", T_INT32, None),
    col("events", T_INT32, None),
    col("revenue_cents", T_INT64, None),
    ColSpec { optional: true, ..col("top_category", T_BYTES, Some(CT_UTF8)) },
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

/// Rows per data page: a page body stays cache-sized and one buffer is reused for every page.
const PAGE_ROWS: usize = 16384;
/// Room reserved in front of a page body for its Thrift page header, which is written backwards from here.
const HDR_ROOM: usize = 48;

/// `fill` is called with this `lo` for the dictionary page of a dictionary-encoded column.
const DICT_PAGE: usize = usize::MAX;

/// A Thrift PageHeader into `page[at..HDR_ROOM]`, written backwards from `HDR_ROOM`; returns `at`.
/// `dict`: a dictionary page of `nvals` entries; else a data page of `nvals` values in encoding `enc`.
fn page_header(page: &mut [u8], dict: bool, nvals: usize, enc: i32) -> usize {
    let size = (page.len() - HDR_ROOM) as i32;
    let mut h = Thrift::new();
    h.begin();
    h.i32(1, if dict { 2 } else { 0 }); // DICTIONARY_PAGE, DATA_PAGE
    h.i32(2, size);
    h.i32(3, size);
    if dict {
        h.sub(7);
        h.i32(1, nvals as i32);
        h.i32(2, 0); // PLAIN
    } else {
        h.sub(5);
        h.i32(1, nvals as i32);
        h.i32(2, enc);
        h.i32(3, 3); // definition levels: RLE
        h.i32(4, 3); // repetition levels: RLE
    }
    h.end();
    h.end();
    assert!(h.b.len() <= HDR_ROOM);
    let at = HDR_ROOM - h.b.len();
    page[at..HDR_ROOM].copy_from_slice(&h.b);
    at
}

/// Writes `path` with one row group of `n` rows. `fill(i, lo, hi, buf)` appends the body of the data page of column i
/// holding rows lo..hi to `buf` (PLAIN, or RLE_DICTIONARY for a dictionary column: bit width, then RLE/bit-packed
/// indices); for the optional column the definition levels come first, as a length-prefixed RLE/bit-packed run.
/// A dictionary column (`dict`, when n > 0) first gets `fill(i, DICT_PAGE, 0, buf)`: the PLAIN dictionary page, whose
/// number of entries `fill` returns.
fn write_parquet(path: &Path, cols: &[ColSpec], n: usize, mut fill: impl FnMut(usize, usize, usize, &mut Vec<u8>) -> usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"PAR1")?;
    let mut off = 4i64;
    let mut page = Vec::new();
    let mut chunks = Vec::with_capacity(cols.len()); // (offset, size, dictionary page offset, first data page offset)
    for (i, c) in cols.iter().enumerate() {
        let (start, use_dict) = (off, c.dict && n > 0);
        let (mut dict_off, mut data_off) = (None, off);
        if use_dict {
            page.clear();
            page.resize(HDR_ROOM, 0);
            let entries = fill(i, DICT_PAGE, 0, &mut page);
            i32::try_from(page.len() - HDR_ROOM).map_err(|_| std::io::Error::other("page over 2 GiB"))?;
            let at = page_header(&mut page, true, entries, 0);
            f.write_all(&page[at..])?;
            dict_off = Some(off);
            off += (page.len() - at) as i64;
            data_off = off;
        }
        let mut lo = 0;
        loop {
            let hi = (lo + PAGE_ROWS).min(n);
            page.clear();
            page.resize(HDR_ROOM, 0);
            fill(i, lo, hi, &mut page);
            i32::try_from(page.len() - HDR_ROOM).map_err(|_| std::io::Error::other("page over 2 GiB"))?;
            let at = page_header(&mut page, false, hi - lo, if use_dict { 8 } else { 0 }); // RLE_DICTIONARY, PLAIN
            f.write_all(&page[at..])?;
            off += (page.len() - at) as i64;
            lo = hi;
            if lo >= n {
                break;
            }
        }
        chunks.push((start, off - start, dict_off, data_off));
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
    for (c, &(o, size, dict_off, data_off)) in cols.iter().zip(&chunks) {
        t.begin();
        t.i64(2, o); // file_offset
        t.sub(3); // ColumnMetaData
        t.i32(1, c.ty);
        if dict_off.is_some() {
            t.list(2, Thrift::I32, 3);
            t.elem_i32(0); // PLAIN: the dictionary page
            t.elem_i32(3); // RLE
            t.elem_i32(8); // RLE_DICTIONARY
        } else {
            t.list(2, Thrift::I32, 2);
            t.elem_i32(0); // PLAIN
            t.elem_i32(3); // RLE
        }
        t.list(3, Thrift::BINARY, 1);
        t.elem_str(c.name);
        t.i32(4, 0); // UNCOMPRESSED
        t.i64(5, n as i64);
        t.i64(6, size);
        t.i64(7, size);
        t.i64(9, data_off); // data_page_offset
        if let Some(d) = dict_off {
            t.i64(11, d); // dictionary_page_offset
        }
        t.end();
        t.end();
    }
    t.i64(2, off - 4); // total_byte_size
    t.i64(3, n as i64);
    t.end();
    t.str(6, "sessions");
    t.end();
    let len = t.b.len() as u32;
    t.b.extend_from_slice(&len.to_le_bytes());
    t.b.extend_from_slice(b"PAR1");
    f.write_all(&t.b)
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

/// The pieces of `segs` (consecutive row slices) that cover rows lo..hi of their concatenation.
fn page_slices<'a>(segs: &[&'a [UserRow]], lo: usize, hi: usize, out: &mut Vec<&'a [UserRow]>) {
    out.clear();
    let mut base = 0;
    for &s in segs {
        let (a, b) = (base, base + s.len());
        if b > lo && a < hi {
            out.push(&s[lo.max(a) - a..hi.min(b) - a]);
        }
        base = b;
    }
}

/// Appends `s` as a PLAIN byte array: 4-byte length, then the bytes. Inline IDs copy 16 bytes in one go.
#[inline]
fn put_name(b: &mut Vec<u8>, s: &SStr) {
    let n = s.0[15];
    if n != SStr::LONG {
        if b.capacity() - b.len() < 20 {
            b.reserve(1 << 16);
        }
        // room for 4 + 16 bytes is there; only the first 4 + n of them stay in the vector
        unsafe {
            let p = b.as_mut_ptr().add(b.len());
            (p as *mut u32).write_unaligned(n as u32);
            std::ptr::copy_nonoverlapping(s.0.as_ptr(), p.add(4), 16);
            b.set_len(b.len() + 4 + n as usize);
        }
    } else {
        let v = s.get();
        b.extend_from_slice(&(v.len() as u32).to_le_bytes());
        b.extend_from_slice(v);
    }
}

/// Appends one `W`-byte little-endian value per row.
fn put_fixed<const W: usize>(b: &mut Vec<u8>, part: &[&[UserRow]], f: impl Fn(&UserRow) -> [u8; W]) {
    let rows: usize = part.iter().map(|s| s.len()).sum();
    let at = b.len();
    b.resize(at + rows * W, 0);
    let mut dst = b[at..].chunks_exact_mut(W);
    for r in part.iter().flat_map(|s| s.iter()) {
        dst.next().unwrap().copy_from_slice(&f(r));
    }
}

/// One run of the RLE/bit-packed hybrid: `run` copies of the `vb`-byte value `idx`.
fn put_run(b: &mut Vec<u8>, run: usize, idx: u32, vb: usize) {
    let mut v = (run as u64) << 1;
    while v >= 0x80 {
        b.push(v as u8 | 0x80);
        v >>= 7;
    }
    b.push(v as u8);
    b.extend_from_slice(&idx.to_le_bytes()[..vb]);
}

fn write_users(path: &Path, segs: &[&[UserRow]], cat_names: &[Bytes]) -> std::io::Result<()> {
    let n: usize = segs.iter().map(|s| s.len()).sum();
    let mut part: Vec<&[UserRow]> = Vec::new();
    let mut run: Vec<u8> = Vec::new();
    // user_id is dictionary encoded: one entry per run of equal IDs (a person's days are consecutive rows), and
    // the data pages are runs of that entry's index
    let (mut entries, mut idx, mut last): (usize, u32, Option<SStr>) = (0, u32::MAX, None);
    // sessions and events stay far below 2^31: main() refuses inputs with that many events
    write_parquet(path, &USER_COLS, n, |c, lo, hi, b| {
        if c == 0 && lo == DICT_PAGE {
            let mut prev: Option<SStr> = None;
            for r in segs.iter().flat_map(|s| s.iter()) {
                if prev != Some(r.name) {
                    put_name(b, &r.name);
                    entries += 1;
                    prev = Some(r.name);
                }
            }
            (idx, last) = (u32::MAX, None);
            return entries;
        }
        page_slices(segs, lo, hi, &mut part);
        match c {
            0 => {
                let w = (usize::BITS - (entries.max(2) - 1).leading_zeros()) as usize;
                b.push(w as u8);
                let mut count = 0;
                for r in part.iter().flat_map(|s| s.iter()) {
                    if last == Some(r.name) {
                        count += 1;
                    } else {
                        if count > 0 {
                            put_run(b, count, idx, w.div_ceil(8));
                        }
                        idx = idx.wrapping_add(1);
                        last = Some(r.name);
                        count = 1;
                    }
                }
                if count > 0 {
                    put_run(b, count, idx, w.div_ceil(8));
                }
            }
            1 => put_fixed(b, &part, |r| r.day.to_le_bytes()),
            2 => put_fixed(b, &part, |r| (r.sessions as i32).to_le_bytes()),
            3 => put_fixed(b, &part, |r| (r.events as i32).to_le_bytes()),
            4 => put_fixed(b, &part, |r| r.revenue.to_le_bytes()),
            _ => {
                // definition levels: one bit-packed run of ceil(rows / 8) groups of 8 bits (1 = present)
                let groups = (hi - lo).div_ceil(8);
                run.clear();
                let mut v = ((groups as u64) << 1) | 1;
                while v >= 0x80 {
                    run.push(v as u8 | 0x80);
                    v >>= 7;
                }
                run.push(v as u8);
                let (mut cur, mut k) = (0u8, 0);
                for r in part.iter().flat_map(|s| s.iter()) {
                    cur |= (r.top.is_some() as u8) << k;
                    k += 1;
                    if k == 8 {
                        run.push(cur);
                        (cur, k) = (0, 0);
                    }
                }
                if k > 0 {
                    run.push(cur);
                }
                b.extend_from_slice(&((run.len()) as u32).to_le_bytes());
                b.extend_from_slice(&run);
                for r in part.iter().flat_map(|s| s.iter()) {
                    if let Some(t) = r.top {
                        let v = cat_names[t as usize];
                        b.extend_from_slice(&(v.len() as u32).to_le_bytes());
                        b.extend_from_slice(v);
                    }
                }
            }
        }
        0
    })
}

fn write_camps(path: &Path, rows: &[CampRow]) -> std::io::Result<()> {
    write_parquet(path, &CAMP_COLS, rows.len(), |c, lo, hi, b| {
        let rows = &rows[lo..hi];
        match c {
            0 => plain_bytes(b, rows.iter().map(|r| r.0)),
            1 => plain_bytes(b, rows.iter().map(|r| r.1)),
            2 => rows.iter().for_each(|r| b.extend_from_slice(&r.2.to_le_bytes())),
            3 => plain_i64(b, rows.iter().map(|r| r.3)),
            4 => plain_i64(b, rows.iter().map(|r| r.4)),
            5 => plain_i64(b, rows.iter().map(|r| r.5)),
            _ => plain_i64(b, rows.iter().map(|r| r.6)),
        }
        0
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
        // this thread is the last of the `threads` workers: no more runnable threads than the CPU quota allows
        while n + 1 < threads.clamp(1, 32) {
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
                // never return: thread exit would free into the zeroed heap; sleep until the process ends
                loop {
                    unsafe { libc::pause() };
                }
            });
            if spawned.is_err() {
                break;
            }
            n += 1;
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
        loop {
            let i = next.fetch_add(1, Relaxed);
            if i >= cnt {
                break;
            }
            unsafe { libc::madvise(pieces[i][0].load(Relaxed) as *mut libc::c_void, pieces[i][1].load(Relaxed), libc::MADV_DONTNEED) };
            done.fetch_add(1, Release);
        }
        while done.load(Acquire) < cnt {
            std::hint::spin_loop();
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
        let mut products = BytesMap::default();
        products.insert(&b"p1"[..], 0u16);
        products.insert(&b"p2"[..], 1u16);
        Ctx {
            ws: parse_ts(b"2028-02-20T00:00:00Z").unwrap(),
            we: parse_ts(b"2028-03-05T00:00:00Z").unwrap(),
            day0: 0,
            ndays: 14,
            ncamp: 0,
            products,
            campaigns: BytesMap::default(),
            unknown_cat: 2,
            cat_rank: vec![0, 1, 2],
        }
    }

    fn fresh() -> EvParse {
        EvParse { t: 3, ub: (0..UB).map(|_| Vec::new()).collect(), bad: Vec::new(), eb: (0..EB).map(|_| Vec::new()).collect() }
    }

    type Flat = (Vec<(u64, Vec<u8>, Vec<u8>, i64, i64, u32, u16, u8)>, Vec<(Vec<u8>, i64)>, Vec<(u64, u64)>);
    fn flat(st: &EvParse) -> Flat {
        (
            st.ub.iter().flatten().map(|e| (e.uh, e.uid.get().to_vec(), e.eid.get().to_vec(), e.ingest, e.ts, e.price, e.cat, e.kind)).collect(),
            st.bad.iter().map(|(e, i)| (e.get().to_vec(), *i)).collect(),
            st.eb.iter().flatten().map(|r| (r.h, r.loc)).collect(),
        )
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
            let (mut a, mut b) = (fresh(), fresh());
            for w in cuts.windows(2) {
                parse_events(&c, file, w[0], w[1], &mut a);
                for_records::<7>(file, w[0], w[1], |f| {
                    if let Some(e) = parse_event(&c, f) {
                        b.add(e);
                    }
                });
            }
            let (fa, fb) = (flat(&a), flat(&b));
            kept += fb.0.len();
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
    fn civil_at_matches_civil_across_years() {
        let mut r = Rng(99);
        for _ in 0..300_000 {
            let s = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", 1890 + r.below(280), r.below(14), r.below(33), r.below(25), r.below(61), r.below(61));
            let mut buf = s.clone().into_bytes();
            buf.extend_from_slice(&[b'7'; 32]);
            assert_eq!(unsafe { civil_at(buf.as_ptr()) }, civil(s.as_bytes()), "{s}");
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
