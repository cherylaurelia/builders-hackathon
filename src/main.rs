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
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
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

/// Key whose hash was computed earlier; equality is still exact on the bytes.
#[derive(Clone, Copy)]
struct PreHashed {
    h: u64,
    s: Bytes,
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

#[inline]
fn uid_bucket(h: u64) -> usize {
    ((h >> 44) as usize) & (UB - 1)
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
    let mut out = Vec::new();
    for &f in files {
        let mut s = 0;
        while s < f.len() {
            let e = (s + CHUNK).min(f.len());
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
    let each = total / (threads * n).max(1) * 5 / 4 + 16;
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

/// A surviving event, routed to its person's bucket.
#[derive(Clone, Copy)]
struct PEvent {
    nh: u64,
    name: Bytes,
    ts: i64,
    price: u32,
    cat: u16,
    kind: u8,
}

/// A valid click, routed to its person's bucket.
#[derive(Clone, Copy)]
struct PClick {
    nh: u64,
    name: Bytes,
    ts: i64,
    cid: Bytes,
    camp: u16,
}

struct UserRow {
    name: Bytes,
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

    // 1. links, on their own thread: nothing needs them until phase 4b
    let link_files = csv_files(&data.join("links"));
    let links_job = std::thread::spawn(move || build_links(&link_files, (threads / 4).max(1)));

    let ctx = Ctx { ws, we, day0, ndays, ncamp, products, campaigns, unknown_cat, cat_rank };

    // 2. events: parse into event_id buckets
    let ev_files = csv_files(&data.join("events"));
    let ev_chunks = chunks(&ev_files);
    let ev_est = ev_files.iter().map(|f| f.len()).sum::<usize>() / 60; // lines are ~75 bytes
    let parsed = par(
        threads,
        ev_chunks.len(),
        || sized::<RawEvent>(EB, ev_est, threads),
        |buckets, i| {
            let (f, s, e) = ev_chunks[i];
            for_records::<7>(f, s, e, |fields| {
                if let Some(r) = parse_event(&ctx, fields) {
                    buckets[(r.h >> 56) as usize % EB].push(r);
                }
            });
        },
    );
    lap("events parsed");

    // 3. dedup per event bucket, route survivors to person buckets
    let n_parsed: usize = parsed.iter().flatten().map(|v| v.len()).sum();
    let routed = par(
        threads,
        EB,
        || (sized::<PEvent>(UB, n_parsed, threads), Vec::<u64>::new(), Vec::<u64>::new(), Vec::<RawEvent>::new()),
        |(out, table, dupkeys, v), b| {
            // Pass 1: find hashes seen more than once (2-4% of events). Key = hash | 1 so 0 marks an empty slot;
            // merging two hashes that differ only in bit 0 just sends both to the exact path below.
            let n: usize = parsed.iter().map(|t| t[b].len()).sum();
            let cap = (n * 2).next_power_of_two().max(16);
            let mask = cap - 1;
            table.clear();
            table.resize(cap, 0);
            dupkeys.clear();
            for t in &parsed {
                for r in &t[b] {
                    let k = r.h | 1;
                    let mut s = (k as usize >> 1) & mask;
                    loop {
                        let x = table[s];
                        if x == 0 {
                            table[s] = k;
                            break;
                        }
                        if x == k {
                            dupkeys.push(k);
                            break;
                        }
                        s = (s + 1) & mask;
                    }
                }
            }
            // small set of duplicated keys, reusing the front of the table
            let dcap = (dupkeys.len() * 4).next_power_of_two().max(16);
            let dmask = dcap - 1;
            table[..dcap].fill(0);
            for &k in dupkeys.iter() {
                let mut s = (k as usize >> 1) & dmask;
                while table[s] != 0 && table[s] != k {
                    s = (s + 1) & dmask;
                }
                table[s] = k;
            }
            let is_dup = |k: u64| {
                let mut s = (k as usize >> 1) & dmask;
                loop {
                    let x = table[s];
                    if x == k {
                        return true;
                    }
                    if x == 0 {
                        return false;
                    }
                    s = (s + 1) & dmask;
                }
            };
            let route = |out: &mut Vec<Vec<PEvent>>, r: &RawEvent| {
                // routed by user_id for now; phase 4b swaps in the person
                let uid = r.uid.get();
                let uh = hash_bytes(uid);
                out[uid_bucket(uh)].push(PEvent { nh: uh, name: uid, ts: r.ts, price: r.price, cat: r.cat, kind: r.kind });
            };
            // Pass 2: a hash seen once is a single copy: keep it if valid. Others go to the exact path.
            v.clear();
            for t in &parsed {
                for r in &t[b] {
                    if dupkeys.is_empty() || !is_dup(r.h | 1) {
                        if r.keep {
                            route(out, r);
                        }
                    } else {
                        v.push(*r);
                    }
                }
            }
            v.sort_unstable_by_key(|r| (r.h, r.ingest));
            let mut seen: Vec<Bytes> = Vec::new(); // distinct event_ids among equal hashes
            let mut i = 0;
            while i < v.len() {
                let mut j = i + 1;
                while j < v.len() && v[j].h == v[i].h {
                    j += 1;
                }
                // ordered by ingest, the first copy of each distinct event_id is the kept copy
                seen.clear();
                for r in &v[i..j] {
                    if seen.contains(&r.eid.get()) {
                        continue;
                    }
                    seen.push(r.eid.get());
                    if r.keep {
                        route(out, r);
                    }
                }
                i = j;
            }
        },
    );
    let routed: Vec<Vec<Vec<PEvent>>> = routed.into_iter().map(|(o, ..)| o).collect();
    drop(parsed);
    lap("events deduped");

    // 4. clicks
    let ck_files = csv_files(&data.join("clicks"));
    let ck_chunks = chunks(&ck_files);
    let ck_est = ck_files.iter().map(|f| f.len()).sum::<usize>() / 40; // lines are ~50 bytes
    let clicks = par(
        threads,
        ck_chunks.len(),
        || sized::<PClick>(UB, ck_est, threads),
        |out, i| {
            let (f, s, e) = ck_chunks[i];
            for_records::<4>(f, s, e, |fields| {
                if let Some(c) = parse_click(&ctx, fields) {
                    out[uid_bucket(c.nh)].push(c);
                }
            });
        },
    );
    lap("clicks parsed");

    let links = links_job.join().unwrap_or_else(|_| die("link builder panicked"));
    lap(&format!("links joined: {} linked ids", links.iter().map(|m| m.len()).sum::<usize>()));

    // 4b. per user_id bucket: user_id -> person through that bucket's slice of the link table
    let n_ev: usize = routed.iter().flatten().map(|v| v.len()).sum();
    let n_ck: usize = clicks.iter().flatten().map(|v| v.len()).sum();
    let resolved = par(
        threads,
        UB,
        || (sized::<PEvent>(PB, n_ev, threads), sized::<PClick>(PB, n_ck, threads)),
        |(oe, oc), u| {
            let map = &links[u];
            let person = |h: u64, s: Bytes| map.get(&PreHashed { h, s }).copied().unwrap_or((s, h));
            for t in &routed {
                for e in &t[u] {
                    let (name, nh) = person(e.nh, e.name);
                    oe[person_bucket(nh)].push(PEvent { nh, name, ..*e });
                }
            }
            for t in &clicks {
                for c in &t[u] {
                    let (name, nh) = person(c.nh, c.name);
                    oc[person_bucket(nh)].push(PClick { nh, name, ..*c });
                }
            }
        },
    );
    drop(routed);
    drop(clicks);
    let (routed, clicks): (Vec<_>, Vec<_>) = resolved.into_iter().unzip();
    lap("user ids resolved");

    // 5. per person bucket
    let width = ctx.ndays;
    let results = par(
        threads,
        PB,
        || Agg {
            rows: Vec::new(),
            cred: vec![[0i64; 3]; (ncamp + 1) * width],
            assist: vec![0i64; (ncamp + 1) * width],
            stamp: vec![u32::MAX; ncamp + 1],
            iv: vec![(0i64, 0i64); ncamp + 1],
        },
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
    write_outputs(&out, &user_parts, &camp_rows, &cat_names, threads);
    lap(&format!("written: {} user rows, {} campaign rows", user_parts.iter().map(|p| p.len()).sum::<usize>(), camp_rows.len()));
}

fn build_links(files: &[Bytes], threads: usize) -> Vec<PreMap<(Bytes, u64)>> {
    let ch = chunks(files);
    let parts = par(
        threads,
        ch.len(),
        Vec::<(Bytes, Bytes)>::new,
        |v, i| {
            let (f, s, e) = ch[i];
            for_records::<2>(f, s, e, |[a, b]| {
                if !a.is_empty() && !b.is_empty() {
                    v.push((a, b));
                }
            });
        },
    );
    let mut ids: BytesMap<u32> = BytesMap::default();
    let mut names: Vec<Bytes> = Vec::new();
    let mut parent: Vec<u32> = Vec::new();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let g = parent[parent[x as usize] as usize];
            parent[x as usize] = g;
            x = g;
        }
        x
    }
    let mut intern = |s: Bytes, names: &mut Vec<Bytes>, parent: &mut Vec<u32>| -> u32 {
        *ids.entry(s).or_insert_with(|| {
            names.push(s);
            parent.push(parent.len() as u32);
            (names.len() - 1) as u32
        })
    };
    for v in &parts {
        for &(a, b) in v {
            let x = intern(a, &mut names, &mut parent);
            let y = intern(b, &mut names, &mut parent);
            let (rx, ry) = (find(&mut parent, x), find(&mut parent, y));
            if rx != ry {
                parent[rx.max(ry) as usize] = rx.min(ry);
            }
        }
    }
    let n = names.len();
    let mut best: Vec<u32> = (0..n as u32).collect();
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        if names[i] < names[best[r] as usize] {
            best[r] = i as u32;
        }
    }
    let mut maps: Vec<PreMap<(Bytes, u64)>> = (0..UB).map(|_| PreMap::default()).collect();
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        let name = names[best[r] as usize];
        let h = hash_bytes(names[i]);
        maps[uid_bucket(h)].insert(PreHashed { h, s: names[i] }, (name, hash_bytes(name)));
    }
    maps
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
    Some(PClick { nh: hash_bytes(uid), name: uid, ts: t, cid, camp })
}

struct Agg {
    rows: Vec<UserRow>,
    cred: Vec<[i64; 3]>,
    assist: Vec<i64>,
    stamp: Vec<u32>,
    iv: Vec<(i64, i64)>,
}

struct Sess {
    start: i64,
    d: usize,
    purchased: bool,
    revenue: i64,
}

fn person_bucket_work(ctx: &Ctx, routed: &[Vec<Vec<PEvent>>], clicks: &[Vec<Vec<PClick>>], p: usize, agg: &mut Agg) {
    let base = ctx.ws - WEEK; // every kept ts is in [base, base + 2^31)
    let width = ctx.ndays;
    let organic = ctx.ncamp;

    // person ids, exact on the name bytes
    let mut pid: PreMap<u32> = PreMap::default();
    let mut names: Vec<Bytes> = Vec::new();
    struct E {
        key: u64,
        price: u32,
        cat: u16,
        kind: u8,
    }
    let n_ev: usize = routed.iter().map(|t| t[p].len()).sum();
    let mut ev: Vec<E> = Vec::with_capacity(n_ev);
    for t in routed {
        for e in &t[p] {
            let next = names.len() as u32;
            let id = *pid.entry(PreHashed { h: e.nh, s: e.name }).or_insert_with(|| {
                names.push(e.name);
                next
            });
            ev.push(E { key: ((id as u64) << 32) | (e.ts - base) as u64, price: e.price, cat: e.cat, kind: e.kind });
        }
    }
    ev.sort_unstable_by_key(|e| e.key);

    struct C {
        key: u64,
        cid: Bytes,
        camp: u16,
    }
    let mut ck: Vec<C> = Vec::new();
    for t in clicks {
        for c in &t[p] {
            if let Some(&id) = pid.get(&PreHashed { h: c.nh, s: c.name }) {
                ck.push(C { key: ((id as u64) << 32) | (c.ts - base) as u64, cid: c.cid, camp: c.camp });
            }
        }
    }
    ck.sort_unstable_by_key(|c| c.key);

    let mut sessions: Vec<Sess> = Vec::new();
    let mut day_groups: Vec<(usize, usize)> = Vec::new(); // (d, first session index)
    let mut cats: Vec<(u16, i64)> = Vec::new();
    let mut touched: Vec<usize> = Vec::new();
    let mut ci = 0;
    let mut i = 0;
    while i < ev.len() {
        let person = (ev[i].key >> 32) as u32;
        let mut j = i;
        while j < ev.len() && (ev[j].key >> 32) as u32 == person {
            j += 1;
        }
        let name = names[person as usize];

        // sessions + user_daily rows
        sessions.clear();
        day_groups.clear();
        let mut row: Option<UserRow> = None;
        let mut prev = i64::MIN;
        for e in &ev[i..j] {
            let ts = (e.key & 0xffff_ffff) as i64 + base;
            if prev == i64::MIN || ts - prev > GAP {
                let d = ((ts.div_euclid(DAY)) - ctx.day0) as usize;
                sessions.push(Sess { start: ts, d, purchased: false, revenue: 0 });
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
            prev = ts;
            let r = row.as_mut().unwrap();
            r.events += 1;
            if e.kind == PURCHASE {
                let s = sessions.last_mut().unwrap();
                s.purchased = true;
                s.revenue += e.price as i64;
                r.revenue += e.price as i64;
                match cats.iter_mut().find(|c| c.0 == e.cat) {
                    Some(c) => c.1 += e.price as i64,
                    None => cats.push((e.cat, e.price as i64)),
                }
            }
        }
        if let Some(r) = row.take() {
            agg.rows.push(finish_row(r, &mut cats, &ctx.cat_rank));
        }

        // this person's clicks
        while ci < ck.len() && ((ck[ci].key >> 32) as u32) < person {
            ci += 1;
        }
        let mut cj = ci;
        while cj < ck.len() && (ck[cj].key >> 32) as u32 == person {
            cj += 1;
        }
        let cs = &ck[ci..cj];
        let cts = |c: &C| (c.key & 0xffff_ffff) as i64 + base;

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
                        if cs[k].cid < best.cid {
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
        let flush = |agg: &mut Agg, k: usize, lo: i64, hi: i64, sessions: &[Sess], day_groups: &[(usize, usize)]| {
            let a = sessions.partition_point(|s| s.start < lo);
            let b = sessions.partition_point(|s| s.start <= hi);
            if a >= b {
                return;
            }
            for (g, &(d, first)) in day_groups.iter().enumerate() {
                let end = day_groups.get(g + 1).map(|x| x.1).unwrap_or(sessions.len());
                let n = end.min(b) as i64 - first.max(a) as i64;
                if n > 0 {
                    agg.assist[k * width + d] += n;
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
                flush(agg, k, lo, hi, &sessions, &day_groups);
                agg.iv[k] = (t, t + WEEK);
            }
        }
        for &k in &touched {
            let (lo, hi) = agg.iv[k];
            flush(agg, k, lo, hi, &sessions, &day_groups);
            agg.stamp[k] = u32::MAX;
        }
        ci = cj;
        i = j;
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

use parquet::basic::Compression;
use parquet::data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

fn props() -> Arc<WriterProperties> {
    Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
            .build(),
    )
}

fn ba(b: &[u8]) -> ByteArray {
    ByteArray::from(b.to_vec())
}

type CampRow = (Bytes, Bytes, i32, i64, i64, i64, i64);

fn write_outputs(out: &Path, users: &[&[UserRow]], camps: &[CampRow], cat_names: &[Bytes], threads: usize) {
    let ud = out.join("user_daily");
    let cd = out.join("campaign_daily");
    for d in [&ud, &cd] {
        std::fs::create_dir_all(d).unwrap_or_else(|e| die(&format!("mkdir {}: {e}", d.display())));
    }
    // split user rows into `threads` files of about equal size
    let total: usize = users.iter().map(|p| p.len()).sum();
    let per = total.div_ceil(threads).max(1);
    let mut groups: Vec<Vec<&UserRow>> = vec![Vec::new()];
    for p in users {
        for r in p.iter() {
            if groups.last().unwrap().len() >= per {
                groups.push(Vec::new());
            }
            groups.last_mut().unwrap().push(r);
        }
    }
    std::thread::scope(|sc| {
        for (gi, g) in groups.iter().enumerate() {
            let ud = &ud;
            sc.spawn(move || {
                write_users(&ud.join(format!("part-{gi}.parquet")), g, cat_names)
                    .unwrap_or_else(|e| die(&format!("write user_daily: {e}")))
            });
        }
        write_camps(&cd.join("part-0.parquet"), camps).unwrap_or_else(|e| die(&format!("write campaign_daily: {e}")));
    });
}

fn write_users(path: &Path, rows: &[&UserRow], cat_names: &[Bytes]) -> parquet::errors::Result<()> {
    let schema = Arc::new(parse_message_type(
        "message user_daily {
            required binary user_id (STRING);
            required int32 day (DATE);
            required int64 sessions;
            required int64 events;
            required int64 revenue_cents;
            optional binary top_category (STRING);
        }",
    )?);
    let file = std::fs::File::create(path)?;
    let mut w = SerializedFileWriter::new(std::io::BufWriter::with_capacity(1 << 20, file), schema, props())?;
    let mut rg = w.next_row_group()?;
    let mut col = 0;
    while let Some(mut c) = rg.next_column()? {
        match col {
            0 => {
                let v: Vec<ByteArray> = rows.iter().map(|r| ba(r.name)).collect();
                c.typed::<ByteArrayType>().write_batch(&v, None, None)?;
            }
            1 => {
                let v: Vec<i32> = rows.iter().map(|r| r.day).collect();
                c.typed::<Int32Type>().write_batch(&v, None, None)?;
            }
            2..=4 => {
                let v: Vec<i64> = rows.iter().map(|r| [r.sessions, r.events, r.revenue][col - 2]).collect();
                c.typed::<Int64Type>().write_batch(&v, None, None)?;
            }
            _ => {
                let v: Vec<ByteArray> = rows.iter().filter_map(|r| r.top.map(|t| ba(cat_names[t as usize]))).collect();
                let def: Vec<i16> = rows.iter().map(|r| r.top.is_some() as i16).collect();
                c.typed::<ByteArrayType>().write_batch(&v, Some(&def), None)?;
            }
        }
        c.close()?;
        col += 1;
    }
    rg.close()?;
    w.close()?;
    Ok(())
}

fn write_camps(path: &Path, rows: &[CampRow]) -> parquet::errors::Result<()> {
    let schema = Arc::new(parse_message_type(
        "message campaign_daily {
            required binary campaign_id (STRING);
            required binary channel (STRING);
            required int32 day (DATE);
            required int64 sessions;
            required int64 purchasing_sessions;
            required int64 revenue_cents;
            required int64 assisted_sessions;
        }",
    )?);
    let file = std::fs::File::create(path)?;
    let mut w = SerializedFileWriter::new(std::io::BufWriter::new(file), schema, props())?;
    let mut rg = w.next_row_group()?;
    let mut col = 0;
    while let Some(mut c) = rg.next_column()? {
        match col {
            0 | 1 => {
                let v: Vec<ByteArray> = rows.iter().map(|r| ba(if col == 0 { r.0 } else { r.1 })).collect();
                c.typed::<ByteArrayType>().write_batch(&v, None, None)?;
            }
            2 => {
                let v: Vec<i32> = rows.iter().map(|r| r.2).collect();
                c.typed::<Int32Type>().write_batch(&v, None, None)?;
            }
            _ => {
                let v: Vec<i64> = rows.iter().map(|r| [r.3, r.4, r.5, r.6][col - 3]).collect();
                c.typed::<Int64Type>().write_batch(&v, None, None)?;
            }
        }
        c.close()?;
        col += 1;
    }
    rg.close()?;
    w.close()?;
    Ok(())
}
