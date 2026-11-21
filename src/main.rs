//! Shopping Sessions pipeline. rules.md is the spec; each phase names the rule it implements.
//!
//! One shuffle. Every user_id is resolved to its person while its line is parsed, so a kept event is written
//! once, 12 bytes, straight into the bucket of its person, and never moved again.
//!
//!   1. links -> one global ID table (open addressing, exact keys, filled concurrently) + a lock-free union-find
//!      that always links the root with the larger ID under the one with the smaller ID, so every root is the
//!      byte-wise smallest ID of its person. Each linked ID's slot then holds its person id = its root's slot.
//!   2. events + clicks, one parallel parse. An ID not in the table is a person on its own: it gets a slot (and so a
//!      person id) the first time any thread sees it. A kept event goes to person bucket `pid % 256` (12 B); every
//!      parsed event also leaves (event_id, ingest_ts, where its record went) in its event_id bucket (24 B).
//!   3. per event_id bucket: copies of one event_id -> the kept copy is the one with the smallest ingest_ts; the
//!      records of all other copies are listed as dropped for their person bucket.
//!   4. per person bucket: counting sort by person, sort each person's events by ts, sessionize, credit, assist.
//!   5. write Parquet.

use memmap2::Mmap;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::time::Instant;

type Bytes = &'static [u8];

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;
const CHUNK: usize = 2 << 20;
const LINK_CHUNK: usize = 256 << 10;
const NB: usize = 256; // person buckets: the low 8 bits of the person id
const EB: usize = 256; // event_id buckets: the top 8 bits of hash(event_id)
const MAXT: usize = 16; // record locations hold the thread in 4 bits
const NONE: u32 = u32::MAX;

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

/// A string of up to 15 bytes stored inline (zero padded, length in the top byte), else (pointer, length | LONG).
/// The inline form is canonical, so two inline values are equal exactly when their 16 bytes are.
#[derive(Clone, Copy)]
#[repr(C)]
struct SStr(u64, u64);
const LONG: u64 = 0xFF;
impl SStr {
    #[inline]
    fn is_long(self) -> bool {
        self.1 >> 56 == LONG
    }
    #[inline]
    fn new(b: &'static [u8]) -> SStr {
        if b.len() <= 15 {
            let mut x = [0u8; 16];
            x[..b.len()].copy_from_slice(b);
            x[15] = b.len() as u8;
            SStr(u64::from_le_bytes(x[..8].try_into().unwrap()), u64::from_le_bytes(x[8..].try_into().unwrap()))
        } else {
            SStr(b.as_ptr() as u64, b.len() as u64 | LONG << 56)
        }
    }
    /// Same as `new`, but reads 16 bytes at the start of `b`: the caller guarantees they are mapped.
    #[inline(always)]
    unsafe fn fast(b: &'static [u8]) -> SStr {
        let n = b.len();
        if n > 15 {
            return SStr(b.as_ptr() as u64, n as u64 | LONG << 56);
        }
        let p = b.as_ptr();
        let lo = (p as *const u64).read_unaligned();
        let hi = (p.add(8) as *const u64).read_unaligned();
        if n >= 8 {
            SStr(lo, (hi & ((1u64 << ((n - 8) * 8)) - 1)) | (n as u64) << 56)
        } else {
            SStr(lo & ((1u64 << (n * 8)) - 1), (n as u64) << 56)
        }
    }
    #[inline]
    fn get(&self) -> &[u8] {
        if !self.is_long() {
            unsafe { std::slice::from_raw_parts(self as *const SStr as *const u8, (self.1 >> 56) as usize) }
        } else {
            unsafe { std::slice::from_raw_parts(self.0 as *const u8, (self.1 & ((1 << 56) - 1)) as usize) }
        }
    }
    #[inline]
    fn hash(self) -> u64 {
        if self.is_long() {
            hash_bytes(self.get())
        } else {
            mix(self.0 ^ self.1.rotate_left(29).wrapping_mul(0x9E3779B97F4A7C15))
        }
    }
    /// Byte-wise order. Inline: big-endian (bytes, zero padding, length) orders exactly like the bytes.
    #[inline]
    fn less(self, o: SStr) -> bool {
        if !self.is_long() && !o.is_long() {
            (self.0.swap_bytes(), self.1.swap_bytes()) < (o.0.swap_bytes(), o.1.swap_bytes())
        } else {
            self.get() < o.get()
        }
    }
}
impl PartialEq for SStr {
    #[inline]
    fn eq(&self, o: &Self) -> bool {
        if !self.is_long() {
            self.0 == o.0 && self.1 == o.1
        } else {
            o.is_long() && self.get() == o.get()
        }
    }
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

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0)
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

// ---------------------------------------------------------------- memory

fn map_anon(bytes: usize) -> *mut u8 {
    const HP: usize = 2 << 20;
    let len = bytes + HP;
    let p = unsafe {
        libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE, -1, 0)
    };
    if p == libc::MAP_FAILED {
        die(&format!("mmap {bytes} bytes failed"));
    }
    let a = (p as usize + HP - 1) & !(HP - 1);
    #[cfg(target_os = "linux")]
    if std::env::var_os("C2_NOHP").is_none() {
        unsafe { libc::madvise(a as *mut libc::c_void, bytes, libc::MADV_HUGEPAGE) };
    }
    a as *mut u8
}

/// Bump allocator over 64 MB huge-page regions; nothing is freed before exit.
struct Arena {
    cur: usize,
    end: usize,
}
impl Arena {
    fn alloc(&mut self, bytes: usize) -> *mut u8 {
        if self.cur + bytes > self.end {
            let sz = (64usize << 20).max(bytes);
            self.cur = map_anon(sz) as usize;
            self.end = self.cur + sz;
        }
        let p = self.cur;
        self.cur += bytes;
        p as *mut u8
    }
}

const BS: usize = 2048; // elements per block

/// An append-only list in blocks of BS elements: no copying on growth.
struct Blocks<T> {
    blks: Vec<*mut T>,
    len: usize,
}
unsafe impl<T> Send for Blocks<T> {}
unsafe impl<T> Sync for Blocks<T> {}
impl<T: Copy> Blocks<T> {
    fn new() -> Self {
        Blocks { blks: Vec::new(), len: 0 }
    }
    #[inline]
    fn push(&mut self, a: &mut Arena, v: T) -> usize {
        let i = self.len;
        let o = i % BS;
        if o == 0 {
            self.blks.push(a.alloc(BS * std::mem::size_of::<T>()) as *mut T);
        }
        unsafe { (*self.blks.last().unwrap_unchecked()).add(o).write(v) };
        self.len = i + 1;
        i
    }
    #[inline]
    fn get(&self, i: usize) -> &T {
        unsafe { &*self.blks[i / BS].add(i % BS) }
    }
    fn slices(&self) -> impl Iterator<Item = &[T]> + '_ {
        self.blks.iter().enumerate().map(move |(k, &p)| unsafe { std::slice::from_raw_parts(p, BS.min(self.len - k * BS)) })
    }
}

// ---------------------------------------------------------------- the ID table

/// One slot: state 0 empty, 1 being written, 2 ready. `pid` is the person id: the slot of the person's smallest ID.
#[repr(C, align(32))]
struct Slot {
    st: AtomicU32,
    pid: AtomicU32,
    lo: AtomicU64,
    hi: AtomicU64,
}

struct Table {
    s: &'static [Slot],
    mask: usize,
    shift: u32,
}
impl Table {
    fn new(cap: usize) -> Table {
        let p = map_anon(cap * std::mem::size_of::<Slot>()) as *const Slot;
        Table { s: unsafe { std::slice::from_raw_parts(p, cap) }, mask: cap - 1, shift: 64 - cap.trailing_zeros() }
    }
    #[inline]
    fn home(&self, h: u64) -> usize {
        (h >> self.shift) as usize
    }
    #[inline]
    fn key(&self, i: usize) -> SStr {
        SStr(self.s[i].lo.load(Relaxed), self.s[i].hi.load(Relaxed))
    }
    /// The person id of ID `k` (hash `h`); an ID seen for the first time becomes its own person.
    #[inline]
    fn get_or_insert(&self, k: SStr, h: u64) -> u32 {
        let mut i = self.home(h);
        let mut probes = 0;
        loop {
            let sl = &self.s[i];
            let mut st = sl.st.load(Acquire);
            if st == 0 {
                match sl.st.compare_exchange(0, 1, Acquire, Acquire) {
                    Ok(_) => {
                        sl.lo.store(k.0, Relaxed);
                        sl.hi.store(k.1, Relaxed);
                        sl.pid.store(i as u32, Relaxed);
                        sl.st.store(2, Release);
                        return i as u32;
                    }
                    Err(x) => st = x,
                }
            }
            while st == 1 {
                std::hint::spin_loop();
                st = sl.st.load(Acquire);
            }
            if SStr(sl.lo.load(Relaxed), sl.hi.load(Relaxed)) == k {
                return sl.pid.load(Relaxed);
            }
            i = (i + 1) & self.mask;
            probes += 1;
            if probes > self.mask {
                die("ID table full");
            }
        }
    }
}

/// Lock-free union-find over table slots: parent 0 = root, else parent slot + 1.
fn find(parent: &[AtomicU32], mut x: u32) -> u32 {
    loop {
        let p = parent[x as usize].load(Relaxed);
        if p == 0 {
            return x;
        }
        let g = parent[(p - 1) as usize].load(Relaxed);
        if g == 0 {
            return p - 1;
        }
        parent[x as usize].store(g, Relaxed); // path halving: g is an ancestor of x
        x = g - 1;
    }
}

/// Joins the persons of slots a and b. The root with the larger ID goes under the root with the smaller ID, so
/// links only ever point to smaller IDs: no cycles, and every root is the smallest ID of its tree.
fn union(tab: &Table, parent: &[AtomicU32], a: u32, b: u32) {
    loop {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra == rb {
            return;
        }
        let (lo, hi) = if tab.key(ra as usize).less(tab.key(rb as usize)) { (ra, rb) } else { (rb, ra) };
        if parent[hi as usize].compare_exchange(0, lo + 1, AcqRel, Relaxed).is_ok() {
            return;
        }
    }
}

/// Small read-only map from a short string to a u16 (campaigns).
struct STab {
    keys: Vec<SStr>,
    vals: Vec<u16>,
    mask: usize,
}
impl STab {
    fn new(items: &[(Bytes, u16)]) -> STab {
        let cap = (items.len() * 4).next_power_of_two().max(16);
        let mut t = STab { keys: vec![SStr(0, 0); cap], vals: vec![0; cap], mask: cap - 1 };
        for &(k, v) in items {
            let s = SStr::new(k);
            let mut i = s.hash() as usize & t.mask;
            loop {
                if t.keys[i].1 == 0 {
                    t.keys[i] = s;
                    t.vals[i] = v;
                    break;
                }
                if t.keys[i] == s {
                    break; // first one wins
                }
                i = (i + 1) & t.mask;
            }
        }
        t
    }
    #[inline]
    fn get(&self, s: SStr) -> Option<u16> {
        let mut i = s.hash() as usize & self.mask;
        loop {
            let k = self.keys[i];
            if k.1 == 0 {
                return None; // an empty slot (a non-empty key never has hi == 0: its length byte is set)
            }
            if k == s {
                return Some(self.vals[i]);
            }
            i = (i + 1) & self.mask;
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

/// Bit masks of `,`, `"` and `\n` over the 128 bytes at p.
#[inline(always)]
unsafe fn masks128(p: *const u8) -> (u128, u128, u128) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        let (cm, qt, nl) = (_mm256_set1_epi8(b',' as i8), _mm256_set1_epi8(b'"' as i8), _mm256_set1_epi8(b'\n' as i8));
        let mut c = 0u128;
        let mut q = 0u128;
        let mut n = 0u128;
        for k in 0..4 {
            let v = _mm256_loadu_si256(p.add(32 * k) as *const __m256i);
            c |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, cm)) as u32 as u128) << (32 * k);
            q |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, qt)) as u32 as u128) << (32 * k);
            n |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, nl)) as u32 as u128) << (32 * k);
        }
        return (c, q, n);
    }
    #[allow(unreachable_code)]
    {
        let (mut c, mut q, mut n) = (0u128, 0u128, 0u128);
        for k in 0..128 {
            let x = *p.add(k);
            c |= ((x == b',') as u128) << k;
            q |= ((x == b'"') as u128) << k;
            n |= ((x == b'\n') as u128) << k;
        }
        (c, q, n)
    }
}

enum Tok<const N: usize> {
    Ok(usize, [u32; N], [u32; N]),
    Bad(usize),
    Long,
}

/// The line at p (its newline within 128 bytes): N fields as unquoted [start, end) offsets, or malformed.
#[inline(always)]
unsafe fn tokenize<const N: usize>(p: *const u8) -> Tok<N> {
    let (c, q, nl) = masks128(p);
    if nl == 0 {
        return Tok::Long;
    }
    let len = nl.trailing_zeros() as usize;
    let lm = (1u128 << len) - 1;
    let c = c & lm;
    let q = q & lm;
    if c.count_ones() as usize != N - 1 {
        return Tok::Bad(len);
    }
    let mut s = [0u32; N];
    let mut e = [0u32; N];
    let mut cm = c;
    let mut start = 0u32;
    for i in 0..N - 1 {
        let pos = cm.trailing_zeros();
        cm &= cm - 1;
        s[i] = start;
        e[i] = pos;
        start = pos + 1;
    }
    s[N - 1] = start;
    e[N - 1] = len as u32;
    if q != 0 {
        for i in 0..N {
            let (a, b) = (s[i], e[i]);
            let fl = b - a;
            let qf = (q >> a) & ((1u128 << fl) - 1);
            if qf != 0 {
                if fl < 2 || qf != (1 | (1u128 << (fl - 1))) {
                    return Tok::Bad(len);
                }
                s[i] = a + 1;
                e[i] = b - 1;
            }
        }
    }
    Tok::Ok(len, s, e)
}

/// The well-formed lines that start in [start, end) of `file`, header skipped, as N unquoted fields.
/// `fast`: 16 bytes from the start of every field are mapped (the line ends at least 32 bytes before the file does).
struct Lines<const N: usize> {
    file: Bytes,
    p: usize,
    end: usize,
}
impl<const N: usize> Lines<N> {
    fn new(file: Bytes, start: usize, end: usize) -> Self {
        let first = if start == 0 { 0 } else { start - 1 };
        let p = match memchr::memchr(b'\n', &file[first.min(file.len())..]) {
            Some(nl) => first + nl + 1,
            None => usize::MAX,
        };
        Lines { file, p, end: end.min(file.len()) }
    }
    #[inline(always)]
    fn next(&mut self) -> Option<([Bytes; N], bool)> {
        let (file, len) = (self.file, self.file.len());
        loop {
            let p = self.p;
            if p >= self.end {
                return None;
            }
            if p + 160 <= len {
                match unsafe { tokenize::<N>(file.as_ptr().add(p)) } {
                    Tok::Ok(ll, s, e) => {
                        self.p = p + ll + 1;
                        return Some((std::array::from_fn(|i| unsafe { file.get_unchecked(p + s[i] as usize..p + e[i] as usize) }), true));
                    }
                    Tok::Bad(ll) => {
                        self.p = p + ll + 1;
                        continue;
                    }
                    Tok::Long => {}
                }
            }
            let e = memchr::memchr(b'\n', &file[p..]).map(|i| p + i).unwrap_or(len);
            self.p = e + 1;
            if let Some(fields) = split::<N>(&file[p..e]) {
                return Some((fields, false));
            }
        }
    }
}

/// Calls `f` for every line that starts in [start, end) of `file`, skipping the header line.
fn for_lines(file: Bytes, mut f: impl FnMut(Bytes)) {
    let len = file.len();
    let mut p = match memchr::memchr(b'\n', file) {
        Some(i) => i + 1,
        None => return,
    };
    while p < len {
        let e = memchr::memchr(b'\n', &file[p..]).map(|i| p + i).unwrap_or(len);
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

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `YYYY-MM-DDTHH:MM:SS` (the first 19 bytes) as epoch ms, calendar-checked.
fn civil(b: &[u8]) -> Option<i64> {
    let b: &[u8; 19] = b[..19].try_into().ok()?;
    let d = |i: usize| b[i].wrapping_sub(b'0') as u32;
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

/// F1 `...Z`, F2 `...+HH:MM`, F3 digits = epoch ms. None = bad timestamp. (General, slow path.)
fn parse_ts(b: &[u8]) -> Option<i64> {
    match b.len() {
        20 if b[19] == b'Z' => civil(b),
        25 if (b[19] == b'+' || b[19] == b'-') && b[22] == b':' => {
            let (oh, om) = offset(b)?;
            let off = (oh * 60 + om) * 60_000;
            let t = civil(b)?;
            Some(if b[19] == b'+' { t - off } else { t + off })
        }
        0 => None,
        _ => f3(b),
    }
}

#[inline]
fn offset(b: &[u8]) -> Option<(i64, i64)> {
    let (h1, h2, m1, m2) = (b[20].wrapping_sub(b'0'), b[21].wrapping_sub(b'0'), b[23].wrapping_sub(b'0'), b[24].wrapping_sub(b'0'));
    if h1 > 9 || h2 > 9 || m1 > 9 || m2 > 9 {
        return None;
    }
    let (oh, om) = ((h1 * 10 + h2) as i64, (m1 * 10 + m2) as i64);
    if oh > 23 || om > 59 {
        return None;
    }
    Some((oh, om))
}

/// F3: digits only. More than 18 significant digits is far outside any window: None either way.
#[inline]
fn f3(b: &[u8]) -> Option<i64> {
    let mut v: i64 = 0;
    let mut i = 0;
    while i < b.len() && b[i] == b'0' {
        i += 1;
    }
    if b.len() - i > 18 {
        return None;
    }
    for &c in &b[i..] {
        let x = c.wrapping_sub(b'0');
        if x > 9 {
            return None;
        }
        v = v * 10 + x as i64;
    }
    Some(v)
}

/// The calendar dates near the window, by `YYYY-MM-` prefix: a date found here is valid; one not found is either
/// invalid or so far from the window that any timestamp on it is out of every window (the caller falls back).
struct Dates {
    pre: Vec<(u64, i64, u32, u32)>, // ("YYYY-MM-" as le u64, epoch day of day 0 of that month, first dd, last dd)
}
impl Dates {
    fn new(d0: i64, d1: i64) -> Dates {
        let mut pre: Vec<(u64, i64, u32, u32)> = Vec::new();
        for day in d0..=d1 {
            let (y, m, d) = civil_from_days(day);
            if !(0..=9999).contains(&y) {
                continue;
            }
            let k = u64::from_le_bytes(format!("{y:04}-{m:02}-").as_bytes().try_into().unwrap());
            match pre.iter_mut().find(|p| p.0 == k) {
                Some(p) => p.3 = d as u32,
                None => pre.push((k, day - d, d as u32, d as u32)),
            }
        }
        Dates { pre }
    }
    /// `YYYY-MM-DDTHH:MM:SS` (first 19 bytes of b) as epoch seconds; None if invalid or not a listed date.
    #[inline(always)]
    fn secs(&self, b: &[u8]) -> Option<i64> {
        let b: &[u8; 19] = b[..19].try_into().ok()?;
        let k = u64::from_le_bytes(b[..8].try_into().unwrap());
        let p = self.pre.iter().find(|p| p.0 == k)?;
        let (d1, d2) = (b[8].wrapping_sub(b'0') as u32, b[9].wrapping_sub(b'0') as u32);
        if d1 > 9 || d2 > 9 || b[10] != b'T' {
            return None;
        }
        let dd = d1 * 10 + d2;
        if dd < p.2 || dd > p.3 {
            return None;
        }
        let x = u64::from_le_bytes(b[11..19].try_into().unwrap());
        let d = x.wrapping_sub(0x3030_3030_3030_3030);
        // bytes: H H : M M : S S -> colons at 2 and 5 become 0x0A; digits must be <= 9
        const DIG: u64 = 0x8080_0080_8000_8080;
        if (d >> 16) & 0xFF != 0x0A || (d >> 40) & 0xFF != 0x0A || (d | d.wrapping_add(0x7676_7676_7676_7676)) & DIG != 0 {
            return None;
        }
        let h = (d & 0xFF) * 10 + ((d >> 8) & 0xFF);
        let m = ((d >> 24) & 0xFF) * 10 + ((d >> 32) & 0xFF);
        let s = ((d >> 48) & 0xFF) * 10 + (d >> 56);
        if h > 23 || m > 59 || s > 59 {
            return None;
        }
        Some((p.1 + dd as i64) * 86_400 + (h * 3600 + m * 60 + s) as i64)
    }
}

/// An event or click `ts` as epoch ms. None = bad, or a date so far from the window that it is out of it anyway.
#[inline(always)]
fn parse_ts_near(b: &[u8], dt: &Dates) -> Option<i64> {
    match b.len() {
        20 if b[19] == b'Z' => Some(dt.secs(b)? * 1000),
        25 if (b[19] == b'+' || b[19] == b'-') && b[22] == b':' => {
            let (oh, om) = offset(b)?;
            let off = (oh * 60 + om) * 60;
            let t = dt.secs(b)?;
            Some(1000 * if b[19] == b'+' { t - off } else { t + off })
        }
        0 => None,
        _ => f3(b),
    }
}

/// Digits only, 1..=10,000,000.
#[inline]
fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() || b.len() > 12 {
        return None;
    }
    let mut v = 0u64;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (c - b'0') as u64;
    }
    if v == 0 || v > 10_000_000 {
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

/// (file, start, end) pieces of at most `size` bytes.
fn chunks(files: &[Bytes], size: usize) -> Vec<(Bytes, usize, usize)> {
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

/// Like `par`, over existing per-thread states.
fn par_mut<S: Send>(states: &mut [S], n: usize, work: impl Fn(&mut S, usize) + Sync) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for s in states.iter_mut() {
            let (next, work) = (&next, &work);
            sc.spawn(move || loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= n {
                    break;
                }
                work(s, i);
            });
        }
    });
}

/// Parses a small clean file (products.csv / campaigns.csv) into its 2-field rows.
fn small_table(path: &Path) -> Vec<(Bytes, Bytes)> {
    let f = map_file(path);
    let mut rows = Vec::new();
    for_lines(f, |l| {
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

/// A kept event in its person bucket. `lc` = person id >> 8 (24 bits) | category >> 8 (8 bits);
/// `pay` = 0 for view/cart, else price (24 bits) | category & 0xFF << 24. ts is ms since ctx.base.
#[derive(Clone, Copy)]
#[repr(C)]
struct PRec {
    lc: u32,
    ts: u32,
    pay: u32,
}

/// Every parsed event: its event_id, ingest_ts (s since ctx.ing_base) and where its record went (NONE = not kept).
#[derive(Clone, Copy)]
#[repr(C)]
struct ERec {
    eid: SStr,
    ingest: u32,
    loc: u32, // thread << 28 | person bucket << 20 | index
}

/// A valid click in its person bucket.
#[derive(Clone, Copy)]
#[repr(C)]
struct CRec {
    lp: u32,
    ts: u32,
    cid: Packed,
    camp: u16,
}

struct TS {
    t: usize,
    arena: Arena,
    pb: Vec<Blocks<PRec>>,
    eb: Vec<Blocks<ERec>>,
    cb: Vec<Blocks<CRec>>,
    edges: Vec<(u32, u32)>,
}

struct Ctx {
    ws: i64,
    we: i64,
    base: i64,     // ws - 7 days: every kept ts is in [base, base + 2^31)
    ing_base: i64, // seconds
    day0: i64,
    ndays: usize,
    ncamp: usize, // organic is index ncamp
    products: BytesMap<u16>,
    camps: STab,
    unknown_cat: u16,
    cat_rank: Vec<u32>,
    dates: Dates,
}

struct UserRow {
    name: SStr,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: Option<u16>,
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
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8))
        .min(MAXT);
    let lap = |what: &str| eprintln!("[{:7.3}s] {what}", t0.elapsed().as_secs_f64());

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
    if cat_names.len() > 1 << 16 {
        die("more than 65536 categories");
    }
    let mut order: Vec<usize> = (0..cat_names.len()).collect();
    order.sort_by(|&a, &b| cat_names[a].cmp(cat_names[b]));
    let mut cat_rank = vec![0u32; cat_names.len()];
    for (r, &i) in order.iter().enumerate() {
        cat_rank[i] = r as u32;
    }

    // campaigns
    let mut camp_ids: Vec<Bytes> = Vec::new();
    let mut camp_channel: Vec<Bytes> = Vec::new();
    let mut seen: BytesMap<u16> = BytesMap::default();
    for (c, ch) in small_table(&data.join("campaigns.csv")) {
        if seen.contains_key(c) {
            continue;
        }
        seen.insert(c, camp_ids.len() as u16);
        camp_ids.push(c);
        camp_channel.push(ch);
    }
    let ncamp = camp_ids.len();
    if ncamp >= u16::MAX as usize {
        die("too many campaigns");
    }
    let camps = STab::new(&camp_ids.iter().enumerate().map(|(i, &c)| (c, i as u16)).collect::<Vec<_>>());

    let base = ws - WEEK;
    if we - base >= 1 << 31 {
        die("window too long");
    }
    // dates any in-window ts (F2 local dates: +-1 day) or a typical ingest_ts can fall on
    let dates = Dates::new(base.div_euclid(DAY) - 2, (we - 1).div_euclid(DAY) + 7);
    let ctx = Ctx { ws, we, base, ing_base: ws.div_euclid(1000) - (1 << 31), day0, ndays, ncamp, products, camps, unknown_cat, cat_rank, dates };

    let link_files = csv_files(&data.join("links"));
    let ev_files = csv_files(&data.join("events"));
    let ck_files = csv_files(&data.join("clicks"));
    let link_bytes: usize = link_files.iter().map(|f| f.len()).sum();
    let ev_bytes: usize = ev_files.iter().map(|f| f.len()).sum();
    let ck_bytes: usize = ck_files.iter().map(|f| f.len()).sum();
    // distinct linked IDs (~1 per 15 bytes of links) + IDs on their own (~1 person per 30 events of ~75 bytes,
    // about half of them with one ID); the table stays under ~2/3 full even at 1.6x this estimate
    let est = link_bytes / 15 + ev_bytes / 75 / 60 + ck_bytes / 50 / 200;
    let cap = (est * 3 / 2).next_power_of_two().max(1 << 16);
    let tab = Table::new(cap);
    let parent: &[AtomicU32] = unsafe { std::slice::from_raw_parts(map_anon(cap * 4) as *const AtomicU32, cap) };
    // touch every page by writing, in parallel: a first read would map the zero page, and the write after it
    // would cost a copy-on-write fault plus a TLB shootdown of every thread
    {
        let regions = [(tab.s.as_ptr() as usize, cap * std::mem::size_of::<Slot>()), (parent.as_ptr() as usize, cap * 4)];
        let pieces: Vec<(usize, usize)> = regions.iter().flat_map(|&(p, n)| (0..n).step_by(1 << 21).map(move |o| (p + o, (n - o).min(1 << 21)))).collect();
        par(threads, pieces.len(), || (), |_, i| {
            let (p, n) = pieces[i];
            for o in (0..n).step_by(4096) {
                unsafe { std::ptr::write_volatile((p + o) as *mut u8, 0) };
            }
        });
    }
    let mut st: Vec<TS> = (0..threads)
        .map(|t| TS {
            t,
            arena: Arena { cur: 0, end: 0 },
            pb: (0..NB).map(|_| Blocks::new()).collect(),
            eb: (0..EB).map(|_| Blocks::new()).collect(),
            cb: (0..NB).map(|_| Blocks::new()).collect(),
            edges: Vec::new(),
        })
        .collect();
    lap(&format!("setup: id table {cap} slots"));

    // 1. links
    let lk = chunks(&link_files, LINK_CHUNK);
    par_mut(&mut st, lk.len(), |s, i| link_chunk(&tab, s, lk[i]));
    {
        let edges: Vec<&[(u32, u32)]> = st.iter().flat_map(|s| s.edges.chunks(16 << 10)).collect();
        par(threads, edges.len(), || (), |_, i| {
            for &(a, b) in edges[i] {
                union(&tab, parent, a, b);
            }
        });
        let per = cap / (threads * 4);
        par(threads, cap.div_ceil(per), || (), |_, i| {
            for x in i * per..((i + 1) * per).min(cap) {
                if parent[x].load(Relaxed) != 0 {
                    tab.s[x].pid.store(find(parent, x as u32), Relaxed);
                }
            }
        });
    }
    lap(&format!("links: {} edges", st.iter().map(|s| s.edges.len()).sum::<usize>()));

    // 2. events + clicks
    let mut work: Vec<(Bytes, usize, usize, bool)> = chunks(&ev_files, CHUNK).into_iter().map(|(f, s, e)| (f, s, e, true)).collect();
    work.extend(chunks(&ck_files, CHUNK).into_iter().map(|(f, s, e)| (f, s, e, false)));
    work.sort_by_key(|w| std::cmp::Reverse(w.2 - w.1));
    par_mut(&mut st, work.len(), |s, i| {
        let (f, a, b, ev) = work[i];
        if ev {
            event_chunk(&ctx, &tab, s, f, a, b)
        } else {
            click_chunk(&ctx, &tab, s, f, a, b)
        }
    });
    let n_ev: usize = st.iter().flat_map(|s| s.pb.iter()).map(|b| b.len).sum();
    let n_er: usize = st.iter().flat_map(|s| s.eb.iter()).map(|b| b.len).sum();
    let n_ck: usize = st.iter().flat_map(|s| s.cb.iter()).map(|b| b.len).sum();
    lap(&format!("parsed: {n_er} events, {n_ev} kept, {n_ck} clicks"));
    if std::env::var_os("C2_DIAG").is_some() {
        let used = tab.s.iter().filter(|s| s.st.load(Relaxed) != 0).count();
        eprintln!("id table: {used} of {cap} slots used");
    }

    // 3. duplicates
    let drops = par(
        threads,
        EB,
        || (Vec::<u64>::new(), Vec::<(u32, u32)>::new(), (0..NB).map(|_| Vec::<u32>::new()).collect::<Vec<_>>()),
        |(tbl, dups, drops), e| dedup_bucket(&st, e, tbl, dups, drops),
    );
    let drops: Vec<Vec<Vec<u32>>> = drops.into_iter().map(|d| d.2).collect();
    lap(&format!("duplicates: {} records dropped", drops.iter().flatten().map(|v| v.len()).sum::<usize>()));

    // 4. per person bucket
    let nlp = (cap >> 8) + 1;
    let width = ndays;
    let mut order: Vec<usize> = (0..NB).collect();
    let sizes: Vec<usize> = (0..NB).map(|b| st.iter().map(|s| s.pb[b].len).sum()).collect();
    order.sort_by_key(|&b| std::cmp::Reverse(sizes[b]));
    let results = par(
        threads,
        NB,
        || Work::new(nlp, ncamp, width),
        |w, i| person_bucket(&ctx, &tab, &st, &drops, order[i], w),
    );
    lap("sessions aggregated");

    // 5. outputs
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

// ---------------------------------------------------------------- parse

const BATCH: usize = 32;

#[inline(always)]
fn sstr(b: Bytes, fast: bool) -> SStr {
    if fast {
        unsafe { SStr::fast(b) }
    } else {
        SStr::new(b)
    }
}

fn link_chunk(tab: &Table, s: &mut TS, (f, a, b): (Bytes, usize, usize)) {
    let mut batch = [(SStr(0, 0), 0u64, SStr(0, 0), 0u64); BATCH];
    let mut n = 0;
    let mut lines = Lines::<2>::new(f, a, b);
    loop {
        let next = lines.next();
        let done = next.is_none();
        if let Some(([x, y], fast)) = next {
            // Linked IDs, rule 1: drop lines where either field is empty
            if !x.is_empty() && !y.is_empty() {
                let (x, y) = (sstr(x, fast), sstr(y, fast));
                let (hx, hy) = (x.hash(), y.hash());
                prefetch(&tab.s[tab.home(hx)]);
                prefetch(&tab.s[tab.home(hy)]);
                batch[n] = (x, hx, y, hy);
                n += 1;
            }
        }
        if n == BATCH || done {
            for &(x, hx, y, hy) in &batch[..n] {
                let (i, j) = (tab.get_or_insert(x, hx), tab.get_or_insert(y, hy));
                if i != j {
                    s.edges.push((i, j));
                }
            }
            n = 0;
        }
        if done {
            return;
        }
    }
}

#[derive(Clone, Copy)]
struct PendEv {
    eid: SStr,
    uid: SStr,
    uh: u64,
    eh: u64,
    ingest: u32,
    ts: u32,
    pay: u32,
    cathi: u32,
}

/// `ingest_ts` as epoch ms; None = not a timestamp (the line is then treated as malformed).
#[inline(always)]
fn ingest_ms(ctx: &Ctx, b: &[u8]) -> Option<i64> {
    if b.len() == 20 && b[19] == b'Z' {
        if let Some(s) = ctx.dates.secs(b) {
            return Some(s * 1000);
        }
    }
    parse_ts(b)
}

fn event_chunk(ctx: &Ctx, tab: &Table, s: &mut TS, f: Bytes, a: usize, b: usize) {
    let t = s.t as u32;
    let z = SStr(0, 0);
    let mut batch = [PendEv { eid: z, uid: z, uh: 0, eh: 0, ingest: 0, ts: 0, pay: 0, cathi: 0 }; BATCH];
    let mut n = 0;
    let mut lines = Lines::<7>::new(f, a, b);
    loop {
        let next = lines.next();
        let done = next.is_none();
        if let Some(([eid, uid, ts, ing, ety, pid, price], fast)) = next {
            'line: {
                // Events 1: drop lines whose event_id is empty
                if eid.is_empty() {
                    break 'line;
                }
                // ingest_ts is always valid F1; a line where it is not is treated as malformed
                let Some(ingest_ms) = ingest_ms(ctx, ing) else { break 'line };
                let ingest = (ingest_ms.div_euclid(1000) - ctx.ing_base).clamp(0, u32::MAX as i64) as u32;
                let es = sstr(eid, fast);
                let eh = es.hash();
                // Events 3 + 4: the kept copy must be valid and not late
                'drop: {
                    if uid.is_empty() || pid.is_empty() {
                        break 'drop;
                    }
                    let Some(ts) = parse_ts_near(ts, &ctx.dates) else { break 'drop };
                    if ts < ctx.ws || ts >= ctx.we || ingest_ms - ts > DAY {
                        break 'drop;
                    }
                    let (pay, cathi) = if ety == b"view" || ety == b"cart" {
                        (0, 0)
                    } else if ety == b"purchase" {
                        let Some(price) = parse_price(price) else { break 'drop };
                        let cat = ctx.products.get(pid).copied().unwrap_or(ctx.unknown_cat) as u32;
                        (price | (cat & 0xFF) << 24, cat >> 8)
                    } else {
                        break 'drop;
                    };
                    let us = sstr(uid, fast);
                    let uh = us.hash();
                    prefetch(&tab.s[tab.home(uh)]);
                    batch[n] = PendEv { eid: es, uid: us, uh, eh, ingest, ts: (ts - ctx.base) as u32, pay, cathi };
                    n += 1;
                    break 'line;
                }
                s.eb[(eh >> 56) as usize].push(&mut s.arena, ERec { eid: es, ingest, loc: NONE });
            }
        }
        if n == BATCH || done {
            for p in &batch[..n] {
                let pid = tab.get_or_insert(p.uid, p.uh);
                let bk = pid as usize & (NB - 1);
                let idx = s.pb[bk].push(&mut s.arena, PRec { lc: (pid >> 8) | p.cathi << 24, ts: p.ts, pay: p.pay });
                if idx >= 1 << 20 {
                    die("person bucket overflow");
                }
                let loc = t << 28 | (bk as u32) << 20 | idx as u32;
                s.eb[(p.eh >> 56) as usize].push(&mut s.arena, ERec { eid: p.eid, ingest: p.ingest, loc });
            }
            n = 0;
        }
        if done {
            return;
        }
    }
}

#[derive(Clone, Copy)]
struct PendCk {
    uid: SStr,
    uh: u64,
    ts: u32,
    camp: u16,
    cid: Packed,
}

fn click_chunk(ctx: &Ctx, tab: &Table, s: &mut TS, f: Bytes, a: usize, b: usize) {
    let mut batch = [PendCk { uid: SStr(0, 0), uh: 0, ts: 0, camp: 0, cid: Packed(0) }; BATCH];
    let mut n = 0;
    let mut lines = Lines::<4>::new(f, a, b);
    loop {
        let next = lines.next();
        let done = next.is_none();
        if let Some(([cid, uid, camp, ts], fast)) = next {
            'line: {
                // Clicks 1 + 3. Copies of a click are identical, so keeping all of them changes nothing (rule 2).
                if cid.is_empty() || uid.is_empty() || camp.is_empty() {
                    break 'line;
                }
                let Some(t) = parse_ts_near(ts, &ctx.dates) else { break 'line };
                if t < ctx.ws - WEEK || t >= ctx.we {
                    break 'line;
                }
                let Some(ci) = ctx.camps.get(sstr(camp, fast)) else { break 'line };
                let us = sstr(uid, fast);
                let uh = us.hash();
                prefetch(&tab.s[tab.home(uh)]);
                batch[n] = PendCk { uid: us, uh, ts: (t - ctx.base) as u32, camp: ci, cid: Packed::new(cid) };
                n += 1;
            }
        }
        if n == BATCH || done {
            for p in &batch[..n] {
                let pid = tab.get_or_insert(p.uid, p.uh);
                s.cb[pid as usize & (NB - 1)].push(&mut s.arena, CRec { lp: pid >> 8, ts: p.ts, cid: p.cid, camp: p.camp });
            }
            n = 0;
        }
        if done {
            return;
        }
    }
}

// ---------------------------------------------------------------- dedup

/// Events 2: among the copies of one event_id the one with the smallest ingest_ts is kept (copies with equal
/// ingest_ts are identical); the records of every other copy that was kept by the parse are dropped.
fn dedup_bucket(st: &[TS], e: usize, tbl: &mut Vec<u64>, dups: &mut Vec<(u32, u32)>, drops: &mut [Vec<u32>]) {
    let n: usize = st.iter().map(|s| s.eb[e].len).sum();
    let cap = (n * 2).next_power_of_two().max(64);
    let mask = cap - 1;
    tbl.clear();
    tbl.resize(cap, 0);
    dups.clear();
    let rec = |r: u32| -> &ERec { st[(r >> 24) as usize].eb[e].get((r & 0xFF_FFFF) as usize) };
    for s in st {
        let mut i = 0u32;
        for sl in s.eb[e].slices() {
            for r in sl {
                let rf = (s.t as u32) << 24 | i;
                i += 1;
                let h = r.eid.hash();
                let tag = (h as u32 | 1) as u64;
                let mut k = (h >> 32) as usize & mask;
                loop {
                    let v = tbl[k];
                    if v == 0 {
                        tbl[k] = tag << 32 | rf as u64;
                        break;
                    }
                    if v >> 32 == tag && rec(v as u32).eid == r.eid {
                        dups.push((v as u32, rf));
                        break;
                    }
                    k = (k + 1) & mask;
                }
            }
        }
    }
    if dups.is_empty() {
        return;
    }
    dups.sort_unstable();
    let mut i = 0;
    while i < dups.len() {
        let first = dups[i].0;
        let mut j = i;
        while j < dups.len() && dups[j].0 == first {
            j += 1;
        }
        // winner: smallest ingest, first copy on a tie (equal ingest_ts copies are identical)
        let mut win = first;
        let mut wi = rec(first).ingest;
        for &(_, c) in &dups[i..j] {
            let ci = rec(c).ingest;
            if ci < wi {
                win = c;
                wi = ci;
            }
        }
        for c in std::iter::once(first).chain(dups[i..j].iter().map(|d| d.1)) {
            if c != win {
                let loc = rec(c).loc;
                if loc != NONE {
                    drops[((loc >> 20) & 0xFF) as usize].push(loc);
                }
            }
        }
        i = j;
    }
}

// ---------------------------------------------------------------- sessions

struct Work {
    rows: Vec<UserRow>,
    cred: Vec<[i64; 3]>,
    assist: Vec<i64>,
    stamp: Vec<u32>,
    iv: Vec<(i64, i64)>,
    epoch: u32,
    cnt: Vec<u32>,
    cur: Vec<u32>,
    out: Vec<u64>,
    purch: Vec<(u32, u16)>,
    ccnt: Vec<u32>,
    ccur: Vec<u32>,
    cout: Vec<u64>,
    cinfo: Vec<(u16, Packed)>,
    dr: Vec<u32>,
    sessions: Vec<Sess>,
    day_groups: Vec<(usize, usize)>,
    cats: Vec<(u16, i64)>,
    touched: Vec<usize>,
}

impl Work {
    fn new(nlp: usize, ncamp: usize, width: usize) -> Work {
        Work {
            rows: Vec::new(),
            cred: vec![[0i64; 3]; (ncamp + 1) * width],
            assist: vec![0i64; (ncamp + 1) * width],
            stamp: vec![0; ncamp + 1],
            iv: vec![(0i64, 0i64); ncamp + 1],
            epoch: 0,
            cnt: vec![0; nlp + 1],
            cur: vec![0; nlp + 1],
            out: Vec::new(),
            purch: Vec::new(),
            ccnt: vec![0; nlp + 1],
            ccur: vec![0; nlp + 1],
            cout: Vec::new(),
            cinfo: Vec::new(),
            dr: Vec::new(),
            sessions: Vec::new(),
            day_groups: Vec::new(),
            cats: Vec::new(),
            touched: Vec::new(),
        }
    }
}

struct Sess {
    start: i64,
    d: usize,
    purchased: bool,
    revenue: i64,
}

#[inline]
fn sort_small(v: &mut [u64]) {
    if v.len() <= 24 {
        for i in 1..v.len() {
            let x = v[i];
            let mut j = i;
            while j > 0 && v[j - 1] > x {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = x;
        }
    } else {
        v.sort_unstable();
    }
}

/// Turns counts in c[0..n] into start offsets in c and cur (c[n] = total).
fn offsets(c: &mut [u32], cur: &mut [u32]) {
    let mut acc = 0u32;
    for (x, y) in c.iter_mut().zip(cur.iter_mut()) {
        let v = *x;
        *x = acc;
        *y = acc;
        acc += v;
    }
}

fn person_bucket(ctx: &Ctx, tab: &Table, st: &[TS], drops: &[Vec<Vec<u32>>], b: usize, w: &mut Work) {
    let nlp = w.cnt.len() - 1;
    w.dr.clear();
    for d in drops {
        w.dr.extend_from_slice(&d[b]);
    }
    w.dr.sort_unstable();
    w.dr.push(u32::MAX);

    // events: counting sort by person
    w.cnt.fill(0);
    let mut di = 0;
    for s in st {
        let lb = (s.t as u32) << 28 | (b as u32) << 20;
        let mut i = 0u32;
        for sl in s.pb[b].slices() {
            for r in sl {
                if lb | i == w.dr[di] {
                    di += 1;
                } else {
                    w.cnt[(r.lc & 0xFF_FFFF) as usize] += 1;
                }
                i += 1;
            }
        }
    }
    offsets(&mut w.cnt, &mut w.cur);
    let total = w.cnt[nlp] as usize;
    w.out.clear();
    w.out.resize(total, 0);
    w.purch.clear();
    di = 0;
    for s in st {
        let lb = (s.t as u32) << 28 | (b as u32) << 20;
        let mut i = 0u32;
        for sl in s.pb[b].slices() {
            for r in sl {
                if lb | i == w.dr[di] {
                    di += 1;
                } else {
                    let lp = (r.lc & 0xFF_FFFF) as usize;
                    let mut pr = 0u32;
                    if r.pay != 0 {
                        w.purch.push((r.pay & 0xFF_FFFF, ((r.pay >> 24) | (r.lc >> 24) << 8) as u16));
                        pr = w.purch.len() as u32;
                    }
                    let c = &mut w.cur[lp];
                    w.out[*c as usize] = (r.ts as u64) << 32 | pr as u64;
                    *c += 1;
                }
                i += 1;
            }
        }
    }

    // clicks of persons with events
    w.ccnt.fill(0);
    for s in st {
        for sl in s.cb[b].slices() {
            for c in sl {
                let lp = c.lp as usize;
                if w.cnt[lp + 1] != w.cnt[lp] {
                    w.ccnt[lp] += 1;
                }
            }
        }
    }
    offsets(&mut w.ccnt, &mut w.ccur);
    w.cout.clear();
    w.cout.resize(w.ccnt[nlp] as usize, 0);
    w.cinfo.clear();
    for s in st {
        for sl in s.cb[b].slices() {
            for c in sl {
                let lp = c.lp as usize;
                if w.cnt[lp + 1] != w.cnt[lp] {
                    let cur = &mut w.ccur[lp];
                    w.cout[*cur as usize] = (c.ts as u64) << 32 | w.cinfo.len() as u64;
                    *cur += 1;
                    w.cinfo.push((c.camp, c.cid));
                }
            }
        }
    }

    let mut ev = std::mem::take(&mut w.out);
    let mut ck = std::mem::take(&mut w.cout);
    for lp in 0..nlp {
        let (a, z) = (w.cnt[lp] as usize, w.cnt[lp + 1] as usize);
        if a == z {
            continue;
        }
        let pid = lp << 8 | b;
        let es = &mut ev[a..z];
        sort_small(es);
        let cs = &mut ck[w.ccnt[lp] as usize..w.ccnt[lp + 1] as usize];
        sort_small(cs);
        person(ctx, tab.key(pid), es, cs, w);
    }
    w.out = ev;
    w.cout = ck;
}

/// Events 5-7 and Clicks 4-7 for one person: `ev` = (ts << 32 | purchase ref) sorted, `cs` = (ts << 32 | click ref) sorted.
fn person(ctx: &Ctx, name: SStr, ev: &[u64], cs: &[u64], w: &mut Work) {
    let base = ctx.base;
    let width = ctx.ndays;
    let organic = ctx.ncamp;

    w.sessions.clear();
    w.day_groups.clear();
    let mut row: Option<UserRow> = None;
    let mut prev = i64::MIN;
    for &e in ev {
        let ts = (e >> 32) as i64 + base;
        if prev == i64::MIN || ts - prev > GAP {
            let d = (ts.div_euclid(DAY) - ctx.day0) as usize;
            w.sessions.push(Sess { start: ts, d, purchased: false, revenue: 0 });
            let day = (ctx.day0 + d as i64) as i32;
            if row.as_ref().map(|r| r.day != day).unwrap_or(true) {
                if let Some(r) = row.take() {
                    w.rows.push(finish_row(r, &mut w.cats, &ctx.cat_rank));
                }
                row = Some(UserRow { name, day, sessions: 0, events: 0, revenue: 0, top: None });
                w.day_groups.push((d, w.sessions.len() - 1));
            }
            row.as_mut().unwrap().sessions += 1;
        }
        prev = ts;
        let r = row.as_mut().unwrap();
        r.events += 1;
        let pr = e as u32;
        if pr != 0 {
            let (price, cat) = w.purch[pr as usize - 1];
            let price = price as i64;
            let s = w.sessions.last_mut().unwrap();
            s.purchased = true;
            s.revenue += price;
            r.revenue += price;
            match w.cats.iter_mut().find(|c| c.0 == cat) {
                Some(c) => c.1 += price,
                None => w.cats.push((cat, price)),
            }
        }
    }
    if let Some(r) = row.take() {
        w.rows.push(finish_row(r, &mut w.cats, &ctx.cat_rank));
    }

    let cts = |c: u64| (c >> 32) as i64 + base;
    // credit: latest click with start - 7d <= ts <= start; tie -> smallest click_id
    let mut q = 0;
    for s in &w.sessions {
        while q < cs.len() && cts(cs[q]) <= s.start {
            q += 1;
        }
        let mut camp = organic;
        if q > 0 {
            let t = cts(cs[q - 1]);
            if s.start - t <= WEEK {
                let mut best = w.cinfo[cs[q - 1] as u32 as usize];
                let mut k = q - 1;
                while k > 0 && cts(cs[k - 1]) == t {
                    k -= 1;
                    let c = w.cinfo[cs[k] as u32 as usize];
                    if c.1.get() < best.1.get() {
                        best = c;
                    }
                }
                camp = best.0 as usize;
            }
        }
        let a = &mut w.cred[camp * width + s.d];
        a[0] += 1;
        a[1] += s.purchased as i64;
        a[2] += s.revenue;
    }

    // assists: a click at t assists sessions starting in [t, t + 7d]; merge per campaign
    fn flush(assist: &mut [i64], width: usize, k: usize, lo: i64, hi: i64, sessions: &[Sess], day_groups: &[(usize, usize)]) {
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
    w.epoch += 1;
    let ep = w.epoch;
    w.touched.clear();
    for &c in cs {
        let k = w.cinfo[c as u32 as usize].0 as usize;
        let t = cts(c);
        if w.stamp[k] != ep {
            w.stamp[k] = ep;
            w.iv[k] = (t, t + WEEK);
            w.touched.push(k);
        } else if t <= w.iv[k].1 {
            w.iv[k].1 = t + WEEK;
        } else {
            let (lo, hi) = w.iv[k];
            flush(&mut w.assist, width, k, lo, hi, &w.sessions, &w.day_groups);
            w.iv[k] = (t, t + WEEK);
        }
    }
    for &k in &w.touched {
        let (lo, hi) = w.iv[k];
        flush(&mut w.assist, width, k, lo, hi, &w.sessions, &w.day_groups);
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::{Acquire, Release}};
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

