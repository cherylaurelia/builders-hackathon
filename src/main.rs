//! Shopping Sessions pipeline. rules.md is the spec; each phase names the rule it implements.
//!
//! One shuffle, no pointers into the input. Every ID (event, user, click) becomes an exact 64-bit key: an ID of up
//! to 10 bytes from [0-9A-Z_a-z] packs losslessly, any other ID is interned. The input is read with pread into a
//! small per-thread buffer, so no input page is ever mapped.
//!
//! Phases:
//!   1. links  -> every link ID goes into one lock-free ID table; a lock-free union-find over its slots hangs the
//!      larger name under the smaller, so the root of a person is their name (byte-wise smallest ID)
//!   2. events + clicks -> parsed from one chunk queue. A user ID is found (or added) in the ID table; its person's
//!      slot number is the person. A valid event is a 12-byte record in its person partition; every event also
//!      leaves (event key, ingest_ts, record address) in its event-key bucket. Clicks go to person partitions too
//!      (copies are identical: no dedup needed)
//!   3. per event-key bucket: a cache-sized bit filter finds the keys seen twice; those few are sorted and every
//!      copy but the earliest ingested is marked dropped in place
//!   4. per person partition: counting sort by person, sort by ts, sessionize, credit, assist, aggregate; each
//!      thread then writes its own Parquet file

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::Instant;

type Bytes = &'static [u8];

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;

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

// ---------------------------------------------------------------- IDs as 64-bit keys

/// Bit 63 of a key: the ID did not pack, the rest is its number in the `Interner`.
const INTERNED: u64 = 1 << 63;

/// Byte -> digit 1..=63 for [0-9A-Z_a-z] in byte order, 0 for any other byte.
const fn pack_table() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut d = 1u8;
    let mut c = 0usize;
    while c < 256 {
        let b = c as u8;
        if (b >= b'0' && b <= b'9') || (b >= b'A' && b <= b'Z') || b == b'_' || (b >= b'a' && b <= b'z') {
            t[c] = d;
            d += 1;
        }
        c += 1;
    }
    t
}
static PACK: [u8; 256] = pack_table();

const fn unpack_table() -> [u8; 64] {
    let p = pack_table();
    let mut t = [0u8; 64];
    let mut c = 0usize;
    while c < 256 {
        if p[c] != 0 {
            t[p[c] as usize] = c as u8;
        }
        c += 1;
    }
    t
}
static UNPACK: [u8; 64] = unpack_table();

/// An ID of 1..=10 bytes from [0-9A-Z_a-z] as a number: six bits a byte, first byte highest, zero filled.
/// 0 = it does not pack. Digits rise with the byte values, so two packed IDs compare as their bytes do.
#[inline]
fn pack(b: &[u8]) -> u64 {
    if b.len() > 10 {
        return 0;
    }
    let mut v = 0u64;
    for &c in b {
        let d = PACK[c as usize];
        if d == 0 {
            return 0;
        }
        v = (v << 6) | d as u64;
    }
    v << (6 * (10 - b.len()))
}

/// `pack` of the `len` (<= 16) bytes whose `load16` is `v`.
#[inline(always)]
fn pack16(v: u128, len: usize) -> u64 {
    if len > 10 {
        return 0;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
    unsafe {
        use std::arch::x86_64::*;
        let c = _mm_set_epi64x((v >> 64) as i64, v as i64);
        let rng = |lo: u8, hi: u8| _mm_and_si128(_mm_cmpgt_epi8(c, _mm_set1_epi8(lo as i8 - 1)), _mm_cmpgt_epi8(_mm_set1_epi8(hi as i8 + 1), c));
        let (dg, up, lw) = (rng(b'0', b'9'), rng(b'A', b'Z'), rng(b'a', b'z'));
        let us = _mm_cmpeq_epi8(c, _mm_set1_epi8(b'_' as i8));
        let valid = _mm_movemask_epi8(_mm_or_si128(_mm_or_si128(dg, up), _mm_or_si128(us, lw))) as u32;
        let need = (1u32 << len) - 1;
        if valid & need != need {
            return 0;
        }
        // digit = byte - offset of its class; the zero bytes past `len` are in no class and stay 0
        let off = _mm_or_si128(
            _mm_or_si128(_mm_and_si128(dg, _mm_set1_epi8(47)), _mm_and_si128(up, _mm_set1_epi8(54))),
            _mm_or_si128(_mm_and_si128(us, _mm_set1_epi8(58)), _mm_and_si128(lw, _mm_set1_epi8(59))),
        );
        let d = _mm_sub_epi8(c, off);
        let w = _mm_maddubs_epi16(d, _mm_set1_epi16(0x0140)); // two digits -> 12 bits
        let l = _mm_madd_epi16(w, _mm_set1_epi32(0x0001_1000)); // four digits -> 24 bits
        let lo = _mm_cvtsi128_si64(l) as u64;
        let l2 = _mm_extract_epi32(l, 2) as u32 as u64;
        return ((lo & 0xffff_ffff) << 36) | ((lo >> 32) << 12) | (l2 >> 12);
    }
    #[allow(unreachable_code)]
    {
        pack(&v.to_le_bytes()[..len])
    }
}

/// The IDs that do not pack (longer than 10 bytes, or a byte outside [0-9A-Z_a-z]): each gets a number, exact on
/// its bytes. Locked and slow; generated data never comes here.
struct Interner {
    shards: Vec<Mutex<(HashMap<Box<[u8]>, u32, BuildHasherDefault<BytesHasher>>, Vec<Box<[u8]>>)>>,
}

impl Interner {
    const SHARDS: usize = 64;

    fn new() -> Interner {
        Interner { shards: (0..Self::SHARDS).map(|_| Mutex::new((HashMap::default(), Vec::new()))).collect() }
    }

    #[cold]
    #[inline(never)]
    fn intern(&self, b: &[u8]) -> u64 {
        let s = (hash_bytes(b) >> 58) as usize;
        let mut g = self.shards[s].lock().unwrap_or_else(|_| die("interner poisoned"));
        let i = match g.0.get(b) {
            Some(&i) => i,
            None => {
                let i = u32::try_from(g.1.len()).unwrap_or_else(|_| die("too many long IDs"));
                g.1.push(b.into());
                g.0.insert(b.into(), i);
                i
            }
        };
        INTERNED | (s as u64) << 32 | i as u64
    }

    /// The key of an ID.
    #[inline]
    fn key(&self, b: &[u8]) -> u64 {
        match pack(b) {
            0 => self.intern(b),
            k => k,
        }
    }

    /// `key` of the `len` bytes at `p` (16 bytes must be readable).
    #[inline(always)]
    unsafe fn key_at(&self, p: *const u8, len: usize) -> u64 {
        if len <= 10 {
            let k = pack16(load16(p, len), len);
            if k != 0 {
                return k;
            }
        }
        self.intern(std::slice::from_raw_parts(p, len))
    }

    /// Appends the ID of `key` to `out`.
    #[inline]
    fn bytes(&self, key: u64, out: &mut Vec<u8>) {
        if key & INTERNED != 0 {
            let g = self.shards[((key >> 32) as usize) & (Self::SHARDS - 1)].lock().unwrap_or_else(|_| die("interner poisoned"));
            out.extend_from_slice(&g.1[key as u32 as usize]);
            return;
        }
        for i in 0..10 {
            let d = (key >> (54 - 6 * i)) & 63;
            if d == 0 {
                break;
            }
            out.push(UNPACK[d as usize]);
        }
    }

    /// Byte-wise `a < b` of the two IDs.
    #[inline]
    fn less(&self, a: u64, b: u64) -> bool {
        if (a | b) & INTERNED == 0 {
            return a < b;
        }
        let (mut x, mut y) = (Vec::new(), Vec::new());
        self.bytes(a, &mut x);
        self.bytes(b, &mut y);
        x < y
    }
}

// ---------------------------------------------------------------- memory

/// `bytes` of zeroed anonymous memory, never freed; huge pages where the kernel offers them.
fn map_anon(bytes: usize) -> *mut u8 {
    let p = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
    if p == libc::MAP_FAILED {
        die(&format!("mmap of {bytes} bytes: {}", std::io::Error::last_os_error()));
    }
    #[cfg(target_os = "linux")]
    unsafe {
        libc::madvise(p, bytes, libc::MADV_HUGEPAGE); // a hint: fine if refused
    }
    p as *mut u8
}

/// A thread's bump allocator for stream blocks.
struct Arena {
    cur: usize,
    end: usize,
}

impl Arena {
    const CHUNK: usize = 32 << 20;

    #[inline]
    fn alloc(&mut self, bytes: usize) -> *mut u8 {
        if self.cur + bytes > self.end {
            let n = bytes.max(Self::CHUNK);
            self.cur = map_anon(n) as usize;
            self.end = self.cur + n;
        }
        let p = self.cur;
        self.cur += bytes;
        p as *mut u8
    }
}

/// An append-only list of `T` in blocks of `per` records; a record never moves.
struct Stream<T> {
    cur: *mut T,
    end: *mut T,
    blocks: Vec<*mut T>,
}

impl<T> Stream<T> {
    fn new() -> Stream<T> {
        Stream { cur: std::ptr::null_mut(), end: std::ptr::null_mut(), blocks: Vec::new() }
    }

    #[inline(always)]
    fn push(&mut self, arena: &mut Arena, per: usize, v: T) -> *mut T {
        if self.cur == self.end {
            self.grow(arena, per);
        }
        let p = self.cur;
        // SAFETY: cur < end, inside the block `grow` took from the arena
        unsafe {
            p.write(v);
            self.cur = p.add(1);
        }
        p
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, arena: &mut Arena, per: usize) {
        let p = arena.alloc(per * std::mem::size_of::<T>()) as *mut T;
        self.blocks.push(p);
        self.cur = p;
        self.end = unsafe { p.add(per) };
    }

    fn len(&self, per: usize) -> usize {
        match self.blocks.last() {
            None => 0,
            Some(&b) => (self.blocks.len() - 1) * per + (unsafe { self.cur.offset_from(b) }) as usize,
        }
    }

    fn slices(&self, per: usize) -> impl Iterator<Item = &[T]> {
        let n = self.blocks.len();
        self.blocks.iter().enumerate().map(move |(i, &b)| {
            let len = if i + 1 == n { (unsafe { self.cur.offset_from(b) }) as usize } else { per };
            unsafe { std::slice::from_raw_parts(b, len) }
        })
    }
}

/// Asks for the cache line at `p`; a hint only.
#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

// ---------------------------------------------------------------- the ID table

/// One user ID. `up` = 0: the ID is its own person (or the root of its link tree); else 1 + the slot above it.
#[repr(C)]
struct Slot {
    key: AtomicU64,
    up: AtomicU32,
    _pad: u32,
}

/// Every user ID seen in links, events and clicks: open addressing on the key, lock free, never resized.
/// A slot's number is the ID's number; a person is the number of the slot of their name.
struct IdTable {
    slots: &'static [Slot],
    bits: u32,
    count: AtomicUsize,
    limit: usize,
}

impl IdTable {
    fn new(bits: u32) -> IdTable {
        let n = 1usize << bits;
        let p = map_anon(n * std::mem::size_of::<Slot>()) as *const Slot;
        // SAFETY: zeroed memory is n empty slots
        IdTable { slots: unsafe { std::slice::from_raw_parts(p, n) }, bits, count: AtomicUsize::new(0), limit: n / 4 * 3 }
    }

    /// Writes to every 2 MiB of the table, on `threads` threads. A page first touched by a read (a lookup) maps
    /// the shared zero page, and the first write there then leaves the whole 2 MiB in 4 KiB pages: every later
    /// lookup would miss the TLB. A write first gets a huge page (where the kernel gives them).
    fn populate(&self, threads: usize) {
        const HUGE: usize = 2 << 20;
        let bytes = self.slots.len() * std::mem::size_of::<Slot>();
        par(
            threads,
            bytes.div_ceil(HUGE),
            || (),
            |_, i| {
                // a zero stored over a zero: the slot stays empty
                self.slots[i * HUGE / std::mem::size_of::<Slot>()].up.store(0, Relaxed);
            },
        );
    }

    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        (mix(key) >> (64 - self.bits)) as usize
    }

    /// `home(key)`, with its cache line on the way: the lookup comes a few lines of input later.
    #[inline(always)]
    fn aim(&self, key: u64) -> u32 {
        let h = self.home(key);
        prefetch(unsafe { self.slots.as_ptr().add(h) });
        h as u32
    }

    /// The slot of `key`, added if new. `ins` is the caller's count of additions not yet reported.
    #[cfg(test)]
    fn slot(&self, key: u64, ins: &mut usize) -> u32 {
        self.slot_from(self.home(key) as u32, key, ins)
    }

    /// `slot` given `home(key)`.
    #[inline(always)]
    fn slot_from(&self, home: u32, key: u64, ins: &mut usize) -> u32 {
        let mask = self.slots.len() - 1;
        let mut i = home as usize;
        loop {
            // SAFETY: i <= mask
            let s = unsafe { self.slots.get_unchecked(i) };
            let k = s.key.load(Relaxed);
            if k == key {
                return i as u32;
            }
            if k == 0 {
                match s.key.compare_exchange(0, key, Relaxed, Relaxed) {
                    Ok(_) => {
                        self.added(ins);
                        return i as u32;
                    }
                    Err(now) if now == key => return i as u32,
                    Err(_) => {}
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// Counts an addition; a table three quarters full restarts the program with a larger one.
    #[inline(always)]
    fn added(&self, ins: &mut usize) {
        *ins += 1;
        if *ins == 256 {
            *ins = 0;
            if self.count.fetch_add(256, Relaxed) + 256 > self.limit {
                restart_bigger(self.bits);
            }
        }
    }

    /// The person of the ID in `slot`, once the links are flattened.
    #[inline(always)]
    fn person(&self, slot: u32) -> u32 {
        match unsafe { self.slots.get_unchecked(slot as usize) }.up.load(Relaxed) {
            0 => slot,
            u => u - 1,
        }
    }

    /// The root above `x`, halving the path on the way.
    fn find(&self, mut x: u32) -> u32 {
        loop {
            let u = self.slots[x as usize].up.load(Relaxed);
            if u == 0 {
                return x;
            }
            let p = u - 1;
            let g = self.slots[p as usize].up.load(Relaxed);
            if g != 0 {
                let _ = self.slots[x as usize].up.compare_exchange(u, g, Relaxed, Relaxed);
            }
            x = p;
        }
    }

    /// Joins the trees of `a` and `b`: the root with the larger name goes under the other, so every step up is to
    /// a smaller name, no cycle can form, and the root of a tree is its smallest name.
    fn union(&self, names: &Interner, mut a: u32, mut b: u32) {
        loop {
            a = self.find(a);
            b = self.find(b);
            if a == b {
                return;
            }
            let (ka, kb) = (self.slots[a as usize].key.load(Relaxed), self.slots[b as usize].key.load(Relaxed));
            let (hi, lo) = if names.less(kb, ka) { (a, b) } else { (b, a) };
            if self.slots[hi as usize].up.compare_exchange(0, lo + 1, Relaxed, Relaxed).is_ok() {
                return;
            }
        }
    }
}

/// Replaces the process with itself and a 4x larger ID table (nothing has been written yet).
#[cold]
fn restart_bigger(bits: u32) -> ! {
    use std::os::unix::process::CommandExt;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if ONCE.swap(true, Relaxed) {
        loop {
            std::thread::park(); // another thread is already restarting
        }
    }
    if bits + 2 > 36 {
        die("ID table would pass 2^36 slots");
    }
    eprintln!("ID table of 2^{bits} slots filled up: restarting with 2^{}", bits + 2);
    let exe = std::env::current_exe().unwrap_or_else(|e| die(&format!("current_exe: {e}")));
    let err = std::process::Command::new(exe).args(std::env::args_os().skip(1)).env("SESS_ID_BITS", (bits + 2).to_string()).exec();
    die(&format!("restart: {err}"))
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

/// A date already checked: its `YYYY-MM-DDT` (11 bytes) and the epoch ms of its midnight.
#[derive(Clone, Copy)]
struct Day {
    a: u64, // bytes 0..8
    b: u32, // bytes 8..11; `NO_DAY.b` matches no input
    ms: i64,
}
const NO_DAY: Day = Day { a: 0, b: u32::MAX, ms: 0 };

/// `ts_at` that remembers the dates it has checked (by day of month), so an F1 or F2 timestamp of a known date
/// costs two compares and the time of day.
#[inline(always)]
unsafe fn ts_memo(p: *const u8, len: usize, days: &mut [Day; 32]) -> Option<i64> {
    let at = |i: usize| *p.add(i);
    let f1 = len == 20 && at(19) == b'Z';
    if !f1 && !(len == 25 && (at(19) == b'+' || at(19) == b'-') && at(22) == b':') {
        return ts_at(p, len);
    }
    let a = (p as *const u64).read_unaligned();
    let b = (p.add(8) as *const u32).read_unaligned() & 0xFF_FFFF;
    let day = &mut days[(((b & 15) * 10 + ((b >> 8) & 15)) & 31) as usize];
    if day.a != a || day.b != b {
        // a date not seen yet, or a bad one: the full check; a good one is remembered
        let t = ts_at(p, len)?;
        *day = Day { a, b, ms: civil_at(p)?.div_euclid(DAY) * DAY };
        return Some(t);
    }
    // HH:MM:SS as in `civil_at`
    let x = (p.add(11) as *const u64).read_unaligned() ^ u64::from_le_bytes(*b"00:00:00");
    const ADD: u64 = u64::from_le_bytes([0x76, 0x76, 0x7F, 0x76, 0x76, 0x7F, 0x76, 0x76]);
    if (x.wrapping_add(ADD) | x) & HI != 0 {
        return None;
    }
    let pb = x * 10 + (x >> 8);
    let (h, mi, sec) = (pb & 0xFF, (pb >> 24) & 0xFF, (pb >> 48) & 0xFF);
    if (h > 23) | (mi > 59) | (sec > 59) {
        return None;
    }
    let t = day.ms + (h * 3_600_000 + mi * 60_000 + sec * 1000) as i64;
    if f1 {
        return Some(t);
    }
    let (h1, h2, m1, m2) = (at(20).wrapping_sub(b'0'), at(21).wrapping_sub(b'0'), at(23).wrapping_sub(b'0'), at(24).wrapping_sub(b'0'));
    if h1 > 9 || h2 > 9 || m1 > 9 || m2 > 9 {
        return None;
    }
    let (oh, om) = ((h1 * 10 + h2) as i64, (m1 * 10 + m2) as i64);
    if oh > 23 || om > 59 {
        return None;
    }
    let off = (oh * 60 + om) * 60_000;
    Some(if at(19) == b'+' { t - off } else { t + off })
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

struct InFile {
    f: File,
    len: usize,
}

/// The `*.csv` files of `dir`, by name.
fn csv_files(dir: &Path) -> Vec<InFile> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "csv").unwrap_or(false) && p.is_file())
            .collect(),
        Err(e) => die(&format!("read_dir {}: {e}", dir.display())),
    };
    paths.sort();
    paths
        .iter()
        .map(|p| {
            let f = File::open(p).unwrap_or_else(|e| die(&format!("open {}: {e}", p.display())));
            let len = f.metadata().unwrap_or_else(|e| die(&format!("stat {}: {e}", p.display()))).len() as usize;
            InFile { f, len }
        })
        .collect()
}

/// (kind, file, start, end) pieces of about `size` bytes.
fn chunks_of(kind: u8, files: &[InFile], size: usize, out: &mut Vec<(u8, usize, usize, usize)>) {
    for (i, f) in files.iter().enumerate() {
        let mut s = 0;
        while s < f.len {
            let e = (s + size).min(f.len);
            out.push((kind, i, s, e));
            s = e;
        }
    }
}

/// True unless a byte of `f` at `off` can be read without waiting for the disk.
#[cfg(target_os = "linux")]
fn uncached(f: &InFile, off: usize) -> bool {
    use std::os::unix::io::AsRawFd;
    let mut b = 0u8;
    let iov = libc::iovec { iov_base: &mut b as *mut u8 as *mut libc::c_void, iov_len: 1 };
    // 1 = read from the page cache; EAGAIN = not cached; any other answer (an old kernel) counts as not cached
    unsafe { libc::preadv2(f.f.as_raw_fd(), &iov, 1, off as libc::off_t, libc::RWF_NOWAIT) != 1 }
}

/// Cold page cache: asks the kernel for the whole input at once, in the order it will be parsed, from a few
/// threads of their own, so the disk works at full speed from the first moment whatever the parse is doing.
/// An input that is cached already (16 probes all answer without the disk) is left alone.
#[cfg(target_os = "linux")]
fn read_ahead(queue: Vec<(&'static InFile, usize, usize)>, threads: usize) {
    use std::os::unix::io::AsRawFd;
    let n = queue.len();
    if n == 0 || !(0..16).any(|i| uncached(queue[i * n / 16].0, (queue[i * n / 16].1 + queue[i * n / 16].2) / 2)) {
        return;
    }
    let queue = std::sync::Arc::new(queue);
    let next = std::sync::Arc::new(AtomicUsize::new(0));
    for _ in 0..threads.div_ceil(2) {
        let (queue, next) = (queue.clone(), next.clone());
        // never joined: these only start reads, and the end of the process ends them. A thread that cannot start,
        // or a hint the kernel refuses, costs speed only: every byte is still read by `read_span`.
        let _ = std::thread::Builder::new().spawn(move || loop {
            let i = next.fetch_add(1, Relaxed);
            if i >= queue.len() {
                break;
            }
            let (f, s, e) = queue[i];
            // 128 KiB a call: the kernel caps one request at its readahead size
            let mut o = s;
            while o < e {
                let l = (e - o).min(128 << 10);
                unsafe { libc::posix_fadvise(f.f.as_raw_fd(), o as libc::off_t, l as libc::off_t, libc::POSIX_FADV_WILLNEED) };
                o += l;
            }
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn read_ahead(_queue: Vec<(&'static InFile, usize, usize)>, _threads: usize) {}

/// Bytes past the data that `read_span` keeps readable (and starts with a `\n`): `scan_lines` loads 144 past a line start.
const PAD: usize = 160;

/// Reads into `buf` every line of `f` that starts in [s, e): from byte s - 1 (to see whether a line starts at s)
/// through the `\n` of the last such line, or the end of the file. Returns (len, start, end) with the data in
/// buf[..len] and the lines to take those that start in [start, end) of it.
fn read_span(f: &InFile, s: usize, e: usize, buf: &mut Vec<u8>) -> (usize, usize, usize) {
    let bs = s.saturating_sub(1);
    let mut want = (e - bs + 512).min(f.len - bs);
    let mut got = 0;
    let len = loop {
        if buf.len() < want + PAD {
            buf.resize(want + PAD, 0);
        }
        while got < want {
            match f.f.read_at(&mut buf[got..want], (bs + got) as u64) {
                Ok(0) => die("an input file shrank while it was read"),
                Ok(n) => got += n,
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                Err(err) => die(&format!("read: {err}")),
            }
        }
        // the last line that starts before e ends at the first newline at or after e - 1
        let from = e - 1 - bs;
        if let Some(i) = memchr::memchr(b'\n', &buf[from..got]) {
            break from + i + 1;
        }
        if bs + got >= f.len {
            break got;
        }
        want = (want + want / 2 + 4096).min(f.len - bs);
    };
    buf[len..len + PAD].fill(b'\n');
    (len, s - bs, e - bs)
}

/// Calls `fast(st, line, e, commas, quotes)` for every line that starts in [start, end) of the `len` bytes at `buf`
/// and has its `\n` in its first 128 bytes (line[e]; the masks cover line[..e]); a longer line goes whole to `slow`.
/// The header line is skipped. `buf` must be readable for 144 bytes past `len`, and buf[len] must be `\n`.
#[inline(always)]
unsafe fn scan_lines<S>(
    buf: *const u8,
    start: usize,
    end: usize,
    len: usize,
    st: &mut S,
    fast: impl Fn(&mut S, *const u8, usize, u128, u128),
    slow: impl Fn(&mut S, Bytes),
) {
    let file = std::slice::from_raw_parts(buf, len);
    let first = start.saturating_sub(1);
    let Some(nl) = memchr::memchr(b'\n', &file[first..]) else { return };
    let mut p = first + nl + 1;
    while p < end && p < len {
        let line = buf.add(p);
        let [c0, q0, n0] = masks64(line);
        let [c1, q1, n1] = masks64(line.add(64));
        let n = n0 as u128 | (n1 as u128) << 64;
        if n != 0 {
            let e = n.trailing_zeros() as usize;
            let lm = (1u128 << e) - 1;
            fast(st, line, e, (c0 as u128 | (c1 as u128) << 64) & lm, (q0 as u128 | (q1 as u128) << 64) & lm);
            p += e + 1;
        } else {
            let e = memchr::memchr(b'\n', &file[p..]).map_or(len, |i| p + i);
            slow(st, std::mem::transmute::<&[u8], Bytes>(&file[p..e]));
            p = e + 1;
        }
    }
}

/// The N fields of the line of `e` (< 128) bytes at `line` as (start, length), quotes removed, given its comma and
/// quote bits; None when the line is malformed (`split::<N>` says the same).
#[inline(always)]
unsafe fn fast_split<const N: usize>(line: *const u8, e: usize, c: u128, q: u128) -> Option<[(usize, usize); N]> {
    if c.count_ones() as usize != N - 1 {
        return None;
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
        return None;
    }
    let mut f = [(0usize, 0usize); N];
    let mut cc = c;
    let mut at = 0;
    for i in 0..N {
        let k = if i + 1 == N { e } else { cc.trailing_zeros() as usize };
        cc &= cc.wrapping_sub(1);
        // a quoted field starts with `"` (an empty unquoted field starts on its `,` or `\n`)
        let qd = (*line.add(at) == b'"') as usize;
        f[i] = (at + qd, k - at - 2 * qd);
        at = k + 1;
    }
    Some(f)
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

/// A valid, non-late event in its person partition. `person` = `DROPPED` once dedup finds an earlier copy.
#[repr(C)]
#[derive(Clone, Copy)]
struct ERec {
    person: u32,
    ts: u32,  // ms after `Ctx::base`
    pay: u32, // 0, or category << 24 | price_cents for a purchase
}
const DROPPED: u32 = u32::MAX;

/// One parsed event line in its event-key bucket: `loc` is the address of its `ERec`, 0 for an invalid or late event
/// (it still takes part in dedup).
#[repr(C)]
#[derive(Clone, Copy)]
struct LRec {
    key: u64,
    ingest: i64,
    loc: u64,
}

/// A valid click in its person partition.
#[repr(C)]
#[derive(Clone, Copy)]
struct CRec {
    cid: u64,
    person: u32,
    ts: u32, // ms after `Ctx::base`
    camp: u32,
    _pad: u32,
}

const EV_BLOCK: usize = 512;
const LG_BLOCK: usize = 256;
const CK_BLOCK: usize = 128;

/// A valid event or click whose user ID is on its way from memory (`IdTable::aim`).
#[derive(Clone, Copy, Default)]
struct Pend {
    a: u64, // event key, or click key
    b: i64, // ingest_ts (events)
    ukey: u64,
    home: u32,
    ts: u32,
    x: u32, // pay, or campaign
}

/// Lines between asking for an ID's cache line and using it.
const RING: usize = 16;

/// One parse thread's output.
struct Shard {
    arena: Arena,
    ev: Vec<Stream<ERec>>,
    ck: Vec<Stream<CRec>>,
    lg: Vec<Stream<LRec>>,
    buf: Vec<u8>,
    ins: usize,
    ring: [Pend; RING],
    head: usize, // events or clicks put in the ring since the last flush
    days: [Day; 32],
}
// SAFETY: the streams own their blocks; after the parse they are only read, except for `ERec::person` of distinct
// records written by the dedup phase while nothing reads them
unsafe impl Send for Shard {}
unsafe impl Sync for Shard {}

struct UserRow {
    name: u64,
    revenue: i64,
    sessions: u32,
    events: u32,
    day: i32,
    top: u16, // NO_CAT when there is no purchase
}
const NO_CAT: u16 = u16::MAX;

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
    ids: IdTable,
    names: Interner,
    pshift: u32, // person >> pshift = person partition
    qshift: u32, // mix(event key) >> qshift = event-key bucket
}

impl Shard {
    fn new(parts: usize, buckets: usize) -> Shard {
        Shard {
            arena: Arena { cur: 0, end: 0 },
            ev: (0..parts).map(|_| Stream::new()).collect(),
            ck: (0..parts).map(|_| Stream::new()).collect(),
            lg: (0..buckets).map(|_| Stream::new()).collect(),
            buf: Vec::new(),
            ins: 0,
            ring: [Pend::default(); RING],
            head: 0,
            days: [NO_DAY; 32],
        }
    }

    /// Queues `p`; returns the one queued RING entries ago, whose slot is in the cache by now.
    #[inline(always)]
    fn queue(&mut self, p: Pend) -> Option<Pend> {
        let i = self.head & (RING - 1);
        let old = if self.head >= RING { Some(self.ring[i]) } else { None };
        self.ring[i] = p;
        self.head += 1;
        old
    }

    /// Finishes everything still queued: events when `events`, else clicks. Call at the end of every chunk.
    fn flush(&mut self, cx: &Ctx, events: bool) {
        for k in self.head.saturating_sub(RING)..self.head {
            let p = self.ring[k & (RING - 1)];
            if events {
                self.finish_event(cx, p);
            } else {
                self.finish_click(cx, p);
            }
        }
        self.head = 0;
    }

    /// Any parsed event: its dedup record. `loc` = 0 for an invalid or late one.
    #[inline(always)]
    fn log(&mut self, cx: &Ctx, ekey: u64, ingest: i64, loc: u64) {
        let q = (mix(ekey) >> cx.qshift) as usize;
        self.lg[q].push(&mut self.arena, LG_BLOCK, LRec { key: ekey, ingest, loc });
    }

    /// A valid, non-late event: into its person's partition, plus its dedup record, once its user ID is at hand.
    #[inline(always)]
    fn keep(&mut self, cx: &Ctx, ekey: u64, ingest: i64, ukey: u64, ts: i64, pay: u32) {
        if let Some(p) = self.queue(Pend { a: ekey, b: ingest, ukey, home: cx.ids.aim(ukey), ts: (ts - cx.base) as u32, x: pay }) {
            self.finish_event(cx, p);
        }
    }

    #[inline(always)]
    fn finish_event(&mut self, cx: &Ctx, p: Pend) {
        let person = cx.ids.person(cx.ids.slot_from(p.home, p.ukey, &mut self.ins));
        let rec = self.ev[(person >> cx.pshift) as usize].push(&mut self.arena, EV_BLOCK, ERec { person, ts: p.ts, pay: p.x });
        self.log(cx, p.a, p.b, rec as u64);
    }

    /// A valid click, into its person's partition once its user ID is at hand.
    #[inline(always)]
    fn click(&mut self, cx: &Ctx, cid: u64, ukey: u64, ts: i64, camp: u16) {
        if let Some(p) = self.queue(Pend { a: cid, b: 0, ukey, home: cx.ids.aim(ukey), ts: (ts - cx.base) as u32, x: camp as u32 }) {
            self.finish_click(cx, p);
        }
    }

    #[inline(always)]
    fn finish_click(&mut self, cx: &Ctx, p: Pend) {
        let person = cx.ids.person(cx.ids.slot_from(p.home, p.ukey, &mut self.ins));
        self.ck[(person >> cx.pshift) as usize].push(&mut self.arena, CK_BLOCK, CRec { cid: p.a, person, ts: p.ts, camp: p.x, _pad: 0 });
    }
}

// ---------------------------------------------------------------- main

fn env_num(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}

/// log2 of the power of two nearest to `n` (at least 1).
fn log2_near(n: usize) -> u32 {
    let n = n.max(1);
    let lo = usize::BITS - 1 - n.leading_zeros();
    lo + (n >= (3usize << lo) / 2) as u32
}

fn main() {
    let t0 = Instant::now();
    let data = PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let threads: usize = env_num("NCPU")
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
    if we - (ws - WEEK) > u32::MAX as i64 {
        die("manifest: window longer than 42 days");
    }
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1) as usize;

    let files = |dir: &str| -> &'static [InFile] { Box::leak(csv_files(&data.join(dir)).into_boxed_slice()) };
    let (link_files, ev_files, ck_files) = (files("links"), files("events"), files("clicks"));
    let bytes = |v: &[InFile]| v.iter().map(|f| f.len).sum::<usize>();
    let (link_bytes, ev_bytes) = (bytes(&link_files), bytes(&ev_files));

    // sizes: a link line is ~21 bytes and names ~1.4 new IDs; ~1 in 40 event lines brings an unlinked ID; an event
    // line is ~75 bytes. A wrong guess costs speed, never rows: the ID table restarts the run when it fills up.
    let id_est = link_bytes / 14 + ev_bytes / 3000 + bytes(&ck_files) / 3000;
    let id_bits = env_num("SESS_ID_BITS").map(|v| v as u32).unwrap_or(usize::BITS - (id_est * 7 / 5).leading_zeros()).clamp(16, 36);
    // about 90,000 events per person partition and per event-key bucket: more of them slow the parse (one open
    // block each per thread), fewer leave the later phases with work that no longer fits a core's cache
    let ev_est = ev_bytes / 72;
    let pbits = (env_num("SESS_PBITS").map(|v| v as u32).unwrap_or_else(|| log2_near(ev_est / 90_000))).clamp(4, 12.min(id_bits - 4));
    let qbits = (env_num("SESS_QBITS").map(|v| v as u32).unwrap_or_else(|| log2_near(ev_est / 90_000))).clamp(4, 12);
    let chunk = env_num("SESS_CHUNK").filter(|&c| c >= 4096).unwrap_or(512 << 10);

    // the parse queue: events first, then clicks
    let mut work = Vec::new();
    chunks_of(0, ev_files, chunk, &mut work);
    chunks_of(1, ck_files, chunk, &mut work);
    read_ahead(
        link_files
            .iter()
            .map(|f| (f, 0, f.len))
            .chain(work.iter().map(|&(kind, fi, s, e)| (if kind == 0 { &ev_files[fi] } else { &ck_files[fi] }, s, e)))
            .collect(),
        threads,
    );

    // 1. links, on every thread; this one reads the small tables meanwhile
    let ids = IdTable::new(id_bits);
    let names = Interner::new();
    let (n_links, small) = std::thread::scope(|sc| {
        let links = sc.spawn(|| {
            ids.populate(threads);
            build_links(&ids, &names, link_files, threads)
        });
        let small = small_tables(&data);
        (links.join().unwrap_or_else(|_| die("link builder panicked")), small)
    });
    let Small { products, campaigns, cat_names, cat_rank, unknown_cat, camp_ids, camp_channel } = small;
    let ncamp = camp_ids.len();
    let cx = Ctx {
        ws,
        we,
        base: ws - WEEK,
        day0,
        ndays,
        ncamp,
        products,
        campaigns,
        unknown_cat,
        cat_rank,
        ids,
        names,
        pshift: id_bits - pbits,
        qshift: 64 - qbits,
    };
    lap(&format!("small tables, links joined: {n_links} lines; 2^{id_bits} id slots, 2^{pbits} person partitions, 2^{qbits} event buckets"));

    // 2. events and clicks, from the one queue
    let (np, nq) = (1usize << pbits, 1usize << qbits);
    let shards = par(
        threads,
        work.len(),
        || Shard::new(np, nq),
        |st, i| {
            let (kind, fi, s, e) = work[i];
            let f = if kind == 0 { &ev_files[fi] } else { &ck_files[fi] };
            let mut buf = std::mem::take(&mut st.buf);
            let (len, start, end) = read_span(f, s, e, &mut buf);
            // SAFETY: read_span leaves PAD readable bytes past len, the first a newline
            unsafe {
                if kind == 0 {
                    parse_events(&cx, buf.as_ptr(), start, end, len, st);
                } else {
                    parse_clicks(&cx, buf.as_ptr(), start, end, len, st);
                }
            }
            st.buf = buf;
        },
    );
    let n_ev: usize = shards.iter().flat_map(|s| s.ev.iter()).map(|s| s.len(EV_BLOCK)).sum();
    let n_ck: usize = shards.iter().flat_map(|s| s.ck.iter()).map(|s| s.len(CK_BLOCK)).sum();
    lap(&format!("parsed: {n_ev} valid events, {n_ck} valid clicks"));

    // 3. duplicates
    let n_dups = mark_duplicates(&shards, nq, threads);
    lap(&format!("duplicates marked: {n_dups} candidates"));

    // 4. per person partition, the largest first; each thread writes its rows when the queue is empty
    let ud = out.join("user_daily");
    let cd = out.join("campaign_daily");
    for d in [&ud, &cd] {
        std::fs::create_dir_all(d).unwrap_or_else(|e| die(&format!("mkdir {}: {e}", d.display())));
    }
    let mut order: Vec<(usize, usize)> = (0..np).map(|p| (shards.iter().map(|s| s.ev[p].len(EV_BLOCK)).sum(), p)).collect();
    order.sort_unstable_by(|a, b| b.cmp(a));
    let width = ndays;
    let next = AtomicUsize::new(0);
    let n_rows = AtomicUsize::new(0);
    let results: Vec<Work> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let (cx, shards, order, next, n_rows, ud, cat_names) = (&cx, &shards, &order, &next, &n_rows, &ud, &cat_names);
                sc.spawn(move || {
                    let mut w = Work::new(cx);
                    loop {
                        let i = next.fetch_add(1, Relaxed);
                        if i >= order.len() {
                            break;
                        }
                        partition_work(cx, shards, order[i].1, &mut w);
                    }
                    // thread 0 writes even with no rows: the table needs at least one file
                    if t == 0 || !w.rows.is_empty() {
                        write_users(&ud.join(format!("part-{t}.parquet")), &w.rows, cat_names, &cx.names)
                            .unwrap_or_else(|e| die(&format!("write user_daily: {e}")));
                    }
                    n_rows.fetch_add(w.rows.len(), Relaxed);
                    w
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| die("a worker panicked"))).collect()
    });
    lap(&format!("sessions aggregated, {} user rows written", n_rows.load(Relaxed)));

    // campaign_daily
    let mut cred = vec![[0i64; 3]; (ncamp + 1) * width];
    let mut assist = vec![0i64; (ncamp + 1) * width];
    for r in &results {
        for (a, b) in cred.iter_mut().zip(&r.cred) {
            a[0] += b.sessions as i64;
            a[1] += b.purchasing as i64;
            a[2] += b.revenue;
        }
        for (a, b) in assist.iter_mut().zip(&r.assist) {
            *a += *b as i64;
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
    write_camps(&cd.join("part-0.parquet"), &camp_rows).unwrap_or_else(|e| die(&format!("write campaign_daily: {e}")));
    lap(&format!("written: {} campaign rows", camp_rows.len()));
    if std::env::var_os("SESS_DEBUG").is_some() {
        if let Ok(t) = std::fs::read_to_string("/proc/self/smaps_rollup") {
            for l in t.lines().filter(|l| l.starts_with("Rss") || l.starts_with("AnonHuge") || l.starts_with("Anonymous")) {
                eprintln!("{l}");
            }
        }
    }
    // every output file is written and closed: end here, without destructors. The kernel frees the memory, about
    // 1 GB per 20M events and most of it in huge pages, in a few ms; no input file is mapped.
    unsafe { libc::_exit(0) }
}

/// An exact map from the IDs of a small clean file to a number: packed IDs in an open-addressing table on their
/// key, any other ID in a byte-string map.
struct IdMap {
    slots: Vec<(u64, u16)>, // key 0 = empty
    other: BytesMap<u16>,
}

impl IdMap {
    fn new(ids: &[(Bytes, u16)]) -> IdMap {
        let cap = (ids.len() * 2).next_power_of_two().max(16);
        let mut m = IdMap { slots: vec![(0, 0); cap], other: BytesMap::default() };
        for &(id, v) in ids {
            match pack(id) {
                0 => {
                    m.other.insert(id, v);
                }
                k => {
                    let mut i = mix(k) as usize & (cap - 1);
                    while m.slots[i].0 != 0 {
                        i = (i + 1) & (cap - 1);
                    }
                    m.slots[i] = (k, v);
                }
            }
        }
        m
    }

    #[inline(always)]
    fn packed(&self, k: u64) -> Option<u16> {
        let mask = self.slots.len() - 1;
        let mut i = mix(k) as usize & mask;
        loop {
            // SAFETY: i <= mask
            let (key, v) = unsafe { *self.slots.get_unchecked(i) };
            if key == k {
                return Some(v);
            }
            if key == 0 {
                return None;
            }
            i = (i + 1) & mask;
        }
    }

    #[inline]
    fn get(&self, id: &[u8]) -> Option<u16> {
        match pack(id) {
            0 => self.other.get(id).copied(),
            k => self.packed(k),
        }
    }

    /// `get` of the `len` bytes at `p` (16 bytes must be readable).
    #[inline(always)]
    unsafe fn get_at(&self, p: *const u8, len: usize) -> Option<u16> {
        if len <= 10 {
            let k = pack16(load16(p, len), len);
            if k != 0 {
                return self.packed(k);
            }
        }
        self.other.get(std::slice::from_raw_parts(p, len)).copied()
    }
}

struct Small {
    products: IdMap,
    campaigns: IdMap,
    cat_names: Vec<Bytes>,
    cat_rank: Vec<u32>,
    unknown_cat: u16,
    camp_ids: Vec<Bytes>,
    camp_channel: Vec<Bytes>,
}

/// products.csv and campaigns.csv.
fn small_tables(data: &Path) -> Small {
    // products -> categories; "unknown" is its own category, never one of the file's
    let mut cat_names: Vec<Bytes> = Vec::new();
    let mut cat_idx: BytesMap<u16> = BytesMap::default();
    let mut products: Vec<(Bytes, u16)> = Vec::new();
    let mut seen: BytesMap<()> = BytesMap::default();
    for (p, c) in small_table(&data.join("products.csv")) {
        let n = cat_names.len() as u16;
        let ci = *cat_idx.entry(c).or_insert_with(|| {
            cat_names.push(c);
            n
        });
        products.push((p, ci));
    }
    // product_id is unique per rules.md; were one repeated, its last line would count
    products.reverse();
    products.retain(|&(p, _)| seen.insert(p, ()).is_none());
    let unknown_cat = cat_names.len() as u16;
    cat_names.push(b"unknown");
    if cat_names.len() > 256 {
        die("over 255 categories"); // ERec::pay keeps the category in 8 bits
    }
    let mut order: Vec<usize> = (0..cat_names.len()).collect();
    order.sort_by(|&a, &b| cat_names[a].cmp(cat_names[b]));
    let mut cat_rank = vec![0u32; cat_names.len()];
    for (r, &i) in order.iter().enumerate() {
        cat_rank[i] = r as u32;
    }

    // campaigns: a repeated campaign_id keeps its first line
    let mut camp_ids: Vec<Bytes> = Vec::new();
    let mut camp_channel: Vec<Bytes> = Vec::new();
    let mut campaigns: Vec<(Bytes, u16)> = Vec::new();
    seen.clear();
    for (c, ch) in small_table(&data.join("campaigns.csv")) {
        if seen.insert(c, ()).is_some() {
            continue;
        }
        if camp_ids.len() >= u16::MAX as usize {
            die("over 65534 campaigns");
        }
        campaigns.push((c, camp_ids.len() as u16));
        camp_ids.push(c);
        camp_channel.push(ch);
    }
    Small { products: IdMap::new(&products), campaigns: IdMap::new(&campaigns), cat_names, cat_rank, unknown_cat, camp_ids, camp_channel }
}

/// rules.md "Linked IDs": every link line joins its two IDs in the ID table. Returns the number of lines joined.
fn build_links(ids: &IdTable, names: &Interner, files: &[InFile], threads: usize) -> usize {
    let mut work = Vec::new();
    chunks_of(0, files, 128 << 10, &mut work);
    // A. parse, add both IDs, join them
    //    in three steps HALF lines apart, so the cache lines each step needs are on their way before it:
    //    keys -> (aim) -> slots -> (aim at what they hang under) -> union
    const HALF: usize = RING / 2;
    let parts = par(
        threads,
        work.len(),
        || (Vec::<u8>::new(), 0usize, 0usize),
        |(buf, lines, ins), i| {
            let (_, fi, s, e) = work[i];
            let (len, start, end) = read_span(&files[fi], s, e, buf);
            // SAFETY: the fields live in `buf`, which outlives this call; nothing keeps them
            let text: Bytes = unsafe { std::mem::transmute(&buf[..len]) };
            let mut keys = [(0u64, 0u64, 0u32, 0u32); HALF];
            let mut ends = [(0u32, 0u32); HALF];
            let mut n = 0usize;
            let mut step = |n: usize, new: Option<(u64, u64)>, ins: &mut usize| {
                let i = n % HALF;
                if n >= 2 * HALF {
                    let (x, y) = ends[i];
                    if x != y {
                        ids.union(names, x, y);
                    }
                }
                if n >= HALF {
                    let (ka, kb, ha, hb) = keys[i];
                    if ka != 0 {
                        let (x, y) = (ids.slot_from(ha, ka, ins), ids.slot_from(hb, kb, ins));
                        for z in [x, y] {
                            let u = ids.slots[z as usize].up.load(Relaxed);
                            if u != 0 {
                                prefetch(&ids.slots[u as usize - 1]);
                            }
                        }
                        ends[i] = (x, y);
                    } else {
                        ends[i] = (0, 0);
                    }
                }
                keys[i] = match new {
                    Some((ka, kb)) => (ka, kb, ids.aim(ka), ids.aim(kb)),
                    None => (0, 0, 0, 0),
                };
            };
            for_records::<2>(text, start, end, |[a, b]| {
                if a.is_empty() || b.is_empty() {
                    return;
                }
                step(n, Some((names.key(a), names.key(b))), ins);
                n += 1;
                *lines += 1;
            });
            for _ in 0..2 * HALF {
                step(n, None, ins);
                n += 1;
            }
        },
    );
    // B. flatten, one pass over the table: every linked ID points straight at its root
    const SPAN: usize = 1 << 15;
    const AHEAD: usize = 24;
    let n = ids.slots.len();
    par(
        threads,
        n.div_ceil(SPAN),
        || (),
        |_, t| {
            for i in t * SPAN..((t + 1) * SPAN).min(n) {
                if i + AHEAD < n {
                    let u = ids.slots[i + AHEAD].up.load(Relaxed);
                    if u != 0 {
                        prefetch(&ids.slots[u as usize - 1]);
                    }
                }
                if ids.slots[i].up.load(Relaxed) != 0 {
                    // i hangs under something, so its root is another slot
                    ids.slots[i].up.store(ids.find(i as u32) + 1, Relaxed);
                }
            }
        },
    );
    parts.iter().map(|p| p.1).sum()
}

/// rules.md "Events" 1, 3, 4 for one line given as its 7 fields: the generic path of `fast_event`.
fn event_fields(cx: &Ctx, [eid, uid, ts, ingest, etype, pid, price]: [Bytes; 7], st: &mut Shard) {
    if eid.is_empty() {
        return;
    }
    let Some(ingest) = parse_ts(ingest) else { return }; // always valid F1 per rules.md
    let ekey = cx.names.key(eid);
    let valid = || -> Option<(i64, u32)> {
        if uid.is_empty() || pid.is_empty() {
            return None;
        }
        let t = parse_ts(ts)?;
        if t < cx.ws || t >= cx.we || ingest - t > DAY {
            return None;
        }
        let pay = match etype {
            b"view" | b"cart" => 0,
            b"purchase" => (cx.products.get(pid).unwrap_or(cx.unknown_cat) as u32) << 24 | parse_price(price)?,
            _ => return None,
        };
        Some((t, pay))
    };
    match valid() {
        Some((t, pay)) => st.keep(cx, ekey, ingest, cx.names.key(uid), t, pay),
        None => st.log(cx, ekey, ingest, 0),
    }
}

/// `split::<7>` + `event_fields` for the line of `e` (< 128) bytes at `line`, given its comma and quote bits;
/// line[e] is its `\n` and line[..144] is readable.
#[inline(always)]
unsafe fn fast_event(cx: &Ctx, line: *const u8, e: usize, c: u128, q: u128, st: &mut Shard) {
    let Some(f) = fast_split::<7>(line, e, c, q) else { return };
    let (ep, el) = (line.add(f[0].0), f[0].1);
    if el == 0 {
        return;
    }
    let Some(ingest) = ts_memo(line.add(f[3].0), f[3].1, &mut st.days) else { return };
    let ekey = cx.names.key_at(ep, el);
    let (ul, pl) = (f[1].1, f[5].1);
    let ts = match ts_memo(line.add(f[2].0), f[2].1, &mut st.days) {
        Some(t) if ul != 0 && pl != 0 && t >= cx.ws && t < cx.we && ingest - t <= DAY => t,
        _ => return st.log(cx, ekey, ingest, 0),
    };
    let (kp, kl) = (line.add(f[4].0), f[4].1);
    let w = (kp as *const u64).read_unaligned();
    let pay = match kl {
        4 if w as u32 == u32::from_le_bytes(*b"view") || w as u32 == u32::from_le_bytes(*b"cart") => 0,
        8 if w == u64::from_le_bytes(*b"purchase") => {
            let Some(p) = parse_price(std::slice::from_raw_parts(line.add(f[6].0), f[6].1)) else { return st.log(cx, ekey, ingest, 0) };
            let cat = cx.products.get_at(line.add(f[5].0), pl).unwrap_or(cx.unknown_cat);
            (cat as u32) << 24 | p
        }
        _ => return st.log(cx, ekey, ingest, 0),
    };
    let ukey = cx.names.key_at(line.add(f[1].0), ul);
    st.keep(cx, ekey, ingest, ukey, ts, pay);
}

/// Parses the event lines that start in [start, end) of the `len` bytes at `buf` (see `scan_lines`) into `st`.
unsafe fn parse_events(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    scan_lines(
        buf,
        start,
        end,
        len,
        st,
        |st, line, e, c, q| fast_event(cx, line, e, c, q, st),
        |st, l| {
            if let Some(f) = split::<7>(l) {
                event_fields(cx, f, st);
            }
        },
    );
    st.flush(cx, true);
}

/// rules.md "Clicks" 1, 3 for one line given as its 4 fields: the generic path of `fast_click`.
fn click_fields(cx: &Ctx, [cid, uid, camp, ts]: [Bytes; 4], st: &mut Shard) {
    if cid.is_empty() || uid.is_empty() {
        return;
    }
    let Some(t) = parse_ts(ts) else { return };
    if t < cx.base || t >= cx.we {
        return;
    }
    let Some(camp) = cx.campaigns.get(camp) else { return };
    st.click(cx, cx.names.key(cid), cx.names.key(uid), t, camp);
}

#[inline(always)]
unsafe fn fast_click(cx: &Ctx, line: *const u8, e: usize, c: u128, q: u128, st: &mut Shard) {
    let Some(f) = fast_split::<4>(line, e, c, q) else { return };
    if f[0].1 == 0 || f[1].1 == 0 {
        return;
    }
    let t = match ts_memo(line.add(f[3].0), f[3].1, &mut st.days) {
        Some(t) if t >= cx.base && t < cx.we => t,
        _ => return,
    };
    let Some(camp) = cx.campaigns.get_at(line.add(f[2].0), f[2].1) else { return };
    st.click(cx, cx.names.key_at(line.add(f[0].0), f[0].1), cx.names.key_at(line.add(f[1].0), f[1].1), t, camp);
}

unsafe fn parse_clicks(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    scan_lines(
        buf,
        start,
        end,
        len,
        st,
        |st, line, e, c, q| fast_click(cx, line, e, c, q, st),
        |st, l| {
            if let Some(f) = split::<4>(l) {
                click_fields(cx, f, st);
            }
        },
    );
    st.flush(cx, false);
}

/// rules.md "Events" 2: per event-key bucket, among the copies of one event_id the smallest ingest_ts is kept and
/// every other valid copy is marked dropped. Returns the number of filter candidates.
///
/// The filter is one 64-bit word per two events: 32 "seen" bits (even) each with a "twice" bit (odd). A key sets
/// two seen bits of its word; a key that finds both set sets their twice bits. All copies of a key meet the same
/// bits, so every copy of a repeated key has both twice bits set at the end; a few other keys do too.
fn mark_duplicates(shards: &[Shard], buckets: usize, threads: usize) -> usize {
    par(
        threads,
        buckets,
        || (Vec::<u64>::new(), Vec::<(u64, i64, u64)>::new(), 0usize),
        |(filter, cand, total), b| {
            let n: usize = shards.iter().map(|s| s.lg[b].len(LG_BLOCK)).sum();
            let words = (n / 2).next_power_of_two().max(64);
            filter.clear();
            filter.resize(words, 0);
            let at = |k: u64| {
                let h = mix(k);
                ((h >> 10) as usize & (words - 1), 1u64 << (2 * (h & 31)) | 1u64 << (2 * ((h >> 5) & 31)))
            };
            for s in shards {
                for blk in s.lg[b].slices(LG_BLOCK) {
                    for r in blk {
                        let (w, m) = at(r.key);
                        let f = &mut filter[w];
                        *f |= if *f & m == m { m << 1 } else { m };
                    }
                }
            }
            cand.clear();
            for s in shards {
                for blk in s.lg[b].slices(LG_BLOCK) {
                    for r in blk {
                        let (w, m) = at(r.key);
                        if (filter[w] >> 1) & m == m {
                            cand.push((r.key, r.ingest, r.loc));
                        }
                    }
                }
            }
            *total += cand.len();
            cand.sort_unstable_by_key(|c| c.0);
            let mut i = 0;
            while i < cand.len() {
                // the copies of one key: the one with the smallest ingest_ts is kept
                let mut j = i + 1;
                let mut keep = i;
                while j < cand.len() && cand[j].0 == cand[i].0 {
                    if cand[j].1 < cand[keep].1 {
                        keep = j;
                    }
                    j += 1;
                }
                for k in i..j {
                    if k != keep && cand[k].2 != 0 {
                        // SAFETY: the address of an ERec in a live arena; no other thread touches this record now
                        unsafe { (*(cand[k].2 as *mut ERec)).person = DROPPED };
                    }
                }
                i = j;
            }
        },
    )
    .iter()
    .map(|x| x.2)
    .sum()
}

#[derive(Clone, Copy, Default)]
struct Cred {
    sessions: u32,
    purchasing: u32,
    revenue: i64,
}

/// A click of the person at hand: `k` = ts << 32 | campaign.
#[derive(Clone, Copy)]
struct Ck {
    k: u64,
    cid: u64,
}

struct Sess {
    start: i64, // ms after `Ctx::base`
    d: usize,
    purchased: bool,
    revenue: i64,
}

/// One sessions thread: its results, and scratch reused across partitions.
struct Work {
    rows: Vec<UserRow>,
    cred: Vec<Cred>,
    assist: Vec<u32>,
    stamp: Vec<bool>,
    iv: Vec<(i64, i64)>,
    seen: Vec<u32>,
    sctr: u32,
    start: Vec<u32>,
    fill: Vec<u32>,
    cstart: Vec<u32>,
    cfill: Vec<u32>,
    ev: Vec<u64>,
    tmp: Vec<u64>,
    ck: Vec<Ck>,
    sessions: Vec<Sess>,
    day_groups: Vec<(usize, usize)>, // (d, first session index)
    cats: Vec<(u16, i64)>,
    touched: Vec<usize>,
}

impl Work {
    fn new(cx: &Ctx) -> Work {
        let cells = (cx.ncamp + 1) * cx.ndays;
        let local = (1usize << cx.pshift) + 1;
        Work {
            rows: Vec::new(),
            cred: vec![Cred::default(); cells],
            assist: vec![0; cells],
            stamp: vec![false; cx.ncamp + 1],
            iv: vec![(0, 0); cx.ncamp + 1],
            seen: vec![0; cx.ncamp + 1],
            sctr: 0,
            start: vec![0; local],
            fill: vec![0; local],
            cstart: vec![0; local],
            cfill: vec![0; local],
            ev: Vec::new(),
            tmp: Vec::new(),
            ck: Vec::new(),
            sessions: Vec::new(),
            day_groups: Vec::new(),
            cats: Vec::new(),
            touched: Vec::new(),
        }
    }
}

/// Sorts `ts << 32 | pay` values by ts.
fn sort_events(v: &mut [u64], tmp: &mut Vec<u64>) {
    if v.len() < 1024 {
        v.sort_unstable();
        return;
    }
    // LSD radix on the 32 ts bits, 11 bits a pass
    let n = v.len();
    tmp.clear();
    tmp.resize(n, 0);
    let mut hist = vec![[0u32; 2048]; 3];
    for &x in v.iter() {
        let t = x >> 32;
        hist[0][(t & 2047) as usize] += 1;
        hist[1][((t >> 11) & 2047) as usize] += 1;
        hist[2][(t >> 22) as usize] += 1;
    }
    for h in hist.iter_mut() {
        let mut sum = 0;
        for c in h.iter_mut() {
            let k = *c;
            *c = sum;
            sum += k;
        }
    }
    for &x in v.iter() {
        let d = ((x >> 32) & 2047) as usize;
        tmp[hist[0][d] as usize] = x;
        hist[0][d] += 1;
    }
    for &x in tmp.iter() {
        let d = ((x >> 43) & 2047) as usize;
        v[hist[1][d] as usize] = x;
        hist[1][d] += 1;
    }
    for &x in v.iter() {
        let d = (x >> 54) as usize;
        tmp[hist[2][d] as usize] = x;
        hist[2][d] += 1;
    }
    v.copy_from_slice(tmp);
}

/// rules.md "Events" 5-7 and "Clicks" 4-7 for person partition `p`: the people whose slot number starts with p.
fn partition_work(cx: &Ctx, shards: &[Shard], p: usize, w: &mut Work) {
    let local = 1usize << cx.pshift;
    let lmask = (local - 1) as u32;

    // events: counting sort by person (the low bits of the slot number)
    w.start[..=local].fill(0);
    for s in shards {
        for blk in s.ev[p].slices(EV_BLOCK) {
            for r in blk {
                if r.person != DROPPED {
                    w.start[(r.person & lmask) as usize + 1] += 1;
                }
            }
        }
    }
    let mut sum = 0;
    for l in 0..local {
        sum += w.start[l + 1];
        w.start[l + 1] = sum;
        w.fill[l] = w.start[l];
    }
    let n = sum as usize;
    if n == 0 {
        return;
    }
    if w.ev.len() < n {
        w.ev.resize(n, 0);
    }
    for s in shards {
        for blk in s.ev[p].slices(EV_BLOCK) {
            for r in blk {
                if r.person != DROPPED {
                    let f = &mut w.fill[(r.person & lmask) as usize];
                    w.ev[*f as usize] = (r.ts as u64) << 32 | r.pay as u64;
                    *f += 1;
                }
            }
        }
    }

    // clicks of the people with events, the same way
    w.cstart[..=local].fill(0);
    for s in shards {
        for blk in s.ck[p].slices(CK_BLOCK) {
            for r in blk {
                let l = (r.person & lmask) as usize;
                w.cstart[l + 1] += (w.start[l + 1] != w.start[l]) as u32;
            }
        }
    }
    let mut sum = 0;
    for l in 0..local {
        sum += w.cstart[l + 1];
        w.cstart[l + 1] = sum;
        w.cfill[l] = w.cstart[l];
    }
    if w.ck.len() < sum as usize {
        w.ck.resize(sum as usize, Ck { k: 0, cid: 0 });
    }
    for s in shards {
        for blk in s.ck[p].slices(CK_BLOCK) {
            for r in blk {
                let l = (r.person & lmask) as usize;
                if w.start[l + 1] != w.start[l] {
                    let f = &mut w.cfill[l];
                    w.ck[*f as usize] = Ck { k: (r.ts as u64) << 32 | r.camp as u64, cid: r.cid };
                    *f += 1;
                }
            }
        }
    }

    let mut ev = std::mem::take(&mut w.ev);
    let mut ck = std::mem::take(&mut w.ck);
    for l in 0..local {
        let (a, b) = (w.start[l] as usize, w.start[l + 1] as usize);
        if a == b {
            continue;
        }
        let name = cx.ids.slots[(p << cx.pshift) | l].key.load(Relaxed);
        let (ca, cb) = (w.cstart[l] as usize, w.cstart[l + 1] as usize);
        sort_events(&mut ev[a..b], &mut w.tmp);
        ck[ca..cb].sort_unstable_by_key(|c| c.k);
        person_work(cx, name, &ev[a..b], &ck[ca..cb], w);
    }
    w.ev = ev;
    w.ck = ck;
}

/// One person: `evs` = their surviving events as `ts << 32 | pay` by ts, `cs` = their valid clicks by ts.
fn person_work(cx: &Ctx, name: u64, evs: &[u64], cs: &[Ck], w: &mut Work) {
    let width = cx.ndays;
    let organic = cx.ncamp;
    let Work { rows, cred, assist, stamp, iv, seen, sctr, sessions, day_groups, cats, touched, .. } = w;

    // sessions + user_daily rows
    sessions.clear();
    day_groups.clear();
    let mut row: Option<UserRow> = None;
    let mut prev = i64::MIN / 2;
    for &e in evs {
        let ts = (e >> 32) as i64;
        if ts - prev > GAP {
            let d = ((ts + cx.base).div_euclid(DAY) - cx.day0) as usize;
            sessions.push(Sess { start: ts, d, purchased: false, revenue: 0 });
            let day = (cx.day0 + d as i64) as i32;
            if row.as_ref().map(|r| r.day != day).unwrap_or(true) {
                if let Some(r) = row.take() {
                    rows.push(finish_row(r, cats, &cx.cat_rank));
                }
                row = Some(UserRow { name, day, sessions: 0, events: 0, revenue: 0, top: NO_CAT });
                day_groups.push((d, sessions.len() - 1));
            }
            row.as_mut().unwrap().sessions += 1;
        }
        prev = ts;
        let r = row.as_mut().unwrap();
        r.events += 1;
        let pay = e as u32;
        if pay != 0 {
            let (price, cat) = ((pay & 0xff_ffff) as i64, (pay >> 24) as u16);
            let s = sessions.last_mut().unwrap();
            s.purchased = true;
            s.revenue += price;
            r.revenue += price;
            match cats.iter_mut().find(|c| c.0 == cat) {
                Some(c) => c.1 += price,
                None => cats.push((cat, price)),
            }
        }
    }
    if let Some(r) = row.take() {
        rows.push(finish_row(r, cats, &cx.cat_rank));
    }

    let cts = |c: &Ck| (c.k >> 32) as i64;
    let ccamp = |c: &Ck| (c.k & 0xffff_ffff) as usize;
    let mut credit = |camp: usize, s: &Sess| {
        let a = &mut cred[camp * width + s.d];
        a.sessions += 1;
        a.purchasing += s.purchased as u32;
        a.revenue += s.revenue;
    };
    // the campaign of the latest click among cs[lo..hi] (all at or before the session start); tie -> smallest click_id
    let latest = |lo: usize, hi: usize| -> usize {
        let t = cts(&cs[hi - 1]);
        let mut best = &cs[hi - 1];
        let mut k = hi - 1;
        while k > lo && cts(&cs[k - 1]) == t {
            k -= 1;
            if cx.names.less(cs[k].cid, best.cid) {
                best = &cs[k];
            }
        }
        ccamp(best)
    };
    let (nc, ns) = (cs.len(), sessions.len());
    if nc == 0 {
        for s in sessions.iter() {
            credit(organic, s);
        }
    } else if nc * ns <= 16384 {
        // few clicks or few sessions: per session, look at the clicks of its 7-day window [start - 7d, start]
        // (a contiguous run of cs) for the credit and for the distinct assisting campaigns
        let (mut lo, mut hi) = (0, 0);
        for s in sessions.iter() {
            while hi < nc && cts(&cs[hi]) <= s.start {
                hi += 1;
            }
            while lo < hi && s.start - cts(&cs[lo]) > WEEK {
                lo += 1;
            }
            let mut camp = organic;
            if hi > lo {
                camp = latest(lo, hi);
                // assists: each distinct campaign of the window once
                if *sctr == u32::MAX {
                    seen.fill(0);
                    *sctr = 0;
                }
                *sctr += 1;
                for c in &cs[lo..hi] {
                    let kk = ccamp(c);
                    if seen[kk] != *sctr {
                        seen[kk] = *sctr;
                        assist[kk * width + s.d] += 1;
                    }
                }
            }
            credit(camp, s);
        }
    } else {
        // credit: latest click with start - 7d <= ts <= start
        let mut q = 0;
        for s in sessions.iter() {
            while q < nc && cts(&cs[q]) <= s.start {
                q += 1;
            }
            let mut camp = organic;
            if q > 0 && s.start - cts(&cs[q - 1]) <= WEEK {
                camp = latest(0, q);
            }
            credit(camp, s);
        }
        // assists: a click at t assists sessions starting in [t, t + 7d]; merge per campaign
        let flush = |assist: &mut [u32], k: usize, lo: i64, hi: i64, sessions: &[Sess], day_groups: &[(usize, usize)]| {
            let a = sessions.partition_point(|s| s.start < lo);
            let b = sessions.partition_point(|s| s.start <= hi);
            if a >= b {
                return;
            }
            for (g, &(d, first)) in day_groups.iter().enumerate() {
                let end = day_groups.get(g + 1).map(|x| x.1).unwrap_or(sessions.len());
                let n = end.min(b) as i64 - first.max(a) as i64;
                if n > 0 {
                    assist[k * width + d] += n as u32;
                }
            }
        };
        touched.clear();
        for c in cs {
            let k = ccamp(c);
            let t = cts(c);
            if !stamp[k] {
                stamp[k] = true;
                iv[k] = (t, t + WEEK);
                touched.push(k);
            } else if t <= iv[k].1 {
                iv[k].1 = t + WEEK;
            } else {
                let (lo, hi) = iv[k];
                flush(assist, k, lo, hi, sessions, day_groups);
                iv[k] = (t, t + WEEK);
            }
        }
        for &k in touched.iter() {
            let (lo, hi) = iv[k];
            flush(assist, k, lo, hi, sessions, day_groups);
            stamp[k] = false;
        }
    }
}

fn finish_row(mut r: UserRow, cats: &mut Vec<(u16, i64)>, rank: &[u32]) -> UserRow {
    r.top = cats
        .iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(rank[b.0 as usize].cmp(&rank[a.0 as usize])))
        .map_or(NO_CAT, |c| c.0);
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

fn write_users(path: &Path, rows: &[UserRow], cat_names: &[Bytes], names: &Interner) -> std::io::Result<()> {
    let n = rows.len();
    write_parquet(path, &USER_COLS, n, |c, b| match c {
        0 => {
            b.reserve(n * 16);
            let mut id = Vec::with_capacity(16);
            for r in rows {
                id.clear();
                names.bytes(r.name, &mut id);
                b.extend_from_slice(&(id.len() as u32).to_le_bytes());
                b.extend_from_slice(&id);
            }
        }
        1 => {
            b.reserve(n * 4);
            rows.iter().for_each(|r| b.extend_from_slice(&r.day.to_le_bytes()))
        }
        2 => plain_i64(b, rows.iter().map(|r| r.sessions as i64)),
        3 => plain_i64(b, rows.iter().map(|r| r.events as i64)),
        4 => plain_i64(b, rows.iter().map(|r| r.revenue)),
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
            for (i, r) in rows.iter().enumerate() {
                run[at + i / 8] |= ((r.top != NO_CAT) as u8) << (i % 8);
            }
            b.extend_from_slice(&(run.len() as u32).to_le_bytes());
            b.extend_from_slice(&run);
            plain_bytes(b, rows.iter().filter(|r| r.top != NO_CAT).map(|r| cat_names[r.top as usize]))
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
        let products = IdMap::new(&[(&b"p1"[..], 0u16), (&b"p2"[..], 1u16)]);
        let ws = parse_ts(b"2028-02-20T00:00:00Z").unwrap();
        Ctx {
            ws,
            we: parse_ts(b"2028-03-05T00:00:00Z").unwrap(),
            base: ws - WEEK,
            day0: 0,
            ndays: 14,
            ncamp: 0,
            products,
            campaigns: IdMap::new(&[]),
            unknown_cat: 2,
            cat_rank: vec![0, 1, 2],
            ids: IdTable::new(16),
            names: Interner::new(),
            pshift: 12,
            qshift: 60,
        }
    }

    /// (events as (person, ts, pay), dedup records as (key, ingest, has a record)), each sorted
    type Flat = (Vec<(u32, u32, u32)>, Vec<(u64, i64, bool)>);
    fn flat(st: &Shard) -> Flat {
        let mut ev: Vec<_> = st.ev.iter().flat_map(|s| s.slices(EV_BLOCK)).flatten().map(|r| (r.person, r.ts, r.pay)).collect();
        let mut lg: Vec<_> = st.lg.iter().flat_map(|s| s.slices(LG_BLOCK)).flatten().map(|r| (r.key, r.ingest, r.loc != 0)).collect();
        ev.sort();
        lg.sort();
        (ev, lg)
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
            let len = text.len();
            let mut padded = text.clone().into_bytes();
            padded.extend_from_slice(&[b'\n'; PAD]);
            let file: Bytes = Box::leak(text.into_bytes().into_boxed_slice());
            // the same random chunking for both
            let mut cuts = vec![0, len];
            for _ in 0..r.below(4) {
                cuts.push(r.below(len + 1));
            }
            cuts.sort();
            let (mut a, mut b) = (Shard::new(16, 16), Shard::new(16, 16));
            for w in cuts.windows(2) {
                unsafe { parse_events(&c, padded.as_ptr(), w[0], w[1], len, &mut a) };
                for_records::<7>(file, w[0], w[1], |f| event_fields(&c, f, &mut b));
                b.flush(&c, true);
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
        let mut days = [NO_DAY; 32];
        let mut hits = 0;
        for _ in 0..200_000 {
            let s = ts_value(&mut r);
            let mut buf = s.clone().into_bytes();
            buf.extend_from_slice(&[b'7'; 32]);
            let want = parse_ts(s.as_bytes());
            assert_eq!(unsafe { ts_at(buf.as_ptr(), s.len()) }, want, "{s:?}");
            let before = days;
            assert_eq!(unsafe { ts_memo(buf.as_ptr(), s.len(), &mut days) }, want, "memo {s:?}");
            hits += (want.is_some() && s.len() >= 20 && before.iter().zip(&days).all(|(x, y)| x.a == y.a && x.b == y.b)) as usize;
        }
        assert!(hits > 20_000, "the memo was hit only {hits} times");
    }

    #[test]
    fn pack16_matches_pack_and_keeps_order() {
        let mut r = Rng(42);
        let alphabet: Vec<u8> = (0u8..=255).filter(|&c| PACK[c as usize] != 0).collect();
        assert_eq!(alphabet.len(), 63);
        let names = Interner::new();
        let mut prev: Option<(Vec<u8>, u64)> = None;
        for i in 0..200_000 {
            let len = r.below(13);
            let mut id: Vec<u8> = (0..len).map(|_| alphabet[r.below(63)]).collect();
            if i % 5 == 0 && len > 0 {
                id[r.below(len)] = r.next() as u8; // usually a byte that does not pack
            }
            let mut buf = id.clone();
            buf.extend((0..32).map(|_| r.next() as u8));
            let k = pack(&id);
            assert_eq!(pack16(unsafe { load16(buf.as_ptr(), len) }, len), k, "{id:?}");
            if k == 0 {
                continue;
            }
            let mut back = Vec::new();
            names.bytes(k, &mut back);
            assert_eq!(back, id);
            if let Some((pid, pk)) = &prev {
                assert_eq!(pid.cmp(&id), pk.cmp(&k), "{pid:?} {id:?}");
            }
            prev = Some((id, k));
        }
        // IDs that do not pack: exact, and ordered by their bytes against packed ones
        let (a, b, c) = (names.key(b"user-with-dash"), names.key(b"user-with-dash"), names.key(b" u1"));
        assert!(a == b && a != c && a & INTERNED != 0);
        assert!(names.less(c, pack(b"u1")) && names.less(pack(b"u1"), a) && !names.less(a, a));
    }

    #[test]
    fn links_name_people_by_their_smallest_id() {
        let c = ctx();
        let mut ins = 0;
        let ids: Vec<u32> = [&b"u5"[..], b"u3", b"u9", b"U7", b"u1", b"zz", b"a-long-user-id"].iter().map(|b| c.ids.slot(c.names.key(b), &mut ins)).collect();
        for (a, b) in [(0, 1), (2, 1), (3, 2), (5, 6), (4, 4)] {
            if a != b {
                c.ids.union(&c.names, ids[a], ids[b]);
            }
        }
        let root = |i: usize| c.ids.find(ids[i]);
        // U7 sorts before every lower-case ID
        assert!(root(0) == ids[3] && root(1) == ids[3] && root(2) == ids[3] && root(3) == ids[3]);
        assert!(root(4) == ids[4]);
        assert!(root(5) == ids[6] && root(6) == ids[6]);
    }
}
