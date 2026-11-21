//! Shopping Sessions pipeline. rules.md is the spec; each phase names the rule it implements.
//!
//! One shuffle, no pointers into the input. Every ID (event, user, click) becomes an exact 64-bit key: an ID of up
//! to 10 bytes from [0-9A-Z_a-z] packs losslessly, any other ID is interned. The input is read with pread into a
//! small per-thread buffer, so no input page is ever mapped; with a cold page cache a few extra threads ask the
//! kernel for the whole input up front (`read_ahead`).
//!
//! Phases:
//!   1. links  -> every link ID goes into one lock-free ID table; a lock-free union-find over its slots hangs the
//!      larger name under the smaller, so the root of a person is their name (byte-wise smallest ID). The table
//!      never grows: when it fills up the program restarts itself with a larger one (`restart_bigger`)
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

/// A hash to take the top bits of (a table slot, a bucket): one multiply, where `mix` is two and five more steps.
#[inline(always)]
fn spread(x: u64) -> u64 {
    x.wrapping_mul(0x9E37_79B9_7F4A_7C15)
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
        // A byte is in the alphabet when its two nibbles agree on a class (one bit each): high nibble 3 with low
        // 0..=9 (bit 0), 4 or 6 with 1..=F (bit 1), 5 with 0..=A or F (bit 2), 7 with 0..=A (bit 3). A byte of
        // 0x80 or more looks up nothing with its low nibble.
        const LO: [u8; 16] = [13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 2, 2, 2, 2, 6];
        const HI: [u8; 16] = [0, 0, 0, 1, 2, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0];
        // digit = byte - offset: 47 for a digit, 54 for A-Z, 59 for a-z by the high nibble alone, 4 more for `_`
        const OFF: [u8; 16] = [0, 0, 0, 47, 54, 54, 59, 59, 0, 0, 0, 0, 0, 0, 0, 0];
        let tbl = |t: &[u8; 16]| _mm_loadu_si128(t.as_ptr() as *const __m128i);
        let hi = _mm_and_si128(_mm_srli_epi16(c, 4), _mm_set1_epi8(0x0F));
        let class = _mm_and_si128(_mm_shuffle_epi8(tbl(&LO), c), _mm_shuffle_epi8(tbl(&HI), hi));
        let invalid = _mm_movemask_epi8(_mm_cmpeq_epi8(class, _mm_setzero_si128())) as u32;
        if invalid & ((1u32 << len) - 1) != 0 {
            return 0;
        }
        // the zero bytes past `len` have offset 0 and stay 0
        let us = _mm_cmpeq_epi8(c, _mm_set1_epi8(b'_' as i8));
        let off = _mm_add_epi8(_mm_shuffle_epi8(tbl(&OFF), hi), _mm_and_si128(us, _mm_set1_epi8(4)));
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

/// `pack` of two IDs at once: the `la` bytes at `pa` and the `lb` bytes at `pb` (16 bytes readable at each), one in
/// each half of a 256-bit register, so the pair costs little more than one `pack16`.
#[inline(always)]
unsafe fn pack2(pa: *const u8, la: usize, pb: *const u8, lb: usize) -> (u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        debug_assert!(la <= 128 && lb <= 128);
        // the first `len` bytes of an ID's 16 (all 16 from `len` = 16 on): KEEP read from 128 - len
        static KEEP: [u8; 144] = {
            let mut k = [0u8; 144];
            let mut i = 0;
            while i < 128 {
                k[i] = 255;
                i += 1;
            }
            k
        };
        let two = |pa: *const u8, pb: *const u8| _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(pa as *const __m128i)), _mm_loadu_si128(pb as *const __m128i), 1);
        let keep = two(KEEP.as_ptr().add(128 - la), KEEP.as_ptr().add(128 - lb));
        let c = _mm256_and_si256(two(pa, pb), keep);
        // classes and offsets as in `pack16`
        const LO: [u8; 16] = [13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 2, 2, 2, 2, 6];
        const HI: [u8; 16] = [0, 0, 0, 1, 2, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0];
        const OFF: [u8; 16] = [0, 0, 0, 47, 54, 54, 59, 59, 0, 0, 0, 0, 0, 0, 0, 0];
        let tbl = |t: &[u8; 16]| _mm256_broadcastsi128_si256(_mm_loadu_si128(t.as_ptr() as *const __m128i));
        let hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), _mm256_set1_epi8(0x0F));
        // no class from the 11th byte on: an ID of more than 10 bytes does not pack
        const TEN: [u8; 16] = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0];
        let class = _mm256_and_si256(_mm256_and_si256(_mm256_shuffle_epi8(tbl(&LO), c), _mm256_shuffle_epi8(tbl(&HI), hi)), tbl(&TEN));
        // a kept byte without a class is outside the alphabet, or past the 10th
        if _mm256_testz_si256(_mm256_cmpeq_epi8(class, _mm256_setzero_si256()), keep) == 0 {
            // one of them does not pack: each on its own (one of more than 10 bytes as an empty one: 0)
            let (la, lb) = (if la > 10 { 0 } else { la }, if lb > 10 { 0 } else { lb });
            return (pack16(load16(pa, la), la), pack16(load16(pb, lb), lb));
        }
        let us = _mm256_cmpeq_epi8(c, _mm256_set1_epi8(b'_' as i8));
        let off = _mm256_add_epi8(_mm256_shuffle_epi8(tbl(&OFF), hi), _mm256_and_si256(us, _mm256_set1_epi8(4)));
        let d = _mm256_sub_epi8(c, off);
        let w = _mm256_maddubs_epi16(d, _mm256_set1_epi16(0x0140));
        // per ID, three numbers of 24 bits: digits 0..4 in bytes 0..3, digits 4..8 in bytes 4..7, digits 8..10
        // (above 12 zero bits) in bytes 8..11
        let l = _mm256_madd_epi16(w, _mm256_set1_epi32(0x0001_1000));
        // their bytes side by side: bits 4.. are the key (the 4 below are zero bits of the third number)
        const SIDE: [u8; 16] = [9, 10, 4, 5, 6, 0, 1, 2, 128, 128, 128, 128, 128, 128, 128, 128];
        let k = _mm256_srli_epi64(_mm256_shuffle_epi8(l, tbl(&SIDE)), 4);
        return (_mm256_extract_epi64(k, 0) as u64, _mm256_extract_epi64(k, 2) as u64);
    }
    #[allow(unreachable_code)]
    {
        let (la, lb) = (if la > 10 { 0 } else { la }, if lb > 10 { 0 } else { lb });
        (pack16(load16(pa, la), la), pack16(load16(pb, lb), lb))
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

    /// The key of the `len` bytes at `p`, given `pack2`'s answer for them (0 = they do not pack).
    #[inline(always)]
    unsafe fn key_or(&self, packed: u64, p: *const u8, len: usize) -> u64 {
        if packed != 0 {
            return packed;
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

/// Every region `map_anon` has handed out, as (address, bytes): what `release_regions` gives back.
static REGIONS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// `bytes` of zeroed anonymous memory, never unmapped; huge pages where the kernel offers them.
fn map_anon(bytes: usize) -> *mut u8 {
    let p = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
    if p == libc::MAP_FAILED {
        die(&format!("mmap of {bytes} bytes: {}", std::io::Error::last_os_error()));
    }
    #[cfg(target_os = "linux")]
    unsafe {
        libc::madvise(p, bytes, libc::MADV_HUGEPAGE); // a hint: fine if refused
    }
    REGIONS.lock().unwrap_or_else(|_| die("region list poisoned")).push((p as usize, bytes));
    p as *mut u8
}

/// `n` copies of `v` in memory written now. `vec![0; n]` gets untouched pages instead, which map the shared zero
/// page on a first read; the write that follows must then flush that mapping from every CPU running a thread of
/// this process (an IPI to each, per 4 KiB page).
fn filled<T: Clone>(n: usize, v: T) -> Vec<T> {
    let mut x = Vec::with_capacity(n);
    x.resize(n, v);
    x
}

/// Gives the pages of every `map_anon` region back to the kernel, on `threads` threads. Call when nothing reads
/// them any more: they read as zeros afterwards. Whatever is left, the exit frees on its one thread, and the run
/// is not over before that. With huge pages that is a few ms either way; where the kernel gave 4 KiB pages (THP
/// off) it is 36 ms at the exit for 20M events, against 18 ms here.
fn release_regions(threads: usize) {
    const STEP: usize = 8 << 20;
    // The output is written by now, so nothing here may end the run: a list that cannot be had, a thread that
    // cannot start or a refused hint leaves memory to the exit, which only takes longer.
    let Ok(mut list) = REGIONS.lock() else { return };
    let regions = std::mem::take(&mut *list);
    drop(list);
    let mut pieces = Vec::new();
    for (p, n) in regions {
        let mut o = 0;
        while o < n {
            pieces.push((p + o, (n - o).min(STEP)));
            o += STEP;
        }
    }
    let next = AtomicUsize::new(0);
    let work = || loop {
        let i = next.fetch_add(1, Relaxed);
        if i >= pieces.len() {
            break;
        }
        // SAFETY: a piece of a live anonymous mapping that nothing reads any more
        unsafe { libc::madvise(pieces[i].0 as *mut libc::c_void, pieces[i].1, libc::MADV_DONTNEED) };
    };
    std::thread::scope(|sc| {
        for _ in 1..threads {
            let _ = std::thread::Builder::new().spawn_scoped(sc, &work);
        }
        work();
    });
}

/// Set at the start when the kernel never gives huge pages (transparent huge pages switched off).
static SMALL_PAGES: AtomicBool = AtomicBool::new(false);

/// With `SMALL_PAGES`: asks the kernel for the pages of the `bytes` at `p` (page aligned, in a `map_anon` region)
/// in one call, where their first writes would take one page fault per 4 KiB. False when nothing was asked for:
/// huge pages are on (a fault then brings 2 MiB at once, and this call on top was measured to cost, not save: 20M
/// events, cold input, 0.68 -> 0.73 s), or the kernel does not know the call (before Linux 5.14).
fn populate(p: usize, bytes: usize) -> bool {
    if !SMALL_PAGES.load(Relaxed) {
        return false;
    }
    #[cfg(target_os = "linux")]
    // SAFETY: a hint on a live anonymous mapping; its pages stay zero
    unsafe {
        libc::madvise(p as *mut libc::c_void, bytes, libc::MADV_POPULATE_WRITE) == 0
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (p, bytes);
        false
    }
}

/// The address range every arena chunk comes from, reserved once and handed out a chunk at a time. Its size keeps
/// a record's address within 32 bits (`loc_of`): 2^32 - 3 steps of 4 bytes.
const ARENA_SPAN: usize = 1 << 34;
static ARENA_BASE: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new(); // (address, bytes)
static ARENA_USED: AtomicUsize = AtomicUsize::new(0);

/// The start of the arena range (2 MiB aligned) and its size: reserved on the first call, no memory behind it yet.
fn arena_range() -> (usize, usize) {
    *ARENA_BASE.get_or_init(|| {
        const ALIGN: usize = 2 << 20;
        // a smaller range where the full one is refused: enough for 100M events at 4 GiB
        for span in [ARENA_SPAN, ARENA_SPAN / 2, ARENA_SPAN / 4] {
            let p = unsafe {
                libc::mmap(std::ptr::null_mut(), span + ALIGN, libc::PROT_NONE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE, -1, 0)
            };
            if p != libc::MAP_FAILED {
                return ((p as usize + ALIGN - 1) & !(ALIGN - 1), span);
            }
        }
        die(&format!("reserving the record address range: {}", std::io::Error::last_os_error()))
    })
}

/// `bytes` (a multiple of 2 MiB) of zeroed memory from the arena range, never unmapped; huge pages where offered.
fn arena_chunk(bytes: usize) -> *mut u8 {
    let (base, span) = arena_range();
    let off = ARENA_USED.fetch_add(bytes, Relaxed);
    if off + bytes > span {
        die("the records need more memory than the arena range holds");
    }
    let p = (base + off) as *mut libc::c_void;
    if unsafe { libc::mprotect(p, bytes, libc::PROT_READ | libc::PROT_WRITE) } != 0 {
        die(&format!("arena chunk of {bytes} bytes: {}", std::io::Error::last_os_error()));
    }
    #[cfg(target_os = "linux")]
    unsafe {
        libc::madvise(p, bytes, libc::MADV_HUGEPAGE); // a hint: fine if refused
    }
    REGIONS.lock().unwrap_or_else(|_| die("region list poisoned")).push((p as usize, bytes));
    p as *mut u8
}

/// The address of an `ERec` in an arena as a number from 1 (0 = no record); `abase` = `arena_range().0`.
#[inline(always)]
fn loc_of(abase: usize, rec: *mut ERec) -> u32 {
    ((rec as usize - abase) >> 2) as u32 + 1
}

/// The `ERec` of a `loc_of` number. For 0 the address is not a record: fine for a prefetch, never to be read.
#[inline(always)]
fn rec_at(abase: usize, loc: u32) -> *mut ERec {
    abase.wrapping_add((loc as usize) << 2).wrapping_sub(4) as *mut ERec
}

/// A thread's bump allocator for stream blocks.
struct Arena {
    cur: usize,
    end: usize,
    ready: usize, // the chunk's pages below this address have been asked for (`populate`)
}

impl Arena {
    // the unit tests make hundreds of arenas, each with its own chunk of the arena range: small ones there
    const CHUNK: usize = if cfg!(test) { 2 << 20 } else { 32 << 20 };
    const STEP: usize = 2 << 20;

    fn new() -> Arena {
        Arena { cur: 0, end: 0, ready: 0 }
    }

    #[inline]
    fn alloc(&mut self, bytes: usize) -> *mut u8 {
        debug_assert!(bytes % 64 == 0); // blocks stay 64-byte aligned (see `Stream`)
        if self.cur + bytes > self.end {
            let n = bytes.max(Self::CHUNK).next_multiple_of(Self::STEP);
            self.cur = arena_chunk(n) as usize;
            self.end = self.cur + n;
            self.ready = self.cur;
        }
        let p = self.cur;
        self.cur += bytes;
        while self.ready < self.cur {
            let n = (self.end - self.ready).min(Self::STEP);
            populate(self.ready, n);
            self.ready += n;
        }
        p as *mut u8
    }
}

/// Bytes a stream writes at a time: three cache lines, a whole number of records of every size here (12, 16).
const GROUP: usize = 192;

/// An append-only list of `T` in blocks of `per` records; a record never moves.
///
/// Records reach memory a group at a time: a thread appends to some 800 streams at once, and a record stored
/// directly first reads its cache line from memory (read for ownership) and crowds the cache. A stream keeps the
/// records of its current group (block offset a multiple of GROUP: blocks are 64-byte aligned, whole groups
/// long) and writes the full group with non-temporal stores, three whole lines that are never read. A record's
/// address is known when it is pushed; until the push after its group is full, or `finish`, its bytes are only in
/// the stage.
/// The caller fences (`sfence`) before others read, and calls `finish` once the pushing is over.
#[repr(C, align(64))]
struct Stream<T> {
    base: *mut T,                // where the current group belongs in its block; null before the first block
    end: *mut T,
    n: usize,                    // records of the current group, in `stage`; K = full (or no block yet)
    blocks: Vec<*mut T>,
    _pad: [u64; 2],              // the stage on its own three cache lines
    stage: [u64; GROUP / 8],     // 8-byte aligned, as every record here
    _t: std::marker::PhantomData<T>,
}

impl<T> Stream<T> {
    const K: usize = GROUP / std::mem::size_of::<T>();

    fn new() -> Stream<T> {
        const { assert!(GROUP % std::mem::size_of::<T>() == 0 && std::mem::align_of::<T>() <= 8 && !std::mem::needs_drop::<T>()) };
        Stream { base: std::ptr::null_mut(), end: std::ptr::null_mut(), n: Self::K, blocks: Vec::new(), _pad: [0; 2], stage: [0; GROUP / 8], _t: std::marker::PhantomData }
    }

    /// Appends `v`; returns where it will be in memory. A push costs one test, the record and the count: a full
    /// group is written (and a full block followed by a new one) by the push that comes after it.
    #[inline(always)]
    fn push(&mut self, arena: &mut Arena, per: usize, v: T) -> *mut T {
        if self.n == Self::K {
            self.spill(arena, per);
        }
        let n = self.n;
        self.n = n + 1;
        // SAFETY: n < K slots of the stage; the group [base, base + K) is inside the block `grow` took
        unsafe {
            (self.stage.as_mut_ptr() as *mut T).add(n).write(v);
            self.base.add(n)
        }
    }

    /// Makes room in the stage: writes the full group, if there is one, and moves on to the next (in a new block
    /// when this one is full, or there is none yet).
    #[inline(always)]
    fn spill(&mut self, arena: &mut Arena, per: usize) {
        if !self.base.is_null() {
            // SAFETY: the group at base is GROUP bytes at a 64-byte aligned address of the block
            unsafe {
                stream_group(self.base as *mut u8, self.stage.as_ptr() as *const u8);
                self.base = self.base.add(Self::K);
            }
        }
        if self.base == self.end {
            self.grow(arena, per);
        }
        self.n = 0;
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, arena: &mut Arena, per: usize) {
        debug_assert!(per % Self::K == 0);
        let p = arena.alloc(per * std::mem::size_of::<T>()) as *mut T;
        debug_assert!(p as usize % 64 == 0);
        self.blocks.push(p);
        self.base = p;
        self.end = unsafe { p.add(per) };
    }

    /// Writes the records still staged to memory with plain stores. Pushing may go on afterwards.
    fn finish(&mut self) {
        if !self.base.is_null() {
            // SAFETY: the n (<= K) staged records belong at [base, base + n), inside the block
            unsafe { std::ptr::copy_nonoverlapping(self.stage.as_ptr() as *const T, self.base, self.n) };
        }
    }

    /// Records in the last block.
    fn tail(&self, b: *mut T) -> usize {
        (unsafe { self.base.offset_from(b) }) as usize + self.n
    }

    fn len(&self, per: usize) -> usize {
        match self.blocks.last() {
            None => 0,
            Some(&b) => (self.blocks.len() - 1) * per + self.tail(b),
        }
    }

    fn slices(&self, per: usize) -> impl Iterator<Item = &[T]> {
        let n = self.blocks.len();
        self.blocks.iter().enumerate().map(move |(i, &b)| {
            let len = if i + 1 == n { self.tail(b) } else { per };
            unsafe { std::slice::from_raw_parts(b, len) }
        })
    }
}

/// Copies a GROUP of bytes to `dst` (64-byte aligned) with non-temporal stores where available.
#[inline(always)]
unsafe fn stream_group(dst: *mut u8, src: *const u8) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        for i in 0..GROUP / 32 {
            _mm256_stream_si256(dst.add(32 * i) as *mut __m256i, _mm256_loadu_si256(src.add(32 * i) as *const __m256i));
        }
        return;
    }
    #[allow(unreachable_code)]
    std::ptr::copy_nonoverlapping(src, dst, GROUP);
}

/// Orders this thread's non-temporal stores before its later stores.
#[inline(always)]
fn sfence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence()
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
/// `px` = this slot's number xor its person's, once the links are flattened: 0 for a root and for every ID added
/// later, so a lookup turns a slot into a person with one xor and no branch.
#[repr(C)]
struct Slot {
    key: AtomicU64,
    up: AtomicU32,
    px: AtomicU32,
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
    /// lookup would miss the TLB. A write first gets a huge page (where the kernel gives them). Where it gives
    /// none, all the pages are asked for here instead: no lookup or insert then waits for a page fault.
    fn populate(&self, threads: usize) {
        const HUGE: usize = 2 << 20;
        let bytes = self.slots.len() * std::mem::size_of::<Slot>();
        let base = self.slots.as_ptr() as usize;
        par(
            threads,
            bytes.div_ceil(HUGE),
            || (),
            |_, i| {
                if !populate(base + i * HUGE, (bytes - i * HUGE).min(HUGE)) {
                    // a zero stored over a zero: the slot stays empty
                    self.slots[i * HUGE / std::mem::size_of::<Slot>()].up.store(0, Relaxed);
                }
            },
        );
    }

    /// Where the search for `key` starts: the first slot of a cache line (4 slots).
    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        (spread(key) >> (64 - self.bits)) as usize & !3
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

    /// `slot` given `home(key)`. A key sits in the first free slot from its home at the time it was added, and
    /// slots never empty again, so it is in the four slots of its home line unless the line was full by then:
    /// those four are compared without a branch between them (which of them holds the key is a coin toss to the
    /// branch predictor), and only a key not among them takes the scan.
    #[inline(always)]
    fn slot_from(&self, home: u32, key: u64, ins: &mut usize) -> u32 {
        // SAFETY: home is a multiple of 4 below the (power of two, >= 2^16) slot count
        let s = unsafe { self.slots.as_ptr().add(home as usize) };
        let eq = |j: usize| (unsafe { (*s.add(j)).key.load(Relaxed) } == key) as u32;
        let m = eq(0) | eq(1) << 1 | eq(2) << 2 | eq(3) << 3;
        if m != 0 {
            return home + m.trailing_zeros();
        }
        self.slot_scan(home, key, ins)
    }

    /// `slot_from` that does not leave the line at `home`: None when the line is full of other keys, so the key
    /// is (or will be added) further on. The caller asks for the next line (`next_line`) and comes back when
    /// it is in the cache, instead of waiting for it here.
    ///
    /// None is exact: a slot never changes once it holds a key, so four keys read here that are not `key` are
    /// the line's four keys for good. A slot read as empty may have been filled since (by another thread, even
    /// with this very key), so any empty one sends the lookup through `slot_scan`, which takes each slot in turn.
    #[inline(always)]
    fn slot_in_line(&self, home: u32, key: u64, ins: &mut usize) -> Option<u32> {
        // SAFETY: as in `slot_from`
        let s = unsafe { self.slots.as_ptr().add(home as usize) };
        // the four keys compared at once: a slot is 16 bytes, its key the first 8, so the keys are the even
        // 64-bit words of the line. A word read while another thread stores a key there is the key or zero.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            use std::arch::x86_64::*;
            let kv = _mm256_set1_epi64x(key as i64);
            let a = _mm256_cmpeq_epi64(kv, _mm256_loadu_si256(s as *const __m256i));
            let b = _mm256_cmpeq_epi64(kv, _mm256_loadu_si256(s.add(2) as *const __m256i));
            let m = (_mm256_movemask_pd(_mm256_castsi256_pd(a)) | _mm256_movemask_pd(_mm256_castsi256_pd(b)) << 4) as u32 & 0x55;
            if m != 0 {
                return Some(home + m.trailing_zeros() / 2);
            }
        }
        let k = [0, 1, 2, 3].map(|j| unsafe { (*s.add(j)).key.load(Relaxed) });
        let m = (k[0] == key) as u32 | ((k[1] == key) as u32) << 1 | ((k[2] == key) as u32) << 2 | ((k[3] == key) as u32) << 3;
        if m != 0 {
            return Some(home + m.trailing_zeros());
        }
        if (k[0] != 0) & (k[1] != 0) & (k[2] != 0) & (k[3] != 0) {
            return None;
        }
        Some(self.slot_scan(home, key, ins))
    }

    /// The line after the one at `home`, with its cache line on the way.
    #[inline(always)]
    fn next_line(&self, home: u32) -> u32 {
        let h = (home as usize + 4) & (self.slots.len() - 1);
        prefetch(unsafe { self.slots.as_ptr().add(h) });
        h as u32
    }

    /// `slot_from` for a key that is not in its home line (yet): linear probing from `home`, adding the key in the
    /// first free slot.
    #[inline(never)]
    fn slot_scan(&self, home: u32, key: u64, ins: &mut usize) -> u32 {
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
        slot ^ unsafe { self.slots.get_unchecked(slot as usize) }.px.load(Relaxed)
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

/// `masks64` for the 32 bytes at `p`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn masks32(p: *const u8) -> [u64; 3] {
    use std::arch::x86_64::*;
    let a = _mm256_loadu_si256(p as *const __m256i);
    let m = |c: u8| _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, _mm256_set1_epi8(c as i8))) as u32 as u64;
    [m(b','), m(b'"'), m(b'\n')]
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

/// `parse_ts` on the 13 bytes at `p`, F3 of today's length: its first five digits behind three zero bytes, and its
/// last eight.
#[inline(always)]
unsafe fn ts13_at(p: *const u8) -> Option<i64> {
    const Z: u64 = u64::from_le_bytes([b'0'; 8]);
    let a = ((p as *const u64).read_unaligned() << 24) ^ (Z << 24);
    let b = (p.add(5) as *const u64).read_unaligned() ^ Z;
    if (a.wrapping_add(0x7676_7676_7676_7676) | a | b.wrapping_add(0x7676_7676_7676_7676) | b) & HI != 0 {
        return None;
    }
    Some((swar8(a) * 100_000_000 + swar8(b)) as i64)
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
    if len == 13 {
        return ts13_at(p);
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
/// costs two compares and the time of day. A 13-digit F3 goes straight to `ts13_at`.
#[inline(always)]
unsafe fn ts_memo(p: *const u8, len: usize, days: &mut [Day; 64]) -> Option<i64> {
    let at = |i: usize| *p.add(i);
    let f1 = len == 20 && at(19) == b'Z';
    if !f1 {
        if len == 13 {
            return ts13_at(p);
        }
        if !(len == 25 && (at(19) == b'+' || at(19) == b'-') && at(22) == b':') {
            return ts_at(p, len);
        }
    }
    let a = (p as *const u64).read_unaligned();
    let b = (p.add(8) as *const u32).read_unaligned() & 0xFF_FFFF;
    // the slot: the low two bits of the day's tens digit and the low four of its units digit, apart for 01..=31
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    let slot = std::arch::x86_64::_pext_u32(b, 0x0F03);
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    let slot = (b & 3) | ((b >> 6) & 0x3C);
    let day = &mut days[(slot & 63) as usize];
    if day.a != a || day.b != b {
        // a date not seen yet, or a bad one: the full check; a good one is remembered
        let t = ts_at(p, len)?;
        *day = Day { a, b, ms: civil_at(p)?.div_euclid(DAY) * DAY };
        return Some(t);
    }
    // HH:MM:SS as in `civil_at`
    let x = (p.add(11) as *const u64).read_unaligned() ^ u64::from_le_bytes(*b"00:00:00");
    const ADD: u64 = u64::from_le_bytes([0x76, 0x76, 0x7F, 0x76, 0x76, 0x7F, 0x76, 0x76]);
    // hours in byte 0, minutes in byte 3, seconds in byte 6, each 0..=99 when x is all digits
    let pb = (x.wrapping_mul(10).wrapping_add(x >> 8)) & 0x00FF_0000_FF00_00FF;
    // adding 127 - 23 (127 - 59) carries an hour over 23 (a minute or second over 59) into its byte's high bit:
    // one test for the digits and the three ranges
    if ((x.wrapping_add(ADD) | x) & HI) | (pb.wrapping_add(0x0044_0000_4400_0068) & 0x0080_0000_8000_0080) != 0 {
        return None;
    }
    // one multiply: h * 3600 + mi * 60 + sec (17 bits) lands at bit 48 of the product; the other terms end below
    // bit 36 or start at bit 72
    let secs = ((pb as u128 * ((3600u128 << 48) | (60 << 24) | 1)) >> 48) as u64 & 0x1_FFFF;
    let t = day.ms + (secs * 1000) as i64;
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

const READ_AHEAD_THREADS: usize = 2;

/// Cold page cache: asks the kernel for the whole input at once, in the order it will be parsed, from a few
/// threads of their own, so the disk works at full speed from the first moment whatever the parse is doing.
/// An input that is cached already (16 probes all answer without the disk) is left alone.
#[cfg(target_os = "linux")]
fn read_ahead(queue: Vec<(&'static InFile, usize, usize)>, _threads: usize) {
    use std::os::unix::io::AsRawFd;
    let n = queue.len();
    if n == 0 || !(0..16).any(|i| uncached(queue[i * n / 16].0, (queue[i * n / 16].1 + queue[i * n / 16].2) / 2)) {
        return;
    }
    let queue = std::sync::Arc::new(queue);
    let next = std::sync::Arc::new(AtomicUsize::new(0));
    for _ in 0..READ_AHEAD_THREADS {
        let (queue, next) = (queue.clone(), next.clone());
        // never joined: these only start reads, and the end of the process ends them. A thread that cannot start,
        // or a hint the kernel refuses, costs speed only: every byte is still read by `read_span`.
        let _ = std::thread::Builder::new().spawn(move || loop {
            let i = next.fetch_add(1, Relaxed);
            if i >= queue.len() {
                break;
            }
            let (f, s, e) = queue[i];
            // readahead(2) in 512 KiB calls: cold A/B on 20M (drop_caches before every run), median of 16:
            // 2 threads x 512 KiB 444 ms, 1 thread fadvise 449, 4 threads x 128 KiB fadvise 469. Calls of 1.5 MiB and
            // up were much slower on another disk (the kernel stops queueing them as one read)
            let mut o = s;
            while o < e {
                let l = (e - o).min(512 << 10);
                unsafe { libc::readahead(f.f.as_raw_fd(), o as libc::off64_t, l) };
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

/// For every line that starts in [start, end) of the `len` bytes at `buf`: `fast64(st, line, e, commas, quotes)`
/// when its `\n` is in its first 64 bytes (line[e]; the masks cover line[..e]), `fast(st, line, e, [c0, q0, c1, q1])`
/// when it is in the next 64 (the commas and quotes of line[..64] and of line[64..e]); a longer line goes whole to
/// `slow`. The header line is skipped. `buf` must be readable for 144 bytes past `len`, and buf[len] must be `\n`.
#[inline(always)]
unsafe fn scan_lines<S>(
    buf: *const u8,
    start: usize,
    end: usize,
    len: usize,
    st: &mut S,
    fast: impl Fn(&mut S, *const u8, usize, [u64; 4]),
    fast64: impl Fn(&mut S, *const u8, usize, u64, u64),
    slow: impl Fn(&mut S, Bytes),
) {
    let file = std::slice::from_raw_parts(buf, len);
    let first = start.saturating_sub(1);
    let Some(nl) = memchr::memchr(b'\n', &file[first..]) else { return };
    let mut p = first + nl + 1;
    while p < end && p < len {
        let line = buf.add(p);
        let [c0, q0, n0] = masks64(line);
        if n0 != 0 {
            // the line ends in its first 64 bytes: one word of masks, the 64 bytes after it never read
            let e = n0.trailing_zeros() as usize;
            let lm = !n0 & n0.wrapping_sub(1); // the bits below the first newline
            fast64(st, line, e, c0 & lm, q0 & lm);
            p += e + 1;
            continue;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let [c1, q1, n1] = {
            let [mut c1, mut q1, mut n1] = masks32(line.add(64));
            if n1 == 0 {
                let [c, q, n] = masks32(line.add(96));
                (c1, q1, n1) = (c1 | c << 32, q1 | q << 32, n1 | n << 32);
            }
            [c1, q1, n1]
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let [c1, q1, n1] = masks64(line.add(64));
        if n1 != 0 {
            // n0 == 0 here: the line ends in its second 64 bytes, and the first 64 need no masking
            let e1 = n1.trailing_zeros() as usize;
            let lm = !n1 & n1.wrapping_sub(1); // the bits below the newline
            fast(st, line, 64 + e1, [c0, q0, c1 & lm, q1 & lm]);
            p += 64 + e1 + 1;
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
#[allow(dead_code)] // with BMI2, only the tests use it
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
    // the commas in order: those of the first 64 bytes, then the rest (N - 1 in all, so `w` never runs dry twice)
    let (mut w, hi) = (c as u64, (c >> 64) as u64);
    let mut base = 0;
    let mut at = 0;
    for i in 0..N {
        let k = if i + 1 == N {
            e
        } else {
            if w == 0 {
                w = hi;
                base = 64;
            }
            let k = base + w.trailing_zeros() as usize;
            w &= w.wrapping_sub(1);
            k
        };
        // a quoted field starts with `"` (an empty unquoted field starts on its `,` or `\n`)
        let qd = (*line.add(at) == b'"') as usize;
        f[i] = (at + qd, k - at - 2 * qd);
        at = k + 1;
    }
    Some(f)
}

/// The N (<= 8) fields of a line of fewer than 128 bytes: byte i of `s` is where field i starts and byte i of `l`
/// its length, quotes removed.
#[derive(Clone, Copy)]
struct Fields {
    s: u64,
    l: u64,
}

impl Fields {
    /// (start, length) of field `i`.
    #[inline(always)]
    fn at(&self, i: usize) -> (usize, usize) {
        (((self.s >> (8 * i)) & 0xFF) as usize, ((self.l >> (8 * i)) & 0xFF) as usize)
    }
}

/// `fast_fields` for a line that ends in its first 64 bytes: one word of masks.
#[inline(always)]
unsafe fn fast_fields64<const N: usize>(line: *const u8, e: usize, c: u64, q: u64) -> Option<Fields> {
    const { assert!(N >= 2 && N <= 8) };
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2", target_feature = "popcnt"))]
    {
        use std::arch::x86_64::{_pdep_u64, _pext_u64};
        let _ = line;
        if c.count_ones() as usize != N - 1 {
            return None;
        }
        let b = c | (1u64 << e); // the N boundaries
        let s = (c << 1) | 1; // the N field starts
        let z = b >> 1; // the byte before each boundary
        let y = q << 1; // the byte after each quote
        let qs = _pext_u64(q, s);
        let qe = _pext_u64(y, b);
        if (q & !(s | z)) | (q & s & z) | (qs ^ qe) != 0 {
            return None;
        }
        // byte i of k = where field i ends: the commas in order, then the line's end
        let (mut w, mut k) = (c, (e as u64) << (8 * (N - 1)));
        for i in 0..N - 1 {
            k |= (w.trailing_zeros() as u64) << (8 * i);
            w &= w.wrapping_sub(1);
        }
        let quoted = _pdep_u64(qs, 0x0101_0101_0101_0101);
        let s = (k << 8).wrapping_add(0x0101_0101_0101_0100).wrapping_add(quoted);
        return Some(Fields { s, l: k.wrapping_sub(s).wrapping_sub(quoted) });
    }
    #[allow(unreachable_code)]
    {
        let f = fast_split::<N>(line, e, c as u128, q as u128)?;
        let (mut s, mut l) = (0u64, 0u64);
        for i in 0..N {
            s |= (f[i].0 as u64) << (8 * i);
            l |= (f[i].1 as u64) << (8 * i);
        }
        Some(Fields { s, l })
    }
}

/// `fast_split` with all the fields in two words, and no byte of the line read.
///
/// Fields in order are the bits of `s` (where a field starts: the line's first byte and the byte after each comma)
/// and of `b` (the boundary that ends it: each comma and the line's end), so one PEXT over each gives, per field,
/// "starts with a quote" and "a quote sits right before its end". The line is well quoted when every quote is one
/// of those two, none is both (a lone `"`), and the two answers agree for every field. The comma positions then go
/// into the bytes of one word and the starts and lengths of all fields come from three additions on it.
#[inline(always)]
unsafe fn fast_fields<const N: usize>(line: *const u8, e: usize, c: u128, q: u128) -> Option<Fields> {
    const { assert!(N >= 2 && N <= 8) };
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2", target_feature = "popcnt"))]
    {
        use std::arch::x86_64::{_pdep_u64, _pext_u64};
        let _ = line;
        let (c0, c1, q0, q1) = (c as u64, (c >> 64) as u64, q as u64, (q >> 64) as u64);
        if (c0.count_ones() + c1.count_ones()) as usize != N - 1 {
            return None;
        }
        let (e0, e1) = if e < 64 { (1u64 << e, 0) } else { (0, 1u64 << (e - 64)) };
        let (b0, b1) = (c0 | e0, c1 | e1); // the N boundaries
        let (s0, s1) = ((c0 << 1) | 1, (c1 << 1) | (c0 >> 63)); // the N field starts
        let (z0, z1) = ((b0 >> 1) | (b1 << 63), b1 >> 1); // the byte before each boundary
        let (y0, y1) = (q0 << 1, (q1 << 1) | (q0 >> 63)); // the byte after each quote
        let qs = _pext_u64(q0, s0) | _pext_u64(q1, s1) << s0.count_ones();
        let qe = _pext_u64(y0, b0) | _pext_u64(y1, b1) << b0.count_ones();
        if (q0 & !(s0 | z0)) | (q1 & !(s1 | z1)) | (q0 & s0 & z0) | (q1 & s1 & z1) | (qs ^ qe) != 0 {
            return None;
        }
        // byte i of k = where field i ends: the commas in order (those of the first 64 bytes, then the rest;
        // N - 1 in all, so `w` never runs dry twice), then the line's end
        let (mut w, mut base, mut k) = (c0, 0u64, (e as u64) << (8 * (N - 1)));
        for i in 0..N - 1 {
            if w == 0 {
                w = c1;
                base = 64;
            }
            k |= (base + w.trailing_zeros() as u64) << (8 * i);
            w &= w.wrapping_sub(1);
        }
        // start = the end before + 1 + quoted (at most 129), length = end - start - quoted (a quoted field holds
        // its two quotes): no carry or borrow leaves a byte below N
        let quoted = _pdep_u64(qs, 0x0101_0101_0101_0101);
        let s = (k << 8).wrapping_add(0x0101_0101_0101_0100).wrapping_add(quoted);
        return Some(Fields { s, l: k.wrapping_sub(s).wrapping_sub(quoted) });
    }
    #[allow(unreachable_code)]
    {
        let f = fast_split::<N>(line, e, c, q)?;
        let (mut s, mut l) = (0u64, 0u64);
        for i in 0..N {
            s |= (f[i].0 as u64) << (8 * i);
            l |= (f[i].1 as u64) << (8 * i);
        }
        Some(Fields { s, l })
    }
}

/// Whether the quotes `q` of a stretch of whole fields are well placed, given its boundaries `b` (the comma or line
/// end that ends each field; bit 0 is a field's first byte), and the quotes that open a field. As in `fast_split`:
/// a quote must be a field's first byte or its last, not both (`s ^ z`), and per field an opening quote comes with
/// a closing one: y - x borrows across exactly the quoted fields' bytes, so a field with only one of the two leaves
/// a bit on a boundary.
#[inline(always)]
fn quotes_ok(b: u64, q: u64) -> (bool, u64, u64) {
    let s = (b << 1) | 1;
    let z = b >> 1;
    let x = q & s;
    let y = (q << 1) & b;
    ((q & !(s ^ z)) | (y.wrapping_sub(x) & b) == 0, x, s)
}

/// The 7 fields of an event line: the first three as in `Fields`, the last four in `s2` and `l2` with their starts
/// counted from line[base2].
#[derive(Clone, Copy)]
struct EvFields {
    s: u64,
    l: u64,
    s2: u64,
    l2: u64,
    base2: usize,
}

impl EvFields {
    #[inline(always)]
    fn whole(f: Fields) -> EvFields {
        EvFields { s: f.s, l: f.l, s2: f.s >> 24, l2: f.l >> 24, base2: 0 }
    }

    /// (pointer, length) of field `i` of the line at `line`.
    #[inline(always)]
    unsafe fn at(&self, line: *const u8, i: usize) -> (*const u8, usize) {
        if i < 3 {
            (line.add(((self.s >> (8 * i)) & 0xFF) as usize), ((self.l >> (8 * i)) & 0xFF) as usize)
        } else {
            let j = 8 * (i - 3);
            (line.add(self.base2 + ((self.s2 >> j) & 0xFF) as usize), ((self.l2 >> j) & 0xFF) as usize)
        }
    }
}

/// `fast_fields::<7>` for a line of 64 <= `e` < 128 bytes given the commas and quotes of its first 64 bytes (c0, q0)
/// and of the rest (c1, q1). When the first three fields end in the first 64 bytes and the other four are under 64
/// bytes together (any line with IDs and timestamps of the usual lengths), the line is taken as two stretches of
/// whole fields, each with its commas and quotes in one word: the second word is the masks shifted down to the
/// fourth field's first byte. Any other line goes through `fast_fields`.
#[inline(always)]
unsafe fn fields7_long(line: *const u8, e: usize, [c0, q0, c1, q1]: [u64; 4]) -> Option<EvFields> {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2", target_feature = "popcnt"))]
    {
        use std::arch::x86_64::{_pdep_u64, _pext_u64};
        if (c0.count_ones() + c1.count_ones()) as usize != 6 {
            return None;
        }
        let w1 = c0 & c0.wrapping_sub(1);
        let w2 = w1 & w1.wrapping_sub(1);
        let (k1, k2, k3) = (c0.trailing_zeros() as u64, w1.trailing_zeros() as u64, w2.trailing_zeros() as u64);
        // the line's end counted from the fourth field's first byte (wraps to a huge number when w2 is 0, k3 = 64)
        let e2 = (e as u64).wrapping_sub(k3 + 1);
        if (w2 != 0) & (e2 < 64) {
            let low = w2 ^ w2.wrapping_sub(1); // bits 0..=k3
            // bits k3 + 1.. of the first word, then the second word's: shifts by 0..=63 only
            let down = |x0: u64, x1: u64| (x0 >> k3 >> 1) | (x1 << (63 - k3));
            let (cb, qb) = (down(c0, c1), down(q0, q1));
            let (ok_a, xa, sa) = quotes_ok(c0 & low, q0 & low);
            let (ok_b, xb, sb) = quotes_ok(cb | 1u64 << e2, qb);
            if !(ok_a & ok_b) {
                return None;
            }
            let v1 = cb & cb.wrapping_sub(1);
            let v2 = v1 & v1.wrapping_sub(1);
            let ka = k1 | k2 << 8 | k3 << 16;
            let kb = cb.trailing_zeros() as u64 | (v1.trailing_zeros() as u64) << 8 | (v2.trailing_zeros() as u64) << 16 | e2 << 24;
            // starts and lengths as in `fast_fields`, per stretch
            let (qa, qb) = (_pdep_u64(_pext_u64(xa, sa), 0x0101_0101), _pdep_u64(_pext_u64(xb, sb), 0x0101_0101));
            let s = (ka << 8).wrapping_add(0x0101_0100).wrapping_add(qa);
            let s2 = (kb << 8).wrapping_add(0x0101_0100).wrapping_add(qb);
            return Some(EvFields { s, l: ka.wrapping_sub(s).wrapping_sub(qa), s2, l2: kb.wrapping_sub(s2).wrapping_sub(qb), base2: k3 as usize + 1 });
        }
    }
    let (c, q) = (c0 as u128 | (c1 as u128) << 64, q0 as u128 | (q1 as u128) << 64);
    fast_fields::<7>(line, e, c, q).map(EvFields::whole)
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

/// One parsed event line in its event-key bucket, 16 bytes: `loc` is its `ERec` (`loc_of`), 0 for an invalid or
/// late event (it still takes part in dedup); `ing` is its ingest_ts (`Ctx::ing`).
#[repr(C)]
#[derive(Clone, Copy)]
struct LRec {
    key: u64,
    ing: u32,
    loc: u32,
}

/// `LRec::ing` of an ingest_ts above `Ctx::ihi`: such an event is late or invalid whatever its ts (a valid ts is
/// below the window end, more than a day before it), so among the copies of an event it only has to rank after
/// every copy that is not.
const ING_LATE: u32 = u32::MAX;
/// `LRec::ing` of an ingest_ts that 32 bits cannot hold (more than 49 days before `Ctx::ihi`): the exact value is
/// in `Shard::far`.
const ING_FAR: u32 = 0;
/// Milliseconds from `Ctx::ilo` through `Ctx::ihi`: `ing` = ms above `ilo` + 1, in 1..ING_LATE.
const ING_SPAN: i64 = (u32::MAX - 2) as i64;

/// A valid click in its person partition.
#[repr(C)]
#[derive(Clone, Copy)]
struct CRec {
    cid: u64,
    w: u64, // ts (ms after `Ctx::base`) | campaign << 32 | the person's number within its partition << 48
}

impl CRec {
    #[inline(always)]
    fn new(cid: u64, local: u32, ts: u32, camp: u32) -> CRec {
        CRec { cid, w: ts as u64 | (camp as u64) << 32 | (local as u64) << 48 }
    }
    #[inline(always)]
    fn ts(self) -> u32 {
        self.w as u32
    }
    #[inline(always)]
    fn camp(self) -> u32 {
        (self.w >> 32) as u16 as u32
    }
    #[inline(always)]
    fn local(self) -> u32 {
        (self.w >> 48) as u32
    }
}

const EV_BLOCK: usize = 512;
const LG_BLOCK: usize = 384; // records: a whole number of 192-byte groups of 16-byte LRecs (6144 bytes, as before)
const CK_BLOCK: usize = 192; // records: a whole number of 192-byte groups of 16-byte CRecs (3072 bytes, as before)

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
    far: Vec<(u64, u32, i64)>, // (event key, loc, ingest_ts) of the records whose `ing` is ING_FAR
    buf: Vec<u8>,
    ins: usize,
    ring: [Pend; RING],
    head: usize, // events or clicks put in the ring since the last flush
    days: [Day; 64],
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
    qshift: u32, // spread(event key) >> qshift = event-key bucket
    abase: usize, // `arena_range().0`: where `LRec::loc` counts from
    ihi: i64,     // window end + 24 h: an event ingested later is late or invalid
    ilo: i64,     // ihi - (ING_SPAN - 1) ms: the earliest ingest_ts an `LRec` holds itself
}

impl Ctx {
    /// (abase, ihi, ilo) for a window that ends at `we`.
    fn record_fields(we: i64) -> (usize, i64, i64) {
        (arena_range().0, we + DAY, we + DAY - (ING_SPAN - 1))
    }

    /// The ingest_ts of a dedup record, exact except above `ihi` (see ING_LATE); `far` = every shard's far list as
    /// a map (`far_map`).
    #[inline(always)]
    fn ingest_of(&self, r: &LRec, far: &HashMap<(u64, u32), i64>) -> i64 {
        match r.ing {
            ING_FAR => far[&(r.key, r.loc)],
            ING_LATE => i64::MAX,
            v => self.ilo + (v as i64 - 1),
        }
    }
}

/// The far ingest_ts of every shard by (event key, loc). The records of one key without an `ERec` (loc 0) share an
/// entry, the earliest of them: none of them can be dropped, and for the others of the key only the earliest counts.
fn far_map(shards: &[Shard]) -> HashMap<(u64, u32), i64> {
    let mut m = HashMap::new();
    for &(key, loc, ingest) in shards.iter().flat_map(|s| s.far.iter()) {
        m.entry((key, loc)).and_modify(|v: &mut i64| *v = (*v).min(ingest)).or_insert(ingest);
    }
    m
}

impl Shard {
    fn new(parts: usize, buckets: usize) -> Shard {
        Shard {
            arena: Arena::new(),
            ev: (0..parts).map(|_| Stream::new()).collect(),
            ck: (0..parts).map(|_| Stream::new()).collect(),
            lg: (0..buckets).map(|_| Stream::new()).collect(),
            far: Vec::new(),
            buf: Vec::new(),
            ins: 0,
            ring: [Pend::default(); RING],
            head: 0,
            days: [NO_DAY; 64],
        }
    }

    /// Queues `p`; returns the one queued RING entries ago, whose slot is in the cache by now.
    #[inline(always)]
    fn queue(&mut self, p: Pend) -> Option<Pend> {
        let old = std::mem::replace(&mut self.ring[self.head & (RING - 1)], p);
        self.head += 1;
        if self.head > RING {
            Some(old)
        } else {
            None
        }
    }

    /// Finishes everything still queued: events when `events`, else clicks. Call at the end of every chunk.
    fn flush(&mut self, cx: &Ctx, events: bool) {
        for k in self.head.saturating_sub(RING)..self.head {
            let p = self.ring[k & (RING - 1)];
            let person = cx.ids.person(cx.ids.slot_from(p.home, p.ukey, &mut self.ins));
            if events {
                self.put_event(cx, p, person);
            } else {
                self.put_click(cx, p, person);
            }
        }
        self.head = 0;
    }

    /// The person of the queued `p`, whose line of the ID table is in the cache by now. An ID that is not in that
    /// line goes back in the queue to wait for the next line, and the entry that leaves the queue for it is looked
    /// up instead; None when nothing left the queue.
    #[inline(always)]
    fn resolve(&mut self, cx: &Ctx, mut p: Pend) -> Option<(Pend, u32)> {
        loop {
            if let Some(slot) = cx.ids.slot_in_line(p.home, p.ukey, &mut self.ins) {
                return Some((p, cx.ids.person(slot)));
            }
            p.home = cx.ids.next_line(p.home);
            p = self.queue(p)?;
        }
    }

    /// Any parsed event: its dedup record. `loc` = 0 for an invalid or late one.
    #[inline(always)]
    fn log(&mut self, cx: &Ctx, ekey: u64, ingest: i64, loc: u32) {
        let q = (spread(ekey) >> cx.qshift) as usize;
        let ing = if ingest > cx.ihi {
            ING_LATE
        } else {
            // ms above ilo (a negative difference is a huge unsigned one)
            let d = ingest.wrapping_sub(cx.ilo) as u64;
            if d < ING_SPAN as u64 {
                d as u32 + 1
            } else {
                self.far.push((ekey, loc, ingest));
                ING_FAR
            }
        };
        // SAFETY: q < 2^qbits, the number of streams (`Shard::new`)
        unsafe { self.lg.get_unchecked_mut(q) }.push(&mut self.arena, LG_BLOCK, LRec { key: ekey, ing, loc });
    }

    /// A valid, non-late event: into its person's partition, plus its dedup record, once its user ID is at hand.
    #[inline(always)]
    fn keep(&mut self, cx: &Ctx, ekey: u64, ingest: i64, ukey: u64, ts: i64, pay: u32) {
        if let Some(p) = self.queue(Pend { a: ekey, b: ingest, ukey, home: cx.ids.aim(ukey), ts: (ts - cx.base) as u32, x: pay }) {
            if let Some((p, person)) = self.resolve(cx, p) {
                self.put_event(cx, p, person);
            }
        }
    }

    #[inline(always)]
    fn put_event(&mut self, cx: &Ctx, p: Pend, person: u32) {
        // SAFETY: a person is a slot number, below 2^id_bits; shifted, below 2^pbits, the number of streams
        let rec = unsafe { self.ev.get_unchecked_mut((person >> cx.pshift) as usize) }.push(&mut self.arena, EV_BLOCK, ERec { person, ts: p.ts, pay: p.x });
        self.log(cx, p.a, p.b, loc_of(cx.abase, rec));
    }

    /// A valid click, into its person's partition once its user ID is at hand.
    #[inline(always)]
    fn click(&mut self, cx: &Ctx, cid: u64, ukey: u64, ts: i64, camp: u16) {
        if let Some(p) = self.queue(Pend { a: cid, b: 0, ukey, home: cx.ids.aim(ukey), ts: (ts - cx.base) as u32, x: camp as u32 }) {
            if let Some((p, person)) = self.resolve(cx, p) {
                self.put_click(cx, p, person);
            }
        }
    }

    #[inline(always)]
    fn put_click(&mut self, cx: &Ctx, p: Pend, person: u32) {
        // SAFETY: as in `put_event`
        unsafe { self.ck.get_unchecked_mut((person >> cx.pshift) as usize) }.push(&mut self.arena, CK_BLOCK, CRec::new(p.a, person & ((1 << cx.pshift) - 1), p.ts, p.x));
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

    // no huge pages from this kernel: memory is asked for in bulk (`populate`). An unreadable setting counts as on.
    if std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled").is_ok_and(|s| s.contains("[never]")) {
        SMALL_PAGES.store(true, Relaxed);
    }

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
    // and no more than 2^9 of each: past that a thread's open streams (3 per partition pair) outgrow its half of an
    // L2 shared by two hyperthreads. Intel Cascade Lake, 2 cores x 2 threads, 40M events: 2^10 + 2^10 +2.3% wall
    // against 2^9 + 2^9, while 2^8 + 2^8 is even with it (164K events a partition); box, 75M events, 8 threads
    // on 4 CPUs: 2^10 3.03 s, 2^9 2.76 s. Only inputs over ~69M events change. The partitions' floor (a click
    // keeps its person's number in the partition in 16 bits) stays above the cap.
    let pbits = (env_num("SESS_PBITS").map(|v| v as u32).unwrap_or_else(|| log2_near(ev_est / 90_000).min(9))).clamp(4, 12.min(id_bits - 4)).max(id_bits.saturating_sub(16));
    let qbits = (env_num("SESS_QBITS").map(|v| v as u32).unwrap_or_else(|| log2_near(ev_est / 90_000).min(9))).clamp(4, 12);
    // 128 KiB a read: the copy from the page cache passes the piece through the core's cache twice (as source and
    // as copy), and a larger piece pushes the thread's open stream groups out of it (L2 is 1 MiB, and on the scorer
    // two threads share it). tmpfs, 20M, 21 runs each, parse CPU: 512 KiB 1.66 s, 128 KiB 1.62 s, 64 KiB 1.60 s,
    // 32 KiB 1.58 s; wall medians 0.340, 0.336, 0.333, 0.354 s
    let chunk = env_num("SESS_CHUNK").filter(|&c| c >= 4096).unwrap_or(128 << 10);

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
    let (abase, ihi, ilo) = Ctx::record_fields(we);
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
        abase,
        ihi,
        ilo,
    };
    lap(&format!("small tables, links joined: {n_links} lines; 2^{id_bits} id slots, 2^{pbits} person partitions, 2^{qbits} event buckets"));

    // 2. events and clicks, from the one queue
    let (np, nq) = (1usize << pbits, 1usize << qbits);
    let mut shards = par(
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
    // the records of groups not yet full, written by this thread (each parse thread fenced its own before the join)
    for s in &mut shards {
        s.ev.iter_mut().for_each(Stream::finish);
        s.ck.iter_mut().for_each(Stream::finish);
        s.lg.iter_mut().for_each(Stream::finish);
    }
    let n_ev: usize = shards.iter().flat_map(|s| s.ev.iter()).map(|s| s.len(EV_BLOCK)).sum();
    let n_ck: usize = shards.iter().flat_map(|s| s.ck.iter()).map(|s| s.len(CK_BLOCK)).sum();
    lap(&format!("parsed: {n_ev} valid events, {n_ck} valid clicks"));
    if n_ev >= i32::MAX as usize {
        die("too many events for 32-bit session and event counts"); // user_daily writes them as INT32
    }

    // 3. duplicates
    let n_dups = mark_duplicates(&cx, &shards, nq, threads);
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
    // A thread that has emptied the queue hands its campaign cells to this thread, then writes its user file; this
    // thread merges the cells and writes campaign_daily meanwhile instead of after the last file.
    let (tx, rx) = std::sync::mpsc::channel::<(Vec<Cred>, Vec<i64>, Vec<u32>)>();
    let (camp_rows, _results) = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let (cx, shards, order, next, n_rows, ud, cat_names) = (&cx, &shards, &order, &next, &n_rows, &ud, &cat_names);
                let tx = tx.clone();
                sc.spawn(move || {
                    let mut w = Work::new(cx);
                    loop {
                        let i = next.fetch_add(1, Relaxed);
                        if i >= order.len() {
                            break;
                        }
                        partition_work(cx, shards, order[i].1, &mut w);
                    }
                    tx.send((std::mem::take(&mut w.cred), std::mem::take(&mut w.rev), std::mem::take(&mut w.assist))).unwrap_or_else(|_| die("campaign merger gone"));
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
        drop(tx);
        // campaign_daily
        let mut cred = vec![[0i64; 3]; (ncamp + 1) * width];
        let mut assist = filled((ncamp + 1) * width, 0i64);
        for _ in 0..threads {
            let (c, r, a) = rx.recv().unwrap_or_else(|_| die("a worker ended without its campaign cells"));
            for ((x, y), r) in cred.iter_mut().zip(&c).zip(&r) {
                x[0] += y.sessions as i64;
                x[1] += y.purchasing as i64;
                x[2] += r;
            }
            for (x, y) in assist.iter_mut().zip(&a) {
                *x += *y as i64;
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
        let results: Vec<Work> = hs.into_iter().map(|h| h.join().unwrap_or_else(|_| die("a worker panicked"))).collect();
        (camp_rows, results)
    });
    lap(&format!("sessions aggregated, {} user rows and {} campaign rows written", n_rows.load(Relaxed), camp_rows.len()));
    // every worker is joined: the records and the ID table have been read for the last time, so their memory goes back
    // to the kernel on all threads (arena c4; where THP is off the exit would free 4 KiB pages on one thread)
    release_regions(threads);
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
    shift: u32,             // spread(key) >> shift = where the search for a key starts
    other: BytesMap<u16>,
    // the IDs of up to 7 bytes by their raw bytes (`short_key`), for `get_short`; empty unless `with_short`
    rkeys: Vec<u64>, // 0 = empty
    rvals: Vec<u16>,
    rshift: u32,
}

/// An ID of `len` < 8 bytes read as the `len` low bytes of `raw`, with `len + 1` above them: never 0, and the
/// same for two IDs only when their bytes are.
#[inline(always)]
fn short_key(raw: u64, len: usize) -> u64 {
    (raw & ((1u64 << (8 * len)) - 1)) | (len as u64 + 1) << 56
}

impl IdMap {
    fn new(ids: &[(Bytes, u16)]) -> IdMap {
        let cap = (ids.len() * 2).next_power_of_two().max(16);
        let mut m = IdMap { slots: filled(cap, (0, 0)), shift: 64 - cap.trailing_zeros(), other: BytesMap::default(), rkeys: Vec::new(), rvals: Vec::new(), rshift: 0 };
        for &(id, v) in ids {
            match pack(id) {
                0 => {
                    m.other.insert(id, v);
                }
                k => {
                    let mut i = (spread(k) >> m.shift) as usize;
                    while m.slots[i].0 != 0 {
                        i = (i + 1) & (cap - 1);
                    }
                    m.slots[i] = (k, v);
                }
            }
        }
        m
    }

    /// Adds the table `get_short` reads: the IDs of 1 to 7 bytes by their raw bytes, in a table of no more than a
    /// fourth full (a lookup of an ID not in it ends at the first empty slot, nearly always the first).
    fn with_short(mut self, ids: &[(Bytes, u16)]) -> IdMap {
        let cap = (ids.len() * 4).next_power_of_two().max(16);
        self.rkeys = vec![0; cap];
        self.rvals = vec![0; cap];
        self.rshift = 64 - cap.trailing_zeros();
        for &(id, v) in ids {
            if id.len() >= 1 && id.len() < 8 {
                let mut raw = [0u8; 8];
                raw[..id.len()].copy_from_slice(id);
                let k = short_key(u64::from_le_bytes(raw), id.len());
                let mut i = (spread(k) >> self.rshift) as usize;
                while self.rkeys[i] != 0 {
                    i = (i + 1) & (cap - 1);
                }
                self.rkeys[i] = k;
                self.rvals[i] = v;
            }
        }
        self
    }

    /// `get_at` for a map made `with_short`.
    #[inline(always)]
    unsafe fn get_short(&self, p: *const u8, len: usize) -> Option<u16> {
        if len >= 8 {
            return self.get_at(p, len);
        }
        let k = short_key((p as *const u64).read_unaligned(), len);
        let mask = self.rkeys.len() - 1;
        let mut i = (spread(k) >> self.rshift) as usize;
        loop {
            let key = *self.rkeys.get_unchecked(i);
            if key == k {
                return Some(*self.rvals.get_unchecked(i));
            }
            if key == 0 {
                return None;
            }
            i = (i + 1) & mask;
        }
    }

    #[inline(always)]
    fn packed(&self, k: u64) -> Option<u16> {
        let mask = self.slots.len() - 1;
        let mut i = (spread(k) >> self.shift) as usize;
        loop {
            // SAFETY: i <= mask (the shift leaves as many bits as the mask has)
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
    Small { products: IdMap::new(&products), campaigns: IdMap::new(&campaigns).with_short(&campaigns), cat_names, cat_rank, unknown_cat, camp_ids, camp_channel }
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
                    let root = ids.find(i as u32);
                    ids.slots[i].up.store(root + 1, Relaxed);
                    ids.slots[i].px.store(i as u32 ^ root, Relaxed);
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

/// `split::<7>` + `event_fields` for the line at `line` given its fields (`fast_fields64`, `fields7_long`);
/// line[..144] is readable.
#[inline(always)]
unsafe fn fast_event(cx: &Ctx, line: *const u8, f: Option<EvFields>, st: &mut Shard) {
    let Some(f) = f else { return };
    let (ep, el) = f.at(line, 0);
    if el == 0 {
        return;
    }
    let (ip, il) = f.at(line, 3);
    let Some(ingest) = ts_memo(ip, il, &mut st.days) else { return };
    let ((up, ul), pl) = (f.at(line, 1), f.at(line, 5).1);
    // both IDs at once; the user ID is interned (when it does not pack) only once the event is known to be kept
    let (ek, uk) = pack2(ep, el, up, ul);
    let ekey = cx.names.key_or(ek, ep, el);
    let (tp, tl) = f.at(line, 2);
    let ts = match ts_memo(tp, tl, &mut st.days) {
        Some(t) if ul != 0 && pl != 0 && t >= cx.ws && t < cx.we && ingest - t <= DAY => t,
        _ => return st.log(cx, ekey, ingest, 0),
    };
    let (kp, kl) = f.at(line, 4);
    let w = (kp as *const u64).read_unaligned();
    let pay = match kl {
        4 if w as u32 == u32::from_le_bytes(*b"view") || w as u32 == u32::from_le_bytes(*b"cart") => 0,
        8 if w == u64::from_le_bytes(*b"purchase") => {
            let (rp, rl) = f.at(line, 6);
            let Some(p) = parse_price(std::slice::from_raw_parts(rp, rl)) else { return st.log(cx, ekey, ingest, 0) };
            let cat = cx.products.get_at(f.at(line, 5).0, pl).unwrap_or(cx.unknown_cat);
            (cat as u32) << 24 | p
        }
        _ => return st.log(cx, ekey, ingest, 0),
    };
    let ukey = cx.names.key_or(uk, up, ul);
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
        |st, line, e, m| fast_event(cx, line, fields7_long(line, e, m), st),
        |st, line, e, c, q| fast_event(cx, line, fast_fields64::<7>(line, e, c, q).map(EvFields::whole), st),
        |st, l| {
            if let Some(f) = split::<7>(l) {
                event_fields(cx, f, st);
            }
        },
    );
    st.flush(cx, true);
    sfence();
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
    let Some(f) = fast_fields::<4>(line, e, c, q) else { return };
    if f.at(0).1 == 0 || f.at(1).1 == 0 {
        return;
    }
    let t = match ts_memo(line.add(f.at(3).0), f.at(3).1, &mut st.days) {
        Some(t) if t >= cx.base && t < cx.we => t,
        _ => return,
    };
    let Some(camp) = cx.campaigns.get_short(line.add(f.at(2).0), f.at(2).1) else { return };
    let (cp, cl, up, ul) = (line.add(f.at(0).0), f.at(0).1, line.add(f.at(1).0), f.at(1).1);
    let (ck, uk) = pack2(cp, cl, up, ul);
    st.click(cx, cx.names.key_or(ck, cp, cl), cx.names.key_or(uk, up, ul), t, camp);
}

/// `fast_click` for a line that ends in its first 64 bytes.
#[inline(always)]
unsafe fn fast_click64(cx: &Ctx, line: *const u8, e: usize, c: u64, q: u64, st: &mut Shard) {
    let Some(f) = fast_fields64::<4>(line, e, c, q) else { return };
    if f.at(0).1 == 0 || f.at(1).1 == 0 {
        return;
    }
    let t = match ts_memo(line.add(f.at(3).0), f.at(3).1, &mut st.days) {
        Some(t) if t >= cx.base && t < cx.we => t,
        _ => return,
    };
    let Some(camp) = cx.campaigns.get_short(line.add(f.at(2).0), f.at(2).1) else { return };
    let (cp, cl, up, ul) = (line.add(f.at(0).0), f.at(0).1, line.add(f.at(1).0), f.at(1).1);
    let (ck, uk) = pack2(cp, cl, up, ul);
    st.click(cx, cx.names.key_or(ck, cp, cl), cx.names.key_or(uk, up, ul), t, camp);
}

unsafe fn parse_clicks(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    scan_lines(
        buf,
        start,
        end,
        len,
        st,
        |st, line, e, [c0, q0, c1, q1]| fast_click(cx, line, e, c0 as u128 | (c1 as u128) << 64, q0 as u128 | (q1 as u128) << 64, st),
        |st, line, e, c, q| fast_click64(cx, line, e, c as u64, q as u64, st),
        |st, l| {
            if let Some(f) = split::<4>(l) {
                click_fields(cx, f, st);
            }
        },
    );
    st.flush(cx, false);
    sfence();
}

/// A cache line of the duplicate finder's table: 8 tags (0 = empty), then their 8 positions + 1.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct DLine([u32; 16]);

/// (bit k set when tag k of `line` is `tag`, bit k set when tag k is empty)
#[inline(always)]
fn line_match(line: &[u32; 16], tag: u32) -> (u32, u32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let v = _mm256_load_si256(line.as_ptr() as *const __m256i);
        let m = |x: __m256i| _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, x))) as u32;
        return (m(_mm256_set1_epi32(tag as i32)), m(_mm256_setzero_si256()));
    }
    #[allow(unreachable_code)]
    (0..8).fold((0, 0), |(e, z), k| (e | ((line[k] == tag) as u32) << k, z | ((line[k] == 0) as u32) << k))
}

/// One record into the duplicate finder's table (`lines` = mask + 1 lines at `tp`, a record's line = fib(key) >> shift);
/// a repeat of a tag joins (or starts) that tag's group: see `mark_duplicates`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dup_insert(tp: *mut DLine, mask: usize, shift: u32, qb: u32, r: &LRec, me: u32, mem: &mut Vec<(LRec, u32)>, groups: &mut Vec<(u32, u32)>, firsts: &mut Vec<(u32, u32)>) {
    let h = r.key.wrapping_mul(0x9E37_79B9_7F4A_7C15); // `fib` of mark_duplicates
    let tag = (h >> 8) as u32 | 1;
    // the event-key bucket is the top `qb` bits of this same product: take the line from the bits below them, or every record of a
    // bucket would pile into 1/2^qb of the lines (20M events in 17 s)
    let mut l = ((h << qb) >> shift) as usize;
    loop {
        // SAFETY: l <= mask, a line of the table
        let line = &mut (*tp.add(l)).0;
        let (eq, empty) = line_match(line, tag);
        if eq != 0 {
            let at = 8 + eq.trailing_zeros() as usize;
            let x = mem.len() as u32;
            if line[at] < DUP_GRP {
                // the tag's first repeat: a new group of the record it met (copied later) and this one
                if groups.len() as u32 >= u32::MAX - DUP_GRP {
                    die("too many duplicate groups in one event-key bucket");
                }
                firsts.push((x, line[at]));
                mem.push((*r, x + 2));
                mem.push((*r, 0));
                line[at] = DUP_GRP + groups.len() as u32;
                groups.push((x, x + 1));
            } else {
                let g = &mut groups[(line[at] - DUP_GRP) as usize];
                mem[g.1 as usize].1 = x + 1;
                mem.push((*r, 0));
                g.1 = x;
            }
            return;
        }
        if empty != 0 {
            let at = empty.trailing_zeros() as usize;
            line[at] = tag;
            line[8 + at] = me;
            return;
        }
        l = (l + 1) & mask;
    }
}

/// A record's position in `mark_duplicates`: shard << DUP_POS_T | index in that shard's stream of the bucket. With at
/// most 127 shards a position + 1 is at most 127 << DUP_POS_T < DUP_GRP: a table slot at or above DUP_GRP holds a
/// group number + DUP_GRP instead.
const DUP_POS_T: u32 = 25;
const DUP_GRP: u32 = (127 << DUP_POS_T) + 1;

/// rules.md "Events" 2: per event-key bucket, among the copies of one event_id the smallest ingest_ts is kept and
/// every other valid copy is marked dropped. Returns the number of records whose tag repeats.
///
/// One pass over the bucket's records puts each in a table of 64-byte lines, 8 tags + 8 positions each, probed
/// with one SIMD compare: a record's line is the top bits of key * 2^64 / phi, its tag 31 lower bits | 1, a full line spills
/// into the next. A record whose tag is already there joins that tag's group (the slot then holds the group, and
/// the group starts with the record it met), chained in `mem`. All copies of a key share its tag, so they all meet
/// in one group; a few other keys do too and are told apart by key.
fn mark_duplicates(cx: &Ctx, shards: &[Shard], buckets: usize, threads: usize) -> usize {
    const POS_T: u32 = DUP_POS_T;
    let far = far_map(shards); // empty unless an ingest_ts is far from the window
    let (far, abase) = (&far, cx.abase);
    if shards.len() > 127 || shards.iter().any(|s| s.lg.iter().any(|l| l.len(LG_BLOCK) >> POS_T != 0)) {
        die("too many events for the duplicate finder");
    }
    par(
        threads,
        buckets,
        || (Vec::<DLine>::new(), Vec::<(LRec, u32)>::new(), Vec::<(u32, u32)>::new(), Vec::<(u32, u32)>::new(), Vec::<LRec>::new(), 0usize),
        |(table, mem, groups, firsts, grp, total), b| {
            let n: usize = shards.iter().map(|s| s.lg[b].len(LG_BLOCK)).sum();
            let lines = (n / 6).next_power_of_two().max(4); // at most 3/4 full
            let mask = lines - 1;
            // Fibonacci hashing: the line is the top bits of key * 2^64 / phi, which depend on every bit of the key
            let shift = 64 - lines.trailing_zeros();
            let fib = |k: u64| k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            table.clear();
            table.resize(lines, DLine([0; 16]));
            mem.clear();
            groups.clear();
            firsts.clear();
            let at = |pos: u32| {
                let (t, i) = ((pos - 1) as usize >> POS_T, (pos - 1) as usize & ((1 << POS_T) - 1));
                // SAFETY: i < the stream's length, so block i / LG_BLOCK holds it
                unsafe { shards[t].lg[b].blocks[i / LG_BLOCK].add(i % LG_BLOCK) as *const LRec }
            };
            let tp = table.as_mut_ptr();
            // one record into the table; a repeat of a tag joins (or starts) that tag's group
            let qb = buckets.trailing_zeros();
            let mut insert = |r: &LRec, me: u32| unsafe { dup_insert(tp, mask, shift, qb, r, me, mem, groups, firsts) };
            let line_of = |r: &LRec| unsafe { tp.add(((fib(r.key) << qb) >> shift) as usize) };
            // ahead of record i: its stream RA records on, and the table line of the record LA on
            const RA: usize = 40;
            const LA: usize = 8;
            for (t, s) in shards.iter().enumerate() {
                let mut me = ((t as u32) << POS_T) + 1;
                let blks: Vec<&[LRec]> = s.lg[b].slices(LG_BLOCK).collect();
                for (k, &blk) in blks.iter().enumerate() {
                    let n = blk.len();
                    let p = blk.as_ptr();
                    // all but the last RA records: the records ahead are in this block
                    let body = n.saturating_sub(RA) & !3;
                    for i in (0..body).step_by(4) {
                        // SAFETY: i + 3 + RA < n
                        unsafe {
                            prefetch(p.add(i + RA));
                            for j in i..i + 4 {
                                prefetch(line_of(&*p.add(j + LA)));
                                insert(&*p.add(j), me);
                                me += 1;
                            }
                        }
                    }
                    // the rest: the records ahead may be in the next block
                    let nb: &[LRec] = blks.get(k + 1).copied().unwrap_or(&[]);
                    let after = |i: usize| if i < n { blk.get(i) } else { nb.get(i - n) };
                    for i in body..n {
                        if i % 4 == 0 {
                            if let Some(f) = after(i + RA) {
                                prefetch(f);
                            }
                        }
                        if let Some(f) = after(i + LA) {
                            prefetch(line_of(f));
                        }
                        insert(&blk[i], me);
                        me += 1;
                    }
                }
            }
            *total += mem.len();
            // the records the groups started with, asked for AHEAD before use
            const AHEAD: usize = 16;
            for k in 0..firsts.len() {
                if let Some(&(_, p)) = firsts.get(k + AHEAD) {
                    prefetch(at(p));
                }
                let (x, p) = firsts[k];
                // SAFETY: `at` is the address of a record of a live stream
                mem[x as usize].0 = unsafe { *at(p) };
            }
            // per group, the events it may mark asked for AHEAD groups before
            let ahead = |g: (u32, u32)| {
                prefetch(rec_at(abase, mem[g.0 as usize].0.loc)); // loc 0 is a harmless hint
                prefetch(rec_at(abase, mem[g.0 as usize + 1].0.loc));
            };
            groups.iter().take(AHEAD).for_each(|&g| ahead(g));
            for gi in 0..groups.len() {
                if let Some(&g) = groups.get(gi + AHEAD) {
                    ahead(g);
                }
                // the records of one tag, in table order; mostly the copies of one key
                grp.clear();
                let mut x = groups[gi].0 as usize + 1;
                while x != 0 {
                    grp.push(mem[x - 1].0);
                    x = mem[x - 1].1 as usize;
                }
                if grp.len() > 16 {
                    // a big group (one event_id with many copies, or many keys on one tag): sorted, not pairwise
                    grp.sort_unstable_by_key(|r| (r.key, cx.ingest_of(r, far)));
                    for k in 1..grp.len() {
                        if grp[k].loc != 0 && grp[k - 1].key == grp[k].key {
                            // SAFETY: as below
                            unsafe { (*rec_at(abase, grp[k].loc)).person = DROPPED };
                        }
                    }
                    continue;
                }
                for (k, r) in grp.iter().enumerate() {
                    // dropped: a valid copy with an earlier copy of its key (smaller ingest_ts, or equal and before it:
                    // copies with equal ingest_ts are identical, so exactly one of them is kept)
                    let ri = cx.ingest_of(r, far);
                    let earlier = |(x, o): (usize, &LRec)| {
                        let oi = cx.ingest_of(o, far);
                        o.key == r.key && (oi < ri || (oi == ri && x < k))
                    };
                    if r.loc != 0 && grp.iter().enumerate().any(earlier) {
                        // SAFETY: the address of an ERec in a live arena; no other thread touches this record now
                        unsafe { (*rec_at(abase, r.loc)).person = DROPPED };
                    }
                }
            }
        },
    )
    .iter()
    .map(|x| x.5)
    .sum()
}

#[derive(Clone, Copy, Default)]
struct Cred {
    sessions: u32,
    purchasing: u32,
}

/// A click of the person at hand: `k` = ts << 32 | campaign.
#[derive(Clone, Copy)]
struct Ck {
    k: u64,
    cid: u64,
}

/// All of one person's events in one 30-minute bin of the window (`ts` is in ms after `Ctx::base`, the bin is
/// `ts / GAP`): two events in a bin are never a session break, two bins apart always are, adjacent bins break
/// exactly when `min` of the later minus `max` of the earlier exceeds GAP. A free bin is (MAX, 0, 0).
#[derive(Clone, Copy)]
struct Bin {
    min: u32,
    max: u32,
    cnt: u32,
}
const FREE_BIN: Bin = Bin { min: u32::MAX, max: 0, cnt: 0 };

/// A person's sessions of one day: one `user_daily` row.
struct DayAcc {
    d: usize,
    sessions: u32,
    events: u32,
    revenue: i64,
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
    rev: Vec<i64>, // revenue of the credited sessions, per cell of `cred`
    assist: Vec<u32>,
    stamp: Vec<bool>,
    iv: Vec<(i64, i64)>,
    seen: Vec<u32>,
    sctr: u32,
    start: Vec<u32>,  // see partition_work
    cstart: Vec<u32>,
    ev: Vec<u64>,
    ck: Vec<Ck>,
    sessions: Vec<Sess>,
    day_groups: Vec<(usize, usize)>, // (d, first session index)
    cats: Vec<(u16, i64)>,
    touched: Vec<usize>,
    bins: Vec<Bin>, // the bins of the person at hand (all free between people)
    bits: Vec<u64>, // which bins are taken
    bin_sess: Vec<u32>, // the session of each bin taken by the person at hand
    purch: Vec<(u32, u32)>, // (bin, pay) of the person's purchases
    sess_day: Vec<u32>,     // the DayAcc of each session
    days: Vec<DayAcc>,
    dcats: Vec<(u32, u16, i64)>, // (DayAcc, category, price sum)
    small: Win,
    last: Vec<u32>,
    lctr: u32,
}

impl Work {
    fn new(cx: &Ctx) -> Work {
        let cells = (cx.ncamp + 1) * cx.ndays;
        let local = (1usize << cx.pshift) + 1;
        Work {
            rows: Vec::new(),
            cred: vec![Cred::default(); cells],
            rev: filled(cells, 0),
            assist: filled(cells + 4, 0), // + cells for the unused steps of the vector window
            stamp: vec![false; cx.ncamp + 1],
            iv: vec![(0, 0); cx.ncamp + 1],
            seen: filled(cx.ncamp + 1, 0),
            sctr: 0,
            start: filled(local + 2, 0),
            cstart: filled(local + 2, 0),
            ev: Vec::new(),
            ck: Vec::new(),
            sessions: Vec::new(),
            day_groups: Vec::new(),
            cats: Vec::new(),
            touched: Vec::new(),
            bins: vec![FREE_BIN; ((cx.we - cx.base) / GAP) as usize + 2],
            bits: vec![0; ((cx.we - cx.base) / GAP) as usize / 64 + 2],
            bin_sess: vec![0; ((cx.we - cx.base) / GAP) as usize + 2],
            purch: Vec::new(),
            sess_day: Vec::new(),
            days: Vec::new(),
            dcats: Vec::new(),
            small: Win { ts: A32([0; 72]), nxt: A32([0; 72]), camp: [0; 72], cid: [0; 72] },
            last: vec![0; cx.ncamp + 1],
            lctr: 0,
        }
    }
}

/// Rotation `r` of eight lanes, and the lanes whose rotated-in lane (i + r) % 8 comes before them (for breaking ties).
#[allow(dead_code)] // only the AVX2 build uses them
const ROT: [[i32; 8]; 8] = {
    let mut t = [[0; 8]; 8];
    let mut r = 0;
    while r < 8 {
        let mut i = 0;
        while i < 8 {
            t[r][i] = ((i + r) % 8) as i32;
            i += 1;
        }
        r += 1;
    }
    t
};
#[allow(dead_code)]
const TIE: [[i32; 8]; 8] = {
    let mut t = [[0; 8]; 8];
    let mut r = 0;
    while r < 8 {
        let mut i = 0;
        while i < 8 {
            t[r][i] = if (i + r) % 8 < i { -1 } else { 0 };
            i += 1;
        }
        r += 1;
    }
    t
};

/// `rank[i]` = slot of key i in sorted order, for 2 <= k <= 64 keys (for k <= 8 the ranks of eight lanes, the ones
/// past k included: always a permutation of 0..8): the number of keys below it plus the equal keys before it, so no
/// data-dependent branches and every slot is used once.
#[inline]
fn ranks_small(k: usize, key: impl Fn(usize) -> u32, rank: &mut [u8; 64]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        if k <= 8 {
            // one vector: every key against the other seven, by rotating the vector, with a fixed trip count.
            // Lanes past k hold the largest key (and any index into `key` is clamped), so they only rank last.
            let mut buf = [0i32; 8];
            for i in 0..8 {
                let kv = (key(i.min(k - 1)) ^ 0x8000_0000) as i32;
                buf[i] = if i < k { kv } else { i32::MAX };
            }
            let v = _mm256_loadu_si256(buf.as_ptr() as *const __m256i);
            let mut acc = _mm256_setzero_si256();
            for r in 1..8 {
                let rot = _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(ROT[r].as_ptr() as *const __m256i)); // lane i = key (i + r) % 8
                let lt = _mm256_cmpgt_epi32(v, rot);
                let tie = _mm256_and_si256(_mm256_cmpeq_epi32(v, rot), _mm256_loadu_si256(TIE[r].as_ptr() as *const __m256i));
                acc = _mm256_sub_epi32(acc, _mm256_or_si256(lt, tie));
            }
            let mut out = [0i32; 8];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, acc);
            for i in 0..8 {
                rank[i] = out[i] as u8;
            }
            return;
        }
        #[repr(align(32))]
        struct K([i32; 64]);
        let mut kt = K([i32::MAX; 64]); // keys flipped to signed order; the padding sorts last
        for i in 0..k {
            kt.0[i] = (key(i) ^ 0x8000_0000) as i32;
        }
        let nch = k.div_ceil(8);
        for i in 0..k {
            let ki = _mm256_set1_epi32(kt.0[i]);
            let mut r = 0u32;
            for c in 0..nch {
                let t = _mm256_load_si256(kt.0.as_ptr().add(c * 8) as *const __m256i);
                let lt = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(ki, t))) as u32;
                let eq = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(ki, t))) as u32;
                let before = (1u32 << i.saturating_sub(c * 8).min(8)) - 1;
                r += lt.count_ones() + (eq & before).count_ones();
            }
            rank[i] = r as u8;
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        for i in 0..k {
            let ki = key(i);
            rank[i] = (0..k).filter(|&j| key(j) < ki || (key(j) == ki && j < i)).count() as u8;
        }
        for i in k..8 {
            rank[i] = i as u8; // the unused lanes of the vector version
        }
    }
}

/// Bit j set when click j (of the first 8 * nch) is in the 7-day window of a session starting at `st`:
/// `st - ts` in 0..=WEEK, as an unsigned wrapping difference.
#[inline(always)]
fn win_mask(ts: &[u32; 72], nch: usize, st: u32) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let s = _mm256_set1_epi32(st as i32);
        let w = _mm256_set1_epi32(WEEK as i32);
        let mut m = 0u64;
        for c in 0..nch {
            let t = _mm256_loadu_si256(ts.as_ptr().add(c * 8) as *const __m256i);
            let d = _mm256_sub_epi32(s, t);
            let ok = _mm256_cmpeq_epi32(_mm256_min_epu32(d, w), d);
            m |= (_mm256_movemask_ps(_mm256_castsi256_ps(ok)) as u32 as u64) << (c * 8);
        }
        return m;
    }
    #[allow(unreachable_code)]
    {
        let mut m = 0u64;
        for j in 0..nch * 8 {
            m |= ((st.wrapping_sub(ts[j]) <= WEEK as u32) as u64) << j;
        }
        m
    }
}

/// Bit j set when `nxt[j] >= hi` (the next click of j's campaign is not before `hi`), for the first 8 * nch.
#[inline(always)]
fn nxt_mask(nxt: &[u32; 72], nch: usize, hi: u32) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let h = _mm256_set1_epi32(hi as i32 - 1);
        let mut m = 0u64;
        for c in 0..nch {
            let t = _mm256_loadu_si256(nxt.as_ptr().add(c * 8) as *const __m256i);
            m |= (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(t, h))) as u32 as u64) << (c * 8);
        }
        return m;
    }
    #[allow(unreachable_code)]
    {
        let mut m = 0u64;
        for j in 0..nch * 8 {
            m |= ((nxt[j] >= hi) as u64) << j;
        }
        m
    }
}

#[repr(align(32))]
struct A32([u32; 72]);

/// The clicks of a person with up to 64 of them, sorted by ts, as arrays for the vector window.
struct Win {
    ts: A32,
    nxt: A32, // index of the next click of the same campaign, else the click count
    camp: [u32; 72],
    cid: [u64; 72],
}

/// rules.md "Events" 5-7 and "Clicks" 4-7 for person partition `p`: the people whose slot number starts with p.
fn partition_work(cx: &Ctx, shards: &[Shard], p: usize, w: &mut Work) {
    let local = 1usize << cx.pshift;
    let lmask = (local - 1) as u32;

    // events: counting sort by person (the low bits of the slot number), with no branch and no bounds check. A
    // dropped copy goes to person `local`, a bin past the real ones whose events are never read. `start[l + 2]`
    // counts person l; then `start[l + 1]` is where l's events begin, and moves to their end as they are placed, so
    // afterwards person l has `start[l]..start[l + 1]`.
    let st = w.start.as_mut_ptr();
    let lm = lmask as usize;
    // SAFETY: start and cstart hold local + 3 entries, and every index below is at most local + 2
    unsafe {
        let ev_bin = |r: &ERec| if r.person == DROPPED { local } else { r.person as usize & lm };
        std::ptr::write_bytes(st, 0, local + 3);
        for s in shards {
            for blk in s.ev[p].slices(EV_BLOCK) {
                for r in blk {
                    *st.add(ev_bin(r) + 2) += 1;
                }
            }
        }
        let mut run = 0;
        for l in 0..=local {
            let c = *st.add(l + 2);
            *st.add(l + 1) = run;
            run += c;
        }
        if *st.add(local + 1) == 0 {
            return;
        }
        if w.ev.len() < run as usize {
            w.ev.resize(run as usize, 0);
        }
        let ev = w.ev.as_mut_ptr();
        for s in shards {
            for blk in s.ev[p].slices(EV_BLOCK) {
                for r in blk {
                    let f = st.add(ev_bin(r) + 1);
                    *ev.add(*f as usize) = (r.ts as u64) << 32 | r.pay as u64;
                    *f += 1;
                }
            }
        }

        // clicks of the people with events, the same way: the others' go to bin `local`
        let ct = w.cstart.as_mut_ptr();
        let ck_bin = |r: &CRec| {
            let l = r.local() as usize & lm; // masked: the index is unchecked
            if *st.add(l + 1) != *st.add(l) { l } else { local }
        };
        std::ptr::write_bytes(ct, 0, local + 3);
        for s in shards {
            for blk in s.ck[p].slices(CK_BLOCK) {
                for r in blk {
                    *ct.add(ck_bin(r) + 2) += 1;
                }
            }
        }
        let mut run = 0;
        for l in 0..=local {
            let c = *ct.add(l + 2);
            *ct.add(l + 1) = run;
            run += c;
        }
        if w.ck.len() < run as usize {
            w.ck.resize(run as usize, Ck { k: 0, cid: 0 });
        }
        let ckp = w.ck.as_mut_ptr();
        for s in shards {
            for blk in s.ck[p].slices(CK_BLOCK) {
                for r in blk {
                    let f = ct.add(ck_bin(r) + 1);
                    *ckp.add(*f as usize) = Ck { k: (r.ts() as u64) << 32 | r.camp() as u64, cid: r.cid };
                    *f += 1;
                }
            }
        }
    }

    let ev = std::mem::take(&mut w.ev);
    let mut ck = std::mem::take(&mut w.ck);
    for l in 0..local {
        let (a, b) = (w.start[l] as usize, w.start[l + 1] as usize);
        if a == b {
            continue;
        }
        let name = cx.ids.slots[(p << cx.pshift) | l].key.load(Relaxed);
        let (ca, cb) = (w.cstart[l] as usize, w.cstart[l + 1] as usize);
        let evs = &ev[a..b];
        // the clicks: sorted in place, or (up to 64) as the arrays of the vector window
        let nc = cb - ca;
        if nc > 64 {
            ck[ca..cb].sort_unstable_by_key(|c| c.k);
        } else if nc > 0 {
            let mut rank = [0u8; 64];
            if nc > 1 {
                ranks_small(nc, |i| (ck[ca + i].k >> 32) as u32, &mut rank);
            }
            let sm = &mut w.small;
            for i in 0..if nc > 1 { nc.max(8) } else { 1 } {
                let (r, c) = (rank[i] as usize, &ck[ca + i.min(nc - 1)]); // lanes past nc land past the end
                sm.ts.0[r] = (c.k >> 32) as u32;
                sm.camp[r] = c.k as u32;
                sm.cid[r] = c.cid;
            }
            // nxt: scanning back, `last` holds the stamp of this person and the index of the later click
            if w.lctr >= 1 << 26 {
                w.last.fill(0);
                w.lctr = 0;
            }
            w.lctr += 1;
            let stamp = w.lctr << 6;
            for j in (0..nc).rev() {
                let v = w.last[sm.camp[j] as usize];
                sm.nxt.0[j] = if v >> 6 << 6 == stamp { v & 63 } else { nc as u32 };
                w.last[sm.camp[j] as usize] = stamp | j as u32;
            }
        }
        person_work(cx, name, evs, &ck[ca..cb], w);
    }
    w.ev = ev;
    w.ck = ck;
}

/// One person: `evs` = their surviving events as `ts << 32 | pay`, `cs` = their valid clicks by ts.
fn person_work(cx: &Ctx, name: u64, evs: &[u64], cs: &[Ck], w: &mut Work) {
    let width = cx.ndays;
    let organic = cx.ncamp;
    let Work { rows, cred, rev, assist, stamp, iv, seen, sctr, sessions, day_groups, cats, touched, small, bins, bits, bin_sess, purch, sess_day, days, dcats, .. } = w;

    // sessions + user_daily rows: the events go into their 30-minute bins, and the bins in order are the sessions
    // through raw pointers taken once (the vectors' fields would be reloaded from memory after every store), with
    // no bounds checks (an event's ts is below we - base, so its bin is below bins.len()), and every event written
    // to the purchase list, which only keeps the ones with a price
    purch.clear();
    purch.reserve(evs.len());
    let (bp, wp, pp) = (bins.as_mut_ptr(), bits.as_mut_ptr(), purch.as_mut_ptr());
    let mut np = 0;
    for &e in evs {
        let ts = (e >> 32) as u32;
        let b = (ts / GAP as u32) as usize;
        debug_assert!(b < bins.len() && b >> 6 < bits.len());
        // SAFETY: b < bins.len() and b / 64 < bits.len() (above), np < evs.len() <= purch's capacity
        unsafe {
            let bin = &mut *bp.add(b);
            bin.cnt += 1;
            bin.min = bin.min.min(ts);
            bin.max = bin.max.max(ts);
            *wp.add(b >> 6) |= 1 << (b & 63);
            *pp.add(np) = (b as u32, e as u32);
        }
        np += (e as u32 != 0) as usize;
    }
    // SAFETY: the first np entries are written
    unsafe { purch.set_len(np) };
    sessions.clear();
    day_groups.clear();
    sess_day.clear();
    days.clear();
    let mut prev_max = i64::MIN / 2;
    for (wi, word) in bits.iter_mut().enumerate() {
        let mut m = std::mem::take(word);
        while m != 0 {
            let b = wi * 64 + m.trailing_zeros() as usize;
            m &= m - 1;
            let bin = std::mem::replace(&mut bins[b], FREE_BIN);
            let (mn, mx) = (bin.min as i64, bin.max as i64);
            if mn - prev_max > GAP {
                let d = ((mn + cx.base).div_euclid(DAY) - cx.day0) as usize;
                sessions.push(Sess { start: mn, d, purchased: false, revenue: 0 });
                if days.last().map_or(true, |x| x.d != d) {
                    days.push(DayAcc { d, sessions: 0, events: 0, revenue: 0 });
                    day_groups.push((d, sessions.len() - 1));
                }
                days.last_mut().unwrap().sessions += 1;
                sess_day.push(days.len() as u32 - 1);
            }
            prev_max = prev_max.max(mx);
            days.last_mut().unwrap().events += bin.cnt;
            bin_sess[b] = sessions.len() as u32 - 1;
        }
    }
    // purchases: into their session and day
    dcats.clear();
    for &(b, pay) in purch.iter() {
        let si = bin_sess[b as usize] as usize;
        let (price, cat) = ((pay & 0xff_ffff) as i64, (pay >> 24) as u16);
        let s = &mut sessions[si];
        s.purchased = true;
        s.revenue += price;
        let g = sess_day[si];
        days[g as usize].revenue += price;
        match dcats.iter_mut().find(|c| c.0 == g && c.1 == cat) {
            Some(c) => c.2 += price,
            None => dcats.push((g, cat, price)),
        }
    }
    purch.clear();
    for (g, x) in days.iter().enumerate() {
        if !dcats.is_empty() {
            cats.extend(dcats.iter().filter(|c| c.0 == g as u32).map(|c| (c.1, c.2)));
        }
        let day = (cx.day0 + x.d as i64) as i32;
        rows.push(finish_row(UserRow { name, day, sessions: x.sessions, events: x.events, revenue: x.revenue, top: NO_CAT }, cats, &cx.cat_rank));
    }

    let cts = |c: &Ck| (c.k >> 32) as i64;
    let ccamp = |c: &Ck| (c.k & 0xffff_ffff) as usize;
    let mut credit = |camp: usize, s: &Sess| {
        let a = &mut cred[camp * width + s.d];
        a.sessions += 1;
        a.purchasing += s.purchased as u32;
        if s.purchased {
            // apart from the counts: most sessions buy nothing, and the cells every session touches stay half as big
            rev[camp * width + s.d] += s.revenue;
        }
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
    } else if nc <= 64 {
        // up to 64 clicks (sorted into `small`): per session, one vector pass finds the clicks of its 7-day window
        // [start - 7d, start] (a contiguous run); the credit goes to the last, the assists to the last click of each
        // campaign in the run
        let nch = nc.div_ceil(8);
        let valid = if nc == 64 { !0u64 } else { (1u64 << nc) - 1 };
        let adummy = assist.len() - 4; // absorbs the increments of the unused steps
        for s in sessions.iter() {
            let m = win_mask(&small.ts.0, nch, s.start as u32) & valid;
            let nz = m != 0;
            let jh = (63 - (m | 1).leading_zeros()) as usize; // the last click of the window
            let mut camp = if nz { small.camp[jh] as usize } else { organic };
            if nz && jh > 0 && small.ts.0[jh - 1] == small.ts.0[jh] {
                // credit: tie in ts -> smallest click_id
                let t = small.ts.0[jh];
                let mut best = jh;
                let mut j = jh;
                while j > 0 && small.ts.0[j - 1] == t {
                    j -= 1;
                    if cx.names.less(small.cid[j], small.cid[best]) {
                        best = j;
                    }
                }
                camp = small.camp[best] as usize;
            }
            // assists: the last click of each campaign in the window; three fixed steps, the rest in a loop
            let mut c = m & nxt_mask(&small.nxt.0, nch, jh as u32 + 1);
            macro_rules! bump {
                ($n:expr) => {
                    let j = (c.trailing_zeros() & 63) as usize;
                    let at = if c != 0 { small.camp[j] as usize * width + s.d } else { adummy + $n };
                    assist[at] += 1;
                    c &= c.wrapping_sub(1);
                };
            }
            bump!(0);
            bump!(1);
            bump!(2);
            while c != 0 {
                bump!(3);
            }
            credit(camp, s);
        }
    } else if nc * ns <= 2048 {
        // few sessions: per session, look at the clicks of its 7-day window [start - 7d, start]
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
    dict: bool, // dictionary encoded when the file has rows
    delta: bool, // DELTA_BINARY_PACKED
}

const fn col(name: &'static str, ty: i32, converted: Option<i32>) -> ColSpec {
    ColSpec { name, ty, optional: false, converted, dict: false, delta: false }
}

const USER_COLS: [ColSpec; 6] = [
    ColSpec { dict: true, ..col("user_id", T_BYTES, Some(CT_UTF8)) },
    ColSpec { delta: true, ..col("day", T_INT32, Some(CT_DATE)) },
    ColSpec { delta: true, ..col("sessions", T_INT32, None) },
    ColSpec { delta: true, ..col("events", T_INT32, None) },
    ColSpec { delta: true, ..col("revenue_cents", T_INT64, None) },
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
    // the whole file is built in memory and written by one call: a write(2) per page cost 10 ms a file when eight
    // threads wrote at once (journal and page-cache locks), the one big call does not
    let mut out: Vec<u8> = Vec::with_capacity(n * 12 + (1 << 20));
    out.extend_from_slice(b"PAR1");
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
            out.extend_from_slice(&page[at..]);
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
            let at = page_header(&mut page, false, hi - lo, if use_dict { 8 } else if c.delta { 5 } else { 0 }); // RLE_DICTIONARY, DELTA_BINARY_PACKED, PLAIN
            out.extend_from_slice(&page[at..]);
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
        } else if c.delta {
            t.list(2, Thrift::I32, 2);
            t.elem_i32(5); // DELTA_BINARY_PACKED
            t.elem_i32(3); // RLE
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
    out.extend_from_slice(&t.b);
    f.write_all(&out)
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

/// Appends the ID of key `k` as a PLAIN byte array: 4-byte length, then the bytes.
#[inline]
fn put_name(b: &mut Vec<u8>, names: &Interner, k: u64, tmp: &mut Vec<u8>) {
    if k & INTERNED == 0 {
        // a packed ID: five digit pairs to ten bytes after the length, the zero digits past its length cut off
        let len = 10 - (k.trailing_zeros() / 6) as usize;
        let mut buf = [0u8; 16];
        buf[..4].copy_from_slice(&(len as u32).to_le_bytes());
        for (i, sh) in [48, 36, 24, 12, 0].into_iter().enumerate() {
            buf[4 + 2 * i..6 + 2 * i].copy_from_slice(&PAIRS[((k >> sh) & 0xfff) as usize].to_le_bytes());
        }
        put16(b, buf, 4 + len);
        return;
    }
    tmp.clear();
    names.bytes(k, tmp);
    b.extend_from_slice(&(tmp.len() as u32).to_le_bytes());
    b.extend_from_slice(tmp);
}

/// Appends the first `n` (at most 16) of the bytes `v` with one 16-byte store instead of a copy of `n` bytes.
#[inline(always)]
fn put16(b: &mut Vec<u8>, v: [u8; 16], n: usize) {
    debug_assert!(n <= 16);
    b.reserve(16);
    // SAFETY: the reserve leaves room for 16 bytes past the length, and n <= 16 of them are now written
    unsafe {
        let l = b.len();
        std::ptr::write_unaligned(b.as_mut_ptr().add(l) as *mut [u8; 16], v);
        b.set_len(l + n);
    }
}

/// The 6-bit digit pairs of a packed ID as two bytes: 4096 entries, the first character in the low byte.
static PAIRS: [u16; 4096] = {
    let mut t = [0u16; 4096];
    let mut x = 0;
    while x < 4096 {
        t[x] = UNPACK[x >> 6] as u16 | (UNPACK[x & 63] as u16) << 8;
        x += 1;
    }
    t
};

/// DELTA_BINARY_PACKED body of `v`: blocks of 128 deltas in four miniblocks of 32, each miniblock at the bit width
/// its largest value (delta minus the block's smallest delta) needs.
fn put_delta(b: &mut Vec<u8>, v: &[i64]) {
    fn uleb(b: &mut Vec<u8>, mut x: u64) {
        while x >= 0x80 {
            b.push(x as u8 | 0x80);
            x >>= 7;
        }
        b.push(x as u8);
    }
    let zz = |x: i64| ((x << 1) ^ (x >> 63)) as u64;
    uleb(b, 128);
    uleb(b, 4);
    uleb(b, v.len() as u64);
    uleb(b, zz(v.first().copied().unwrap_or(0)));
    let mut i = 1;
    while i < v.len() {
        let m = (v.len() - i).min(128);
        let mut d = [0i64; 128];
        let mut min = i64::MAX;
        for j in 0..m {
            d[j] = v[i + j].wrapping_sub(v[i + j - 1]);
            min = min.min(d[j]);
        }
        uleb(b, zz(min));
        // a miniblock past the last delta is absent; the last one is padded with zeros (the smallest residual)
        let mut r = [0u64; 128];
        for j in 0..m {
            r[j] = d[j].wrapping_sub(min) as u64;
        }
        let mut w = [0u8; 4];
        for k in 0..m.div_ceil(32) {
            w[k] = (64 - r[k * 32..k * 32 + 32].iter().fold(0u64, |a, &x| a | x).leading_zeros()) as u8;
        }
        b.extend_from_slice(&w);
        for k in 0..m.div_ceil(32) {
            let w = w[k] as u32;
            if w == 0 {
                continue;
            }
            // 32 values of w bits: 4w bytes, written as whole 64-bit words into room reserved for them plus one
            // word (the last, partial word is stored whole and only its bytes in use are kept)
            let bytes = 4 * w as usize;
            b.reserve(bytes + 8);
            // SAFETY: the full words end at or before `bytes` and the last store at most 8 bytes later, all inside
            // the reserved room
            unsafe {
                let base = b.as_mut_ptr().add(b.len());
                let (mut acc, mut nb, mut o) = (0u64, 0u32, 0usize);
                for &x in &r[k * 32..k * 32 + 32] {
                    acc |= x << nb; // nb < 64; bits past 64 come back below
                    let nn = nb + w;
                    if nn >= 64 {
                        std::ptr::write_unaligned(base.add(o) as *mut u64, acc.to_le());
                        o += 8;
                        acc = if nn > 64 { x >> (w - (nn - 64)) } else { 0 };
                        nb = nn - 64;
                    } else {
                        nb = nn;
                    }
                }
                if nb > 0 {
                    std::ptr::write_unaligned(base.add(o) as *mut u64, acc.to_le());
                    o += nb.div_ceil(8) as usize;
                }
                debug_assert_eq!(o, bytes);
                b.set_len(b.len() + bytes);
            }
        }
        i += m;
    }
}

/// One run of the RLE/bit-packed hybrid: `run` copies of the `vb`-byte value `idx`.
#[inline(always)]
fn put_run(b: &mut Vec<u8>, run: usize, idx: u32, vb: usize) {
    // at most 10 bytes of ULEB128 and 4 of the value: one 16-byte store
    let mut buf = [0u8; 16];
    let mut v = (run as u64) << 1;
    let mut n = 0;
    while v >= 0x80 {
        buf[n] = v as u8 | 0x80;
        v >>= 7;
        n += 1;
    }
    buf[n] = v as u8;
    buf[n + 1..n + 5].copy_from_slice(&idx.to_le_bytes());
    put16(b, buf, n + 1 + vb);
}

fn write_users(path: &Path, rows: &[UserRow], cat_names: &[Bytes], names: &Interner) -> std::io::Result<()> {
    let mut tmp = Vec::with_capacity(64);
    let n = rows.len();
    let mut run: Vec<u8> = Vec::new();
    let mut vals: Vec<i64> = Vec::new();
    // user_id is dictionary encoded: one entry per run of equal IDs (a person's days are consecutive rows), and
    // the data pages are runs of that entry's index
    let (mut entries, mut idx, mut last): (usize, u32, Option<u64>) = (0, u32::MAX, None);
    // sessions and events stay far below 2^31: main() refuses inputs with that many events
    write_parquet(path, &USER_COLS, n, |c, lo, hi, b| {
        if c == 0 && lo == DICT_PAGE {
            let mut prev: Option<u64> = None;
            for r in rows {
                if prev != Some(r.name) {
                    put_name(b, names, r.name, &mut tmp);
                    entries += 1;
                    prev = Some(r.name);
                }
            }
            (idx, last) = (u32::MAX, None);
            return entries;
        }
        let part = &rows[lo..hi];
        match c {
            0 => {
                let w = (usize::BITS - (entries.max(2) - 1).leading_zeros()) as usize;
                b.push(w as u8);
                let mut count = 0;
                for r in part.iter() {
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
            1..=4 => {
                vals.clear();
                match c {
                    1 => vals.extend(part.iter().map(|r| r.day as i64)),
                    2 => vals.extend(part.iter().map(|r| r.sessions as i64)),
                    3 => vals.extend(part.iter().map(|r| r.events as i64)),
                    _ => vals.extend(part.iter().map(|r| r.revenue)),
                }
                put_delta(b, &vals);
            }
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
                for r in part.iter() {
                    cur |= ((r.top != NO_CAT) as u8) << k;
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
                for r in part.iter() {
                    if r.top != NO_CAT {
                        let v = cat_names[r.top as usize];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_lookup_matches_get() {
        // IDs of 1..=10 bytes, with a NUL byte, a shared prefix, a trailing NUL: `get_short` agrees with `get`
        let ids: Vec<&'static [u8]> = vec![b"c_1", b"c_12", b"c_123", b"c_1234", b"c_12345", b"c_123456", b"c_1234567", b"c_12345678", b"a", b"a\0", b"a\0\0", b"\0", b"\xff\xfe", b"x_y_z_w"];
        let list: Vec<(Bytes, u16)> = ids.iter().enumerate().map(|(i, &b)| (b, i as u16)).collect();
        let m = IdMap::new(&list).with_short(&list);
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut probes: Vec<Vec<u8>> = ids.iter().map(|b| b.to_vec()).collect();
        for _ in 0..3000 {
            let n = rng.below(11);
            probes.push((0..n).map(|_| [b'c', b'_', b'1', b'2', b'a', 0, 255][rng.below(7)]).collect());
        }
        for p in probes {
            let mut buf = p.clone();
            buf.resize(p.len() + 16, 0x5a); // 16 readable bytes after the ID
            let got = unsafe { m.get_short(buf.as_ptr(), p.len()) };
            assert_eq!(got, m.get(&p), "id {p:?}");
        }
    }

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
            campaigns: IdMap::new(&[]).with_short(&[]),
            unknown_cat: 2,
            cat_rank: vec![0, 1, 2],
            ids: IdTable::new(16),
            names: Interner::new(),
            pshift: 12,
            qshift: 60,
            abase: Ctx::record_fields(0).0,
            ihi: Ctx::record_fields(ws + 14 * DAY).1,
            ilo: Ctx::record_fields(ws + 14 * DAY).2,
        }
    }

    /// (events as (person, ts, pay), dedup records as (key, ingest, has a record)), each sorted
    type Flat = (Vec<(u32, u32, u32)>, Vec<(u64, i64, bool)>);
    fn flat(st: &mut Shard) -> Flat {
        st.ev.iter_mut().for_each(Stream::finish);
        st.lg.iter_mut().for_each(Stream::finish);
        let mut ev: Vec<_> = st.ev.iter().flat_map(|s| s.slices(EV_BLOCK)).flatten().map(|r| (r.person, r.ts, r.pay)).collect();
        // the ingest_ts as the record holds it; a far one as its exact value, from the shard's far list
        let mut lg: Vec<_> = st.lg.iter().flat_map(|s| s.slices(LG_BLOCK)).flatten().filter(|r| r.ing != ING_FAR).map(|r| (r.key, r.ing as i64, r.loc != 0)).collect();
        lg.extend(st.far.iter().map(|&(key, loc, ingest)| (key, ingest, loc != 0)));
        assert_eq!(st.far.len(), st.lg.iter().flat_map(|s| s.slices(LG_BLOCK)).flatten().filter(|r| r.ing == ING_FAR).count());
        ev.sort();
        lg.sort();
        (ev, lg)
    }

    /// `fast_split` (with the masks `scan_lines` computes) agrees with `split` on random lines made of the bytes
    /// that matter (from arena/c2's tests).
    fn fast_split_vs_split<const N: usize>(seed: u64) {
        let mut r = Rng(seed);
        let atoms = ["a", ",", "\"", "\"\"", "xy", "\r", " ", "\"ab\"", "1234567"];
        for _ in 0..200_000 {
            let mut line = String::new();
            for _ in 0..r.below(40) {
                line += r.pick(&atoms);
            }
            line.truncate(126);
            let mut buf = line.clone().into_bytes();
            buf.push(b'\n');
            buf.resize(buf.len() + 200, b'z');
            let want = split::<N>(Box::leak(line.clone().into_bytes().into_boxed_slice())).map(|f| f.map(|x| x.to_vec()));
            let got = unsafe {
                let p = buf.as_ptr();
                let [c0, q0, n0] = masks64(p);
                let [c1, q1, n1] = masks64(p.add(64));
                let e = (n0 as u128 | (n1 as u128) << 64).trailing_zeros() as usize;
                assert_eq!(e, line.len());
                let lm = (1u128 << e) - 1;
                let (c, q) = ((c0 as u128 | (c1 as u128) << 64) & lm, (q0 as u128 | (q1 as u128) << 64) & lm);
                let packed = fast_fields::<N>(p, e, c, q).map(|f| std::array::from_fn::<_, N, _>(|i| f.at(i)));
                assert_eq!(packed, fast_split::<N>(p, e, c, q), "line {line:?}");
                if e < 64 {
                    let p64 = fast_fields64::<N>(p, e, c as u64, q as u64).map(|f| std::array::from_fn::<_, N, _>(|i| f.at(i)));
                    assert_eq!(p64, packed, "line {line:?}");
                }
                packed
            }
            .map(|f| f.map(|(s, l)| buf[s..s + l].to_vec()));
            assert_eq!(got, want, "line {line:?}");
        }
    }

    /// `fast_fields64` and `fields7_long` (with the masks `scan_lines` gives them) agree with `split` on random lines
    /// made of the bytes that matter, weighted so that many have the right number of commas.
    fn length_split_vs_split<const N: usize>(seed: u64) -> usize {
        let mut r = Rng(seed);
        let atoms = ["a", ",", "\"", "\"\"", "xy", "\r", " ", "\"ab\"", "1234567", "\"2028-02-21T05:00:00Z\"", "u1234567"];
        let mut some = 0;
        for round in 0..400_000 {
            let mut line = String::new();
            if round % 2 == 0 {
                for _ in 0..r.below(40) {
                    line += r.pick(&atoms);
                }
            } else {
                // N fields (now and then one more or less), most of them clean
                let n = if r.below(10) == 0 { N - 1 + r.below(3) } else { N };
                for i in 0..n {
                    if i > 0 {
                        line.push(',');
                    }
                    for _ in 0..r.below(4) {
                        line += r.pick(&["a", "xy", "1234567", "2028-02-21T05:00:00Z", "u1234567", "", "\"", " "]);
                    }
                    match r.below(8) {
                        0 => line = format!("{line}\""),
                        1 => {
                            let at = line.rfind(',').map_or(0, |k| k + 1);
                            line.insert(at, '"');
                            line.push('"');
                        }
                        _ => {}
                    }
                }
            }
            line.truncate(126);
            let mut buf = line.clone().into_bytes();
            buf.push(b'\n');
            buf.resize(buf.len() + 200, b'z');
            let want = split::<N>(Box::leak(line.clone().into_bytes().into_boxed_slice())).map(|f| f.map(|x| x.to_vec()));
            let e = line.len();
            let got: Option<[Vec<u8>; N]> = unsafe {
                let p = buf.as_ptr();
                let [c0, q0, n0] = masks64(p);
                let [c1, q1, n1] = masks64(p.add(64));
                if e < 64 {
                    assert_eq!(n0.trailing_zeros() as usize, e);
                    let lm = (1u64 << e) - 1;
                    fast_fields64::<N>(p, e, c0 & lm, q0 & lm).map(|f| std::array::from_fn(|i| f.at(i)).map(|(s, l)| buf[s..s + l].to_vec()))
                } else {
                    assert!(n0 == 0 && n1.trailing_zeros() as usize == e - 64);
                    let lm = (1u64 << (e - 64)) - 1;
                    assert_eq!(N, 7);
                    fields7_long(p, e, [c0, q0, c1 & lm, q1 & lm]).map(|f| {
                        std::array::from_fn(|i| {
                            let (fp, l) = f.at(p, i);
                            std::slice::from_raw_parts(fp, l).to_vec()
                        })
                    })
                }
            };
            some += got.is_some() as usize;
            assert_eq!(got, want, "line {line:?}");
        }
        some
    }

    #[test]
    fn length_split_matches_split() {
        assert!(length_split_vs_split::<7>(11) > 20_000);
        assert!(length_split_vs_split::<7>(12) > 20_000);
    }

    #[test]
    fn fast_split_matches_split() {
        fast_split_vs_split::<2>(1);
        fast_split_vs_split::<4>(2);
        fast_split_vs_split::<7>(3);
    }

    #[test]
    fn prices() {
        assert_eq!(parse_price(b"0250"), Some(250));
        assert_eq!(parse_price(b"000000000000000000000250"), Some(250));
        assert_eq!(parse_price(b"10000000"), Some(10_000_000));
        assert_eq!(parse_price(b"10000001"), None);
        assert_eq!(parse_price(b"0"), None);
        assert_eq!(parse_price(b"12.5"), None);
        assert_eq!(parse_price(b""), None);
        assert_eq!(parse_price(b"99999999999999999999999"), None);
    }

    /// `ts_at` and `ts_memo` agree with `parse_ts` on arena/c2's timestamp mix: F1/F2 with random fields near a
    /// window holding February 29, 13-digit F3, digit strings of any length, and a stray byte now and then.
    #[test]
    fn timestamps_match_general_parser() {
        let mut r = Rng(7);
        let mut days = [NO_DAY; 64];
        for _ in 0..300_000 {
            let mut s: Vec<u8> = match r.below(4) {
                0 => format!("2028-0{}-{:02}T{:02}:{:02}:{:02}Z", 2 + r.below(2), r.below(32), r.below(25), r.below(61), r.below(61)).into_bytes(),
                1 => format!(
                    "2028-0{}-{:02}T{:02}:{:02}:{:02}{}{:02}:{:02}",
                    2 + r.below(2),
                    r.below(32),
                    r.below(25),
                    r.below(61),
                    r.below(61),
                    if r.below(2) == 0 { '+' } else { '-' },
                    r.below(25),
                    r.below(61)
                )
                .into_bytes(),
                2 => format!("18{:011}", r.next() % 100_000_000_000).into_bytes(),
                _ => (0..r.below(22)).map(|_| b'0' + r.below(10) as u8).collect(),
            };
            if r.below(8) == 0 && !s.is_empty() {
                let i = r.below(s.len());
                s[i] = b"xZT:-+ 0"[r.below(8)];
            }
            let mut buf = s.clone();
            buf.extend_from_slice(&[b'7'; 32]);
            let want = parse_ts(&s);
            assert_eq!(unsafe { ts_at(buf.as_ptr(), s.len()) }, want, "{:?}", String::from_utf8_lossy(&s));
            assert_eq!(unsafe { ts_memo(buf.as_ptr(), s.len(), &mut days) }, want, "memo {:?}", String::from_utf8_lossy(&s));
        }
    }

    #[test]
    fn streams_keep_every_record_where_push_said() {
        let mut r = Rng(17);
        let mut arena = Arena::new();
        let mut s12: Vec<Stream<ERec>> = (0..23).map(|_| Stream::new()).collect();
        let mut s24: Vec<Stream<LRec>> = (0..23).map(|_| Stream::new()).collect();
        let mut want12: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); 23];
        let mut want24: Vec<Vec<(u64, u32, u32)>> = vec![Vec::new(); 23];
        let mut at: Vec<(*mut ERec, u32)> = Vec::new();
        for i in 0..400_000u32 {
            let b = r.below(23);
            if r.below(2) == 0 {
                let rec = ERec { person: i, ts: r.next() as u32, pay: b as u32 };
                at.push((s12[b].push(&mut arena, EV_BLOCK, rec), i));
                want12[b].push((rec.person, rec.ts, rec.pay));
            } else {
                let rec = LRec { key: r.next(), ing: i, loc: b as u32 };
                s24[b].push(&mut arena, LG_BLOCK, rec);
                want24[b].push((rec.key, rec.ing, rec.loc));
            }
        }
        sfence();
        s12.iter_mut().for_each(Stream::finish);
        s24.iter_mut().for_each(Stream::finish);
        for b in 0..23 {
            let got: Vec<_> = s12[b].slices(EV_BLOCK).flatten().map(|r| (r.person, r.ts, r.pay)).collect();
            assert_eq!(got, want12[b]);
            assert_eq!(s12[b].len(EV_BLOCK), want12[b].len());
            let got: Vec<_> = s24[b].slices(LG_BLOCK).flatten().map(|r| (r.key, r.ing, r.loc)).collect();
            assert_eq!(got, want24[b]);
        }
        assert!(at.iter().all(|&(p, i)| unsafe { (*p).person } == i));
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
            let (fa, fb) = (flat(&mut a), flat(&mut b));
            kept += fb.0.len();
            assert!(fa == fb, "round {round}: fast parse differs\n{}", std::str::from_utf8(file).unwrap());
        }
        assert!(kept > 1000, "too few kept events ({kept}) to mean anything");
    }

    #[test]
    fn ts_at_matches_parse_ts() {
        let mut r = Rng(0xfeed_beef_1357_9bdf);
        let mut days = [NO_DAY; 64];
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
        // 13 bytes, the F3 length with its own path: digits, and now and then any other byte anywhere
        let mut good = 0;
        for i in 0..300_000 {
            let mut buf: Vec<u8> = (0..13).map(|_| b'0' + r.below(10) as u8).collect();
            if i % 3 == 0 {
                buf[r.below(13)] = *[0u8, b'/', b':', b' ', 0x7f, 0x80, 0xb9, 0xff, b'a', b'-', b'.'].get(r.below(11)).unwrap();
            }
            let want = parse_ts(&buf);
            good += want.is_some() as usize;
            buf.extend_from_slice(&[b'7'; 32]);
            assert_eq!(unsafe { ts_at(buf.as_ptr(), 13) }, want, "{:?}", &buf[..13]);
        }
        assert!(good > 150_000);
    }

    #[test]
    fn pack2_matches_pack() {
        let mut r = Rng(77);
        let alphabet: Vec<u8> = (0u8..=255).filter(|&c| PACK[c as usize] != 0).collect();
        let mut both = 0;
        for i in 0..400_000 {
            let mut ids: Vec<Vec<u8>> = (0..2)
                .map(|_| {
                    let len = r.below(17);
                    (0..len).map(|_| alphabet[r.below(63)]).collect()
                })
                .collect();
            if i % 4 == 0 {
                let which = r.below(2);
                let len = ids[which].len();
                if len > 0 {
                    ids[which][r.below(len)] = r.next() as u8; // usually a byte that does not pack
                }
            }
            let bufs: Vec<Vec<u8>> = ids
                .iter()
                .map(|id| {
                    let mut b = id.clone();
                    b.extend((0..32).map(|_| r.next() as u8));
                    b
                })
                .collect();
            let want = (pack(&ids[0]), pack(&ids[1]));
            let got = unsafe { pack2(bufs[0].as_ptr(), ids[0].len(), bufs[1].as_ptr(), ids[1].len()) };
            assert_eq!(got, want, "{ids:?}");
            both += (want.0 != 0 && want.1 != 0) as usize;
        }
        assert!(both > 50_000, "too few packable pairs ({both}) to mean anything");
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
