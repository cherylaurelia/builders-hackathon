//! Shopping Sessions pipeline. rules.md is the spec; each phase names the rule it implements.
//!
//! One shuffle, no pointers into the input. Every ID (event, user, click) becomes an exact 64-bit key: an ID of up
//! to 9 ASCII bytes packs losslessly (7 bits a byte), any other ID is interned. The input is read with pread into a
//! small per-thread buffer, so no input page is ever mapped; with a cold page cache a few extra threads ask the
//! kernel for the whole input up front (`read_ahead`).
//!
//! Phases:
//!   1. links  -> every link ID goes into one lock-free ID table; a lock-free union-find over its slots hangs the
//!      larger name under the smaller, so the root of a person is their name (byte-wise smallest ID); then every
//!      ID gets its person's number written next to it. The table never grows: when it fills up the program
//!      restarts itself with a larger one (`restart_bigger`)
//!   2. events + clicks -> parsed from one chunk queue. A user ID is found (or added) in the ID table, a cache
//!      line of five IDs and their people. A valid event is a 12-byte record in its person partition; every event
//!      also leaves (event key, ingest_ts, record address) in its event-key bucket. Clicks go to person partitions
//!      too (copies are identical: no dedup needed)
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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::Acquire, Ordering::Relaxed, Ordering::Release};
use std::sync::Mutex;
use std::time::Instant;

type Bytes = &'static [u8];

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
mod wide;

/// The allocator: blocks of 1 MiB and more come from their own 2 MiB-aligned anonymous mappings advised for huge
/// pages, everything smaller from the system allocator. With THP in `madvise` mode the system allocator's large
/// blocks (the rows, the counting-sort buffers, the duplicate finder's tables, the file buffers) were 4 KiB pages:
/// a fault and a zeroed page per 4 KiB written. A block that grows moves its pages (mremap) into a new aligned
/// reservation instead of copying them.
#[cfg(target_os = "linux")]
mod huge_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    const BIG: usize = 1 << 20;
    const HP: usize = 2 << 20;

    pub struct Huge;

    fn big(l: &Layout) -> bool {
        l.size() >= BIG && l.align() <= HP
    }
    fn span(size: usize) -> usize {
        (size + HP - 1) & !(HP - 1)
    }

    /// `len` (a multiple of HP) bytes of address space at a multiple of HP, unbacked; null on failure.
    unsafe fn reserve(len: usize, prot: i32) -> *mut u8 {
        let raw = libc::mmap(std::ptr::null_mut(), len + HP, prot, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE, -1, 0);
        if raw == libc::MAP_FAILED {
            return std::ptr::null_mut();
        }
        let r = raw as usize;
        let a = (r + HP - 1) & !(HP - 1);
        if a > r {
            libc::munmap(raw, a - r);
        }
        if r + HP > a {
            libc::munmap((a + len) as *mut libc::c_void, r + HP - a);
        }
        a as *mut u8
    }

    unsafe fn map(size: usize) -> *mut u8 {
        let len = span(size);
        let p = reserve(len, libc::PROT_READ | libc::PROT_WRITE);
        if !p.is_null() {
            libc::madvise(p as *mut libc::c_void, len, libc::MADV_HUGEPAGE);
        }
        p
    }

    unsafe impl GlobalAlloc for Huge {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            if big(&l) { map(l.size()) } else { System.alloc(l) }
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            if big(&l) { map(l.size()) } else { System.alloc_zeroed(l) } // fresh mappings read as zeros
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            if big(&l) {
                libc::munmap(p as *mut libc::c_void, span(l.size()));
            } else {
                System.dealloc(p, l)
            }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            let nl = Layout::from_size_align_unchecked(new, l.align());
            match (big(&l), big(&nl)) {
                (false, false) => System.realloc(p, l, new),
                (true, true) => {
                    let (old, len) = (span(l.size()), span(new));
                    if len <= old {
                        if len < old {
                            libc::munmap(p.add(len) as *mut libc::c_void, old - len);
                        }
                        return p;
                    }
                    // the pages move to an aligned reservation of the new size: no copy, no new faults for them
                    let to = reserve(len, libc::PROT_NONE);
                    if to.is_null() {
                        return std::ptr::null_mut();
                    }
                    let q = libc::mremap(p as *mut libc::c_void, old, len, libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED, to as *mut libc::c_void);
                    if q == libc::MAP_FAILED {
                        libc::munmap(to as *mut libc::c_void, len);
                        return std::ptr::null_mut();
                    }
                    libc::madvise(q, len, libc::MADV_HUGEPAGE);
                    q as *mut u8
                }
                _ => {
                    let q = self.alloc(nl);
                    if !q.is_null() {
                        std::ptr::copy_nonoverlapping(p, q, l.size().min(new));
                        self.dealloc(p, l);
                    }
                    q
                }
            }
        }
    }
}
#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOC: huge_alloc::Huge = huge_alloc::Huge;

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

/// An ID of 1..=9 bytes, each 1..=127 (ASCII, no NUL), as a number: seven bits a byte, first byte highest, zero
/// filled. 0 = it does not pack. A byte is never 0, so the zero fill ends the ID: distinct IDs give distinct numbers,
/// and two packed IDs compare as their bytes do.
#[inline]
fn pack(b: &[u8]) -> u64 {
    if b.len() > 9 {
        return 0;
    }
    let mut v = 0u64;
    for &c in b {
        if c == 0 || c >= 128 {
            return 0;
        }
        v = (v << 7) | c as u64;
    }
    v << (7 * (9 - b.len()))
}

/// `pack` of the `len` bytes at `p`; 9 bytes at `p` must be readable.
#[inline(always)]
unsafe fn pack_at(p: *const u8, len: usize) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        use std::arch::x86_64::{_bzhi_u64, _pext_u64};
        const ONES: u64 = 0x0101_0101_0101_0101;
        const HIGH: u64 = ONES << 7;
        let lo = (p as *const u64).read_unaligned();
        // Every byte of the ID must be 1..=127: none with its high bit set, none zero. `(x - ONES) & !x` has the
        // high bit of every zero byte of x (and can only be wrong above one).
        if len == 9 {
            // nearly every ID: eight bytes and the ninth, which is 1..=127 when neither it nor it - 1 has bit 7
            let b8 = *p.add(8) as u64;
            if ((lo.wrapping_sub(ONES) & !lo) | lo | b8.wrapping_sub(1) | b8) & HIGH != 0 {
                return 0;
            }
            // first byte highest: swap, then squeeze the seven low bits of each byte together
            return (_pext_u64(lo.swap_bytes(), ONES * 0x7F) << 7) | b8;
        }
        if len.wrapping_sub(1) >= 8 {
            return 0;
        }
        // 1..=8 bytes: the bytes past `len` are cut from the ID and from the zero test
        let n8 = 8 * len as u32;
        let lo = _bzhi_u64(lo, n8);
        if (_bzhi_u64(lo.wrapping_sub(ONES) & !lo, n8) | lo) & HIGH != 0 {
            return 0;
        }
        return _pext_u64(lo.swap_bytes(), ONES * 0x7F) << 7;
    }
    #[allow(unreachable_code)]
    {
        pack(std::slice::from_raw_parts(p, len))
    }
}

/// Bit 62 of an `INTERNED` key: the rest is the ID's slot in the long-ID table (`Interner::long`).
const LONG: u64 = 1 << 62;

/// A slot of the long-ID table: an ID of a short prefix (1..=8 bytes, each 1..=127) and 32 lowercase hex digits,
/// as `meta` = LONG_READY | the prefix's `pack` >> 7 (0: free; LONG_BUSY while its finder writes the digits) and
/// the 16 bytes the digits spell.
#[repr(C, align(32))]
struct LongSlot {
    meta: AtomicU64,
    lo: AtomicU64,
    hi: AtomicU64,
    _pad: u64,
}
const LONG_BUSY: u64 = 1 << 63;
const LONG_READY: u64 = 1 << 62;
/// Slots looked at before an ID goes to the locked maps instead (the table is then nearly full): an ID found or
/// put within this many slots of its home always is, as slots are never emptied.
const LONG_PROBES: usize = 1024;

/// The 32 lowercase hex digits at `p` as 16 bytes (the first two digits the first byte, the low one of `lo`);
/// None when one is not a digit or `a`..=`f`.
#[inline(always)]
unsafe fn hex32(p: *const u8) -> Option<(u64, u64)> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        let x = _mm256_loadu_si256(p as *const _);
        let d = _mm256_sub_epi8(x, _mm256_set1_epi8(b'0' as i8));
        let l = _mm256_sub_epi8(x, _mm256_set1_epi8(b'a' as i8));
        let okd = _mm256_cmpeq_epi8(_mm256_min_epu8(d, _mm256_set1_epi8(9)), d);
        let okl = _mm256_cmpeq_epi8(_mm256_min_epu8(l, _mm256_set1_epi8(5)), l);
        if _mm256_movemask_epi8(_mm256_or_si256(okd, okl)) != -1 {
            return None;
        }
        let nib = _mm256_blendv_epi8(_mm256_add_epi8(l, _mm256_set1_epi8(10)), d, okd);
        // digit pairs: the first times 16 plus the second, then the sixteen words to bytes in order
        let v = _mm256_maddubs_epi16(nib, _mm256_set1_epi16(0x0110));
        let r = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi16(v, v)));
        return Some((_mm_cvtsi128_si64(r) as u64, _mm_extract_epi64::<1>(r) as u64));
    }
    #[allow(unreachable_code)]
    {
        let mut w = [0u64; 2];
        for i in 0..32 {
            let c = *p.add(i);
            let n = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                _ => return None,
            } as u64;
            w[i / 16] |= n << (8 * ((i % 16) / 2) + if i % 2 == 0 { 4 } else { 0 });
        }
        Some((w[0], w[1]))
    }
}

/// The IDs that do not pack (longer than 9 bytes, or a byte that is NUL or not ASCII): each gets a number, exact on
/// its bytes. A prefix and 32 lowercase hex digits (one ID in eight of generated data) goes to a table without
/// locks (`long`); anything else to locked maps, slow (generated data never comes there).
struct Interner {
    shards: Vec<Mutex<(HashMap<Box<[u8]>, u32, BuildHasherDefault<BytesHasher>>, Vec<Box<[u8]>>)>>,
    long: usize, // the table: 2^lbits `LongSlot`s
    lbits: u32,
}

impl Interner {
    const SHARDS: usize = 64;

    #[cfg(test)]
    fn new() -> Interner {
        Interner::sized(12, 1)
    }

    /// With a long-ID table of 2^lbits slots, its pages written on `threads` threads.
    ///
    /// A slot is read before it is written, and a page read first maps the shared zero page, which the write then
    /// replaces with a 4 KiB page: no huge pages, a fault per 4 KiB (v5 20M: ~36K faults). So every 2 MiB gets one
    /// write first, while nothing reads the table (or every page, where there are no huge pages: `populate`).
    fn sized(lbits: u32, threads: usize) -> Interner {
        const HP: usize = 2 << 20;
        let bytes = std::mem::size_of::<LongSlot>() << lbits;
        let long = map_anon(bytes) as usize;
        if !populate(long, bytes) {
            let pages = bytes.div_ceil(HP);
            let next = AtomicUsize::new(0);
            let work = || loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= pages {
                    break;
                }
                // SAFETY: a byte of the fresh mapping (a free slot's meta, already 0), which nothing reads yet
                unsafe { std::ptr::write_volatile((long + i * HP) as *mut u8, 0) };
            };
            std::thread::scope(|sc| {
                for _ in 1..threads.min(pages) {
                    // a thread that cannot start leaves its pages to the others
                    let _ = std::thread::Builder::new().spawn_scoped(sc, &work);
                }
                work();
            });
        }
        Interner { shards: (0..Self::SHARDS).map(|_| Mutex::new((HashMap::default(), Vec::new()))).collect(), long, lbits }
    }

    #[inline(always)]
    fn slot(&self, i: usize) -> &LongSlot {
        // SAFETY: i is masked to the table's size; the mapping lives as long as the run (zero pages are free slots)
        unsafe { &*(self.long as *const LongSlot).add(i & ((1 << self.lbits) - 1)) }
    }

    /// Where the search for the ID whose 32 hex digits are at `p` starts (the prefix plays no part): a hash of
    /// the digits as they are, so that `prefetch_long` need not check or convert them.
    #[inline(always)]
    unsafe fn long_home(&self, p: *const u8) -> usize {
        let w = |k: usize| (p.add(8 * k) as *const u64).read_unaligned();
        (mix(w(0) ^ w(1).rotate_left(16) ^ w(2).rotate_left(32) ^ w(3).rotate_left(48)) >> (64 - self.lbits)) as usize
    }

    /// Asks for the slot where the search for the ID whose 32 hex digits are at `p` starts.
    #[inline(always)]
    unsafe fn prefetch_long(&self, p: *const u8) {
        prefetch(self.slot(self.long_home(p)) as *const LongSlot);
    }

    /// The key of the `len` (33..=40) bytes at `p` from the long-ID table, when they are a prefix and 32 lowercase hex
    /// digits and the table has room near the ID's home; else 0.
    #[inline(never)]
    unsafe fn long(&self, p: *const u8, len: usize) -> u64 {
        let plen = len - 32;
        let pre = pack_at(p, plen) >> 7;
        if pre == 0 {
            return 0;
        }
        let Some((lo, hi)) = hex32(p.add(plen)) else { return 0 };
        let mut i = self.long_home(p.add(plen));
        for _ in 0..LONG_PROBES {
            let s = self.slot(i);
            let mut m = s.meta.load(Acquire);
            if m == 0 {
                match s.meta.compare_exchange(0, LONG_BUSY | pre, Acquire, Acquire) {
                    Ok(_) => {
                        s.lo.store(lo, Relaxed);
                        s.hi.store(hi, Relaxed);
                        s.meta.store(LONG_READY | pre, Release);
                        return INTERNED | LONG | (i & ((1 << self.lbits) - 1)) as u64;
                    }
                    Err(now) => m = now,
                }
            }
            if m & !(LONG_BUSY | LONG_READY) == pre {
                while m & LONG_BUSY != 0 {
                    std::hint::spin_loop();
                    m = s.meta.load(Acquire);
                }
                if s.lo.load(Relaxed) == lo && s.hi.load(Relaxed) == hi {
                    return INTERNED | LONG | (i & ((1 << self.lbits) - 1)) as u64;
                }
            }
            i += 1;
        }
        0
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

    /// The key of an ID that does not pack.
    #[inline(always)]
    fn unpacked(&self, b: &[u8]) -> u64 {
        if (33..=40).contains(&b.len()) {
            // SAFETY: the bytes are a slice of `len`
            let k = unsafe { self.long(b.as_ptr(), b.len()) };
            if k != 0 {
                return k;
            }
        }
        self.intern(b)
    }

    /// The key of an ID.
    #[inline]
    fn key(&self, b: &[u8]) -> u64 {
        match pack(b) {
            0 => self.unpacked(b),
            k => k,
        }
    }

    /// `key` of the `len` (> 0) bytes at `p` (9 bytes must be readable).
    #[inline(always)]
    unsafe fn key_at(&self, p: *const u8, len: usize) -> u64 {
        let k = pack_at(p, len);
        if k != 0 {
            return k;
        }
        self.unpacked(std::slice::from_raw_parts(p, len))
    }

    /// Appends the ID of `key` to `out`.
    #[inline]
    fn bytes(&self, key: u64, out: &mut Vec<u8>) {
        if key & (INTERNED | LONG) == INTERNED | LONG {
            let s = self.slot(key as usize);
            let pre = s.meta.load(Acquire) & !(LONG_BUSY | LONG_READY);
            self.bytes(pre << 7, out);
            const HEX: &[u8; 16] = b"0123456789abcdef";
            for w in [s.lo.load(Relaxed), s.hi.load(Relaxed)] {
                for k in 0..8 {
                    let b = (w >> (8 * k)) as u8;
                    out.push(HEX[(b >> 4) as usize]);
                    out.push(HEX[(b & 15) as usize]);
                }
            }
            return;
        }
        if key & INTERNED != 0 {
            let g = self.shards[((key >> 32) as usize) & (Self::SHARDS - 1)].lock().unwrap_or_else(|_| die("interner poisoned"));
            out.extend_from_slice(&g.1[key as u32 as usize]);
            return;
        }
        for i in 0..9 {
            let d = (key >> (56 - 7 * i)) & 127;
            if d == 0 {
                break;
            }
            out.push(d as u8);
        }
    }

    /// The ID of a packed or long-ID-table key into `out`, without the heap; its length.
    #[inline]
    fn short_bytes(&self, key: u64, out: &mut [u8; 48]) -> usize {
        let mut n = 0;
        let mut put = |k: u64| {
            for i in 0..9 {
                let d = (k >> (56 - 7 * i)) & 127;
                if d == 0 {
                    break;
                }
                out[n] = d as u8;
                n += 1;
            }
        };
        if key & INTERNED == 0 {
            put(key);
            return n;
        }
        let s = self.slot(key as usize);
        put((s.meta.load(Acquire) & !(LONG_BUSY | LONG_READY)) << 7);
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for w in [s.lo.load(Relaxed), s.hi.load(Relaxed)] {
            for k in 0..8 {
                let b = (w >> (8 * k)) as u8;
                out[n] = HEX[(b >> 4) as usize];
                out[n + 1] = HEX[(b & 15) as usize];
                n += 2;
            }
        }
        n
    }

    /// Byte-wise `a < b` of the two IDs.
    #[inline]
    fn less(&self, a: u64, b: u64) -> bool {
        if (a | b) & INTERNED == 0 {
            return a < b;
        }
        // packed or in the long-ID table: compared on the stack
        let heap = |k: u64| k & (INTERNED | LONG) == INTERNED;
        if !heap(a) && !heap(b) {
            let (mut x, mut y) = ([0u8; 48], [0u8; 48]);
            let (nx, ny) = (self.short_bytes(a, &mut x), self.short_bytes(b, &mut y));
            return x[..nx] < y[..ny];
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
    // at a multiple of 2 MiB, so that every 2 MiB of it can be a huge page: the kernel may place a mapping anywhere
    // (v5 20M: the long-ID table on 4 KiB pages took ~65K faults)
    const HP: usize = 2 << 20;
    let raw = unsafe { libc::mmap(std::ptr::null_mut(), bytes + HP, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
    if raw == libc::MAP_FAILED {
        die(&format!("mmap of {bytes} bytes: {}", std::io::Error::last_os_error()));
    }
    let (r, a) = (raw as usize, (raw as usize + HP - 1) & !(HP - 1));
    // SAFETY: the parts before the aligned start and after its `bytes` are this mapping's own, unused
    unsafe {
        if a > r {
            libc::munmap(raw, a - r);
        }
        let end = (a + bytes + 4095) & !4095;
        if r + bytes + HP > end {
            libc::munmap(end as *mut libc::c_void, r + bytes + HP - end);
        }
    }
    let p = a as *mut libc::c_void;
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

/// An empty vector with room for `cap` elements, reserved at once: a block of 1 MiB or more gets huge pages from
/// the allocator (see `huge_alloc`), and one that never grows is never moved.
fn huge_vec<T>(cap: usize) -> Vec<T> {
    Vec::with_capacity(cap)
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

/// `bytes` (a multiple of `Arena::STEP`: 2 MiB outside the tests) of zeroed memory from the arena range, never
/// unmapped; huge pages where offered.
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
    // the unit tests make thousands of arenas (two a `Shard`), each with its own chunks of the arena range: small
    // ones there, or the tests together run out of the range
    const CHUNK: usize = if cfg!(test) { 1 << 18 } else { 32 << 20 };
    const STEP: usize = if cfg!(test) { 1 << 18 } else { 2 << 20 };

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

/// An append-only list of `T` in blocks of `per` records; a record never moves.
///
/// Records reach memory a group at a time: a thread appends to a thousand streams and more at once, and a record
/// stored directly first reads its cache line from memory (read for ownership) and crowds the cache. A stream
/// keeps the records of its current group (L cache lines, at a block offset that is a multiple of GROUP: blocks
/// are 64-byte aligned, whole groups long) and writes the full group with non-temporal stores, whole lines that
/// are never read. A record's address is known when it is pushed; until its group is full and the next record
/// comes, or `finish` runs, its bytes are only in the stage. The caller fences (`sfence`) before others read,
/// and calls `finish` once the pushing is over.
struct Stream<T, const L: usize> {
    grp: *mut u8,            // where the staged group belongs in the current block; null before the first push
    end: *mut u8,            // the end of the current block
    pos: usize,              // bytes staged; GROUP = the group is full (or there is no block yet): `next` first
    stage: [[u64; 8]; L],    // 8-byte aligned, as every record here
    blocks: Vec<*mut T>,
}

impl<T, const L: usize> Stream<T, L> {
    const SIZE: usize = std::mem::size_of::<T>();
    /// Bytes written at a time: L cache lines.
    const GROUP: usize = L * 64;

    fn new() -> Stream<T, L> {
        const { assert!(L > 0 && (L * 64) % std::mem::size_of::<T>() == 0 && std::mem::align_of::<T>() <= 8 && !std::mem::needs_drop::<T>()) };
        Stream { grp: std::ptr::null_mut(), end: std::ptr::null_mut(), pos: Self::GROUP, stage: [[0; 8]; L], blocks: Vec::new() }
    }

    #[inline(always)]
    fn push(&mut self, arena: &mut Arena, per: usize, v: T) -> *mut T {
        if self.pos == Self::GROUP {
            self.next(arena, per);
        }
        let pos = self.pos;
        self.pos = pos + Self::SIZE;
        // SAFETY: pos + SIZE <= GROUP, the size of the stage; [grp, grp + GROUP) is inside the block `grow` took
        unsafe {
            ((self.stage.as_mut_ptr() as *mut u8).add(pos) as *mut T).write(v);
            self.grp.add(pos) as *mut T
        }
    }

    /// The staged group is full: it goes to memory and the group after it is staged, in a new block when this
    /// one is full (or there is none yet).
    #[inline(always)]
    fn next(&mut self, arena: &mut Arena, per: usize) {
        if !self.grp.is_null() {
            // SAFETY: GROUP bytes at a 64-byte aligned address of the block
            unsafe {
                stream_group(self.grp, self.stage.as_ptr() as *const u8, Self::GROUP);
                self.grp = self.grp.add(Self::GROUP);
            }
        }
        if self.grp == self.end {
            self.grow(arena, per);
        }
        self.pos = 0;
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, arena: &mut Arena, per: usize) {
        debug_assert!(per * Self::SIZE % Self::GROUP == 0);
        let p = arena.alloc(per * Self::SIZE);
        debug_assert!(p as usize % 64 == 0);
        self.blocks.push(p as *mut T);
        self.grp = p;
        self.end = unsafe { p.add(per * Self::SIZE) };
    }

    /// Writes the records still staged to memory with plain stores.
    fn finish(&mut self) {
        if !self.grp.is_null() {
            // SAFETY: the staged bytes belong at [grp, grp + pos), inside the block
            unsafe { std::ptr::copy_nonoverlapping(self.stage.as_ptr() as *const u8, self.grp, self.pos) };
        }
    }

    /// Records in the last block (there is one).
    fn tail(&self, b: *mut T) -> usize {
        (self.grp as usize + self.pos - b as usize) / Self::SIZE
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

/// Records in a block of a `Log`: 4 KiB of keys, then 4 KiB of payloads.
const LG_BLOCK: usize = 512;

/// The dedup records of one event-key bucket from one thread: per record a key and a payload of 8 bytes each,
/// append-only, in blocks of LG_BLOCK records whose keys fill one page and whose payloads fill the next. The
/// passes that only look at keys (`mark_duplicates`) then read half the bytes, a page at a time (a hardware
/// prefetcher stops at the end of a page, so it does not fetch the payloads behind the keys).
///
/// Like a `Stream`, eight records at a time reach memory: their keys as one cache line and their payloads as
/// another, with non-temporal stores. The caller fences (`sfence`) before others read, and calls `finish` once
/// the pushing is over.
struct Log {
    kp: *mut u64,      // where the staged keys belong, in the key page of the current block; null before the first push
    end: *mut u64,     // the end of that key page
    pos: usize,        // records staged; 8 = the group is full (or there is no block yet): `next` first
    stage: [u64; 16],  // 8 keys, then their 8 payloads
    blocks: Vec<*mut u64>,
}

impl Log {
    fn new() -> Log {
        Log { kp: std::ptr::null_mut(), end: std::ptr::null_mut(), pos: 8, stage: [0; 16], blocks: Vec::new() }
    }

    /// `arena` must hand out nothing but this type's blocks: they are 8 KiB each, so every block starts on a page.
    #[inline(always)]
    fn push(&mut self, arena: &mut Arena, key: u64, pay: u64) {
        if self.pos == 8 {
            self.next(arena);
        }
        let pos = self.pos;
        self.pos = pos + 1;
        // SAFETY: pos < 8
        unsafe {
            *self.stage.get_unchecked_mut(pos) = key;
            *self.stage.get_unchecked_mut(8 + pos) = pay;
        }
    }

    #[inline(always)]
    fn next(&mut self, arena: &mut Arena) {
        if !self.kp.is_null() {
            // SAFETY: a cache line of the key page and the one as far into the payload page, 64-byte aligned
            unsafe {
                let st = self.stage.as_ptr() as *const u8;
                stream_group(self.kp as *mut u8, st, 64);
                stream_group(self.kp.add(LG_BLOCK) as *mut u8, st.add(64), 64);
                self.kp = self.kp.add(8);
            }
        }
        if self.kp == self.end {
            self.grow(arena);
        }
        self.pos = 0;
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, arena: &mut Arena) {
        let p = arena.alloc(LG_BLOCK * 16) as *mut u64;
        debug_assert!(p as usize % 4096 == 0);
        self.blocks.push(p);
        self.kp = p;
        self.end = unsafe { p.add(LG_BLOCK) };
    }

    /// Writes the records still staged to memory with plain stores.
    fn finish(&mut self) {
        if !self.kp.is_null() {
            // SAFETY: the staged records belong at kp and LG_BLOCK entries on, inside the block
            unsafe {
                std::ptr::copy_nonoverlapping(self.stage.as_ptr(), self.kp, self.pos);
                std::ptr::copy_nonoverlapping(self.stage.as_ptr().add(8), self.kp.add(LG_BLOCK), self.pos);
            }
        }
    }

    /// Records in the last block `b`.
    fn tail(&self, b: *mut u64) -> usize {
        (self.kp as usize - b as usize) / 8 + self.pos
    }

    fn len(&self) -> usize {
        match self.blocks.last() {
            None => 0,
            Some(&b) => (self.blocks.len() - 1) * LG_BLOCK + self.tail(b),
        }
    }

    /// Per block, its keys and their payloads.
    fn blocks(&self) -> impl Iterator<Item = (&[u64], &[u64])> {
        let n = self.blocks.len();
        self.blocks.iter().enumerate().map(move |(i, &b)| {
            let len = if i + 1 == n { self.tail(b) } else { LG_BLOCK };
            unsafe { (std::slice::from_raw_parts(b as *const u64, len), std::slice::from_raw_parts(b.add(LG_BLOCK) as *const u64, len)) }
        })
    }
}

/// Copies `bytes` bytes (whole cache lines) to `dst` (64-byte aligned) with non-temporal stores where available.
#[inline(always)]
unsafe fn stream_group(dst: *mut u8, src: *const u8, bytes: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        for i in 0..bytes / 32 {
            _mm256_stream_si256(dst.add(32 * i) as *mut __m256i, _mm256_loadu_si256(src.add(32 * i) as *const __m256i));
        }
        return;
    }
    #[allow(unreachable_code)]
    std::ptr::copy_nonoverlapping(src, dst, bytes);
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

/// In `Line::up` while the links are joined: the ID hangs under another, the rest is 1 + that one's slot.
const LINKED: u32 = 1 << 31;

/// A cache line of the ID table: five user IDs, and for each what it hangs under.
///
/// `up` while the links are joined: 0 = the ID is its own person (or the root of its link tree), else LINKED | 1 +
/// the slot above it. Once the links are flattened it is the ID's person (`IdTable::own` of the root's slot,
/// never 0), and 0 only for an ID added since that nobody has looked up yet: such an ID is its own person.
/// A lookup is one 64-byte load, one compare of its eight words with the ID, and one 4-byte load.
#[repr(C, align(64))]
struct Line {
    key: [AtomicU64; 5],
    up: [AtomicU32; 5],
    _pad: u32,
}

/// The IDs of a batch that are not in their home lines (`IdTable::people`): where they are in the batch.
struct Late {
    n: usize,
    at: [u8; BATCH],
}

/// A quarter of a `Line`: what `IdTable::slots` and a home count in, so the parse loops ask for an ID's line with
/// `slots.as_ptr().add(home)`.
#[repr(C)]
struct Slot([u64; 2]);

/// Every user ID seen in links, events and clicks: open addressing on the key, lock free, never resized.
/// An ID's slot is 8 * its line + its place in the line (0..5).
struct IdTable {
    slots: &'static [Slot], // the lines, in quarters
    lines: &'static [Line],
    bits: u32, // 2^bits quarters, 2^(bits - 2) lines
    part: u32, // 2^part lines hold the names of one person partition
    #[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
    wide: bool, // the CPU compares a line with one instruction (AVX-512)
    count: AtomicUsize,
    limit: usize,
}

impl IdTable {
    /// A table of 2^(bits - 2) lines whose people fall into 2^pbits partitions (`own`).
    fn new(bits: u32, pbits: u32) -> IdTable {
        assert!((6..=30).contains(&bits) && pbits + 4 <= bits); // at least four lines a partition (`bins`)
        let n = 1usize << bits;
        let p = map_anon(n * std::mem::size_of::<Slot>());
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
        let wide = wide::available();
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
        let wide = false;
        // SAFETY: zeroed memory (page aligned) is n empty quarters, n / 4 empty lines
        unsafe {
            IdTable {
                slots: std::slice::from_raw_parts(p as *const Slot, n),
                lines: std::slice::from_raw_parts(p as *const Line, n / 4),
                bits,
                part: bits - 2 - pbits,
                wide,
                count: AtomicUsize::new(0),
                limit: n / 4 * 3,
            }
        }
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
                    self.lines[i * HUGE / std::mem::size_of::<Line>()].up[0].store(0, Relaxed);
                }
            },
        );
    }

    /// Where the search for `key` starts: the first quarter of a line.
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

    /// The key in slot `t`: a slot a search returned, or one a `Line::up` names.
    #[inline(always)]
    fn key(&self, t: u32) -> &AtomicU64 {
        debug_assert!(((t >> 3) as usize) < self.lines.len() && t & 7 < 5);
        // SAFETY: such a slot's line is below the line count and its place below 5
        unsafe { self.lines.get_unchecked((t >> 3) as usize).key.get_unchecked((t & 7) as usize) }
    }

    /// What slot `t` hangs under (`Line::up`); `t` as for `key`.
    #[inline(always)]
    fn up(&self, t: u32) -> &AtomicU32 {
        debug_assert!(((t >> 3) as usize) < self.lines.len() && t & 7 < 5);
        // SAFETY: as in `key`
        unsafe { self.lines.get_unchecked((t >> 3) as usize).up.get_unchecked((t & 7) as usize) }
    }

    /// The slot of `key`, added if new. `ins` is the caller's count of additions not yet reported.
    #[cfg(test)]
    fn slot(&self, key: u64, ins: &mut usize) -> u32 {
        self.slot_scan(self.home(key) as u32 * 2, key, ins)
    }

    /// Bit j set when slot j of the line at `home` holds `key`, and bit j set when it is empty. Each key is read
    /// whole (an aligned 8 bytes is never seen half written).
    #[inline(always)]
    fn look(&self, home: u32, key: u64) -> (u32, u32) {
        // SAFETY: home is a multiple of 4 below the (power of two, >= 2^16) quarter count: the first quarter of
        // a line, 64 readable bytes
        let l = unsafe { self.slots.as_ptr().add(home as usize) };
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            use std::arch::x86_64::*;
            let (a, b) = (_mm256_loadu_si256(l as *const __m256i), _mm256_loadu_si256((l as *const __m256i).add(1)));
            // the first four keys, and the fifth: the other three words of `b` are not keys
            let m = |x: __m256i| (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, x))) | (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(b, x))) & 1) << 4) as u32;
            return (m(_mm256_set1_epi64x(key as i64)), m(_mm256_setzero_si256()));
        }
        #[allow(unreachable_code)]
        {
            let k = unsafe { &(*(l as *const Line)).key };
            let m = |x: u64| (0..5).fold(0, |m, j| m | ((k[j].load(Relaxed) == x) as u32) << j);
            (m(key), m(0))
        }
    }

    /// The slot of `key` (added if new) when that is settled by the line at `home`: None when the line is full of
    /// other keys, so the key is (or will be added) further on. The caller asks for the next line (`next_line`)
    /// and comes back when it is in the cache, instead of waiting for it here.
    ///
    /// A key sits in the first free slot from its home at the time it was added, and slots never empty again, so
    /// it is in its home line unless the line was full by then. None is exact: a slot never changes once it holds
    /// a key, so five keys read here that are not `key` are the line's five keys for good. A key that is not in
    /// the line goes straight into the line's first empty slot as it was read, without a second look at the slots
    /// before it (each holds another key, for good). When another thread takes that slot first, `slot_scan` goes
    /// on from it: it may hold this very key now.
    #[inline(always)]
    fn slot_in_line(&self, home: u32, key: u64, ins: &mut usize) -> Option<u32> {
        let (eq, empty) = self.look(home, key);
        if eq != 0 {
            return Some(home * 2 + eq.trailing_zeros());
        }
        if empty == 0 {
            return None;
        }
        let t = home * 2 + empty.trailing_zeros();
        if self.key(t).compare_exchange(0, key, Relaxed, Relaxed).is_ok() {
            self.added(ins);
            return Some(t);
        }
        Some(self.slot_scan(t, key, ins))
    }

    /// The line after the one at `home`, with its cache line on the way.
    #[inline(always)]
    fn next_line(&self, home: u32) -> u32 {
        let h = (home as usize + 4) & (self.slots.len() - 1);
        prefetch(unsafe { self.slots.as_ptr().add(h) });
        h as u32
    }

    /// The slot of `key`, added if new, searching from slot `from` (a slot of its home line with only other keys
    /// before it in the line, or the first slot of a line after a full one): slot by slot through the line and on
    /// through the lines after it, adding the key in the first free slot.
    #[inline(never)]
    fn slot_scan(&self, from: u32, key: u64, ins: &mut usize) -> u32 {
        let mask = self.lines.len() - 1;
        let (mut l, mut j) = ((from >> 3) as usize & mask, (from & 7) as usize);
        loop {
            while j < 5 {
                // SAFETY: l <= mask, j < 5
                let s = unsafe { self.lines.get_unchecked(l).key.get_unchecked(j) };
                let t = (l * 8 + j) as u32;
                let k = s.load(Relaxed);
                if k == key {
                    return t;
                }
                if k == 0 {
                    match s.compare_exchange(0, key, Relaxed, Relaxed) {
                        Ok(_) => {
                            self.added(ins);
                            return t;
                        }
                        Err(now) if now == key => return t,
                        Err(_) => {}
                    }
                }
                j += 1;
            }
            j = 0;
            l = (l + 1) & mask;
        }
    }

    /// Counts an addition; a table with three IDs a line restarts the program with a larger one.
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

    /// The person of the ID in slot `t` when it hangs under nothing, never 0. Its partition (`person >>
    /// (part + 3)`) is the high bits of its line; below that, 1 + (its place in the line << part | the low bits of
    /// its line): the people of a partition are numbered from 1 through 5 << part.
    #[inline(always)]
    fn own(&self, t: u32) -> u32 {
        let (line, j, k) = (t >> 3, t & 7, self.part);
        (((line >> k) << (k + 3)) | ((line & ((1 << k) - 1)) * 5 + j)) + 1
    }

    /// The bins the sessions phase counts the people of a partition in: one past the largest number `own` gives
    /// within a partition, up to a multiple of 8. Below 2^(part + 3) (a partition has at least four lines), so it
    /// is also a number in a partition that no person has: the one the pay records of purchases go by.
    #[inline(always)]
    fn bins(&self) -> usize {
        ((5usize << self.part) + 1).next_multiple_of(8)
    }

    /// The slot of the ID that names `person`: `own` backwards.
    #[inline(always)]
    fn slot_of(&self, person: u32) -> u32 {
        let (x, k) = (person.wrapping_sub(1), self.part);
        let y = x & ((8 << k) - 1);
        ((((x >> (k + 3)) << k) + y / 5) << 3) | (y % 5)
    }

    /// The name of `person`: the key of its root. For a number that is nobody's: 0 or some other key.
    #[inline(always)]
    fn name(&self, person: u32) -> u64 {
        let t = self.slot_of(person);
        match self.lines.get((t >> 3) as usize) {
            Some(l) if t & 7 < 5 => l.key[(t & 7) as usize].load(Relaxed),
            _ => 0,
        }
    }

    /// The person of the ID in slot `t`, once the links are flattened.
    #[inline(always)]
    fn person(&self, t: u32) -> u32 {
        let v = self.up(t).load(Relaxed);
        if v != 0 {
            return v;
        }
        self.first_seen(t)
    }

    /// `person` of an ID added after the links that has no person written down yet: its own, and written down
    /// (whoever comes first writes the same number).
    #[cold]
    #[inline(never)]
    fn first_seen(&self, t: u32) -> u32 {
        let v = self.own(t);
        self.up(t).store(v, Relaxed);
        v
    }

    /// The people of the `n` IDs `key`, whose home lines `home` were asked for a batch ago (`aim`): the first pass
    /// of `Shard::drain`. An ID that is not in its home line (the line was full of other keys when the ID was
    /// added, or is now) asks for the next line and is looked up from there after the others.
    #[inline(always)]
    fn people(&self, key: &[u64; BATCH], home: &mut [u32; BATCH], n: usize, person: &mut [u32; BATCH], ins: &mut usize) {
        let n = n.min(BATCH);
        let mut late = Late { n: 0, at: [0; BATCH] };
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
        if self.wide {
            // SAFETY: `wide` says the CPU has AVX-512 F and VL
            unsafe { self.people_wide(key, home, n, person, &mut late, ins) };
            return self.people_late(key, home, person, &late, ins);
        }
        for i in 0..n {
            self.person_at(i, key, home, person, &mut late, ins);
        }
        self.people_late(key, home, person, &late, ins)
    }

    /// `people` for the ID at `i`, by the line at its home alone: its person, or its place in `late`.
    #[inline(always)]
    fn person_at(&self, i: usize, key: &[u64; BATCH], home: &mut [u32; BATCH], person: &mut [u32; BATCH], late: &mut Late, ins: &mut usize) {
        match self.slot_in_line(home[i], key[i], ins) {
            Some(t) => person[i] = self.person(t),
            None => {
                home[i] = self.next_line(home[i]);
                late.at[late.n & (BATCH - 1)] = i as u8; // n < BATCH: at most one entry per ID of the batch
                late.n += 1;
            }
        }
    }

    /// The IDs of `late`: each from the line after its home line, which holds five other keys for good.
    #[inline(always)]
    fn people_late(&self, key: &[u64; BATCH], home: &[u32; BATCH], person: &mut [u32; BATCH], late: &Late, ins: &mut usize) {
        for &i in &late.at[..late.n.min(BATCH)] {
            let i = i as usize & (BATCH - 1);
            person[i] = self.person(self.slot_scan(home[i] * 2, key[i], ins));
        }
    }

    /// `person_at` out of line, for the 512-bit loop: the few IDs that loop does not settle itself.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
    #[cold]
    #[inline(never)]
    fn person_rare(&self, i: usize, key: &[u64; BATCH], home: &mut [u32; BATCH], person: &mut [u32; BATCH], late: &mut Late, ins: &mut usize) {
        self.person_at(i & (BATCH - 1), key, home, person, late, ins)
    }

    /// The first pass of `people` with one 512-bit compare a line: the line's eight words against the ID, of which
    /// the first five are its keys. An ID found with its person written down is settled here; any other goes
    /// through `person_rare`.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
    #[target_feature(enable = "avx512f,avx512vl,avx2,bmi1,bmi2")]
    #[inline(never)]
    unsafe fn people_wide(&self, key: &[u64; BATCH], home: &mut [u32; BATCH], n: usize, person: &mut [u32; BATCH], late: &mut Late, ins: &mut usize) {
        use std::arch::x86_64::*;
        let base = self.slots.as_ptr();
        for i in 0..n.min(BATCH) {
            // SAFETY: as in `look`: a home is the first quarter of a line, 64 readable bytes; a word of the line
            // is read whole
            let l = base.add(*home.get_unchecked(i) as usize);
            let eq = _mm512_mask_cmpeq_epi64_mask(0x1f, _mm512_loadu_si512(l as *const _), _mm512_set1_epi64(*key.get_unchecked(i) as i64)) as u32;
            if eq != 0 {
                let p = (*(l as *const Line)).up.get_unchecked(eq.trailing_zeros() as usize).load(Relaxed);
                if p != 0 {
                    *person.get_unchecked_mut(i) = p;
                    continue;
                }
            }
            self.person_rare(i, key, home, person, late, ins);
        }
    }

    /// Flattens the links, one pass over the table on `threads` threads: every ID gets its person (`own` of its
    /// root) in place of what it hangs under. The linked IDs of a stretch wait, what they hang under asked for,
    /// until WAIT of them do.
    fn flatten(&self, threads: usize) {
        let ids = self;
        const SPAN: usize = 1 << 13;
        const WAIT: usize = 32;
        let n = ids.lines.len();
        par(
            threads,
            n.div_ceil(SPAN),
            || (),
            |_, t| {
                let mut wait = [(0u32, 0u32); WAIT + 4];
                let mut nw = 0;
                let settle = |wait: &[(u32, u32)]| {
                    for &(i, p) in wait {
                        // i hangs under p, so its root is p or above it
                        ids.up(i).store(ids.own(ids.find(p)), Relaxed);
                    }
                };
                for l in t * SPAN..((t + 1) * SPAN).min(n) {
                    let line = &ids.lines[l];
                    // the IDs of a line are its first slots: an ID goes into the first free one
                    for j in 0..5 {
                        if line.key[j].load(Relaxed) == 0 {
                            break;
                        }
                        let i = (l * 8 + j) as u32;
                        let u = line.up[j].load(Relaxed);
                        if u == 0 {
                            line.up[j].store(ids.own(i), Relaxed);
                        } else {
                            let p = (u & !LINKED) - 1; // an ID that hangs under another does so for good
                            prefetch(&ids.lines[(p >> 3) as usize]);
                            wait[nw] = (i, p);
                            nw += 1;
                        }
                    }
                    if nw >= WAIT {
                        settle(&wait[..nw]);
                        nw = 0;
                    }
                }
                settle(&wait[..nw]);
            },
        );
    }

    /// The root above slot `x`, halving the path on the way. While the links are joined, or while they are
    /// flattened: a slot that already holds its person gives that person's slot.
    fn find(&self, mut x: u32) -> u32 {
        loop {
            let u = self.up(x).load(Relaxed);
            if u & LINKED == 0 {
                return if u == 0 { x } else { self.slot_of(u) };
            }
            let p = (u & !LINKED) - 1;
            let g = self.up(p).load(Relaxed);
            if g & LINKED != 0 {
                let _ = self.up(x).compare_exchange(u, g, Relaxed, Relaxed);
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
            let (ka, kb) = (self.key(a).load(Relaxed), self.key(b).load(Relaxed));
            let (hi, lo) = if names.less(kb, ka) { (a, b) } else { (b, a) };
            if self.up(hi).compare_exchange(0, LINKED | (lo + 1), Relaxed, Relaxed).is_ok() {
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
    if bits + 2 > 30 {
        die("ID table would pass 2^28 lines");
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
#[cfg(test)]
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
/// of `file`, skipping the header line. Same dialect as `split`, in one SIMD pass over the chunk. The tests'
/// reference for the parsers.
#[cfg(test)]
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
    if len == 13 {
        // F3 of today's length: its first five digits behind three zero bytes, and its last eight
        const Z: u64 = u64::from_le_bytes([b'0'; 8]);
        let a = ((p as *const u64).read_unaligned() << 24) ^ (Z << 24);
        let b = (p.add(5) as *const u64).read_unaligned() ^ Z;
        if (a.wrapping_add(0x7676_7676_7676_7676) | a | b.wrapping_add(0x7676_7676_7676_7676) | b) & HI != 0 {
            return None;
        }
        return Some((swar8(a) * 100_000_000 + swar8(b)) as i64);
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
unsafe fn ts_memo(p: *const u8, len: usize, days: &mut [Day; 64]) -> Option<i64> {
    let at = |i: usize| *p.add(i);
    let f1 = len == 20 && at(19) == b'Z';
    if !f1 && !(len == 25 && (at(19) == b'+' || at(19) == b'-') && at(22) == b':') {
        return ts_at(p, len);
    }
    let a = (p as *const u64).read_unaligned();
    let b = (p.add(8) as *const u32).read_unaligned() & 0xFF_FFFF;
    // a slot per day of the month: the low bits of its two digits (any other bytes land somewhere too)
    let day = &mut days[((b ^ (b >> 6)) & 63) as usize];
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
    // hours in byte 0, minutes in byte 3, seconds in byte 6, each 0..=99
    let pb = (x * 10 + (x >> 8)) & 0x00FF_0000_FF00_00FF;
    // adding 127 - 23 (127 - 59) carries an hour over 23 (a minute or second over 59) into its byte's high bit
    if pb.wrapping_add(0x0044_0000_4400_0068) & 0x0080_0000_8000_0080 != 0 {
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

/// A date already checked, for the one-pass event loop: bytes 0..8 and 3..11 of its `YYYY-MM-DDT` and the epoch
/// ms of its midnight. The slot of a date is the low bits of its two day digits (`DAY_BITS` of bytes 3..11).
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
#[derive(Clone, Copy)]
#[repr(C)]
struct DayM {
    a: u64,
    a2: u64,
    ms: i64,
}
/// Matches no input: bytes 3..8 cannot be both 0x00 and 0xFF.
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
const NO_DAYM: DayM = DayM { a: 0, a2: u64::MAX, ms: 0 };
/// In bytes 3..11 of a timestamp: the low two bits of the tens of the day and the low four of its units.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
const DAY_BITS: u64 = (0x03 << 40) | (0x0F << 48);

/// The first five digits of a 13-digit F3 timestamp already checked: those bytes above three zero bytes, and
/// their value in ms (x 10^8). The slot is the low bits of the fourth and fifth digit (`MS_BITS`).
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
#[derive(Clone, Copy)]
#[repr(C)]
struct MsM {
    key: u64,
    val: i64,
}
/// Matches no input: a key's low three bytes are zero.
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
const NO_MSM: MsM = MsM { key: 1, val: 0 };
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
const MS_BITS: u64 = (0x03 << 48) | (0x0F << 56);

/// The 64-bit constants of the one-pass event loop. They are read from memory (the compiler cannot see their
/// values), so each use is an operand of its instruction instead of a ten-byte load of an immediate before it.
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
#[repr(C)]
struct K {
    hi: u64,       // the high bit of every byte
    hms_x: u64,    // "00:00:00"
    hms_add: u64,  // per byte, 0x7F - the largest value allowed: 2 9 : 5 9 : 5 9
    hms_pair: u64, // the bytes that hold hours, minutes and seconds once the digits are paired
    hms_mul: u64,  // 3600, 60 and 1 placed so that the three products meet at bit 48
    zeros: u64,    // "00000000"
    d76: u64,      // per byte, 0x7F - 9
    p7f: u64,      // the low seven bits of every byte
    sw_m: u64,     // `swar8`
    sw_a: u64,
    sw_b: u64,
}

#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
impl K {
    const NEW: K = K {
        hi: HI,
        hms_x: u64::from_le_bytes(*b"00:00:00"),
        hms_add: u64::from_le_bytes([0x7F - 2, 0x76, 0x7F, 0x7F - 5, 0x76, 0x7F, 0x7F - 5, 0x76]),
        hms_pair: 0x00FF_0000_FF00_00FF,
        hms_mul: (3600 << 48) | (60 << 24) | 1,
        zeros: u64::from_le_bytes([b'0'; 8]),
        d76: 0x7676_7676_7676_7676,
        p7f: 0x7F7F_7F7F_7F7F_7F7F,
        sw_m: 0x0000_00FF_0000_00FF,
        sw_a: 100 + (1_000_000 << 32),
        sw_b: 1 + (10_000 << 32),
    };
}

/// Remembers the date of the timestamp at `p` (F1 or F2: its first 19 bytes are a valid date and time). False when
/// they are not.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[cold]
#[inline(never)]
unsafe fn day_fill(p: *const u8, dm: &mut [DayM; 64]) -> bool {
    let Some(t) = civil_at(p) else { return false };
    let (a, a2) = ((p as *const u64).read_unaligned(), (p.add(3) as *const u64).read_unaligned());
    dm[std::arch::x86_64::_pext_u64(a2, DAY_BITS) as usize] = DayM { a, a2, ms: t.div_euclid(DAY) * DAY };
    true
}

/// `ts_at`, for a timestamp the one-pass loop does not take itself; a good one leaves its date (F1, F2) or its
/// first five digits (F3 of 13) remembered for the next.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[cold]
#[inline(never)]
unsafe fn ts_cold(p: *const u8, len: usize, dm: &mut [DayM; 64], fm: &mut [MsM; 64]) -> Option<i64> {
    let t = ts_at(p, len)?;
    if len == 20 || len == 25 {
        day_fill(p, dm);
    } else if len == 13 {
        let key = (p as *const u64).read_unaligned() << 24;
        let d = key ^ (u64::from_le_bytes([b'0'; 8]) << 24);
        fm[std::arch::x86_64::_pext_u64(key, MS_BITS) as usize] = MsM { key, val: (swar8(d) * 100_000_000) as i64 };
    }
    Some(t)
}

/// The time of day of the `HH:MM:SS` at `p` in seconds, and nonzero when a byte is not what it may be. It is a
/// time when that is zero and the seconds are below `DAY_SECS`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn hms_at(k: &K, p: *const u8) -> (u64, u64) {
    let x = (p as *const u64).read_unaligned() ^ k.hms_x;
    let bad = (x.wrapping_add(k.hms_add) | x) & k.hi;
    let pb = (x * 10 + (x >> 8)) & k.hms_pair;
    let secs = ((pb as u128 * k.hms_mul as u128) >> 48) as u64 & 0x1_FFFF;
    (secs, bad)
}

/// Seconds in a day: `hms_at` of an hour of 24 to 29 passes its digit check and gives this many or more.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
const DAY_SECS: u64 = 86_400;

/// The key of the ID of 8 or 9 bytes (`len`) at `p`, every byte of it 1..=127 (`pack_at`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn pack89(k: &K, p: *const u8, len: usize) -> u64 {
    pack_m(k, p, 0u8.wrapping_sub(len as u8 - 8))
}

/// `pack89` given the mask of the ninth byte: 0xFF for an ID of 9 bytes, 0 for one of 8.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn pack_m(k: &K, p: *const u8, ninth: u8) -> u64 {
    let lo = (p as *const u64).read_unaligned();
    (std::arch::x86_64::_pext_u64(lo.swap_bytes(), k.p7f) << 7) | (*p.add(8) & ninth) as u64
}

/// `ts_at` of the `tl` bytes at `tp` for the one-pass loops: F1 and F2 of a date seen before and F3 of 13 digits
/// whose first five were seen before are taken here, the rest by `ts_cold`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn ts_usual(k: &K, dm: *const DayM, fm: *const MsM, st: *mut Shard, tp: *const u8, tl: usize) -> Option<i64> {
    use std::arch::x86_64::_pext_u64;
    if tl == 20 {
        let (ta, ta2) = ((tp as *const u64).read_unaligned(), (tp.add(3) as *const u64).read_unaligned());
        let d = dm.add(_pext_u64(ta2, DAY_BITS) as usize);
        if ((*d).a == ta) & ((*d).a2 == ta2) & (*tp.add(19) == b'Z') {
            let (secs, bad) = hms_at(k, tp.add(11));
            if bad != 0 || secs >= DAY_SECS {
                return None;
            }
            return Some((*d).ms + (secs * 1000) as i64);
        }
    } else if tl == 13 {
        let key = (tp as *const u64).read_unaligned() << 24;
        let m = fm.add(_pext_u64(key, MS_BITS) as usize);
        if (*m).key == key {
            let lo = (tp.add(5) as *const u64).read_unaligned() ^ k.zeros;
            if (lo.wrapping_add(k.d76) | lo) & k.hi != 0 {
                return None;
            }
            let d = lo * 10 + (lo >> 8);
            let v = ((d & k.sw_m).wrapping_mul(k.sw_a)).wrapping_add(((d >> 16) & k.sw_m).wrapping_mul(k.sw_b)) >> 32;
            return Some((*m).val + v as i64);
        }
    } else if tl == 25 {
        let (ta, ta2) = ((tp as *const u64).read_unaligned(), (tp.add(3) as *const u64).read_unaligned());
        let d = dm.add(_pext_u64(ta2, DAY_BITS) as usize);
        let sign = *tp.add(19);
        if ((*d).a == ta) & ((*d).a2 == ta2) & ((sign == b'+') | (sign == b'-')) & (*tp.add(22) == b':') {
            let (secs, bad) = hms_at(k, tp.add(11));
            let at = |i: usize| (*tp.add(i)).wrapping_sub(b'0');
            let (h1, h2, m1, m2) = (at(20), at(21), at(23), at(24));
            let (oh, om) = (h1 as i64 * 10 + h2 as i64, m1 as i64 * 10 + m2 as i64);
            if (bad != 0) | (secs >= DAY_SECS) | (h1 > 9) | (h2 > 9) | (m1 > 9) | (m2 > 9) | (oh > 23) | (om > 59) {
                return None;
            }
            let off = (oh * 60 + om) * 60_000;
            let t = (*d).ms + (secs * 1000) as i64;
            return Some(if sign == b'+' { t - off } else { t + off });
        }
    }
    ts_cold(tp, tl, &mut (*st).dm, &mut (*st).fm)
}

/// One shape of an event line's first three fields, already worked out: the commas `c` and quotes `q` of the
/// line through its third comma, and what `window2` and the checks after it make of them. Lines come in a few
/// hundred such shapes (two ID lengths, three timestamp lengths, quotes or none), so nearly every line finds its
/// own here and skips the work. Only shapes the one-pass loop takes are kept: well quoted, both IDs 8 or 9 bytes.
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Head {
    c: u64, // 0 = no shape here (a shape has three commas)
    q: u64,
    es: u8, // where event_id starts
    us: u8, // where user_id starts
    ts: u8, // where ts starts
    tl: u8, // its length
    e9: u8, // 0xFF when event_id is 9 bytes, 0 when 8: the mask of its ninth byte
    u9: u8, // the same for user_id
}
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
const NO_HEAD: Head = Head { c: 0, q: 0, es: 0, us: 0, ts: 0, tl: 0, e9: 0, u9: 0 };
/// Places for shapes per thread: many more than there are shapes, so that two seldom want the same one (only the
/// places in use are ever touched).
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
const HEADS: usize = 4096;

/// Where the shape (c, q) is looked for first.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
fn head_slot(c: u64, q: u64) -> usize {
    (spread(c ^ (q << 7)) >> (64 - HEADS.trailing_zeros())) as usize
}

/// The shape (c, q) when it is not in its first place: in one of the three places after it, or worked out now and
/// kept (in the first free of the four, else in the first). Null when the one-pass loop does not take such a line.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[cold]
#[inline(never)]
fn head_find(heads: &mut [Head], c: u64, q: u64) -> *const Head {
    let at = |i: usize| (head_slot(c, q) + i) & (HEADS - 1);
    if let Some(i) = (1..4).find(|&i| heads[at(i)].c == c && heads[at(i)].q == q) {
        return &heads[at(i)];
    }
    let (s, e, bad) = window2(c, q);
    let mut f = Cut { s, e };
    let ((es, el), (us, ul), (ts, tl)) = (f.next(), f.next(), f.next());
    if bad != 0 || (el.wrapping_sub(8) | ul.wrapping_sub(8)) > 1 {
        return std::ptr::null();
    }
    let i = at((0..4).find(|&i| heads[at(i)].c == 0).unwrap_or(0));
    heads[i] = Head { c, q, es: es as u8, us: us as u8, ts: ts as u8, tl: tl as u8, e9: if el == 9 { 0xFF } else { 0 }, u9: if ul == 9 { 0xFF } else { 0 } };
    &heads[i]
}

/// `x.trailing_zeros()` (64 for 0) as the bare instruction. The compiler puts a clearing of the result register
/// before every `tzcnt` it emits itself, against a false dependency that the scorer's CPU (Skylake and later) does
/// not have: some fifteen instructions a line in the one-pass loops.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
fn tz(x: u64) -> usize {
    let r: usize;
    // SAFETY: one register-to-register instruction of BMI1, which this build requires
    unsafe { std::arch::asm!("tzcnt {r}, {x}", x = in(reg) x, r = lateout(reg) r, options(pure, nomem, nostack)) };
    r
}

/// `tz` of 32 bits (32 for 0).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
fn tz32(x: u32) -> usize {
    let r: usize;
    // SAFETY: as `tz`; writing the 32-bit register clears the upper half
    unsafe { std::arch::asm!("tzcnt {r:e}, {x:e}", x = in(reg) x, r = lateout(reg) r, options(pure, nomem, nostack)) };
    r
}

/// `window` for the one-pass loop: the same three results from fewer steps. With S the fields' first bytes,
/// Z their last and Y the closing quotes moved onto their boundaries: a quote is in exactly one of S and Z, or it
/// is out of place; an opening quote moves its field's start up by adding its own bit, a closing one moves the end
/// down by subtracting its bit from the boundary's.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
fn window2(b: u64, q: u64) -> (u64, u64, u64) {
    let s = (b << 1) | 1;
    let z = b >> 1;
    let x = q & s;
    let yz = q & z;
    let bad = (q & !(s ^ z)) | ((yz << 1).wrapping_sub(x) & b);
    (s + x, b - yz, bad)
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

/// True when `f` is on a file system that keeps its files in memory (tmpfs, ramfs): nothing of it waits on a disk.
/// Such a file answers the probe of `uncached` with "not supported", which reads as not cached.
#[cfg(target_os = "linux")]
fn in_memory(f: &InFile) -> bool {
    use std::os::unix::io::AsRawFd;
    const TMPFS: i64 = 0x0102_1994;
    const RAMFS: i64 = 0x8584_58f6;
    // SAFETY: fstatfs fills the zeroed struct or fails; a failure counts as a disk
    let mut s: libc::statfs = unsafe { std::mem::zeroed() };
    let ok = unsafe { libc::fstatfs(f.f.as_raw_fd(), &mut s) } == 0;
    ok && matches!(s.f_type as i64, TMPFS | RAMFS)
}

/// Cold page cache: asks the kernel for the whole input at once, in the order it will be parsed, from a few
/// threads of their own, so the disk works at full speed from the first moment whatever the parse is doing.
/// An input that is cached already (16 probes all answer without the disk, or are of files kept in memory) is
/// left alone: on a tmpfs the threads only took two CPUs from the link builder for the calls they made.
#[cfg(target_os = "linux")]
fn read_ahead(queue: Vec<(&'static InFile, usize, usize)>, _threads: usize) {
    use std::os::unix::io::AsRawFd;
    let n = queue.len();
    let cold = |i: usize| {
        let (f, s, e) = queue[i * n / 16];
        !in_memory(f) && uncached(f, (s + e) / 2)
    };
    if n == 0 || !(0..16).any(cold) {
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

/// Bytes past the data that `read_span` keeps readable (and starts with a `\n`): the parsers load 144 past a line start.
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

/// `x` without its lowest set bit.
#[inline(always)]
fn drop_low(x: u64) -> u64 {
    x & x.wrapping_sub(1)
}

/// The fields that end at the boundaries `b` (commas, or the line's `\n`) of a 64-byte window that starts where a
/// field starts, given the window's quotes `q` below its last boundary: (where each field's value starts, where it
/// ends, nonzero when a quote is out of place). Field i's value runs from the i-th set bit of the first word to the
/// i-th set bit of the second, quotes left out; the first word has one more bit, past the last boundary.
///
/// Well quoted <=> every quote sits at a field's first byte (`s`) or right before its boundary (`z`), none is both
/// (a lone `"`), and per field a first-byte quote (`x`) comes with a last-byte quote (`y`, moved onto the boundary):
/// then `y - x` borrows across exactly the quoted fields' bytes and clears every boundary, while a field with only
/// one of the two leaves a bit on the boundary that ends it (or the lowest such field does).
#[inline(always)]
fn window(b: u64, q: u64) -> (u64, u64, u64) {
    let s = (b << 1) | 1;
    let z = b >> 1;
    let x = q & s;
    let y = (q << 1) & b;
    let bad = (q & !(s | z)) | (x & z) | (y.wrapping_sub(x) & b);
    ((s ^ x) | (x << 1), (b & !y) | (y >> 1), bad)
}

/// The fields of a window still to come, in order (`window`).
#[derive(Clone, Copy)]
struct Cut {
    s: u64,
    e: u64,
}

impl Cut {
    /// The next field as (start, length), quotes removed.
    #[inline(always)]
    fn next(&mut self) -> (usize, usize) {
        let (s, e) = (self.s.trailing_zeros() as usize, self.e.trailing_zeros() as usize);
        (self.s, self.e) = (drop_low(self.s), drop_low(self.e));
        (s, e - s)
    }
}

/// `split::<7>` of the event line at `line` (136 bytes must be readable): its first three fields from `line`, the
/// other four from `line + o`, o, and the line's length with its `\n`. None sends the line to `split`: it is
/// malformed, or it does not have its first three fields in its first 64 bytes and the other four in the 64 bytes
/// from the fourth's start.
///
/// Two windows of 64 bytes, so every mask is one word: nearly every line has its third comma by byte 51, and then
/// 55 bytes or fewer to its end.
#[inline(always)]
unsafe fn split_event(line: *const u8) -> Option<(Cut, Cut, usize, usize)> {
    let [ca, qa, na] = masks64(line);
    let ba = ca | na;
    let b2 = drop_low(drop_low(ba));
    if b2 == 0 {
        return None;
    }
    let ma = b2 ^ b2.wrapping_sub(1); // through the third boundary
    let (sa, ea, bad_a) = window(ba & ma, qa & ma);
    // the second window starts after the third comma: its masks are those of the line's bytes 64.. moved down
    // (taken at a fixed place, so they do not wait for the comma to be found). Nearly every line ends in its first
    // 96 bytes: the last 32 of the 128 are looked at only when it does not
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let [c1, q1, n1] = {
        let [mut c1, mut q1, mut n1] = masks32(line.add(64));
        if na | n1 == 0 {
            let [c, q, n] = masks32(line.add(96));
            (c1, q1, n1) = (c1 | c << 32, q1 | q << 32, n1 | n << 32);
        }
        [c1, q1, n1]
    };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    let [c1, q1, n1] = masks64(line.add(64));
    let k2 = b2.trailing_zeros();
    let o = k2 as usize + 1;
    let down = |m0: u64, m1: u64| ((m0 >> 1) >> k2) | (m1 << (63 - k2));
    let (cb, qb, nb) = (down(ca, c1), down(qa, q1), down(na, n1));
    let bb = cb | nb;
    let b3 = drop_low(drop_low(drop_low(bb)));
    if b3 == 0 {
        return None;
    }
    let mb = b3 ^ b3.wrapping_sub(1); // through the fourth boundary
    let (sb, eb, bad_b) = window(bb & mb, qb & mb);
    // the first three boundaries are commas; of the next four only the last is a newline
    if (ba & ma & na) | bad_a | ((nb & mb) ^ (b3 & b3.wrapping_neg())) | bad_b != 0 {
        return None;
    }
    Some((Cut { s: sa, e: ea }, Cut { s: sb, e: eb }, o, o + b3.trailing_zeros() as usize + 1))
}

/// `split::<N>` of the line at `line`, through 64-byte windows that each start where a field starts: every field as
/// (start, length) in the line, quotes removed, and the line's length with its `\n`. None sends the line to `split`:
/// it is malformed, or a field is longer than 63 bytes. 64 bytes past the line's end must be readable.
///
/// For the lines `split_event` and `split_click` do not take: long IDs (v5) push their fields past the first window.
#[inline(always)]
unsafe fn split_n<const N: usize>(line: *const u8) -> Option<([(usize, usize); N], usize)> {
    let mut f = [(0, 0); N];
    let (mut pos, mut i) = (0, 0);
    loop {
        let [c, q, n] = masks64(line.add(pos));
        let b = c | n;
        if b == 0 {
            return None;
        }
        // the boundaries this window gives: all it has, or as many as the line still wants
        let take = (b.count_ones() as usize).min(N - i);
        let mut k = b;
        for _ in 1..take {
            k = drop_low(k);
        }
        let last = k & k.wrapping_neg();
        let m = last ^ last.wrapping_sub(1);
        let (mut fs, mut fe, bad) = window(b & m, q & m);
        // a newline only as the last boundary of all
        let nl = if i + take == N { last } else { 0 };
        if bad | ((n & m) ^ nl) != 0 {
            return None;
        }
        for _ in 0..take {
            let (a, e) = (fs.trailing_zeros() as usize, fe.trailing_zeros() as usize);
            (fs, fe) = (drop_low(fs), drop_low(fe));
            *f.get_unchecked_mut(i) = (pos + a, e - a);
            i += 1;
        }
        pos += last.trailing_zeros() as usize + 1;
        if i == N {
            return Some((f, pos));
        }
    }
}

/// `split::<4>` of the click line at `line` (72 bytes must be readable), as `split_event`: its fields, and its
/// length with its `\n`. None unless the whole line is in its first 64 bytes.
#[inline(always)]
unsafe fn split_click(line: *const u8) -> Option<(Cut, usize)> {
    let [c, q, n] = masks64(line);
    let b = c | n;
    let b3 = drop_low(drop_low(drop_low(b)));
    if b3 == 0 {
        return None;
    }
    let m = b3 ^ b3.wrapping_sub(1);
    let (s, e, bad) = window(b & m, q & m);
    if ((n & m) ^ (b3 & b3.wrapping_neg())) | bad != 0 {
        return None;
    }
    Some((Cut { s, e }, b3.trailing_zeros() as usize + 1))
}

/// `split::<2>` of the link line at `line` (72 bytes must be readable), as `split_click`: its two fields, and its
/// length with its `\n`. None unless the whole line is in its first 64 bytes.
#[inline(always)]
unsafe fn split_link(line: *const u8) -> Option<(Cut, usize)> {
    let [c, q, n] = masks64(line);
    let b = c | n;
    let b1 = drop_low(b);
    if b1 == 0 {
        return None;
    }
    let m = b1 ^ b1.wrapping_sub(1); // through the second boundary
    let (s, e, bad) = window(b & m, q & m);
    // the first boundary is a comma, the second the newline
    if ((n & m) ^ (b1 & b1.wrapping_neg())) | bad != 0 {
        return None;
    }
    Some((Cut { s, e }, b1.trailing_zeros() as usize + 1))
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

/// When the run started (set first thing in `main`).
static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// With SESS_DEBUG set: prints how long after the start a step inside a phase ended.
fn mark(what: &str) {
    if std::env::var_os("SESS_DEBUG").is_some() {
        eprintln!("[{:7.3}s] {what}", START.get().map_or(0.0, |t| t.elapsed().as_secs_f64()));
    }
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

/// A valid, non-late event in its person partition, 8 bytes. `ts` is 0 once dedup finds an earlier copy: no
/// event of the window has ts 0 (the window starts 7 days after `Ctx::base`).
///
/// A purchase is two records: the one right before it in memory (same block) is its pay record, whose `ts` is
/// category << 24 | price_cents and whose person number is one past the partition's last (`IdTable::bins`), a
/// person no event has, so the counting sort puts the pay records away without a branch.
#[repr(C)]
#[derive(Clone, Copy)]
struct ERec {
    ts: u32, // ms after `Ctx::base`
    lw: u32, // the person's number within its partition; HAS_PAY for a purchase
}
/// In `ERec::lw`: the record before this one is its pay record.
const HAS_PAY: u32 = 1 << 31;

/// One parsed event line in its event-key bucket, 16 bytes: `loc` is its `ERec` (`loc_of`), 0 for an invalid or
/// late event (it still takes part in dedup); `ing` is its ingest_ts (`Ctx::ing`). In a `Log` the record is its
/// key and the payload `LRec::pay`.
#[repr(C)]
#[derive(Clone, Copy)]
struct LRec {
    key: u64,
    ing: u32,
    loc: u32,
}

impl LRec {
    /// `ing` and `loc` as the payload of a `Log` record.
    #[inline(always)]
    fn pay(ing: u32, loc: u32) -> u64 {
        ing as u64 | (loc as u64) << 32
    }
    #[inline(always)]
    fn of(key: u64, pay: u64) -> LRec {
        LRec { key, ing: pay as u32, loc: (pay >> 32) as u32 }
    }
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

const EV_BLOCK: usize = 768; // records: 6144 bytes
/// Cache lines a stream of each kind writes at a time (`Stream`); a block is a whole number of them. One line:
/// the stages are what a parse thread keeps in its half of the L2 (Intel VM, 60M: 1, 2 and 4 lines parse alike).
const EV_LINES: usize = 1;
const CK_LINES: usize = 1;
const CK_BLOCK: usize = 192; // records: 3072 bytes

/// Events or clicks whose user IDs are on their way from memory at a time (`Batch`).
const BATCH: usize = 64;

/// The valid events (or clicks) of the last few lines, each waiting for its user ID's line of the ID table
/// (`IdTable::aim`). When BATCH of them wait they are looked up and written out in one loop (`Shard::drain`): by
/// then every line asked for has had the time of the rest of the batch to arrive.
struct Batch {
    n: usize,
    a: [u64; BATCH],    // event key, or click key
    b: [i64; BATCH],    // ingest_ts (events)
    ukey: [u64; BATCH], // user ID
    tx: [u64; BATCH],   // ts (ms after `Ctx::base`) << 32 | pay, or campaign
    home: [u32; BATCH], // where the search for `ukey` starts
}

/// One parse thread's output.
struct Shard {
    arena: Arena,
    ev: Vec<Stream<ERec, EV_LINES>>,
    ck: Vec<Stream<CRec, CK_LINES>>,
    lg: Vec<Log>,
    larena: Arena, // the blocks of `lg`, and nothing else (`Log::push`)
    far: Vec<(u64, u32, i64)>, // (event key, loc, ingest_ts) of the records whose `ing` is ING_FAR
    buf: Vec<u8>,
    ins: usize,
    bt: Batch,
    days: [Day; 64],
    // what the one-pass loops remember and read (`parse_events`, `parse_clicks`)
    #[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
    dm: [DayM; 64],
    #[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
    fm: [MsM; 64],
    #[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
    k: K,
    #[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
    heads: Vec<Head>, // HEADS of them
    // the dates the 512-bit event loop remembers (`wide::parse_events`)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
    dv: wide::DayV,
    // event or click lines the 512-bit loops leave to `event_line` or `click_line`, a few lines later: their long IDs' slots were asked for
    // (`wide::defer`); `pn` of them from `ph` on, in a ring
    #[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
    pend: [usize; PEND],
    #[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
    pn: usize,
    #[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), allow(dead_code))]
    ph: usize,
}
/// Lines `Shard::pend` holds.
const PEND: usize = 8;
// SAFETY: the streams own their blocks; after the parse they are only read, except for `ERec::person` of distinct
// records written by the dedup phase while nothing reads them
unsafe impl Send for Shard {}
unsafe impl Sync for Shard {}

struct UserRow {
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

    /// `LRec::ing` of an ingest_ts; None for one that 32 bits cannot hold (it goes in `Shard::far`, as ING_FAR).
    #[inline(always)]
    fn ing(&self, ingest: i64) -> Option<u32> {
        // ms above ilo (a negative difference is a huge unsigned one): below ING_SPAN through ihi
        let d = ingest.wrapping_sub(self.ilo) as u64;
        if d < ING_SPAN as u64 {
            Some(d as u32 + 1)
        } else if ingest > self.ihi {
            Some(ING_LATE)
        } else {
            None
        }
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
            larena: Arena::new(),
            ev: (0..parts).map(|_| Stream::new()).collect(),
            ck: (0..parts).map(|_| Stream::new()).collect(),
            lg: (0..buckets).map(|_| Log::new()).collect(),
            far: Vec::new(),
            buf: Vec::new(),
            ins: 0,
            bt: Batch { n: 0, a: [0; BATCH], b: [0; BATCH], ukey: [0; BATCH], tx: [0; BATCH], home: [0; BATCH] },
            days: [NO_DAY; 64],
            dm: [NO_DAYM; 64],
            fm: [NO_MSM; 64],
            k: std::hint::black_box(K::NEW),
            heads: vec![NO_HEAD; HEADS],
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
            dv: wide::DayV::new(),
            pend: [0; PEND],
            pn: 0,
            ph: 0,
        }
    }

    /// Looks up the user IDs of the waiting events (`EVENTS`) or clicks and writes their records. Call when the
    /// batch is full and at the end of every chunk.
    ///
    /// Two passes: the first turns every user ID into its person (`IdTable::people`: the table lines were asked
    /// for as the lines were parsed), the second writes the records. The streams' stages are not asked for: they
    /// stay in the L2 (Intel 20M: asking cost 1.4% wall).
    #[inline(never)]
    fn drain<const EVENTS: bool>(&mut self, cx: &Ctx) {
        let Shard { arena, larena, ev, ck, lg, far, ins, bt, .. } = self;
        let n = bt.n.min(BATCH);
        bt.n = 0;
        // what the loops need, in locals: nothing is read again through `cx` or `self` per record
        let (ids, pshift, qshift, abase) = (&cx.ids, cx.pshift, cx.qshift, cx.abase);
        let (lmask, pay_bin) = ((1u32 << pshift) - 1, ids.bins() as u32);
        let (evp, ckp, lgp) = (ev.as_mut_ptr(), ck.as_mut_ptr(), lg.as_mut_ptr());
        let mut person = [0u32; BATCH];
        ids.people(&bt.ukey, &mut bt.home, n, &mut person, ins);
        for i in 0..n {
            let (a, tx, person) = (bt.a[i], bt.tx[i], person[i]);
            let part = (person >> pshift) as usize;
            // SAFETY: a person's partition (`IdTable::own`) is below the 2^pbits streams of `ev` and `ck`;
            // the shift of the event key leaves qbits bits, a bucket below the 2^qbits streams of `lg`
            unsafe {
                if EVENTS {
                    let s = &mut *evp.add(part);
                    let (ts, pay, l) = ((tx >> 32) as u32, tx as u32, person & lmask);
                    let rec = if pay == 0 {
                        s.push(arena, EV_BLOCK, ERec { ts, lw: l })
                    } else {
                        // the pay record and its event in one group, so next to each other in memory: when one
                        // place is left in the group it goes to a pay record of nothing
                        if s.pos == EV_LINES * 64 - 8 {
                            s.push(arena, EV_BLOCK, ERec { ts: 0, lw: pay_bin });
                        }
                        s.push(arena, EV_BLOCK, ERec { ts: pay, lw: pay_bin });
                        s.push(arena, EV_BLOCK, ERec { ts, lw: l | HAS_PAY })
                    };
                    let loc = loc_of(abase, rec);
                    let ing = cx.ing(bt.b[i]).unwrap_or_else(|| {
                        far.push((a, loc, bt.b[i]));
                        ING_FAR
                    });
                    (*lgp.add((spread(a) >> qshift) as usize)).push(larena, a, LRec::pay(ing, loc));
                } else {
                    (*ckp.add(part)).push(arena, CK_BLOCK, CRec::new(a, person & ((1 << pshift) - 1), (tx >> 32) as u32, tx as u32));
                }
            }
        }
    }

    /// A valid event (`events`) or click joins the batch, its user ID's line of the ID table asked for.
    #[inline(always)]
    fn wait(&mut self, cx: &Ctx, events: bool, a: u64, b: i64, ukey: u64, ts: i64, x: u32) {
        let i = self.bt.n;
        // SAFETY: n < BATCH between calls: a full batch is drained at once
        unsafe {
            *self.bt.a.get_unchecked_mut(i) = a;
            *self.bt.b.get_unchecked_mut(i) = b;
            *self.bt.ukey.get_unchecked_mut(i) = ukey;
            *self.bt.tx.get_unchecked_mut(i) = ((ts - cx.base) as u32 as u64) << 32 | x as u64;
            *self.bt.home.get_unchecked_mut(i) = cx.ids.aim(ukey);
        }
        self.bt.n = i + 1;
        if i + 1 == BATCH {
            if events {
                self.drain::<true>(cx);
            } else {
                self.drain::<false>(cx);
            }
        }
    }

    /// Any parsed event: its dedup record. `loc` = 0 for an invalid or late one.
    #[inline(always)]
    fn log(&mut self, cx: &Ctx, ekey: u64, ingest: i64, loc: u32) {
        let q = (spread(ekey) >> cx.qshift) as usize;
        let ing = cx.ing(ingest).unwrap_or_else(|| {
            self.far.push((ekey, loc, ingest));
            ING_FAR
        });
        // SAFETY: q is below the 2^qbits streams of `lg`: the shift leaves qbits bits
        unsafe { self.lg.get_unchecked_mut(q) }.push(&mut self.larena, ekey, LRec::pay(ing, loc));
    }

    /// A valid, non-late event: into its person's partition, plus its dedup record, once its user ID is at hand.
    #[inline(always)]
    fn keep(&mut self, cx: &Ctx, ekey: u64, ingest: i64, ukey: u64, ts: i64, pay: u32) {
        self.wait(cx, true, ekey, ingest, ukey, ts, pay);
    }

    /// A valid click, into its person's partition once its user ID is at hand.
    #[inline(always)]
    fn click(&mut self, cx: &Ctx, cid: u64, ukey: u64, ts: i64, camp: u16) {
        self.wait(cx, false, cid, 0, ukey, ts, camp as u32);
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
    let t0 = *START.get_or_init(Instant::now);
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
    let id_bits = env_num("SESS_ID_BITS").map(|v| v as u32).unwrap_or(usize::BITS - (id_est * 7 / 5).leading_zeros()).clamp(16, 30);
    // about 90,000 events per person partition and per event-key bucket: more of them slow the parse (one open
    // block each per thread), fewer leave the later phases with work that no longer fits a core's cache
    let ev_est = ev_bytes / 72;
    // and no more than 2^9 of each: past that a thread's open streams (3 per partition pair) outgrow its half of an
    // L2 shared by two hyperthreads. Intel Cascade Lake, 2 cores x 2 threads, 40M events: 2^10 + 2^10 +2.3% wall
    // against 2^9 + 2^9, while 2^8 + 2^8 is even with it (164K events a partition); box, 75M events, 8 threads
    // on 4 CPUs: 2^10 3.03 s, 2^9 2.76 s. Only inputs over ~69M events change. The partitions' floor (a click
    // keeps its person's number in the partition in 16 bits) stays above the cap.
    let pbits = (env_num("SESS_PBITS").map(|v| v as u32).unwrap_or_else(|| log2_near(ev_est / 90_000).min(9))).clamp(4, 12.min(id_bits - 4)).max((id_bits + 1).saturating_sub(16));
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
    let ids = IdTable::new(id_bits, pbits);
    // the long-ID table: one ID in eight is long in generated data, some 600 input bytes a distinct one; about
    // half full
    let lbits = env_num("SESS_LONG_BITS").map(|v| v as u32).unwrap_or(usize::BITS - ((ev_bytes + bytes(&ck_files)) / 400).leading_zeros()).clamp(12, 28);
    let names = Interner::sized(lbits, threads);
    mark("files listed, ID table mapped");
    let (n_links, small) = std::thread::scope(|sc| {
        let links = sc.spawn(|| {
            ids.populate(threads);
            mark("ID table populated");
            build_links(&ids, &names, link_files, threads)
        });
        let small = small_tables(&data);
        mark("small tables read");
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
        pshift: id_bits + 1 - pbits,
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
        s.lg.iter_mut().for_each(Log::finish);
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
    // a thread's rows: about one per 9 events, and room for twice its share (more only grows the vector)
    let rows_cap = n_ev / threads / 4 + 1024;
    // the counting sorts' buffers: room for the largest partition at once, in huge pages
    let ev_cap = order.first().map_or(0, |x| x.0);
    let ck_cap = (0..np).map(|p| shards.iter().map(|s| s.ck[p].len(CK_BLOCK)).sum::<usize>()).max().unwrap_or(0);
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
                    let mut w = Work::new(cx, rows_cap, ev_cap, ck_cap);
                    loop {
                        let i = next.fetch_add(1, Relaxed);
                        if i >= order.len() {
                            break;
                        }
                        partition_work(cx, shards, order[i].1, &mut w);
                    }
                    let cred: Vec<Cred> = w.cred.iter().zip(&w.buy).map(|(&c, b)| Cred { sessions: c, purchasing: b.purchasing }).collect();
                    let rev: Vec<i64> = w.buy.iter().map(|b| b.revenue).collect();
                    let assist: Vec<u32> = w.assist.iter().zip(&w.around).map(|(&a, &r)| a as u32 + (r << 16)).collect();
                    tx.send((cred, rev, assist)).unwrap_or_else(|_| die("campaign merger gone"));
                    // thread 0 writes even with no rows: the table needs at least one file
                    if t == 0 || !w.rows.is_empty() {
                        write_users(&ud.join(format!("part-{t}.parquet")), &w.rows, &w.people, cat_names, &cx.names)
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
        // an ID given twice keeps its first number: a lookup stops at the first slot that holds the key
        for &(id, v) in ids {
            match pack(id) {
                0 => {
                    m.other.entry(id).or_insert(v);
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
        let k = pack_at(p, len);
        if k != 0 {
            return self.packed(k);
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
    let rows = small_table(&data.join("products.csv"));
    let mut products: Vec<(Bytes, u16)> = Vec::with_capacity(rows.len());
    for (p, c) in rows {
        let n = cat_names.len() as u16;
        let ci = *cat_idx.entry(c).or_insert_with(|| {
            cat_names.push(c);
            n
        });
        products.push((p, ci));
    }
    // product_id is unique per rules.md; were one repeated, its last line would count: of the entries of one ID
    // an `IdMap` answers with the first it was given
    products.reverse();
    let mut seen: BytesMap<()> = BytesMap::default();
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

/// Link lines whose IDs are on their way from the ID table at a time (`LinkBatch`).
const LINKS: usize = 32;

/// The link lines a thread has parsed whose IDs wait for their lines of the ID table (`IdTable::aim`), as the
/// events of a `Batch` do: when LINKS of them wait, `join` adds their IDs and joins them.
struct LinkBatch {
    n: usize,
    key: [[u64; 2]; LINKS],
    home: [[u32; 2]; LINKS],
    ins: usize,
    lines: usize,
}

impl LinkBatch {
    fn new() -> LinkBatch {
        LinkBatch { n: 0, key: [[0; 2]; LINKS], home: [[0; 2]; LINKS], ins: 0, lines: 0 }
    }

    /// A link line with the IDs `ka` and `kb` joins the batch.
    #[inline(always)]
    fn push(&mut self, ids: &IdTable, names: &Interner, ka: u64, kb: u64) {
        let i = self.n;
        // SAFETY: n < LINKS between calls: a full batch is joined at once
        unsafe {
            *self.key.get_unchecked_mut(i) = [ka, kb];
            *self.home.get_unchecked_mut(i) = [ids.aim(ka), ids.aim(kb)];
        }
        self.n = i + 1;
        self.lines += 1;
        if i + 1 == LINKS {
            self.join(ids, names);
        }
    }

    /// Adds the IDs of the waiting lines to the table and joins the two of each line. Call when the batch is full
    /// and at the end of every chunk.
    ///
    /// Three passes, so nothing waits on a chain of memory: the IDs become slots (an ID that is not in its home
    /// line and finds it full asks for the next line and is looked up from there after the others), then what each
    /// slot hangs under is asked for, then the lines are joined.
    #[inline(never)]
    fn join(&mut self, ids: &IdTable, names: &Interner) {
        let n = self.n.min(LINKS);
        self.n = 0;
        let mut slot = [[0u32; 2]; LINKS];
        let mut late = [0u8; 2 * LINKS];
        let mut nl = 0;
        for i in 0..n {
            for j in 0..2 {
                match ids.slot_in_line(self.home[i][j], self.key[i][j], &mut self.ins) {
                    Some(t) => slot[i][j] = t,
                    None => {
                        self.home[i][j] = ids.next_line(self.home[i][j]);
                        late[nl] = (2 * i + j) as u8;
                        nl += 1;
                    }
                }
            }
        }
        for &t in &late[..nl] {
            let (i, j) = (t as usize / 2, t as usize % 2);
            // its home line holds five other keys for good, so the search may start at the line after it
            slot[i][j] = ids.slot_scan(self.home[i][j] * 2, self.key[i][j], &mut self.ins);
        }
        for &t in slot[..n].iter().flatten() {
            let u = ids.up(t).load(Relaxed);
            if u != 0 {
                prefetch(&ids.lines[(((u & !LINKED) - 1) >> 3) as usize]);
            }
        }
        for &[x, y] in &slot[..n] {
            if x != y {
                ids.union(names, x, y);
            }
        }
    }
}

/// rules.md "Linked IDs": every link line joins its two IDs in the ID table. Returns the number of lines joined.
fn build_links(ids: &IdTable, names: &Interner, files: &[InFile], threads: usize) -> usize {
    let mut work = Vec::new();
    chunks_of(0, files, 128 << 10, &mut work);
    // A. parse, add both IDs, join them: a batch of lines at a time (`LinkBatch`). A line that ends in its first
    //    64 bytes is split with one mask per kind of byte (`split_link`), any other by `split`
    let parts = par(
        threads,
        work.len(),
        || (Vec::<u8>::new(), LinkBatch::new()),
        |(buf, bt), i| {
            let (_, fi, s, e) = work[i];
            let (len, start, end) = read_span(&files[fi], s, e, buf);
            let (base, text) = (buf.as_ptr(), &buf[..len]);
            let first = start.saturating_sub(1);
            if let Some(nl) = memchr::memchr(b'\n', &text[first..]) {
                let mut p = first + nl + 1; // the header line is skipped
                let stop = end.min(len);
                while p < stop {
                    // SAFETY: read_span leaves PAD readable bytes of `buf` past len: split_link reads 64 from the
                    // line's start, and an ID's 9 bytes start before the line's end
                    unsafe {
                        let line = base.add(p);
                        if let Some((mut f, n)) = split_link(line) {
                            let ((sa, la), (sb, lb)) = (f.next(), f.next());
                            if la != 0 && lb != 0 {
                                bt.push(ids, names, names.key_at(line.add(sa), la), names.key_at(line.add(sb), lb));
                            }
                            p += n;
                            continue;
                        }
                    }
                    let eol = memchr::memchr(b'\n', &text[p..]).map_or(len, |i| p + i);
                    // SAFETY: the fields live in `buf`, which outlives this use; nothing keeps them
                    if let Some([a, b]) = split::<2>(unsafe { std::mem::transmute::<&[u8], Bytes>(&text[p..eol]) }) {
                        if !a.is_empty() && !b.is_empty() {
                            bt.push(ids, names, names.key(a), names.key(b));
                        }
                    }
                    p = eol + 1;
                }
            }
            bt.join(ids, names);
        },
    );
    mark("link lines joined");
    // B. every ID gets its person
    ids.flatten(threads);
    parts.iter().map(|p| p.1.lines).sum()
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

/// `event_fields` for the line at `line` with its first three fields `a` and, from `rest`, its other four `b`
/// (`split_event`); line[..144] is readable.
#[inline(always)]
unsafe fn fast_event(cx: &Ctx, line: *const u8, mut a: Cut, rest: *const u8, mut b: Cut, st: &mut Shard) {
    let (es, el) = a.next();
    if el == 0 {
        return;
    }
    let (is, il) = b.next();
    let Some(ingest) = ts_memo(rest.add(is), il, &mut st.days) else { return };
    let ekey = cx.names.key_at(line.add(es), el);
    let (us, ul) = a.next();
    let (ts, tl) = a.next();
    let (ks, kl) = b.next();
    let (ps, pl) = b.next();
    let ts = match ts_memo(line.add(ts), tl, &mut st.days) {
        Some(t) if ul != 0 && pl != 0 && t >= cx.ws && t < cx.we && ingest - t <= DAY => t,
        _ => return st.log(cx, ekey, ingest, 0),
    };
    let w = (rest.add(ks) as *const u64).read_unaligned();
    let pay = match kl {
        4 if w as u32 == u32::from_le_bytes(*b"view") || w as u32 == u32::from_le_bytes(*b"cart") => 0,
        8 if w == u64::from_le_bytes(*b"purchase") => {
            let (cs, cl) = b.next();
            let Some(p) = parse_price(std::slice::from_raw_parts(rest.add(cs), cl)) else { return st.log(cx, ekey, ingest, 0) };
            let cat = cx.products.get_at(rest.add(ps), pl).unwrap_or(cx.unknown_cat);
            (cat as u32) << 24 | p
        }
        _ => return st.log(cx, ekey, ingest, 0),
    };
    st.keep(cx, ekey, ingest, cx.names.key_at(line.add(us), ul), ts, pay);
}

/// `fast_event` for the line at `line` with its fields `f` (`split_n`).
#[inline(always)]
unsafe fn fast_event_n(cx: &Ctx, line: *const u8, f: [(usize, usize); 7], st: &mut Shard) {
    let [(es, el), (us, ul), (ts, tl), (is, il), (ks, kl), (ps, pl), (cs, cl)] = f;
    if el == 0 {
        return;
    }
    let Some(ingest) = ts_memo(line.add(is), il, &mut st.days) else { return };
    let ekey = cx.names.key_at(line.add(es), el);
    let ts = match ts_memo(line.add(ts), tl, &mut st.days) {
        Some(t) if ul != 0 && pl != 0 && t >= cx.ws && t < cx.we && ingest - t <= DAY => t,
        _ => return st.log(cx, ekey, ingest, 0),
    };
    let w = (line.add(ks) as *const u64).read_unaligned();
    let pay = match kl {
        4 if w as u32 == u32::from_le_bytes(*b"view") || w as u32 == u32::from_le_bytes(*b"cart") => 0,
        8 if w == u64::from_le_bytes(*b"purchase") => {
            let Some(p) = parse_price(std::slice::from_raw_parts(line.add(cs), cl)) else { return st.log(cx, ekey, ingest, 0) };
            let cat = cx.products.get_at(line.add(ps), pl).unwrap_or(cx.unknown_cat);
            (cat as u32) << 24 | p
        }
        _ => return st.log(cx, ekey, ingest, 0),
    };
    st.keep(cx, ekey, ingest, cx.names.key_at(line.add(us), ul), ts, pay);
}

/// The event line at `buf + p` through the two-window parser (`split_event`, `fast_event`) or the window walk
/// (`split_n`, `fast_event_n`), or, when neither takes it, as a whole line through `split` and `event_fields`.
/// Returns where the next line starts.
#[inline(never)]
unsafe fn event_line(cx: &Ctx, buf: *const u8, p: usize, len: usize, st: &mut Shard) -> usize {
    let line = buf.add(p);
    if let Some((a, b, o, n)) = split_event(line) {
        fast_event(cx, line, a, line.add(o), b, st);
        return p + n;
    }
    if let Some((f, n)) = split_n::<7>(line) {
        fast_event_n(cx, line, f, st);
        return p + n;
    }
    let file = std::slice::from_raw_parts(buf, len);
    let e = memchr::memchr(b'\n', &file[p..]).map_or(len, |i| p + i);
    if let Some(f) = split::<7>(std::mem::transmute::<&[u8], Bytes>(&file[p..e])) {
        event_fields(cx, f, st);
    }
    e + 1
}

/// Parses the event lines that start in [start, end) of the `len` bytes at `buf` into `st`; the header line is
/// skipped. `buf` must be readable for 144 bytes past `len`, and buf[len] must be `\n` (`read_span`).
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
unsafe fn parse_events(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    let file = std::slice::from_raw_parts(buf, len);
    if let Some(nl) = memchr::memchr(b'\n', &file[start.saturating_sub(1)..]) {
        let mut p = start.saturating_sub(1) + nl + 1;
        while p < end && p < len {
            p = event_line(cx, buf, p, len, st);
        }
    }
    st.drain::<true>(cx);
    sfence();
}

/// Parses the event lines that start in [start, end) of the `len` bytes at `buf` into `st`; the header line is
/// skipped. `buf` must be readable for 144 bytes past `len`, and buf[len] must be `\n` (`read_span`).
///
/// Eight lines at a time where the CPU has AVX-512 (`wide`), else one at a time (`parse_events_v3`). The wide
/// loop counts the seconds between window_start and a date, so it wants a window that starts on a whole second.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
unsafe fn parse_events(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    if wide::available() && cx.ws % 1000 == 0 {
        wide::parse_events(cx, buf, start, end, len, st)
    } else {
        parse_events_v3(cx, buf, start, end, len, st)
    }
}

/// `parse_events` one line at a time.
///
/// One pass per line for the usual shape, everything else goes to `event_line`, which is exact on any line. The
/// usual shape: the line ends in its bytes 64..96 and its first 64 bytes are all 0x20..=0x7F; its first three
/// fields end in those 64 bytes and the other four fit the 64 bytes after them; both IDs are 8 or 9 bytes;
/// ingest_ts is F1 of a date seen before. For such a line this does what `fast_event` does, with the same
/// precedence of the checks. What the line gives is written straight into the next place of the batch
/// (`Shard::wait`), which counts only once the event is known to be kept.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
unsafe fn parse_events_v3(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    use std::arch::x86_64::*;
    let file = std::slice::from_raw_parts(buf, len);
    let first = start.saturating_sub(1);
    if let Some(nl) = memchr::memchr(b'\n', &file[first..]) {
        let mut p = first + nl + 1;
        let stop = end.min(len);
        let (cm, qt, nv, lowest) = (_mm256_set1_epi8(b',' as i8), _mm256_set1_epi8(b'"' as i8), _mm256_set1_epi8(b'\n' as i8), _mm256_set1_epi8(0x20));
        let (ws, span, back) = (cx.ws, (cx.we - cx.ws) as u64, (cx.ws - cx.base) as u64);
        let (slots, hshift) = (cx.ids.slots.as_ptr(), 64 - cx.ids.bits);
        let st: *mut Shard = st;
        let k: *const K = &(*st).k;
        let (dm, fm): (*const DayM, *const MsM) = ((*st).dm.as_ptr(), (*st).fm.as_ptr());
        let heads: *const Head = (*st).heads.as_ptr();
        let bt: *mut Batch = &mut (*st).bt;
        let (ba, bb, bu, bx, bh) = ((*bt).a.as_mut_ptr(), (*bt).b.as_mut_ptr(), (*bt).ukey.as_mut_ptr(), (*bt).tx.as_mut_ptr(), (*bt).home.as_mut_ptr());
        'lines: while p < stop {
            let line = buf.add(p);
            // SAFETY: 144 bytes are readable from every line start: the three loads take 96, and no
            // field read below goes past byte 96 + 32
            let mm = |x: __m256i| _mm256_movemask_epi8(x) as u32;
            let v2 = _mm256_loadu_si256(line.add(64) as *const __m256i);
            let nhi = mm(_mm256_cmpeq_epi8(v2, nv));
            let (v0, v1) = (_mm256_loadu_si256(line as *const __m256i), _mm256_loadu_si256(line.add(32) as *const __m256i));
            // a byte of the first 64 below 0x20 or above 0x7F (as a signed byte, it is below 0x20): no newline there,
            // and both IDs are bytes that pack
            let odd = mm(_mm256_cmpgt_epi8(lowest, _mm256_min_epi8(v0, v1)));
            'usual: {
                if (nhi == 0) | (odd != 0) {
                    break 'usual;
                }
                let clo = mm(_mm256_cmpeq_epi8(v0, cm)) as u64 | (mm(_mm256_cmpeq_epi8(v1, cm)) as u64) << 32;
                let qlo = mm(_mm256_cmpeq_epi8(v0, qt)) as u64 | (mm(_mm256_cmpeq_epi8(v1, qt)) as u64) << 32;
                let thru = nhi ^ nhi.wrapping_sub(1); // through the newline
                let bhi = (mm(_mm256_or_si256(_mm256_cmpeq_epi8(v2, cm), _mm256_cmpeq_epi8(v2, nv))) & thru) as u64;
                let qhi = (mm(_mm256_cmpeq_epi8(v2, qt)) & thru) as u64;
                let e2 = tz32(nhi);
                // the first window: through the third comma
                let c3 = drop_low(drop_low(clo));
                let o = tz(c3) + 1; // 65 when there is none
                // the second window starts after the third comma, at byte o, and must hold the newline (byte 64 + e2)
                if (o > 63) | (e2 >= o) {
                    break 'usual;
                }
                let ma = c3 ^ c3.wrapping_sub(1);
                let (ca, qa) = (clo & ma, qlo & ma);
                let b2 = (clo >> o) | (bhi << (64 - o));
                let q2 = (qlo >> o) | (qhi << (64 - o));
                let (sb, eb, bad_b) = window2(b2, q2);
                // the second window well quoted, and four boundaries from the third comma through the newline:
                // seven fields
                if (bad_b != 0) | (b2.count_ones() != 4) {
                    break 'usual;
                }
                // the first window: its shape is nearly always one seen before (`Head`)
                let mut hd = heads.add(head_slot(ca, qa));
                if ((*hd).c != ca) | ((*hd).q != qa) {
                    hd = head_find(&mut (*st).heads, ca, qa);
                    if hd.is_null() {
                        break 'usual;
                    }
                }
                let next = p + 65 + e2;
                let n = (*bt).n;
                // event_id, user_id: into the batch
                let (tp, tl) = (line.add((*hd).ts as usize), (*hd).tl as usize);
                let ekey = pack_m(&*k, line.add((*hd).es as usize), (*hd).e9);
                let ukey = pack_m(&*k, line.add((*hd).us as usize), (*hd).u9);
                *ba.add(n) = ekey;
                *bu.add(n) = ukey;
                let home = (spread(ukey) >> hshift) as usize & !3;
                prefetch(slots.add(home));
                *bh.add(n) = home as u32;
                // ingest_ts: F1 when its 20 bytes are a date, a time and a Z, and the field ends there. The quotes
                // are in place, so the byte after the Z ends the field when it is a comma or a (closing) quote
                let rest = line.add(o as usize);
                let ip = rest.add(tz(sb));
                let zend = (ip.add(19) as *const u16).read_unaligned();
                if (zend != u16::from_le_bytes(*b"Z,")) & (zend != u16::from_le_bytes(*b"Z\"")) {
                    break 'usual;
                }
                let (ia, ia2) = ((ip as *const u64).read_unaligned(), (ip.add(3) as *const u64).read_unaligned());
                let di = _pext_u64(ia2, DAY_BITS) as usize;
                if ((*dm.add(di)).a != ia) | ((*dm.add(di)).a2 != ia2) {
                    if !day_fill(ip, &mut (*st).dm) {
                        break 'usual;
                    }
                }
                let (isecs, ibad) = hms_at(&*k, ip.add(11));
                if ibad != 0 || isecs >= DAY_SECS {
                    break 'usual;
                }
                let ingest = (*dm.add(di)).ms + (isecs * 1000) as i64;
                *bb.add(n) = ingest;
                'kept: {
                    // event_type: `view` or `cart` when those four bytes are followed by a comma or a (closing) quote
                    let (sb, eb) = (drop_low(sb), drop_low(eb));
                    let w = (rest.add(tz(sb)) as *const u64).read_unaligned();
                    let after = (w >> 32) as u8;
                    let mut pay = 0;
                    if !(((w as u32 == u32::from_le_bytes(*b"view")) | (w as u32 == u32::from_le_bytes(*b"cart"))) & ((after == b',') | (after == b'"'))) {
                        pay = purchase(cx, rest, w, sb, eb);
                        if pay == 0 {
                            break 'kept;
                        }
                    }
                    // product_id: empty when its first byte is where it ends
                    let (sb, eb) = (drop_low(sb), drop_low(eb));
                    if (sb & sb.wrapping_neg()) & eb != 0 {
                        break 'kept;
                    }
                    let Some(t) = ts_usual(&*k, dm, fm, st, tp, tl) else { break 'kept };
                    let tw = t.wrapping_sub(ws) as u64;
                    if (tw >= span) | (ingest.wrapping_sub(t) > DAY) {
                        break 'kept;
                    }
                    *bx.add(n) = ((tw + back) as u32 as u64) << 32 | pay as u64;
                    (*bt).n = n + 1;
                    if n + 1 == BATCH {
                        (*st).drain::<true>(cx);
                    }
                    p = next;
                    continue 'lines;
                }
                // an event that is not kept: its dedup record alone
                (*st).log(cx, *ba.add(n), *bb.add(n), 0);
                p = next;
                continue 'lines;
            }
            p = event_line(cx, buf, p, len, &mut *st);
        }
    }
    st.drain::<true>(cx);
    sfence();
}

/// What `fast_event` makes of an event whose type is neither `view` nor `cart`: category << 24 | price_cents for
/// a `purchase` with a good price, else 0 (a price is never 0). `sb` and `eb` are the second window's fields from
/// the type on, `w` the eight bytes where the type starts.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(never)]
unsafe fn purchase(cx: &Ctx, rest: *const u8, w: u64, sb: u64, eb: u64) -> u32 {
    let mut f = Cut { s: sb, e: eb };
    let (_, kl) = f.next();
    if kl != 8 || w != u64::from_le_bytes(*b"purchase") {
        return 0;
    }
    let (ps, pl) = f.next();
    let (cs, cl) = f.next();
    let Some(price) = parse_price(std::slice::from_raw_parts(rest.add(cs), cl)) else { return 0 };
    (cx.products.get_at(rest.add(ps), pl).unwrap_or(cx.unknown_cat) as u32) << 24 | price
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
unsafe fn fast_click(cx: &Ctx, line: *const u8, mut f: Cut, st: &mut Shard) {
    let (cs, cl) = f.next();
    let (us, ul) = f.next();
    if cl == 0 || ul == 0 {
        return;
    }
    let (ks, kl) = f.next();
    let (ts, tl) = f.next();
    let t = match ts_memo(line.add(ts), tl, &mut st.days) {
        Some(t) if t >= cx.base && t < cx.we => t,
        _ => return,
    };
    let Some(camp) = cx.campaigns.get_short(line.add(ks), kl) else { return };
    st.click(cx, cx.names.key_at(line.add(cs), cl), cx.names.key_at(line.add(us), ul), t, camp);
}

/// `fast_click` for the line at `line` with its fields `f` (`split_n`).
#[inline(always)]
unsafe fn fast_click_n(cx: &Ctx, line: *const u8, f: [(usize, usize); 4], st: &mut Shard) {
    let [(cs, cl), (us, ul), (ks, kl), (ts, tl)] = f;
    if cl == 0 || ul == 0 {
        return;
    }
    let t = match ts_memo(line.add(ts), tl, &mut st.days) {
        Some(t) if t >= cx.base && t < cx.we => t,
        _ => return,
    };
    let Some(camp) = cx.campaigns.get_short(line.add(ks), kl) else { return };
    st.click(cx, cx.names.key_at(line.add(cs), cl), cx.names.key_at(line.add(us), ul), t, camp);
}

/// The click line at `buf + p` through `split_click` and `fast_click` or `split_n` and `fast_click_n`, or, when
/// neither takes it, as a whole line through `split` and `click_fields`. Returns where the next line starts.
#[inline(never)]
unsafe fn click_line(cx: &Ctx, buf: *const u8, p: usize, len: usize, st: &mut Shard) -> usize {
    let line = buf.add(p);
    if let Some((f, n)) = split_click(line) {
        fast_click(cx, line, f, st);
        return p + n;
    }
    if let Some((f, n)) = split_n::<4>(line) {
        fast_click_n(cx, line, f, st);
        return p + n;
    }
    let file = std::slice::from_raw_parts(buf, len);
    let e = memchr::memchr(b'\n', &file[p..]).map_or(len, |i| p + i);
    if let Some(f) = split::<4>(std::mem::transmute::<&[u8], Bytes>(&file[p..e])) {
        click_fields(cx, f, st);
    }
    e + 1
}

/// Parses the click lines that start in [start, end) of the `len` bytes at `buf` into `st`; the header line is
/// skipped. `buf` must be readable for 144 bytes past `len`, and buf[len] must be `\n` (`read_span`).
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
unsafe fn parse_clicks(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    let file = std::slice::from_raw_parts(buf, len);
    if let Some(nl) = memchr::memchr(b'\n', &file[start.saturating_sub(1)..]) {
        let mut p = start.saturating_sub(1) + nl + 1;
        while p < end && p < len {
            p = click_line(cx, buf, p, len, st);
        }
    }
    st.drain::<false>(cx);
    sfence();
}

/// Parses the click lines that start in [start, end) of the `len` bytes at `buf` into `st`; the header line is
/// skipped. `buf` must be readable for 144 bytes past `len`, and buf[len] must be `\n` (`read_span`).
///
/// Eight lines at a time where the CPU has AVX-512 (`wide`), else one at a time (`parse_clicks_v3`), as
/// `parse_events`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
unsafe fn parse_clicks(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    if wide::available() && cx.ws % 1000 == 0 {
        wide::parse_clicks(cx, buf, start, end, len, st)
    } else {
        parse_clicks_v3(cx, buf, start, end, len, st)
    }
}

/// `parse_clicks` one line at a time.
///
/// One pass per line for the usual shape, as `parse_events_v3`; every other line goes to `click_line`. The usual
/// shape: the line ends in its first 64 bytes and every byte before that is 0x20..=0x7F; both IDs are 8 or 9 bytes.
/// For such a line this does what `fast_click` does.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
unsafe fn parse_clicks_v3(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    use std::arch::x86_64::*;
    let file = std::slice::from_raw_parts(buf, len);
    let first = start.saturating_sub(1);
    if let Some(nl) = memchr::memchr(b'\n', &file[first..]) {
        let mut p = first + nl + 1;
        let stop = end.min(len);
        let (cm, qt, lowest) = (_mm256_set1_epi8(b',' as i8), _mm256_set1_epi8(b'"' as i8), _mm256_set1_epi8(0x20));
        let (base, span) = (cx.base, (cx.we - cx.base) as u64);
        let (slots, hshift) = (cx.ids.slots.as_ptr(), 64 - cx.ids.bits);
        let st: *mut Shard = st;
        let k: *const K = &(*st).k;
        let (dm, fm): (*const DayM, *const MsM) = ((*st).dm.as_ptr(), (*st).fm.as_ptr());
        let bt: *mut Batch = &mut (*st).bt;
        let (ba, bu, bx, bh) = ((*bt).a.as_mut_ptr(), (*bt).ukey.as_mut_ptr(), (*bt).tx.as_mut_ptr(), (*bt).home.as_mut_ptr());
        'lines: while p < stop {
            let line = buf.add(p);
            // SAFETY: 144 bytes are readable from every line start: the two loads take 64, and no
            // field read below goes past byte 64 + 32
            let (v0, v1) = (_mm256_loadu_si256(line as *const __m256i), _mm256_loadu_si256(line.add(32) as *const __m256i));
            let m64 = |x: __m256i, y: __m256i| _mm256_movemask_epi8(x) as u32 as u64 | (_mm256_movemask_epi8(y) as u32 as u64) << 32;
            // the bytes below 0x20 or above 0x7F (as signed bytes, they are below 0x20): the first must be the
            // line's newline, so that both IDs are bytes that pack
            let odd = m64(_mm256_cmpgt_epi8(lowest, v0), _mm256_cmpgt_epi8(lowest, v1));
            'usual: {
                let e = tz(odd);
                if (odd == 0) || *line.add(e) != b'\n' {
                    break 'usual;
                }
                let thru = odd ^ odd.wrapping_sub(1); // through the newline
                let b = (m64(_mm256_cmpeq_epi8(v0, cm), _mm256_cmpeq_epi8(v1, cm)) | odd) & thru;
                let q = m64(_mm256_cmpeq_epi8(v0, qt), _mm256_cmpeq_epi8(v1, qt)) & thru;
                let (s, en, bad) = window2(b, q);
                // well quoted, and four boundaries through the newline: four fields
                if (bad != 0) | (b.count_ones() != 4) {
                    break 'usual;
                }
                let next = p + e + 1;
                let (cs, ce) = (tz(s), tz(en));
                let (s, en) = (drop_low(s), drop_low(en));
                let (us, ue) = (tz(s), tz(en));
                let (cl, ul) = (ce - cs, ue - us);
                if (cl.wrapping_sub(8) | ul.wrapping_sub(8)) > 1 {
                    break 'usual;
                }
                let (s, en) = (drop_low(s), drop_low(en));
                let ks = tz(s);
                let kl = tz(en) - ks;
                let (s, en) = (drop_low(s), drop_low(en));
                let ts = tz(s);
                let (tp, tl) = (line.add(ts), tz(en) - ts);
                p = next;
                let Some(t) = ts_usual(&*k, dm, fm, st, tp, tl) else { continue 'lines };
                let tw = t.wrapping_sub(base) as u64;
                if tw >= span {
                    continue 'lines;
                }
                let Some(camp) = cx.campaigns.get_short(line.add(ks), kl) else { continue 'lines };
                let n = (*bt).n;
                let ukey = pack89(&*k, line.add(us), ul);
                *ba.add(n) = pack89(&*k, line.add(cs), cl);
                *bu.add(n) = ukey;
                let home = (spread(ukey) >> hshift) as usize & !3;
                prefetch(slots.add(home));
                *bh.add(n) = home as u32;
                *bx.add(n) = (tw as u32 as u64) << 32 | camp as u64;
                (*bt).n = n + 1;
                if n + 1 == BATCH {
                    (*st).drain::<false>(cx);
                }
                continue 'lines;
            }
            p = click_line(cx, buf, p, len, &mut *st);
        }
    }
    st.drain::<false>(cx);
    sfence();
}

/// rules.md "Events" 2: per event-key bucket, among the copies of one event_id the smallest ingest_ts is kept and
/// every other valid copy is marked dropped. Returns the number of records looked at exactly.
///
/// Two passes over the bucket's records find the few that may have a copy, with two bits per slot of a table that
/// fits the core's cache (a record's slot is the bits of key * 2^64 / phi below the bucket's): the first sets
/// "seen", and "again" when it already was; the second keeps the records whose slot says "again" - every copy of a
/// repeated key, plus the few other keys that share a slot with another. Only those go through an exact table on
/// the key, where the copy with the smaller ingest_ts stays and the other is dropped.
fn mark_duplicates(cx: &Ctx, shards: &[Shard], buckets: usize, threads: usize) -> usize {
    let far = far_map(shards); // empty unless an ingest_ts is far from the window
    let (far, abase) = (&far, cx.abase);
    let qb = buckets.trailing_zeros();
    par(
        threads,
        buckets,
        || (Vec::<u64>::new(), Vec::<(u64, usize)>::new(), Vec::<LRec>::new(), 0usize),
        |(seen, cand, table, total), b| {
            let n: usize = shards.iter().map(|s| s.lg[b].len()).sum();
            if n == 0 {
                return;
            }
            if n >= u32::MAX as usize {
                die("too many events in one event-key bucket");
            }
            // 8 slots a record, 32 slots a word: "seen" bits in the low half, "again" bits in the high half, so that
            // a record touches one cache line of the table
            let sbits = (n * 8).next_power_of_two().trailing_zeros().clamp(5, 64 - qb);
            seen.clear();
            seen.resize(1 << (sbits - 5), 0);
            // A record's slot is the sbits bits of its hash below the qb bits of its bucket, and those are b for
            // every record here: the hash shifted down to the slot still has b in front, which a base as many
            // words back takes away. So one shift gives the bit (its low 5 bits) and the word (the rest).
            let s0 = 64 - qb - sbits;
            let wb = seen.as_mut_ptr().wrapping_sub(b << (sbits - 5));
            // SAFETY (both passes): hash >> s0 is b << sbits | slot, so the word is one of `seen`
            let first = |key: u64| unsafe {
                let x = spread(key) >> s0;
                let w = wb.wrapping_add((x >> 5) as usize);
                let (v, t) = (*w, (1u32 << (x as u32 & 31)) as u64);
                *w = v | t | (v & t) << 32;
            };
            for s in shards {
                for (keys, _) in s.lg[b].blocks() {
                    let mut four = keys.chunks_exact(4);
                    for p in &mut four {
                        first(p[0]);
                        first(p[1]);
                        first(p[2]);
                        first(p[3]);
                    }
                    four.remainder().iter().for_each(|&k| first(k));
                }
            }
            cand.clear();
            cand.reserve(n);
            let cp = cand.as_mut_ptr();
            let mut c = 0;
            // SAFETY: c < n, the room reserved: every record is written, as its key and where its payload is (not
            // read here), and kept when its slot says "again"
            let mut second = |key: &u64, pay: *const u64| unsafe {
                let x = spread(*key) >> s0;
                *cp.add(c) = (*key, pay as usize);
                c += (((*wb.wrapping_add((x >> 5) as usize) >> 32) as u32 >> (x as u32 & 31)) & 1) as usize;
            };
            for s in shards {
                for (keys, pays) in s.lg[b].blocks() {
                    let pp = pays.as_ptr();
                    let mut i = 0;
                    while i + 4 <= keys.len() {
                        // SAFETY: i + 3 < len, of both slices
                        unsafe {
                            second(keys.get_unchecked(i), pp.add(i));
                            second(keys.get_unchecked(i + 1), pp.add(i + 1));
                            second(keys.get_unchecked(i + 2), pp.add(i + 2));
                            second(keys.get_unchecked(i + 3), pp.add(i + 3));
                        }
                        i += 4;
                    }
                    while i < keys.len() {
                        second(&keys[i], unsafe { pp.add(i) });
                        i += 1;
                    }
                }
            }
            // SAFETY: the first c entries are written
            unsafe { cand.set_len(c) };
            *total += c;
            // exact: a table on another hash of the key that holds, per key, its earliest copy so far (key 0 =
            // free; a record with key 0 has `zero` for a table). Of two copies of a key the one with the smaller
            // ingest_ts stays (the earlier one when they are equal: copies with equal ingest_ts are identical) and
            // the other is dropped, so only the earliest copy is never dropped. At most two thirds full.
            let slots = (c + c / 2 + 1).next_power_of_two().max(16);
            let tshift = 64 - slots.trailing_zeros();
            table.clear();
            table.resize(slots, LRec { key: 0, ing: 0, loc: 0 });
            let mut zero: Option<LRec> = None;
            'cands: for &(key, pay) in cand.iter() {
                // SAFETY: the payload of a record of a live block; the candidates come in the order of the blocks,
                // so these reads run forward through the payload pages
                let r = &LRec::of(key, unsafe { *(pay as *const u64) });
                let best: &mut LRec = if r.key == 0 {
                    match &mut zero {
                        Some(z) => z,
                        None => {
                            zero = Some(*r);
                            continue;
                        }
                    }
                } else {
                    let mut h = (r.key.wrapping_mul(0xD6E8_FEB8_6659_FD93) >> tshift) as usize;
                    loop {
                        // SAFETY: h < slots, by the shift and the mask
                        let e = unsafe { &mut *table.as_mut_ptr().add(h) };
                        if e.key == r.key {
                            break e;
                        }
                        if e.key == 0 {
                            *e = *r;
                            continue 'cands;
                        }
                        h = (h + 1) & (slots - 1);
                    }
                };
                let lost = if cx.ingest_of(r, far) < cx.ingest_of(best, far) { std::mem::replace(best, *r) } else { *r };
                if lost.loc != 0 {
                    // dropped: ts 0, an event of bin 0 (person_work) and no purchase (partition_work). Only
                    // stored, never read here: the record's cache line is not waited for
                    // SAFETY: the address of an ERec in a live arena; no other thread touches this record now
                    unsafe { (*rec_at(abase, lost.loc)).ts = 0 };
                }
            }
        },
    )
    .iter()
    .map(|x| x.3)
    .sum()
}

#[derive(Clone, Copy, Default)]
struct Cred {
    sessions: u32,
    purchasing: u32,
}

#[derive(Clone, Copy)]
struct Buy {
    purchasing: u32,
    revenue: i64,
}

#[repr(align(32))]
struct A32([u32; 72]);

/// All of one person's events in one 30-minute bin of the window (`ts` is in ms after `Ctx::base`, the bin is
/// `ts / GAP`): two events in a bin are never a session break, two bins apart always are, adjacent bins break
/// exactly when `min` of the later minus `max` of the earlier exceeds GAP. The largest ts is kept inverted
/// (`nmax` = !max), so that one unsigned minimum of two lanes updates both ends. A free bin is (MAX, MAX, 0);
/// `sess` is the session the bin went to when it was last taken.
#[derive(Clone, Copy)]
#[repr(C)]
struct Bin {
    min: u32,
    nmax: u32,
    cnt: u32,
    sess: u32,
}
const FREE_BIN: Bin = Bin { min: u32::MAX, nmax: u32::MAX, cnt: 0, sess: 0 };

/// A session of the person at hand.
#[derive(Clone, Copy)]
#[repr(C)]
struct Ses {
    start: u32, // ms after `Ctx::base`
    d: u32,     // day of the window
    row: u32,   // its row, counted from the person's first
    rev: i64,   // revenue
}

/// A session of a person with more than 64 clicks (`credit_big`).
struct Sess {
    start: i64, // ms after `Ctx::base`
    d: usize,
    revenue: i64,
}

/// One sessions thread: its results, and scratch reused across partitions.
struct Work {
    rows: Vec<UserRow>,
    people: Vec<(u64, u32)>, // the people of `rows` in order: name, rows
    // Per campaign and day (cell = campaign * days + day), in arrays of their own so that the ones every session
    // touches stay small: sessions credited; sessions assisted, in 16 bits with the times each count went round
    // apart (`around`); and for the credited sessions with a purchase their number and revenue.
    cred: Vec<u32>,
    assist: Vec<u16>,
    around: Vec<u32>,
    buy: Vec<Buy>,
    doff: i64, // ts (ms after `Ctx::base`) + doff = ms after the midnight that starts day 0 of the window
    start: Vec<u32>, // see partition_work
    cstart: Vec<u32>,
    act: Vec<u32>, // the people of the partition at hand who have events: the first `nact` entries
    nact: usize,
    ev: Vec<u32>,           // the partition's events by person: their ts
    pl: Vec<[u32; 3]>,      // its purchases as they are found: person, ts, pay
    pls: Vec<(u32, u32)>,   // and by person: ts, pay
    pstart: Vec<u32>,       // like `start`, for `pls`
    ck: Vec<CRec>,  // the partition's clicks by person
    ckt: Vec<CRec>, // room for the sort of a person's many clicks
    sat: Vec<u32>,  // and its counts per 30-minute bin
    // the ts and the campaign of the clicks of the person at hand (up to 64), as arrays for the vector passes
    cts: A32,
    ccamp: A32,
    bins: Vec<Bin>, // the bins of the person at hand (all free between people)
    bits: Vec<u64>, // which bins are taken
    // the sessions of the person at hand
    ses: Vec<Ses>,
    dcats: Vec<(u32, u16, i64)>, // (row, category, price sum)
    best: Vec<i64>,              // price sum of a row's top category
    eq: [u64; 64],               // per click of the person at hand, the clicks of the same campaign
    // people with more than 64 clicks
    sessions: Vec<Sess>,
    sbit: Vec<u64>, // the 30-minute bins a session of the person starts in
    spre: Vec<u32>, // per word of `sbit`, the bits of the words before it
    sday: Vec<u32>, // per day, and past the last: the person's sessions that start before it
    stamp: Vec<bool>,
    iv: Vec<(i64, i64)>,
    seen: Vec<u32>,
    sctr: u32,
    touched: Vec<usize>,
}

impl Work {
    fn new(cx: &Ctx, rows_cap: usize, ev_cap: usize, ck_cap: usize) -> Work {
        let cells = (cx.ncamp + 1) * cx.ndays;
        let local = (1usize << cx.pshift) + 1;
        let nbins = ((cx.we - cx.base) / GAP) as usize + 2;
        Work {
            rows: huge_vec(rows_cap),
            people: huge_vec(rows_cap / 2),
            cred: filled(cells, 0),
            assist: filled(cells, 0),
            around: filled(cells, 0),
            buy: filled(cells, Buy { purchasing: 0, revenue: 0 }),
            doff: cx.base - cx.day0 * DAY,
            start: filled(local + 2, 0),
            cstart: filled(local + 2, 0),
            act: filled(local + 2, 0),
            nact: 0,
            ev: huge_vec(ev_cap),
            pl: Vec::new(),
            pls: Vec::new(),
            pstart: filled(local + 2, 0),
            ck: huge_vec(ck_cap),
            ckt: Vec::new(),
            sat: vec![0; nbins.next_multiple_of(8) + 3],
            cts: A32([0; 72]),
            ccamp: A32([0; 72]),
            bins: vec![FREE_BIN; nbins],
            bits: vec![0; nbins / 64 + 1],
            ses: vec![Ses { start: 0, d: 0, row: 0, rev: 0 }; nbins + 1],
            dcats: Vec::new(),
            best: vec![0; nbins + 1],
            eq: [0; 64],
            sessions: Vec::new(),
            sbit: vec![0; nbins / 64 + 1],
            spre: vec![0; nbins / 64 + 1],
            sday: vec![0; cx.ndays + 2],
            stamp: vec![false; cx.ncamp + 1],
            iv: vec![(0, 0); cx.ncamp + 1],
            seen: filled(cx.ncamp + 1, 0),
            sctr: 0,
            touched: Vec::new(),
        }
    }
}

/// Eight ones, then eight zeros: read from 8 - k, the lanes of the first k entries of a vector.
#[allow(dead_code)] // only the AVX2 build uses it
static LANES: [i32; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];

/// The clicks at `ts[..nc]` (1 <= nc <= 64, in any order, 8 readable entries past them) against a session starting
/// at `st`: (bit j set when click j is in the session's 7-day window, `st - ts` in 0..=WEEK as an unsigned wrapping
/// difference; of those, the bits of the clicks with the latest ts).
#[inline(always)]
unsafe fn click_window(ts: *const u32, nc: usize, st: u32) -> (u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        let s = _mm256_set1_epi32(st as i32);
        let w = _mm256_set1_epi32(WEEK as i32);
        let hmax = |v: __m256i| {
            let x = _mm_max_epu32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
            let x = _mm_max_epu32(x, _mm_shuffle_epi32(x, 0b01_00_11_10));
            let x = _mm_max_epu32(x, _mm_shuffle_epi32(x, 0b10_11_00_01));
            _mm256_broadcastd_epi32(x)
        };
        let bits = |v: __m256i| _mm256_movemask_ps(_mm256_castsi256_ps(v)) as u32 as u64;
        if nc <= 8 {
            let t = _mm256_loadu_si256(ts as *const __m256i);
            let d = _mm256_sub_epi32(s, t);
            let ok = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_min_epu32(d, w), d), _mm256_loadu_si256(LANES.as_ptr().add(8 - nc) as *const __m256i));
            let m = bits(ok);
            let e = bits(_mm256_cmpeq_epi32(t, hmax(_mm256_and_si256(ok, t))));
            return (m, e & m);
        }
        let nch = nc.div_ceil(8);
        let mut m = 0u64;
        let mut vmax = _mm256_setzero_si256();
        for c in 0..nch {
            let t = _mm256_loadu_si256(ts.add(c * 8) as *const __m256i);
            let d = _mm256_sub_epi32(s, t);
            let mut ok = _mm256_cmpeq_epi32(_mm256_min_epu32(d, w), d);
            if c == nch - 1 {
                ok = _mm256_and_si256(ok, _mm256_loadu_si256(LANES.as_ptr().add(nch * 8 - nc) as *const __m256i));
            }
            m |= bits(ok) << (c * 8);
            vmax = _mm256_max_epu32(vmax, _mm256_and_si256(ok, t));
        }
        let top = hmax(vmax);
        let mut e = 0u64;
        for c in 0..nch {
            e |= bits(_mm256_cmpeq_epi32(_mm256_loadu_si256(ts.add(c * 8) as *const __m256i), top)) << (c * 8);
        }
        return (m, e & m);
    }
    #[allow(unreachable_code)]
    {
        let (mut m, mut top) = (0u64, 0u32);
        for j in 0..nc {
            let t = *ts.add(j);
            if st.wrapping_sub(t) <= WEEK as u32 {
                m |= 1 << j;
                top = top.max(t);
            }
        }
        let mut e = 0u64;
        for j in 0..nc {
            e |= ((*ts.add(j) == top) as u64) << j;
        }
        (m, e & m)
    }
}

/// `eq[j]` = bit i set when click i (of `camp[..nc]`, 1 <= nc <= 64, 8 readable entries past them) is of click
/// j's campaign; bits at and past nc are undefined.
#[inline(always)]
unsafe fn same_campaign(camp: *const u32, nc: usize, eq: &mut [u64; 64]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        let bits = |v: __m256i| _mm256_movemask_ps(_mm256_castsi256_ps(v)) as u32 as u64;
        if nc <= 8 {
            let v = _mm256_loadu_si256(camp as *const __m256i);
            for j in 0..nc {
                *eq.get_unchecked_mut(j) = bits(_mm256_cmpeq_epi32(v, _mm256_set1_epi32(*camp.add(j) as i32)));
            }
            return;
        }
        let nch = nc.div_ceil(8);
        for j in 0..nc {
            let k = _mm256_set1_epi32(*camp.add(j) as i32);
            let mut e = 0u64;
            for c in 0..nch {
                e |= bits(_mm256_cmpeq_epi32(_mm256_loadu_si256(camp.add(c * 8) as *const __m256i), k)) << (c * 8);
            }
            *eq.get_unchecked_mut(j) = e;
        }
        return;
    }
    #[allow(unreachable_code)]
    for j in 0..nc {
        let mut e = 0u64;
        for i in 0..nc {
            e |= ((*camp.add(i) == *camp.add(j)) as u64) << i;
        }
        eq[j] = e;
    }
}

/// The counts of a counting sort into offsets: `st[l + 2]` counts bin l, for l in 0..=local (`local` a multiple of
/// 8), and `st[l + 1]` becomes the number of entries in the bins before l. Returns that number for all bins, and
/// how many of the bins 0..local are not empty: those are listed in `act` (unless it is null; room for `local`).
unsafe fn offsets(st: *mut u32, local: usize, act: *mut u32) -> (u32, usize) {
    #[allow(unused_assignments)]
    let mut run = 0u32;
    let mut na = 0;
    let mut l = 0;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        // eight bins a step: the running sums inside the vector by shifts, then the sum of the bins before
        let high = _mm256_setr_epi32(0, 0, 0, 0, -1, -1, -1, -1);
        let (three, seven) = (_mm256_set1_epi32(3), _mm256_set1_epi32(7));
        let mut before = _mm256_setzero_si256();
        while l < local {
            let c = _mm256_loadu_si256(st.add(l + 2) as *const __m256i);
            let x = _mm256_add_epi32(c, _mm256_slli_si256(c, 4));
            let x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8)); // sums within each half
            let x = _mm256_add_epi32(x, _mm256_and_si256(_mm256_permutevar8x32_epi32(x, three), high));
            _mm256_storeu_si256(st.add(l + 1) as *mut __m256i, _mm256_add_epi32(before, _mm256_sub_epi32(x, c)));
            before = _mm256_add_epi32(before, _mm256_permutevar8x32_epi32(x, seven));
            if !act.is_null() {
                let mut m = !(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(c, _mm256_setzero_si256()))) as u32) & 0xff;
                while m != 0 {
                    *act.add(na) = (l as u32) + m.trailing_zeros();
                    na += 1;
                    m &= m - 1;
                }
            }
            l += 8;
        }
        run = _mm256_cvtsi256_si32(before) as u32;
    }
    while l <= local {
        let c = *st.add(l + 2);
        *st.add(l + 1) = run;
        run += c;
        if !act.is_null() && c != 0 && l != local {
            *act.add(na) = l as u32;
            na += 1;
        }
        l += 1;
    }
    (run, na)
}

/// rules.md "Events" 5-7 and "Clicks" 4-7 for person partition `p`: the people whose slot number starts with p.
#[inline(never)]
fn partition_work(cx: &Ctx, shards: &[Shard], p: usize, w: &mut Work) {
    let lmask = (1u32 << cx.pshift) - 1;
    // the people of a partition are numbered 1..=5 << part (`IdTable::own`): bins 0..local, a multiple of 8, of the
    // 2^pshift that the arrays have room for. A pay record counts as an event of person `local`, one past the last
    let local = cx.ids.bins();

    // events: counting sort by person (the low bits of the slot number), with no branch and no bounds check; a
    // dropped copy stays with its person (person_work leaves it out). `start[l + 2]` counts person l; then
    // `start[l + 1]` is where l's events begin, and moves to their end as they are placed, so afterwards person l
    // has `start[l]..start[l + 1]`. An event goes as its ts alone, 4 bytes: twice the events a cache line of the
    // 8 bytes of ts and pay, both when they are placed and when they are read. The few with a pay, the purchases,
    // are listed apart and sorted by person the same way.
    let st = w.start.as_mut_ptr();
    let lm = lmask as usize;
    // SAFETY: start, cstart and pstart hold 2^pshift + 3 entries, and every index below is at most lm + 2: `lm`
    // lets the number of the pay records through (`local`, below 2^pshift) and takes HAS_PAY off
    unsafe {
        let ev_bin = |r: &ERec| r.lw as usize & lm;
        std::ptr::write_bytes(st, 0, local + 3);
        for s in shards {
            for blk in s.ev[p].slices(EV_BLOCK) {
                // four records a step: the loop's own instructions are a third of one record's
                let mut four = blk.chunks_exact(4);
                for r in &mut four {
                    *st.add(ev_bin(&r[0]) + 2) += 1;
                    *st.add(ev_bin(&r[1]) + 2) += 1;
                    *st.add(ev_bin(&r[2]) + 2) += 1;
                    *st.add(ev_bin(&r[3]) + 2) += 1;
                }
                for r in four.remainder() {
                    *st.add(ev_bin(r) + 2) += 1;
                }
            }
        }
        // the counts turn into offsets, and the people with events are listed
        let (run, na) = offsets(st, local, w.act.as_mut_ptr());
        if na == 0 {
            return;
        }
        w.nact = na;
        if w.ev.len() < run as usize {
            w.ev.resize(run as usize, 0);
        }
        let ev = w.ev.as_mut_ptr();
        let pl = &mut w.pl;
        pl.clear();
        let mut place = |r: &ERec| {
            let l = ev_bin(r);
            let f = st.add(l + 1);
            *ev.add(*f as usize) = r.ts;
            *f += 1;
            if r.lw & HAS_PAY != 0 && r.ts != 0 {
                // SAFETY: a purchase has its pay record right before it, in the same block
                pl.push([l as u32, r.ts, (*(r as *const ERec).sub(1)).ts]);
            }
        };
        for s in shards {
            for blk in s.ev[p].slices(EV_BLOCK) {
                let mut four = blk.chunks_exact(4);
                for r in &mut four {
                    place(&r[0]);
                    place(&r[1]);
                    place(&r[2]);
                    place(&r[3]);
                }
                four.remainder().iter().for_each(&mut place);
            }
        }
        // the purchases by person: `pstart[l]..pstart[l + 1]` of `pls`, as (ts, pay)
        let pt = w.pstart.as_mut_ptr();
        std::ptr::write_bytes(pt, 0, local + 3);
        if !pl.is_empty() {
            for q in pl.iter() {
                *pt.add(q[0] as usize + 2) += 1;
            }
            offsets(pt, local, std::ptr::null_mut());
            if w.pls.len() < pl.len() {
                w.pls.resize(pl.len(), (0, 0));
            }
            let pls = w.pls.as_mut_ptr();
            for q in pl.iter() {
                let f = pt.add(q[0] as usize + 1);
                *pls.add(*f as usize) = (q[1], q[2]);
                *f += 1;
            }
        }

        // clicks the same way, as they are (those of people with no events are never looked at)
        let ct = w.cstart.as_mut_ptr();
        let ck_bin = |r: &CRec| r.local() as usize & lm; // masked: the index is unchecked
        std::ptr::write_bytes(ct, 0, local + 3);
        for s in shards {
            for blk in s.ck[p].slices(CK_BLOCK) {
                for r in blk {
                    *ct.add(ck_bin(r) + 2) += 1;
                }
            }
        }
        let (run, _) = offsets(ct, local, std::ptr::null_mut());
        if w.ck.len() < run as usize {
            w.ck.resize(run as usize, CRec { cid: 0, w: 0 });
        }
        let ckp = w.ck.as_mut_ptr();
        for s in shards {
            for blk in s.ck[p].slices(CK_BLOCK) {
                for r in blk {
                    let f = ct.add(ck_bin(r) + 1);
                    *ckp.add(*f as usize) = *r;
                    *f += 1;
                }
            }
        }
    }

    let ev = std::mem::take(&mut w.ev);
    for i in 0..w.nact {
        let l = w.act[i] as usize;
        let (a, b) = (w.start[l] as usize, w.start[l + 1] as usize);
        let name = cx.ids.name(((p << cx.pshift) | l) as u32);
        let (ca, cb) = (w.cstart[l] as usize, w.cstart[l + 1] as usize);
        let (pa, pb) = (w.pstart[l] as usize, w.pstart[l + 1] as usize);
        person_work(cx, name, &ev[a..b], pa, pb - pa, ca, cb - ca, w);
    }
    w.ev = ev;
}

/// A person's events (their ts) into their 30-minute bins: returns the words of the bitmap at `wp` with a bin
/// taken. On its own, with nothing else to keep in registers.
///
/// SAFETY: every ts / GAP is a bin of `bp` and a bit of `wp`, in one of its first 64 words.
#[inline(never)]
unsafe fn bin_events(evs: &[u32], bp: *mut Bin, wp: *mut u64) -> u64 {
    let mut wm = 0u64;
    let mut one = |ts: u32| {
        let b = (ts / GAP as u32) as usize;
        let bin = bp.add(b);
        (*bin).cnt += 1;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            use std::arch::x86_64::*;
            // lanes (ts, !ts) against the bin's (min, nmax)
            let t = _mm_xor_si128(_mm_set1_epi32(ts as i32), _mm_setr_epi32(0, -1, 0, 0));
            _mm_storel_epi64(bin as *mut __m128i, _mm_min_epu32(_mm_loadl_epi64(bin as *const __m128i), t));
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            (*bin).min = (*bin).min.min(ts);
            (*bin).nmax = (*bin).nmax.min(!ts);
        }
        *wp.add(b >> 6) |= 1 << (b & 63);
        wm |= 1 << (b >> 6);
    };
    let mut two = evs.chunks_exact(2);
    for e in &mut two {
        one(e[0]);
        one(e[1]);
    }
    two.remainder().iter().for_each(|&e| one(e));
    wm
}

/// The taken bins in order are the sessions, and the sessions in order the rows (one per day with a session
/// start), with no data-dependent branch: the next session and the next row are written at every bin, and kept
/// (the count moves past them) when the bin starts one. Frees the bins and the bitmap; returns (sessions written
/// from `sp`, rows written from `rp`).
///
/// SAFETY: `wm` = the words of `wp` that are not zero; `sp` has room for a session per taken bin + 1 and `rp` for
/// as many rows; `doff` + a bin's min is not negative.
#[inline(never)]
unsafe fn scan_bins(bp: *mut Bin, wp: *mut u64, mut wm: u64, sp: *mut Ses, rp: *mut UserRow, day0: i32, doff: i64) -> (usize, usize) {
    let (mut ns, mut nr) = (0usize, 0usize);
    let mut prev_max = i64::MIN / 2;
    let mut prev_d = u32::MAX;
    while wm != 0 {
        let wi = wm.trailing_zeros() as usize;
        wm &= wm - 1;
        let mut m = *wp.add(wi);
        *wp.add(wi) = 0;
        loop {
            let b = wi * 64 + m.trailing_zeros() as usize;
            m &= m - 1;
            let bin = *bp.add(b);
            let mn = bin.min;
            let new = (mn as i64 - prev_max > GAP) as usize;
            prev_max = !bin.nmax as i64;
            // an event's ts is at or after the window start, which is at or after the midnight of day 0
            let d = ((mn as i64 + doff) as u32) / DAY as u32;
            let new_row = new & (d != prev_d) as usize;
            prev_d = if new != 0 { d } else { prev_d };
            sp.add(ns).write(Ses { start: mn, d, row: 0, rev: 0 });
            ns += new;
            rp.add(nr).write(UserRow { revenue: 0, sessions: 0, events: 0, day: day0 + d as i32, top: NO_CAT });
            nr += new_row;
            // the first bin starts a session and a row, so ns - 1 and nr - 1 are entries written here
            let r = &mut *rp.add(nr - 1);
            r.sessions += new as u32;
            r.events += bin.cnt;
            (*sp.add(ns - 1)).row = (nr - 1) as u32;
            *bp.add(b) = Bin { sess: (ns - 1) as u32, ..FREE_BIN };
            if m == 0 {
                break;
            }
        }
    }
    (ns, nr)
}

/// One person: `evs` = the ts of their events, a dropped copy as 0; their `np` purchases are the entries from `pa`
/// of `w.pls`, and their `nc` valid clicks the entries from `ca` of `w.ck`, in any order.
#[inline(never)]
fn person_work(cx: &Ctx, name: u64, evs: &[u32], pa: usize, np: usize, ca: usize, nc: usize, w: &mut Work) {
    let nbins = w.bins.len();
    w.rows.reserve(nbins + 1); // a row per session at most, and one being written
    let row0 = w.rows.len();
    debug_assert!(evs.iter().all(|&e| ((e / GAP as u32) as usize) < nbins) && nbins <= 64 * 64 && w.bits.len() * 64 >= nbins);
    // SAFETY: an event's ts is below we - base, so its bin is below bins.len(), a bit of bits, which has at most
    // 64 words (the window is under 2^32 ms). The sessions (at most one per taken bin, and `ses` holds nbins + 1)
    // and the rows (room for nbins + 1 past row0) are written in order.
    let ns = unsafe {
        let (bp, wp) = (w.bins.as_mut_ptr(), w.bits.as_mut_ptr());
        let mut wm = bin_events(evs, bp, wp);
        // the dropped copies are in bin 0 (ts 0), which no event of the window is in (it starts 7 days after
        // `Ctx::base`): out of the bitmap, and out of `wm` when its word is left empty
        *bp = FREE_BIN;
        let w0 = *wp & !1;
        *wp = w0;
        wm &= !((w0 == 0) as u64);
        let (ns, nr) = scan_bins(bp, wp, wm, w.ses.as_mut_ptr(), w.rows.as_mut_ptr().add(row0), cx.day0 as i32, w.doff);
        w.rows.set_len(row0 + nr);
        if nr != 0 {
            w.people.push((name, nr as u32));
        }
        ns
    };

    // purchases: into their session and row; then each row's top category
    if np != 0 {
        let Work { rows, pls, bins, ses, dcats, best, .. } = w;
        let rows = &mut rows[row0..];
        dcats.clear();
        for &(ts, pay) in &pls[pa..pa + np] {
            // the bin of a kept event is one this person took, so its `sess` is this person's
            let s = &mut ses[bins[(ts / GAP as u32) as usize].sess as usize];
            let (price, cat) = ((pay & 0xff_ffff) as i64, (pay >> 24) as u16);
            s.rev += price;
            let g = s.row;
            rows[g as usize].revenue += price;
            match dcats.iter_mut().find(|c| c.0 == g && c.1 == cat) {
                Some(c) => c.2 += price,
                None => dcats.push((g, cat, price)),
            }
        }
        // the highest sum; a tie goes to the category that sorts first
        let rank = &cx.cat_rank;
        for &(g, cat, sum) in dcats.iter() {
            let (r, top) = (&mut rows[g as usize], &mut best[g as usize]);
            if r.top == NO_CAT || sum > *top || (sum == *top && rank[cat as usize] < rank[r.top as usize]) {
                r.top = cat;
                *top = sum;
            }
        }
    }

    // credit and assists
    if nc == 0 {
        let Work { cred, buy, ses, .. } = w;
        let (width, organic) = (cx.ndays, cx.ncamp);
        for s in &ses[..ns] {
            credit(cred, buy, organic * width + s.d as usize, s.rev);
        }
    } else if nc <= 64 {
        credit_clicks(cx, ca, nc, ns, w);
    } else {
        credit_big(cx, ca, nc, ns, w);
    }
}

/// `n` more sessions assisted by the campaign on the day of cell `at`.
#[inline(always)]
fn assisted(assist: &mut [u16], around: &mut [u32], at: usize, n: u32) {
    let v = assist[at] as u32 + n;
    assist[at] = v as u16;
    if v >> 16 != 0 {
        around[at] += v >> 16;
    }
}

/// A session credited to the campaign and day of cell `at`.
#[inline(always)]
fn credit(cred: &mut [u32], buy: &mut [Buy], at: usize, revenue: i64) {
    cred[at] += 1;
    if revenue != 0 {
        let b = &mut buy[at];
        b.purchasing += 1;
        b.revenue += revenue;
    }
}

/// Credit and assists of the `ns` sessions of a person with 1 to 64 clicks (`nc`, from `ca`), in the order they
/// were read: per session, vector passes find the clicks of its 7-day window [start - 7d, start] and the latest of
/// them, which takes the credit; the assists go to one click of each campaign in the window.
#[inline(never)]
fn credit_clicks(cx: &Ctx, ca: usize, nc: usize, ns: usize, w: &mut Work) {
    let (width, organic) = (cx.ndays, cx.ncamp);
    let Work { cred, assist, around, buy, ses, ck, cts, ccamp, eq, .. } = w;
    let cs = &ck[ca..ca + nc];
    assert!(nc <= 64 && assist.len() == (organic + 1) * width);
    for (j, c) in cs.iter().enumerate() {
        // SAFETY: j < nc <= 64, and the arrays hold 72
        unsafe {
            *cts.0.get_unchecked_mut(j) = c.ts();
            *ccamp.0.get_unchecked_mut(j) = c.camp() * width as u32; // the campaign's first cell
        }
    }
    // the vector passes read up to 8 entries past the last click: the arrays hold 72
    let (tp, cp) = (cts.0.as_ptr(), ccamp.0.as_ptr());
    unsafe { same_campaign(cp, nc, eq) };
    for s in &ses[..ns] {
        let d = s.d as usize;
        // SAFETY: a bit of m or e is below nc, the index of one of the person's clicks; a click's campaign is
        // below ncamp and d below ndays, so the campaign's first cell + d is a cell
        unsafe {
            let (m, e) = click_window(tp, nc, s.start);
            let mut best = (e.trailing_zeros() & 63) as usize;
            if e & e.wrapping_sub(1) != 0 {
                // a tie in ts -> smallest click_id
                let mut rest = e & (e - 1);
                while rest != 0 {
                    let j = rest.trailing_zeros() as usize;
                    rest &= rest - 1;
                    if cx.names.less(cs.get_unchecked(j).cid, cs.get_unchecked(best).cid) {
                        best = j;
                    }
                }
            }
            let at = if m != 0 { *cp.add(best) as usize } else { organic * width } + d;
            // assists: one click of each campaign in the window
            let day = assist.as_mut_ptr().add(d);
            let mut c = m;
            while c != 0 {
                let j = c.trailing_zeros() as usize;
                let a = day.add(*cp.add(j) as usize);
                let v = (*a).wrapping_add(1);
                *a = v;
                if v == 0 {
                    around[*cp.add(j) as usize + d] += 1;
                }
                c &= !*eq.get_unchecked(j);
            }
            credit(cred, buy, at, s.rev);
        }
    }
}

/// A person's many clicks by ts. One counting pass on the 30-minute bin of the ts puts them in order up to the bin
/// (`cnt`: an entry per bin, a multiple of 8 of them, + 3), then an insertion pass orders the few clicks of each
/// bin. With over 32 clicks a bin on average they are sorted by comparisons instead.
fn sort_clicks(v: &mut [CRec], tmp: &mut Vec<CRec>, cnt: &mut [u32]) {
    let (n, bins) = (v.len(), cnt.len() - 3);
    if n > 32 * bins {
        return v.sort_unstable_by_key(|c| c.ts());
    }
    let bin = |c: &CRec| (c.ts() / GAP as u32) as usize;
    assert!(v.iter().all(|c| bin(c) < bins));
    cnt.fill(0);
    for c in v.iter() {
        cnt[bin(c) + 2] += 1;
    }
    // SAFETY: bins + 3 entries, bins a multiple of 8
    unsafe { offsets(cnt.as_mut_ptr(), bins, std::ptr::null_mut()) };
    tmp.clear();
    tmp.resize(n, CRec { cid: 0, w: 0 });
    for c in v.iter() {
        let f = &mut cnt[bin(c) + 1];
        tmp[*f as usize] = *c;
        *f += 1;
    }
    for i in 1..n {
        let c = tmp[i];
        let mut j = i;
        while j > 0 && tmp[j - 1].ts() > c.ts() {
            tmp[j] = tmp[j - 1];
            j -= 1;
        }
        tmp[j] = c;
    }
    v.copy_from_slice(tmp);
}

/// Credit and assists of the `ns` sessions of a person with more than 64 clicks (`nc`, from `ca`).
#[cold]
#[inline(never)]
fn credit_big(cx: &Ctx, ca: usize, nc: usize, ns: usize, w: &mut Work) {
    let width = cx.ndays;
    let organic = cx.ncamp;
    let Work { cred, assist, around, buy, stamp, iv, seen, sctr, sessions, touched, ck, ckt, sat, ses, sbit, spre, sday, .. } = w;
    sort_clicks(&mut ck[ca..ca + nc], ckt, sat);
    let cs: &[CRec] = &ck[ca..ca + nc];
    sessions.clear();
    sessions.extend(ses[..ns].iter().map(|s| Sess { start: s.start as i64, d: s.d as usize, revenue: s.rev }));

    let cts = |c: &CRec| c.ts() as i64;
    let ccamp = |c: &CRec| c.camp() as usize;
    let mut credit = |camp: usize, s: &Sess| credit(cred, buy, camp * width + s.d, s.revenue);
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
    if nc * ns <= 2048 {
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
                        assisted(assist, around, kk * width + s.d, 1);
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
        // assists: a click at t assists sessions starting in [t, t + 7d]; merge per campaign. The sessions of an
        // interval without a search: at most one session starts in a 30-minute bin (two starts are more than 30
        // minutes apart), so the sessions before a time are the ones before its bin (the bits of `sbit` before it:
        // `spre` of its word and a count within the word), plus the one of its bin when that starts before it; and
        // the sessions of a day are the ones between `sday` of it and of the next.
        sbit.fill(0);
        for s in &ses[..ns] {
            let b = (s.start / GAP as u32) as usize;
            sbit[b >> 6] |= 1 << (b & 63);
        }
        let mut acc = 0;
        for (x, w) in spre.iter_mut().zip(sbit.iter()) {
            *x = acc;
            acc += w.count_ones();
        }
        let nb = sbit.len() * 64;
        let (sbit, spre): (&[u64], &[u32]) = (sbit, spre);
        // the sessions that start before bin b
        let before = |b: usize| if b >= nb { ns } else { (spre[b >> 6] + (sbit[b >> 6] & ((1u64 << (b & 63)) - 1)).count_ones()) as usize };
        let mut j = 0;
        for (d, x) in sday.iter_mut().enumerate() {
            while j < ns && (ses[j].d as usize) < d {
                j += 1;
            }
            *x = j as u32;
        }
        let (ses, sday): (&[Ses], &[u32]) = (&ses[..ns], sday);
        let mut flush = |k: usize, lo: i64, hi: i64| {
            let mut a = before((lo / GAP) as usize);
            a += (a < ns && (ses[a].start as i64) < lo) as usize;
            let mut b = before((hi / GAP) as usize);
            b += (b < ns && ses[b].start as i64 <= hi) as usize;
            if a >= b {
                return;
            }
            for d in ses[a].d as usize..=ses[b - 1].d as usize {
                assisted(assist, around, k * width + d, (sday[d + 1] as usize).min(b) as u32 - (sday[d] as usize).max(a) as u32);
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
                flush(k, lo, hi);
                iv[k] = (t, t + WEEK);
            }
        }
        for &k in touched.iter() {
            let (lo, hi) = iv[k];
            flush(k, lo, hi);
            stamp[k] = false;
        }
    }
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
    let mut out: Vec<u8> = huge_vec(n * 12 + (1 << 20));
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

/// Appends the ID of key `k` as a PLAIN byte array: 4-byte length, then the bytes.
#[inline]
fn put_name(b: &mut Vec<u8>, names: &Interner, k: u64, tmp: &mut Vec<u8>) {
    if k & INTERNED == 0 {
        // a packed ID: nine 7-bit bytes, first byte highest; the zero bytes past its length cut off
        let len = 9 - (k.trailing_zeros() / 7) as usize;
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            // length, first byte, and the other eight 7-bit groups spread to a byte each (the highest first):
            // 13 bytes written whole, of which 4 + len count
            b.reserve(16);
            // SAFETY: 16 bytes are reserved past the length
            unsafe {
                let p = b.as_mut_ptr().add(b.len());
                (p as *mut u32).write_unaligned((len as u32).to_le());
                *p.add(4) = (k >> 56) as u8 & 127;
                (p.add(5) as *mut u64).write_unaligned(std::arch::x86_64::_pdep_u64(k, 0x7f7f_7f7f_7f7f_7f7f).to_be());
                b.set_len(b.len() + 4 + len);
            }
            return;
        }
        #[allow(unreachable_code)]
        {
            let mut buf = [0u8; 9];
            for (i, c) in buf.iter_mut().enumerate() {
                *c = ((k >> (56 - 7 * i)) & 127) as u8;
            }
            b.extend_from_slice(&(len as u32).to_le_bytes());
            b.extend_from_slice(&buf[..len]);
            return;
        }
    }
    tmp.clear();
    names.bytes(k, tmp);
    b.extend_from_slice(&(tmp.len() as u32).to_le_bytes());
    b.extend_from_slice(tmp);
}

/// DELTA_BINARY_PACKED body of the `n` values `get(0..n)`: blocks of 128 deltas in four miniblocks of 32, each
/// miniblock at the bit width its largest value (delta minus the block's smallest delta) needs.
#[inline(always)]
fn put_delta(b: &mut Vec<u8>, n: usize, get: impl Fn(usize) -> i64) {
    let zz = |x: i64| ((x << 1) ^ (x >> 63)) as u64;
    // room for everything at once: four varints, then per block a varint, four widths and up to 128 * 8 bytes
    // (+ 8: a packed value is stored as a whole word)
    b.reserve(48 + n.div_ceil(128) * (16 + 1024) + 8);
    // SAFETY: every write below stays inside the room just reserved
    unsafe {
        let start = b.as_mut_ptr().add(b.len());
        let mut p = start;
        let uleb = |p: &mut *mut u8, mut x: u64| {
            while x >= 0x80 {
                p.write(x as u8 | 0x80);
                *p = p.add(1);
                x >>= 7;
            }
            p.write(x as u8);
            *p = p.add(1);
        };
        uleb(&mut p, 128);
        uleb(&mut p, 4);
        uleb(&mut p, n as u64);
        let mut prev = if n > 0 { get(0) } else { 0 };
        uleb(&mut p, zz(prev));
        let mut i = 1;
        while i < n {
            let m = (n - i).min(128);
            // a miniblock past the last delta is absent; the last one is padded with zeros (the smallest residual)
            let mut r = [0u64; 128];
            let mut min = i64::MAX;
            for (j, d) in r[..m].iter_mut().enumerate() {
                let v = get(i + j);
                let x = v.wrapping_sub(prev);
                prev = v;
                *d = x as u64;
                min = min.min(x);
            }
            uleb(&mut p, zz(min));
            for d in r[..m].iter_mut() {
                *d = (*d as i64).wrapping_sub(min) as u64;
            }
            let minis = m.div_ceil(32);
            let mut w = [0u8; 4];
            for k in 0..minis {
                w[k] = (64 - r[k * 32..k * 32 + 32].iter().fold(0u64, |a, &x| a | x).leading_zeros()) as u8;
            }
            (p as *mut [u8; 4]).write_unaligned(w);
            p = p.add(4);
            for k in 0..minis {
                let w = w[k] as u32;
                let vals = &r[k * 32..k * 32 + 32];
                if w == 0 {
                    continue;
                }
                if w <= 56 {
                    // up to 7 bits wait from the value before: the whole bytes go out with every value
                    let (mut acc, mut nb) = (0u64, 0u32);
                    for &x in vals {
                        acc |= x << nb;
                        nb += w;
                        (p as *mut u64).write_unaligned(acc.to_le());
                        let bytes = nb >> 3;
                        p = p.add(bytes as usize);
                        acc >>= bytes * 8;
                        nb &= 7;
                    }
                } else {
                    let (mut acc, mut nb) = (0u128, 0u32);
                    for &x in vals {
                        acc |= (x as u128) << nb;
                        nb += w;
                        if nb >= 64 {
                            (p as *mut u64).write_unaligned((acc as u64).to_le());
                            p = p.add(8);
                            acc >>= 64;
                            nb -= 64;
                        }
                    }
                    // 32 values of w bits are a whole number of bytes
                    std::ptr::copy_nonoverlapping(acc.to_le_bytes().as_ptr(), p, (nb / 8) as usize);
                    p = p.add((nb / 8) as usize);
                }
            }
            i += m;
        }
        let len = b.len() + (p as usize - start as usize);
        b.set_len(len);
    }
}

/// One run of the RLE/bit-packed hybrid: `run` copies of the `vb`-byte value `idx` (vb <= 4).
#[inline(always)]
fn put_run(b: &mut Vec<u8>, run: usize, idx: u32, vb: usize) {
    if run < 64 {
        // one byte of header: header and value go out as one word, of which 1 + vb bytes count
        b.reserve(8);
        // SAFETY: 8 bytes are reserved past the length
        unsafe {
            (b.as_mut_ptr().add(b.len()) as *mut u64).write_unaligned((((run as u64) << 1) | (idx as u64) << 8).to_le());
            b.set_len(b.len() + 1 + vb);
        }
        return;
    }
    let mut v = (run as u64) << 1;
    while v >= 0x80 {
        b.push(v as u8 | 0x80);
        v >>= 7;
    }
    b.push(v as u8);
    b.extend_from_slice(&idx.to_le_bytes()[..vb]);
}

fn write_users(path: &Path, rows: &[UserRow], people: &[(u64, u32)], cat_names: &[Bytes], names: &Interner) -> std::io::Result<()> {
    let segs: &[&[UserRow]] = &[rows];
    let mut tmp = Vec::with_capacity(64);
    let n: usize = segs.iter().map(|s| s.len()).sum();
    let mut part: Vec<&[UserRow]> = Vec::new();
    let mut run: Vec<u8> = Vec::new();
    // user_id is dictionary encoded: one entry per person (a person's days are consecutive rows), and the data
    // pages are runs of that entry's index. `at` = the person the next page starts in, and how many of its rows
    // the pages before took.
    let entries = people.len();
    let mut at = (0usize, 0usize);
    // sessions and events stay far below 2^31: main() refuses inputs with that many events
    write_parquet(path, &USER_COLS, n, |c, lo, hi, b| {
        if c == 0 && lo == DICT_PAGE {
            for &(name, _) in people {
                put_name(b, names, name, &mut tmp);
            }
            return entries;
        }
        page_slices(segs, lo, hi, &mut part);
        match c {
            0 => {
                let w = (usize::BITS - (entries.max(2) - 1).leading_zeros()) as usize;
                b.push(w as u8);
                let mut left = hi - lo;
                while left > 0 {
                    let (i, used) = at;
                    let run = (people[i].1 as usize - used).min(left);
                    put_run(b, run, i as u32, w.div_ceil(8));
                    left -= run;
                    at = if used + run == people[i].1 as usize { (i + 1, 0) } else { (i, used + run) };
                }
            }
            1..=4 => {
                // straight from the rows, one loop per column
                let rs = &rows[lo..hi];
                // SAFETY (the four): put_delta asks for indices below rs.len()
                match c {
                    1 => put_delta(b, rs.len(), |i| unsafe { rs.get_unchecked(i) }.day as i64),
                    2 => put_delta(b, rs.len(), |i| unsafe { rs.get_unchecked(i) }.sessions as i64),
                    3 => put_delta(b, rs.len(), |i| unsafe { rs.get_unchecked(i) }.events as i64),
                    _ => put_delta(b, rs.len(), |i| unsafe { rs.get_unchecked(i) }.revenue),
                }
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
                for r in part.iter().flat_map(|s| s.iter()) {
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
                for r in part.iter().flat_map(|s| s.iter()) {
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
            match r.below(14) {
                0 => String::new(),
                1 => format!("{c}{}", "x".repeat(r.below(40))),
                2 => format!(" {c}{}", r.below(1000)),
                // v5: a prefix and 32 hex digits (now and then not lowercase, or one short)
                3 | 4 => {
                    let h = format!("{c}{:016x}{:016x}", r.next(), r.next());
                    match r.below(8) {
                        0 => h.to_uppercase(),
                        1 => h[..32].to_string(),
                        _ => h,
                    }
                }
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
            ids: IdTable::new(16, 4),
            names: Interner::new(),
            pshift: 13,
            qshift: 60,
            abase: Ctx::record_fields(0).0,
            ihi: Ctx::record_fields(ws + 14 * DAY).1,
            ilo: Ctx::record_fields(ws + 14 * DAY).2,
        }
    }

    type EventParser = unsafe fn(&Ctx, *const u8, usize, usize, usize, &mut Shard);
    /// Every event parser of this build: the one the run uses first.
    fn event_parsers() -> Vec<(&'static str, EventParser)> {
        #[allow(unused_mut)]
        let mut v: Vec<(&'static str, EventParser)> = vec![("parse_events", parse_events)];
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
        v.push(("parse_events_v3", parse_events_v3));
        v
    }

    /// Every click parser of this build: the one the run uses first.
    fn click_parsers() -> Vec<(&'static str, EventParser)> {
        #[allow(unused_mut)]
        let mut v: Vec<(&'static str, EventParser)> = vec![("parse_clicks", parse_clicks)];
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
        v.push(("parse_clicks_v3", parse_clicks_v3));
        v
    }

    /// The clicks of a shard as (click key, the rest of the record), sorted.
    fn click_list(st: &mut Shard) -> Vec<(u64, u64)> {
        st.ck.iter_mut().for_each(Stream::finish);
        let mut v: Vec<_> = st.ck.iter().flat_map(|s| s.slices(CK_BLOCK)).flatten().map(|r| (r.cid, r.w)).collect();
        v.sort();
        v
    }

    /// (events as (person, ts, pay), dedup records as (key, ingest, has a record)), each sorted
    type Flat = (Vec<(u32, u32, u32)>, Vec<(u64, i64, bool)>);
    fn flat(st: &mut Shard) -> Flat {
        st.ev.iter_mut().for_each(Stream::finish);
        st.lg.iter_mut().for_each(Log::finish);
        // an event as (partition << 17 | number in it, ts, pay): a purchase's pay is the record before it; the pay
        // records themselves (the one before a purchase, and the fillers with ts 0) are left out
        let mut ev = Vec::new();
        for (p, s) in st.ev.iter().enumerate() {
            let recs: Vec<ERec> = s.slices(EV_BLOCK).flatten().copied().collect();
            for (i, r) in recs.iter().enumerate() {
                let pays_next = recs.get(i + 1).is_some_and(|n| n.lw & HAS_PAY != 0);
                if r.ts != 0 && !pays_next {
                    let pay = if r.lw & HAS_PAY != 0 { recs[i - 1].ts } else { 0 };
                    ev.push(((p as u32) << 17 | (r.lw & !HAS_PAY), r.ts, pay));
                }
            }
        }
        // the ingest_ts as the record holds it; a far one as its exact value, from the shard's far list
        let recs: Vec<LRec> = st.lg.iter().flat_map(|s| s.blocks()).flat_map(|(k, p)| k.iter().zip(p).map(|(&k, &p)| LRec::of(k, p))).collect();
        let mut lg: Vec<_> = recs.iter().filter(|r| r.ing != ING_FAR).map(|r| (r.key, r.ing as i64, r.loc != 0)).collect();
        lg.extend(st.far.iter().map(|&(key, loc, ingest)| (key, ingest, loc != 0)));
        assert_eq!(st.far.len(), recs.iter().filter(|r| r.ing == ING_FAR).count());
        ev.sort();
        lg.sort();
        (ev, lg)
    }

    /// `split_event` and `split_click` agree with `split` on every line they take, and take the usual ones: random
    /// lines made of the bytes that matter, and lines of the generator's shape with fields quoted at random.
    #[test]
    fn window_split_matches_split() {
        let mut r = Rng(3);
        let atoms = ["a", ",", "\"", "\"\"", "xy", "\r", " ", "\"ab\"", "1234567", "e4ajy5dva", "2027-12-18T22:22:54Z", "\n"];
        let (mut took7, mut took4, mut usual7, mut usual4) = (0, 0, 0, 0);
        for round in 0..600_000 {
            let n = if round % 2 == 0 { 7 } else { 4 };
            let mut line = String::new();
            let usual = round % 3 == 0;
            if usual {
                let ev = ["e4ajy5dva", "u24wwm60w", "2027-12-21T10:03:10-02:00", "2027-12-18T22:22:54Z", "purchase", "p38449", "41607"];
                let ck = ["k9pbmgz1e", "u8nj7nrj5", "c_332", "2027-12-21T10:03:10-02:00"];
                let fields: Vec<String> = (0..n)
                    .map(|i| {
                        let v = if n == 7 { ev[i] } else { ck[i] };
                        let v = if r.below(6) == 0 { "" } else { v };
                        if r.below(3) == 0 { format!("\"{v}\"") } else { v.to_string() }
                    })
                    .collect();
                line = fields.join(",");
            } else if round % 3 == 1 {
                // about the right number of fields, each a few atoms
                let k = if r.below(8) == 0 { n - 1 + r.below(3) } else { n };
                let fields: Vec<String> = (0..k).map(|_| (0..r.below(3)).map(|_| r.pick(&atoms[2..])).collect::<String>()).collect();
                line = fields.join(",");
                line.truncate(126);
            } else {
                for _ in 0..r.below(30) {
                    line += r.pick(&atoms);
                }
                line.truncate(126);
            }
            let mut buf = line.clone().into_bytes();
            buf.push(b'\n');
            // what follows the line must not matter
            let tail = ["z", ",", "\"", "\n"];
            for _ in 0..200 {
                buf.extend_from_slice(r.pick(&tail).as_bytes());
            }
            let first = line.split('\n').next().unwrap().to_string();
            let text: Bytes = Box::leak(first.clone().into_bytes().into_boxed_slice());
            let (got, want) = if n == 7 {
                let got = unsafe { split_event(buf.as_ptr()) }.map(|(mut a, mut b, o, n)| {
                    let mut f: Vec<(usize, usize)> = (0..3).map(|_| a.next()).collect();
                    f.extend((0..4).map(|_| b.next()).map(|(s, l)| (o + s, l)));
                    (f, n)
                });
                (got, split::<7>(text).map(|f| f.map(|x| x.to_vec()).to_vec()))
            } else {
                let got = unsafe { split_click(buf.as_ptr()) }.map(|(mut f, n)| ((0..4).map(|_| f.next()).collect::<Vec<_>>(), n));
                (got, split::<4>(text).map(|f| f.map(|x| x.to_vec()).to_vec()))
            };
            if let Some((f, len)) = &got {
                let len = *len;
                assert_eq!(len, first.len() + 1, "line {line:?}");
                let f: Vec<Vec<u8>> = f.iter().map(|&(s, l)| buf[s..s + l].to_vec()).collect();
                assert_eq!(Some(f), want, "line {line:?}");
                if n == 7 {
                    took7 += 1;
                } else {
                    took4 += 1;
                }
            }
            if usual {
                assert!(got.is_some() && want.is_some(), "line {line:?}");
                if n == 7 {
                    usual7 += 1;
                } else {
                    usual4 += 1;
                }
            }
        }
        assert!(usual7 > 90_000 && usual4 > 90_000 && took7 > usual7 + 10_000 && took4 > usual4 + 10_000, "{took7} {took4} {usual7} {usual4}");
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
        let mut s12: Vec<Stream<ERec, EV_LINES>> = (0..23).map(|_| Stream::new()).collect();
        let mut larena = Arena::new();
        let mut s24: Vec<Log> = (0..23).map(|_| Log::new()).collect();
        let mut want12: Vec<Vec<(u32, u32)>> = vec![Vec::new(); 23];
        let mut want24: Vec<Vec<(u64, u32, u32)>> = vec![Vec::new(); 23];
        let mut at: Vec<(*mut ERec, u32)> = Vec::new();
        for i in 0..400_000u32 {
            let b = r.below(23);
            if r.below(2) == 0 {
                let rec = ERec { ts: r.next() as u32, lw: i };
                at.push((s12[b].push(&mut arena, EV_BLOCK, rec), i));
                want12[b].push((rec.lw, rec.ts));
            } else {
                let rec = LRec { key: r.next(), ing: i, loc: b as u32 };
                s24[b].push(&mut larena, rec.key, LRec::pay(rec.ing, rec.loc));
                want24[b].push((rec.key, rec.ing, rec.loc));
            }
        }
        sfence();
        s12.iter_mut().for_each(Stream::finish);
        s24.iter_mut().for_each(Log::finish);
        for b in 0..23 {
            let got: Vec<_> = s12[b].slices(EV_BLOCK).flatten().map(|r| (r.lw, r.ts)).collect();
            assert_eq!(got, want12[b]);
            assert_eq!(s12[b].len(EV_BLOCK), want12[b].len());
            let got: Vec<_> = s24[b].blocks().flat_map(|(k, p)| k.iter().zip(p).map(|(&k, &p)| LRec::of(k, p))).map(|r| (r.key, r.ing, r.loc)).collect();
            assert_eq!(s24[b].len(), want24[b].len());
            assert_eq!(got, want24[b]);
        }
        assert!(at.iter().all(|&(p, i)| unsafe { (*p).lw } == i));
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
            let mut b = Shard::new(16, 16);
            for w in cuts.windows(2) {
                for_records::<7>(file, w[0], w[1], |f| event_fields(&c, f, &mut b));
                b.drain::<true>(&c);
            }
            let fb = flat(&mut b);
            kept += fb.0.len();
            for (name, parse) in event_parsers() {
                let mut a = Shard::new(16, 16);
                for w in cuts.windows(2) {
                    unsafe { parse(&c, padded.as_ptr(), w[0], w[1], len, &mut a) };
                }
                assert!(flat(&mut a) == fb, "round {round}: {name} differs\n{}", std::str::from_utf8(file).unwrap());
            }
        }
        assert!(kept > 1000, "too few kept events ({kept}) to mean anything");
    }

    /// Lines of the generator's shape (IDs of 8 or 9 bytes, event lines of 64 to 95 bytes), fields quoted at
    /// random and now and then a byte changed: `parse_events` and `parse_clicks` against the generic field parsers.
    #[test]
    fn usual_lines_match_reference() {
        let mut c = ctx();
        let camps: Vec<(Bytes, u16)> = (0..40u16).map(|i| (&*Box::leak(format!("c_{}", i * 37).into_bytes().into_boxed_slice()), i)).collect();
        c.campaigns = IdMap::new(&camps).with_short(&camps);
        c.ncamp = camps.len();
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        fn b36(r: &mut Rng, n: usize) -> String {
            (0..n).map(|_| char::from_digit(r.below(36) as u32, 36).unwrap()).collect()
        }
        // F1 of 2028-02-15T00:00:00Z + ms
        fn f1(ms: u64) -> String {
            let (d, s) = ((ms / 86_400_000) as usize, ms % 86_400_000 / 1000);
            let (mo, dd) = if d < 15 { (2, 15 + d) } else { (3, d - 14) };
            format!("2028-{mo:02}-{dd:02}T{:02}:{:02}:{:02}Z", s / 3600, s / 60 % 60, s % 60)
        }
        let t0 = parse_ts(b"2028-02-15T00:00:00Z").unwrap() as u64;
        fn ts(r: &mut Rng, t0: u64, ms: u64) -> String {
            match r.below(12) {
                0..=5 => f1(ms),
                6..=8 => (t0 + ms).to_string(),
                9 => format!("{}+05:00", &f1(ms + 5 * 3_600_000)[..19]),
                10 => format!("{}-10:30", &f1(ms.saturating_sub(37_800_000))[..19]),
                _ => ts_value(r),
            }
        }
        fn quote(r: &mut Rng, v: String) -> String {
            if r.below(5) == 0 {
                format!("\"{v}\"")
            } else {
                v
            }
        }
        fn id(r: &mut Rng, c: char) -> String {
            match r.below(40) {
                0 => String::new(),
                1 => {
                    let n = 5 + r.below(7);
                    format!("{c}{}", b36(r, n))
                }
                2..=5 => format!("{c}{}", b36(r, 7)),
                6..=10 => format!("{c}{:016x}{:016x}", r.next(), r.next()),
                _ => format!("{c}{}", b36(r, 8)),
            }
        }
        fn mutate(r: &mut Rng, mut l: String) -> String {
            if r.below(10) == 0 && !l.is_empty() {
                let k = r.below(l.len());
                if l.is_char_boundary(k) && l.is_char_boundary(k + 1) {
                    match r.below(3) {
                        0 => l.replace_range(k..k + 1, r.pick(&["\"", ",", "\r", "\u{1}", "é", " ", "0", "Z", ":", "x", "\n", "\u{7f}"])),
                        1 => {
                            l.remove(k);
                        }
                        _ => l.insert(k, ['"', ',', '9', 'T'][r.below(4)]),
                    }
                }
            }
            l
        }
        let (mut kept, mut clicks) = (0, 0);
        for round in 0..200 {
            let mut ev = String::from("event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
            let mut ck = String::from("click_id,user_id,campaign_id,ts\n");
            for _ in 0..r.below(300) {
                let ms = r.below(25 * 86_400_000) as u64;
                let ing = ms / 1000 * 1000 + [30_000, 600_000, 3_000_000, 86_400_000, 86_401_000, 3 * 86_400_000][r.below(6)];
                let f = [
                    id(&mut r, 'e'),
                    id(&mut r, 'u'),
                    ts(&mut r, t0, ms),
                    if r.below(30) == 0 { ts(&mut r, t0, ing) } else { f1(ing) },
                    r.pick(&["view", "view", "view", "view", "cart", "cart", "purchase", "purchase", "Purchase", "views", "car", ""]).to_string(),
                    r.pick(&["p1", "p2", "p39594", "p17893", "p8823", "p310", ""]).to_string(),
                    r.pick(&["", "", "", "41607", "720", "0", "10000000", "10000001", "-5", "1.5"]).to_string(),
                ];
                let l = f.map(|v| quote(&mut r, v)).join(",");
                ev += &mutate(&mut r, l);
                ev.push('\n');
                let ms = r.below(25 * 86_400_000) as u64;
                let f = [id(&mut r, 'k'), id(&mut r, 'u'), format!("c_{}", r.below(45) * 37), ts(&mut r, t0, ms)];
                let l = f.map(|v| quote(&mut r, v)).join(",");
                ck += &mutate(&mut r, l);
                ck.push('\n');
            }
            if round % 3 == 0 {
                ev.pop(); // no trailing newline
                ck.pop();
            }
            for (events, text) in [(true, ev), (false, ck)] {
                let len = text.len();
                let mut padded = text.clone().into_bytes();
                padded.extend_from_slice(&[b'\n'; PAD]);
                let file: Bytes = Box::leak(text.into_bytes().into_boxed_slice());
                let mut cuts = vec![0, len];
                for _ in 0..r.below(4) {
                    cuts.push(r.below(len + 1));
                }
                cuts.sort();
                let (mut a, mut b) = (Shard::new(16, 16), Shard::new(16, 16));
                for w in cuts.windows(2) {
                    if events {
                        unsafe { parse_events(&c, padded.as_ptr(), w[0], w[1], len, &mut a) };
                        for_records::<7>(file, w[0], w[1], |f| event_fields(&c, f, &mut b));
                        b.drain::<true>(&c);
                        // the other event parsers of this build give the same
                        for (name, parse) in event_parsers().into_iter().skip(1) {
                            let (mut a2, mut b2) = (Shard::new(16, 16), Shard::new(16, 16));
                            unsafe { parse(&c, padded.as_ptr(), w[0], w[1], len, &mut a2) };
                            for_records::<7>(file, w[0], w[1], |f| event_fields(&c, f, &mut b2));
                            b2.drain::<true>(&c);
                            assert!(flat(&mut a2) == flat(&mut b2), "round {round}: {name} differs\n{}", String::from_utf8_lossy(file));
                        }
                    } else {
                        unsafe { parse_clicks(&c, padded.as_ptr(), w[0], w[1], len, &mut a) };
                        for_records::<4>(file, w[0], w[1], |f| click_fields(&c, f, &mut b));
                        b.drain::<false>(&c);
                        // the other click parsers of this build give the same
                        for (name, parse) in click_parsers().into_iter().skip(1) {
                            let (mut a2, mut b2) = (Shard::new(16, 16), Shard::new(16, 16));
                            unsafe { parse(&c, padded.as_ptr(), w[0], w[1], len, &mut a2) };
                            for_records::<4>(file, w[0], w[1], |f| click_fields(&c, f, &mut b2));
                            b2.drain::<false>(&c);
                            assert!(click_list(&mut a2) == click_list(&mut b2), "round {round}: {name} differs\n{}", String::from_utf8_lossy(file));
                        }
                    }
                }
                let (ca, cb) = (click_list(&mut a), click_list(&mut b));
                let (fa, fb) = (flat(&mut a), flat(&mut b));
                kept += fb.0.len();
                clicks += cb.len();
                assert!(fa == fb && ca == cb, "round {round}: one-pass parse differs\n{}", String::from_utf8_lossy(file));
            }
        }
        assert!(kept > 3000 && clicks > 5000, "too few kept events ({kept}) or clicks ({clicks}) to mean anything");
    }

    // What the edge tests make their lines of.
    // `YYYY-MM-DDTHH:MM:SS` of a second of January to April 2028
    fn edge_civil(sec: i64) -> String {
        let (mut d, s) = (sec.div_euclid(86_400), sec.rem_euclid(86_400));
        let mut mo = 1;
        for len in [31, 29, 31] {
            if d < len {
                break;
            }
            d -= len;
            mo += 1;
        }
        format!("2028-{mo:02}-{:02}T{:02}:{:02}:{:02}", d + 1, s / 3600, s / 60 % 60, s % 60)
    }
    fn edge_b36(r: &mut Rng, n: usize) -> String {
        (0..n).map(|_| char::from_digit(r.below(36) as u32, 36).unwrap()).collect()
    }
    fn edge_id(r: &mut Rng, c: char, dirty: bool) -> String {
        match if dirty { r.below(12) } else { 6 + r.below(54) } {
            0 => String::new(),
            1 => format!("{c}{}", edge_b36(r, 6)),
            2 => format!("{c}{}", edge_b36(r, 9)),
            3 => format!("{c}{} ", edge_b36(r, 7)),
            4 => format!("{c}{}\u{7f}", edge_b36(r, 7)),
            5 => format!("{c}{}é", edge_b36(r, 6)),
            6..=20 => format!("{c}{}", edge_b36(r, 7)),
            21..=28 => format!("{c}{:016x}{:016x}", r.next(), r.next()),
            _ => format!("{c}{}", edge_b36(r, 8)),
        }
    }
    // a ts for the second `sec` (from 2028-01-01) in one of the three forms, now and then off its rules
    fn edge_ts(r: &mut Rng, jan1: i64, sec: i64, dirty: bool) -> String {
        let mut v = match r.below(10) {
            0..=3 => format!("{}Z", edge_civil(sec)),
            4..=6 => format!("{}{:03}", jan1 + sec, r.below(1000)),
            _ => {
                let (h, m) = match r.below(8) {
                    0 => (0, 0),
                    1 => (23, 59),
                    2 => (12, 0),
                    _ => (r.below(13) as i64, [0, 30, 45][r.below(3)]),
                };
                let off = h * 3600 + m * 60;
                if r.below(2) == 0 {
                    format!("{}+{h:02}:{m:02}", edge_civil(sec + off))
                } else {
                    format!("{}-{h:02}:{m:02}", edge_civil(sec - off))
                }
            }
        };
        match if dirty { r.below(10) } else { 99 } {
            0 if v.len() >= 19 => v.replace_range(11..19, r.pick(&["24:00:00", "23:60:00", "23:59:60", "23:59:59", "00:00:00", "29:59:59", "19:5a:00", "1:00:000"])),
            1 if v.len() == 25 => v.replace_range(19..25, r.pick(&["+24:00", "+23:60", "-23:59", "+05:6 ", " 05:30", "Z05:30", "+0530 ", "+29:00", "-9:300", "+05-30", "+05:3a"])),
            2 if v.len() >= 10 => v.replace_range(5..10, r.pick(&["02-30", "02-29", "13-01", "00-10", "03-00", "03-32", "1a-01", "03 01"])),
            3 => {
                let k = r.below(v.len());
                v.replace_range(k..k + 1, r.pick(&["0", "9", ":", "-", "T", "Z", "z", "x", "/", " ", "+", "\u{7f}", "\r"]));
            }
            4 => v.push_str(r.pick(&["0", "Z", " ", "\r"])),
            5 => {
                v.pop();
            }
            6 => v.insert(0, '0'),
            _ => {}
        }
        v
    }
    fn edge_quote(r: &mut Rng, v: String, dirty: bool) -> String {
        match r.below(if dirty { 26 } else { 100 }) {
            0..=19 => format!("\"{v}\""),
            20 => format!("\"{v}"),
            21 => format!("{v}\""),
            22 => "\"".to_string(),
            _ => v,
        }
    }
    /// Lines of the generator's shape with every field on its edges, in the order that makes the eight-line
    /// parser's lanes differ: timestamps of all three forms across ten weeks (two months share a day of the month),
    /// times and offsets at and past their limits, events ingested exactly 24 hours late and a second more, IDs
    /// of 7 to 10 bytes, every price, fields quoted at random, bytes changed at random. Every event parser of the
    /// build against the field parser, and the eight-line one must take most of the lines itself.
    #[test]
    fn edge_lines_match_reference() {
        let c = ctx();
        let mut r = Rng(0x0123_4567_89ab_cdef);
        let jan1 = parse_ts(b"2028-01-01T00:00:00Z").unwrap() / 1000;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
        let usual0 = wide::USUAL.load(Relaxed);
        let (mut lines, mut kept) = (0, 0);
        for round in 0..300 {
            let mut ev = String::from("event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
            for _ in 0..r.below(200) {
                // 2028-02-05 .. 2028-03-20; the window is 2028-02-20 .. 2028-03-05
                let sec = if r.below(3) == 0 { 35 * 86_400 + r.below(45 * 86_400) as i64 } else { 50 * 86_400 + r.below(14 * 86_400) as i64 };
                let sec = match r.below(30) {
                    0 => 50 * 86_400,     // window_start
                    1 => 50 * 86_400 - 1, // a second before
                    2 => 64 * 86_400,     // window_end
                    3 => 64 * 86_400 - 1,
                    _ => sec,
                };
                let ing = sec + [0, 30, 600, 86_400, 86_401, 86_399, -3600, 5 * 86_400, 40 * 86_400][r.below(9)];
                // one line in four is dirty in its IDs, its ts, its type or its quotes
                let dirty = r.below(4) == 0;
                let f = [
                    edge_id(&mut r, 'e', dirty),
                    edge_id(&mut r, 'u', dirty),
                    edge_ts(&mut r, jan1, sec, dirty),
                    if r.below(25) == 0 { edge_ts(&mut r, jan1, ing, true) } else { format!("{}Z", edge_civil(ing)) },
                    r.pick(&["view", "view", "view", "view", "view", "cart", "cart", "purchase", "purchase", "purchase", "Purchase", "views", "cart ", "", "purchas", "purchased"][..if dirty { 16 } else { 10 }]).to_string(),
                    r.pick(&["p1", "p2", "p2", "p39594", "p17893", "p8823", "p310", "", "P1"]).to_string(),
                    r.pick(&["", "", "", "41607", "720", "0", "10000000", "10000001", "0250", "-5", "1.5", "12\r", "00000000000000000000000001"]).to_string(),
                ];
                let mut l = f.map(|v| edge_quote(&mut r, v, dirty)).join(",");
                match if dirty { r.below(15) } else { 99 } {
                    0 => {
                        let k = r.below(l.len());
                        if l.is_char_boundary(k) && l.is_char_boundary(k + 1) {
                            l.replace_range(k..k + 1, r.pick(&["\"", ",", "\r", "\u{1}", " ", "\u{0}"]));
                        }
                    }
                    1 => l.push_str(",x"),
                    2 => l = l.replacen(',', "", 1),
                    3 => l.push_str(&" ".repeat(r.below(80))),
                    _ => {}
                }
                ev += &l;
                ev.push('\n');
                lines += 1;
            }
            if round % 3 == 0 {
                ev.pop();
            }
            let len = ev.len();
            let mut padded = ev.clone().into_bytes();
            padded.extend_from_slice(&[b'\n'; PAD]);
            let file: Bytes = Box::leak(ev.into_bytes().into_boxed_slice());
            let mut cuts = vec![0, len];
            for _ in 0..r.below(4) {
                cuts.push(r.below(len + 1));
            }
            cuts.sort();
            let mut b = Shard::new(16, 16);
            for w in cuts.windows(2) {
                for_records::<7>(file, w[0], w[1], |f| event_fields(&c, f, &mut b));
                b.drain::<true>(&c);
            }
            let fb = flat(&mut b);
            kept += fb.0.len();
            for (name, parse) in event_parsers() {
                let mut a = Shard::new(16, 16);
                for w in cuts.windows(2) {
                    unsafe { parse(&c, padded.as_ptr(), w[0], w[1], len, &mut a) };
                }
                let fa = flat(&mut a);
                if fa != fb {
                    let (ea, eb): (std::collections::BTreeSet<_>, std::collections::BTreeSet<_>) = (fa.0.iter().collect(), fb.0.iter().collect());
                    let (la, lb): (std::collections::BTreeSet<_>, std::collections::BTreeSet<_>) = (fa.1.iter().collect(), fb.1.iter().collect());
                    panic!(
                        "round {round}: {name} differs: events only here {:?}, only in the reference {:?}; dedup records only here {:?}, only in the reference {:?}\n{}",
                        ea.difference(&eb).take(5).collect::<Vec<_>>(),
                        eb.difference(&ea).take(5).collect::<Vec<_>>(),
                        la.difference(&lb).take(5).collect::<Vec<_>>(),
                        lb.difference(&la).take(5).collect::<Vec<_>>(),
                        String::from_utf8_lossy(file)
                    );
                }
            }
        }
        assert!(kept > 3000, "too few kept events ({kept} of {lines} lines) to mean anything");
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
        if wide::available() {
            let took = wide::USUAL.load(Relaxed) - usual0;
            eprintln!("the eight-line parser took {took} of {lines} lines itself; {kept} events kept");
            assert!(took * 5 > lines, "the eight-line parser took {took} of {lines} lines itself");
        }
    }

    /// `edge_lines_match_reference` for clicks: every click parser of the build against the field parser on click
    /// lines with the same IDs, timestamps and quotes, the click window's two ends (seven days before window_start
    /// counts, window_end does not), campaigns in the file and not, of 0 to 9 bytes.
    #[test]
    fn edge_clicks_match_reference() {
        let mut c = ctx();
        let camps: Vec<(Bytes, u16)> = (0..40u16).map(|i| (&*Box::leak(format!("c_{}", i * 37).into_bytes().into_boxed_slice()), i)).collect();
        c.campaigns = IdMap::new(&camps).with_short(&camps);
        c.ncamp = camps.len();
        let mut r = Rng(0x0fed_cba9_8765_4321);
        let jan1 = parse_ts(b"2028-01-01T00:00:00Z").unwrap() / 1000;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
        let usual0 = wide::USUAL.load(Relaxed);
        let (mut lines, mut kept) = (0, 0);
        for round in 0..300 {
            let mut ck = String::from("click_id,user_id,campaign_id,ts\n");
            for _ in 0..r.below(200) {
                // 2028-02-05 .. 2028-03-20; the click window is 2028-02-13 .. 2028-03-05
                let sec = if r.below(3) == 0 { 35 * 86_400 + r.below(45 * 86_400) as i64 } else { 43 * 86_400 + r.below(21 * 86_400) as i64 };
                let sec = match r.below(30) {
                    0 => 43 * 86_400,     // seven days before window_start
                    1 => 43 * 86_400 - 1, // a second before
                    2 => 64 * 86_400,     // window_end
                    3 => 64 * 86_400 - 1,
                    _ => sec,
                };
                let dirty = r.below(4) == 0;
                let camp = match r.below(24) {
                    0 => String::new(),
                    1 => "c_99999".to_string(),
                    2 => "C_37".to_string(),
                    3 => "c_37 ".to_string(),
                    4 => "c_123456".to_string(),
                    5 => "c_1234567".to_string(),
                    _ => format!("c_{}", r.below(45) * 37),
                };
                let f = [edge_id(&mut r, 'k', dirty), edge_id(&mut r, 'u', dirty), camp, edge_ts(&mut r, jan1, sec, dirty)];
                let mut l = f.map(|v| edge_quote(&mut r, v, dirty)).join(",");
                match if dirty { r.below(15) } else { 99 } {
                    0 => {
                        let k = r.below(l.len());
                        if l.is_char_boundary(k) && l.is_char_boundary(k + 1) {
                            l.replace_range(k..k + 1, r.pick(&["\"", ",", "\r", "\u{1}", " ", "\u{0}"]));
                        }
                    }
                    1 => l.push_str(",x"),
                    2 => l = l.replacen(',', "", 1),
                    3 => l.push_str(&" ".repeat(r.below(80))),
                    _ => {}
                }
                ck += &l;
                ck.push('\n');
                lines += 1;
            }
            if round % 3 == 0 {
                ck.pop();
            }
            let len = ck.len();
            let mut padded = ck.clone().into_bytes();
            padded.extend_from_slice(&[b'\n'; PAD]);
            let file: Bytes = Box::leak(ck.into_bytes().into_boxed_slice());
            let mut cuts = vec![0, len];
            for _ in 0..r.below(4) {
                cuts.push(r.below(len + 1));
            }
            cuts.sort();
            let mut b = Shard::new(16, 16);
            for w in cuts.windows(2) {
                for_records::<4>(file, w[0], w[1], |f| click_fields(&c, f, &mut b));
                b.drain::<false>(&c);
            }
            let cb = click_list(&mut b);
            kept += cb.len();
            for (name, parse) in click_parsers() {
                let mut a = Shard::new(16, 16);
                for w in cuts.windows(2) {
                    unsafe { parse(&c, padded.as_ptr(), w[0], w[1], len, &mut a) };
                }
                let ca = click_list(&mut a);
                if ca != cb {
                    let (sa, sb): (std::collections::BTreeSet<_>, std::collections::BTreeSet<_>) = (ca.iter().collect(), cb.iter().collect());
                    panic!(
                        "round {round}: {name} differs: clicks only here {:x?}, only in the reference {:x?}\n{}",
                        sa.difference(&sb).take(5).collect::<Vec<_>>(),
                        sb.difference(&sa).take(5).collect::<Vec<_>>(),
                        String::from_utf8_lossy(file)
                    );
                }
            }
        }
        assert!(kept > 3000, "too few kept clicks ({kept} of {lines} lines) to mean anything");
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
        if wide::available() {
            let took = wide::USUAL.load(Relaxed) - usual0;
            eprintln!("the eight-line parser took {took} of {lines} click lines itself; {kept} clicks kept");
            assert!(took * 5 > lines, "the eight-line parser took {took} of {lines} click lines itself");
        }
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
    fn pack_at_matches_pack_and_keeps_order() {
        let mut r = Rng(42);
        let names = Interner::new();
        let mut prev: Option<(Vec<u8>, u64)> = None;
        let mut packed = 0;
        for i in 0..300_000 {
            let len = r.below(12);
            let mut id: Vec<u8> = (0..len).map(|_| 1 + r.below(127) as u8).collect();
            if i % 5 == 0 && len > 0 {
                id[r.below(len)] = [0u8, 0x80, 0xff, 0xc3, 0x7f, 1][r.below(6)]; // mostly bytes that do not pack
            }
            let mut buf = id.clone();
            buf.extend((0..32).map(|_| r.next() as u8));
            let k = pack(&id);
            assert_eq!(k != 0, (1..=9).contains(&len) && id.iter().all(|&c| c != 0 && c < 128), "{id:?}");
            assert_eq!(unsafe { pack_at(buf.as_ptr(), len) }, k, "{id:?}");
            if k == 0 {
                continue;
            }
            packed += 1;
            assert!(k & INTERNED == 0);
            let mut back = Vec::new();
            names.bytes(k, &mut back);
            assert_eq!(back, id);
            let (mut out, mut tmp) = (Vec::new(), Vec::new());
            put_name(&mut out, &names, k, &mut tmp);
            assert_eq!(out[..4], (id.len() as u32).to_le_bytes());
            assert_eq!(out[4..], id[..]);
            if let Some((pid, pk)) = &prev {
                assert_eq!(pid.cmp(&id), pk.cmp(&k), "{pid:?} {id:?}");
            }
            prev = Some((id, k));
        }
        assert!(packed > 100_000);
        // a prefix sorts first, and so does an upper-case variant
        assert!(pack(b"u1") < pack(b"u10") && pack(b"U9zzzzzzz") < pack(b"u1") && pack(b"u1") != pack(b"u1\0"));
        assert_eq!(pack(b"0123456789"), 0);
        // IDs that do not pack: exact, and ordered by their bytes against packed ones
        let (a, b, c) = (names.key(b"user-with-dash"), names.key(b"user-with-dash"), names.key(" \u{e9}1".as_bytes()));
        assert!(a == b && a != c && a & INTERNED != 0);
        assert!(names.less(c, pack(b"u1")) && names.less(pack(b"u1"), a) && !names.less(a, a));
        // a prefix and 32 lowercase hex digits: the long-ID table, exact, its bytes back, ordered by its bytes
        let hex = |i: u64| format!("u{:016x}{:016x}", mix(i), mix(i + 1000));
        let mut keys = Vec::new();
        for i in 0..3000u64 {
            let id = hex(i);
            let k = names.key(id.as_bytes());
            assert_eq!(k & (INTERNED | LONG), INTERNED | LONG, "{id}");
            assert_eq!(names.key(format!("{id}").as_bytes()), k);
            let mut back = Vec::new();
            names.bytes(k, &mut back);
            assert_eq!(back, id.as_bytes());
            keys.push((id, k));
        }
        let all: std::collections::HashSet<u64> = keys.iter().map(|x| x.1).collect();
        assert_eq!(all.len(), keys.len());
        for w in keys.windows(2) {
            assert_eq!(names.less(w[0].1, w[1].1), w[0].0 < w[1].0);
            assert_eq!(names.less(w[0].1, pack(b"u5")), w[0].0.as_str() < "u5");
        }
        // a longer prefix too; upper case, another length or a byte that is not a digit: the locked maps
        let k8 = names.key(b"abcdefgh0123456789abcdef0123456789abcdef");
        assert_eq!(k8 & (INTERNED | LONG), INTERNED | LONG);
        let mut back = Vec::new();
        names.bytes(k8, &mut back);
        assert_eq!(back, b"abcdefgh0123456789abcdef0123456789abcdef");
        for odd in [&b"u0123456789ABCDEF0123456789abcdef"[..], b"u0123456789abcdef0123456789abcdeg", b"u0123456789abcdef0123456789abcde", b"\xe90123456789abcdef0123456789abcdef"] {
            let k = names.key(odd);
            assert_eq!(k & (INTERNED | LONG), INTERNED, "{odd:?}");
            let mut back = Vec::new();
            names.bytes(k, &mut back);
            assert_eq!(back, odd);
        }
        // a full table sends the rest to the locked maps, and the keys stay exact
        let small = Interner::sized(4, 2);
        let ks: Vec<u64> = (0..100).map(|i| small.key(hex(i).as_bytes())).collect();
        assert_eq!(ks.iter().filter(|&&k| k & LONG != 0).count(), 16);
        for (i, &k) in ks.iter().enumerate() {
            assert_eq!(small.key(hex(i as u64).as_bytes()), k);
            let mut back = Vec::new();
            small.bytes(k, &mut back);
            assert_eq!(back, hex(i as u64).as_bytes());
        }
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

    /// `IdTable::people` (a batch at a time: with the 512-bit compare where the CPU has it, and without) against
    /// `slot` and `person` one ID at a time: linked IDs, IDs the pass adds itself, IDs whose home line is full of
    /// others. And the numbers of people: `slot_of` finds the slot `own` numbered, in the person's partition.
    #[test]
    fn batches_find_the_people_single_lookups_do() {
        let c = ctx();
        let ids = &c.ids;
        let mut r = Rng(7);
        let mut ins = 0;
        let key = |k: usize| c.names.key(format!("u{k:07}").as_bytes());
        // 24,000 IDs in 2^14 lines, a third of them joined into trees
        let slots: Vec<u32> = (0..24_000).map(|k| ids.slot(key(k), &mut ins)).collect();
        for _ in 0..8_000 {
            ids.union(&c.names, slots[r.below(24_000)], slots[r.below(24_000)]);
        }
        ids.flatten(3);
        let local = (1u32 << c.pshift) - 1;
        for &t in &slots {
            let p = ids.own(t);
            assert_eq!(ids.slot_of(p), t);
            assert_eq!(p >> c.pshift, (t >> 3) >> ids.part);
            assert!((1..=5 << ids.part).contains(&(p & local)), "person {p} of slot {t}");
        }
        let want: Vec<u32> = slots.iter().map(|&t| ids.person(t)).collect();
        let mut linked = 0;
        for (k, &p) in want.iter().enumerate() {
            // a person is the number of its root, which holds the tree's smallest ID and is its own person
            assert!(ids.name(p) <= key(k) && ids.person(ids.slot_of(p)) == p);
            linked += (ids.name(p) != key(k)) as usize;
        }
        assert!(linked > 5_000, "{linked}");
        // the same table without the 512-bit compare
        let narrow = IdTable { slots: ids.slots, lines: ids.lines, bits: ids.bits, part: ids.part, wide: false, count: AtomicUsize::new(0), limit: ids.limit };
        let (mut fresh, mut far) = (0, 0);
        for round in 0..400 {
            let t = if round % 2 == 0 { ids } else { &narrow };
            let n = 1 + r.below(BATCH);
            let which: Vec<usize> = (0..n).map(|_| if r.below(3) == 0 { 24_000 + r.below(12_000) } else { r.below(24_000) }).collect();
            let (mut keys, mut home, mut person) = ([0u64; BATCH], [0u32; BATCH], [0u32; BATCH]);
            for (i, &k) in which.iter().enumerate() {
                keys[i] = key(k);
                home[i] = t.aim(keys[i]);
            }
            t.people(&keys, &mut home, n, &mut person, &mut ins);
            for (i, &k) in which.iter().enumerate() {
                let slot = ids.slot(keys[i], &mut ins);
                far += (slot >> 3 != ids.home(keys[i]) as u32 >> 2) as usize;
                if k < 24_000 {
                    assert_eq!(person[i], want[k], "ID {k} in round {round}");
                } else {
                    fresh += 1;
                    assert_eq!((ids.name(person[i]), ids.person(slot)), (keys[i], person[i]), "new ID {k} in round {round}");
                }
            }
        }
        assert!(fresh > 3_000 && far > 50, "{fresh} {far}");
    }

    /// `mark_duplicates` against the rule itself, per key: every copy with an event that is ingested after the
    /// earliest copy is dropped, and of the earliest copies (identical ones) exactly one stays. Keys with one copy
    /// and with thousands, short IDs, copies without an event, far and late ingest_ts, three shards, four buckets.
    #[test]
    fn duplicates_keep_the_earliest_copy() {
        let mut c = ctx();
        c.qshift = 62;
        let (parts, buckets) = (1usize << (17 - c.pshift), 4usize);
        let mut r = Rng(5);
        let mut shards: Vec<Shard> = (0..3).map(|_| Shard::new(parts, buckets)).collect();
        let mut copies: HashMap<u64, Vec<(i64, *mut ERec)>> = HashMap::new();
        let mut n = 0;
        for k in 0..70_000usize {
            let id = if k < 1200 { format!("e{k}") } else { format!("e{k:08}") };
            let key = c.names.key(id.as_bytes());
            let count = match r.below(100) {
                0..=79 => 1,
                80..=94 => 2,
                95..=98 => 3 + r.below(3),
                _ => 30,
            } + if k == 4242 { 5000 } else { 0 };
            let first = match r.below(100) {
                0 => c.ilo - DAY - 1000 * r.below(1000) as i64, // far before the window: kept exact on the side
                1 => c.ihi + 1000 * (1 + r.below(1000) as i64), // late whatever its ts: never has an event
                _ => c.ws + 1000 * r.below(14 * 86_400) as i64,
            };
            let (mut ingest, mut valid) = (first, r.below(10) != 0 && first <= c.ihi);
            for _ in 0..count {
                // one in ten copies is identical to the one before it; the others are ingested later
                if r.below(10) != 0 {
                    ingest += 1000 * (1 + r.below(600) as i64);
                    valid = r.below(10) != 0 && ingest <= c.ihi;
                }
                let Shard { arena, ev, .. } = &mut shards[r.below(3)];
                let rec = if valid { ev[r.below(parts)].push(arena, EV_BLOCK, ERec { ts: 5, lw: 7 }) } else { std::ptr::null_mut() };
                let loc = if valid { loc_of(c.abase, rec) } else { 0 };
                shards[r.below(3)].log(&c, key, ingest, loc);
                copies.entry(key).or_default().push((ingest, rec));
                n += 1;
            }
        }
        for s in &mut shards {
            s.ev.iter_mut().for_each(Stream::finish);
            s.lg.iter_mut().for_each(Log::finish);
        }
        assert!(n > 100_000 && shards.iter().all(|s| s.lg.iter().all(|l| l.len() > 4 * LG_BLOCK)));
        let cands = mark_duplicates(&c, &shards, buckets, 3);
        assert!(cands > n / 4 && cands <= n, "{cands} of {n}");
        // a dropped event has its ts zeroed (`mark_duplicates`)
        let dropped = |rec: *mut ERec| unsafe { (*rec).ts == 0 };
        let (mut kept, mut gone) = (0, 0);
        for (key, list) in &copies {
            let min = list.iter().map(|x| x.0).min().unwrap();
            let (first, later): (Vec<&(i64, *mut ERec)>, Vec<&(i64, *mut ERec)>) = list.iter().filter(|x| !x.1.is_null()).partition(|x| x.0 == min);
            for x in &later {
                assert!(dropped(x.1), "key {key:x}: a later copy stayed");
            }
            // the earliest copies are identical: all of them have an event or none has, and one of them stays
            if !first.is_empty() {
                assert_eq!(first.len(), list.iter().filter(|x| x.0 == min).count());
                assert_eq!(first.iter().filter(|x| !dropped(x.1)).count(), 1, "key {key:x}: {} earliest copies", first.len());
                kept += 1;
            }
            gone += later.len() + first.len().saturating_sub(1);
        }
        assert!(kept > 50_000 && gone > 20_000, "{kept} {gone}");
    }

    /// `build_links` against a plain union-find over the lines `split` takes: files larger than a chunk, IDs of
    /// every kind (short, 9 bytes, long, not ASCII), quoted fields, self links, empty sides, malformed lines, lines
    /// past the 64-byte masks, a file that is only its header and one without a last newline.
    #[test]
    fn link_files_join_people() {
        let dir = std::env::temp_dir().join(format!("sessions-links-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut r = Rng(11);
        let id = |r: &mut Rng| -> String {
            let k = r.below(12_000);
            match r.below(40) {
                0 => format!("U{k:08}"),
                1 => format!("u{}", k % 1000),
                2 => format!("a-long-user-id-{k}"),
                3 => format!("u\u{e9}{k:06}"),
                _ => format!("u{k:08}"),
            }
        };
        let mut texts = Vec::new();
        for (f, lines) in [40_000, 0, 9_000, 25_000].into_iter().enumerate() {
            let mut t = String::from("user_id_a,user_id_b\n");
            for _ in 0..lines {
                let (a, b) = (id(&mut r), id(&mut r));
                let mut q = |v: &str| if r.below(5) == 0 { format!("\"{v}\"") } else { v.to_string() };
                let (qa, qb) = (q(&a), q(&b));
                t += &match r.below(60) {
                    0 => format!("{a},{a}"),
                    1 => format!(",{b}"),
                    2 => format!("{a},"),
                    3 => format!("{a},{b},{a}"),
                    4 => format!("\"{a},{b}"),
                    5 => String::new(),
                    6 => a.clone(),
                    7 => format!("{a},\"\""),
                    8 => format!("{a} , {b}"),
                    9 => format!("{a},{b}\r"),
                    10 => format!("{a}{},{b}", "x".repeat(70)),
                    11 => format!("{qa},{b}{}", "y".repeat(40)),
                    _ => format!("{qa},{qb}"),
                };
                t.push('\n');
            }
            if f == 2 {
                t.pop(); // no newline at the end of this file
            }
            std::fs::write(dir.join(format!("part-{f}.csv")), &t).unwrap();
            texts.push(t);
        }
        // the reference: every line `split` takes with two non-empty IDs joins them
        let mut num: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut up: Vec<usize> = Vec::new();
        fn root(up: &mut [usize], mut x: usize) -> usize {
            while up[x] != x {
                up[x] = up[up[x]];
                x = up[x];
            }
            x
        }
        let mut joined = 0;
        for t in &texts {
            let t: Bytes = Box::leak(t.clone().into_bytes().into_boxed_slice());
            for line in t.split(|&c| c == b'\n').skip(1) {
                let Some([a, b]) = split::<2>(line) else { continue };
                if a.is_empty() || b.is_empty() {
                    continue;
                }
                joined += 1;
                let [x, y] = [a, b].map(|v| {
                    let n = up.len();
                    let i = *num.entry(v.to_vec()).or_insert(n);
                    if i == n {
                        up.push(n);
                    }
                    i
                });
                let (x, y) = (root(&mut up, x), root(&mut up, y));
                up[x] = y;
            }
        }
        let mut best: HashMap<usize, Vec<u8>> = HashMap::new();
        for (v, &i) in &num {
            let e = best.entry(root(&mut up, i)).or_insert_with(|| v.clone());
            if v < e {
                *e = v.clone();
            }
        }
        assert!(joined > 60_000 && num.len() > 15_000 && best.len() > 100, "{joined} {} {}", num.len(), best.len());

        let c = ctx();
        let files = csv_files(&dir);
        assert_eq!(files.len(), 4);
        assert_eq!(build_links(&c.ids, &c.names, &files, 3), joined);
        std::fs::remove_dir_all(&dir).unwrap();
        let mut ins = 0;
        for (v, &i) in &num {
            let person = c.ids.person(c.ids.slot(c.names.key(v), &mut ins));
            let mut name = Vec::new();
            c.names.bytes(c.ids.name(person), &mut name);
            assert_eq!(&name, &best[&root(&mut up, i)], "ID {:?}", String::from_utf8_lossy(v));
        }
        assert_eq!(ins, 0, "an ID of the links was not in the table");
    }
}
