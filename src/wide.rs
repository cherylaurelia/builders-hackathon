//! Events and clicks eight lines at a time, on a CPU with AVX-512 (F, BW, VL, DQ, CD: every scoring machine
//! has them).
//!
//! `parse_events_v3` in main.rs works one line at a time: some 250 instructions a line, and the core runs them
//! about as fast as it can decode them, so only fewer instructions make it faster. Here a line costs its loads
//! and its compares into mask registers (stored as they are); everything after that is done for eight lines by
//! one instruction: each line is a 64-bit lane of a 512-bit register.
//!
//!   A. per line: the masks of its commas and quotes, where its newline is
//!   B. eight lines: where the fields are (count-leading-zeros gives the place of a comma) and whether the
//!      quotes are in place
//!   C. per line: the bytes of four fields copied to a fixed place (a gather would do this in one instruction,
//!      but on the scorer's CPU it costs 13 ns: the microcode fix for CVE-2022-40982)
//!   D. eight lines: both IDs packed, the timestamps converted, the type compared (the campaign looked up), the
//!      window and the 24 hours checked; what is kept goes to the batch in one compressed store per column
//!
//! A lane that is not the usual shape (see `batch`, `click_batch`) goes to `event_line` or `click_line`, which
//! are exact on any line. What a usual lane gives is what `event_fields` and `click_fields` give: the tests
//! compare them on random lines.
//!
//! Licence: on the scorer's CPU (Cascade Lake, all 8 threads busy) 512-bit logic, shifts, compares, permutes and
//! `vpmuludq` leave the clock alone however dense. Back-to-back 512-bit `vpmaddwd`/`vpmaddubsw` cost 16% of it,
//! one in sixteen instructions costs nothing; here they are ten in some four hundred (`madd2`).

use super::*;
use std::arch::x86_64::*;

/// The three inputs of `vpternlog` as truth-table columns: an expression over these is the table (the immediate)
/// of the same expression over the instruction's three operands.
const TA: i32 = 0xF0;
const TB: i32 = 0xCC;
const TC: i32 = 0xAA;

/// True when this CPU has what the loops here need (and SESS_NARROW is not set).
pub fn available() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        std::env::var_os("SESS_NARROW").is_none()
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512cd")
    })
}

/// How far ahead of a walk its bytes are asked for (a mapped file comes from memory, a page at a time): 256 bytes
pub static PF: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Lines the eight-line paths took themselves (the tests check that it is most of them).
#[cfg(test)]
pub static USUAL: AtomicUsize = AtomicUsize::new(0);

/// The dates seen so far (the `YYYY-MM-DDT` of an F1 or F2 timestamp), by day of the month: two tables of 32
/// that four registers each hold, so a lookup is two permutes and a blend instead of a gather.
#[repr(C, align(64))]
pub struct DayV {
    ka: [u64; 32], // bytes 0..8 of the date
    kb: [u64; 32], // bytes 8..11, bit 24, and in the high half the seconds from window_start to its midnight; 0 = none
}

impl DayV {
    pub fn new() -> DayV {
        DayV { ka: [0; 32], kb: [0; 32] }
    }
}

/// Remembers the date of the F1/F2 timestamp at `p` when its first 19 bytes are a date and a time, and the date
/// is within 64 days of window_start: its seconds from there fit 32 bits many times over, and a date far from
/// the window (a bad ts) does not push out one that the next lines need.
#[cold]
#[inline(never)]
unsafe fn day_fill(ws: i64, p: *const u8, dv: &mut DayV) {
    let Some(t) = civil_at(p) else { return };
    let rel = (t.div_euclid(DAY) * DAY - ws) / 1000;
    if !(-64 * 86_400..=64 * 86_400).contains(&rel) {
        return;
    }
    let (a, b) = ((p as *const u64).read_unaligned(), (p.add(8) as *const u64).read_unaligned());
    let d = b & 0x0F0F;
    let slot = ((d * 10 + (d >> 8)) & 31) as usize;
    dv.ka[slot] = a;
    dv.kb[slot] = (b & 0xFF_FFFF) | 1 << 24 | (rel as i32 as u32 as u64) << 32;
}

// ---------------------------------------------------------------- what both loops are made of

/// A 64-byte field of the lanes (`Lanes`, `ClickLanes`) as a register.
macro_rules! ld {
    ($a:expr) => {
        _mm512_load_si512($a.as_ptr() as *const _)
    };
}
/// A register to a 64-byte field of the lanes.
macro_rules! sv {
    ($a:expr, $v:expr) => {
        _mm512_store_si512($a.as_mut_ptr() as *mut _, $v)
    };
}
/// The 64 bytes at `64 * i` of a field of copies.
macro_rules! z {
    ($a:expr, $i:expr) => {
        _mm512_load_si512($a.as_ptr().add(64 * $i) as *const _)
    };
}
/// A number in every lane.
macro_rules! k {
    ($v:expr) => {
        _mm512_set1_epi64($v as i64)
    };
}
/// Where the lowest set bit of x is (64 for none), given x - 1.
macro_rules! tzv {
    ($x:expr, $xm:expr) => {
        _mm512_sub_epi64(k!(64), _mm512_lzcnt_epi64(_mm512_andnot_si512($x, $xm)))
    };
}
/// The quote bit at place `at` of the mask `q`, as 0 or 1.
macro_rules! qbit {
    ($q:expr, $at:expr) => {
        _mm512_and_si512(_mm512_srlv_epi64($q, $at), k!(1))
    };
}
/// `window2`: nonzero where a quote of `q` is out of place among the boundaries `b`.
macro_rules! misquoted {
    ($b:expr, $q:expr) => {{
        let s = _mm512_or_si512(_mm512_slli_epi64::<1>($b), k!(1));
        let z = _mm512_srli_epi64::<1>($b);
        let x = _mm512_and_si512($q, s);
        let yz = _mm512_and_si512($q, z);
        // a quote neither first nor last of a field, or both; an opening one without its closing one, or the reverse
        _mm512_or_si512(_mm512_andnot_si512(_mm512_xor_si512(s, z), $q), _mm512_and_si512(_mm512_sub_epi64(_mm512_slli_epi64::<1>(yz), x), $b))
    }};
}
/// The copy of a field: 16 or 32 bytes from where the i-th start of `$a` says (read back from memory: taken out
/// of the register it would cost the two ports that the 512-bit instructions have).
macro_rules! copy16 {
    ($to:expr, $buf:expr, $a:expr, $i:expr) => {
        _mm_storeu_si128($to.as_mut_ptr().add(16 * $i) as *mut _, _mm_loadu_si128($buf.add(std::ptr::read_volatile($a.as_ptr().add($i)) as usize) as *const _))
    };
}
macro_rules! copy32 {
    ($to:expr, $buf:expr, $a:expr, $i:expr) => {
        _mm256_storeu_si256($to.as_mut_ptr().add(32 * $i) as *mut _, _mm256_loadu_si256($buf.add(std::ptr::read_volatile($a.as_ptr().add($i)) as usize) as *const _))
    };
}

/// Where eight walks over the lines that start in [start, end) of the `len` bytes at `buf` start, and where the
/// part of each ends; None when no line starts there.
///
/// The piece is cut in eight at line starts and each lane walks its own part: where a line ends is known only
/// from its newline (load, compare, mask to register, count: some 16 cycles), and with one walk the eight lines
/// of a batch would wait for one another. The parts end within a few lines of each other; a lane whose part is
/// done sits out.
unsafe fn walks(buf: *const u8, start: usize, end: usize, len: usize) -> Option<([u64; 8], [u64; 8])> {
    let file = std::slice::from_raw_parts(buf, len);
    let first = start.saturating_sub(1);
    let p0 = first + memchr::memchr(b'\n', &file[first..])? + 1;
    let stop = end.min(len);
    // lane j takes the lines that start in [cut j, cut j + 1): it starts at the first line start from its cut on,
    // which is right after the first newline at or after the byte before the cut. No such newline: no line
    let (mut cur, mut lim) = ([len as u64; 8], [stop as u64; 8]);
    cur[0] = p0 as u64;
    for j in 1..8 {
        let cut = p0 + stop.saturating_sub(p0) * j / 8;
        lim[j - 1] = cut as u64;
        if let Some(i) = memchr::memchr(b'\n', &file[cut - 1..]) {
            cur[j] = (cut + i) as u64;
        }
    }
    Some((cur, lim))
}

/// Digit pairs to numbers: bytes times `w8` added in pairs (16 bits), those times `w16` added in pairs (32 bits).
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
unsafe fn madd2(x: __m512i, w8: __m512i, w16: __m512i) -> __m512i {
    _mm512_madd_epi16(_mm512_maddubs_epi16(x, w8), w16)
}

/// The `HH:MM:SS` (or `HH:MM` + three zero bytes) in each lane as seconds, and a vector that is zero in the
/// lanes where every byte is what it may be: digits within 2 9 : 5 9 : 5 9. An hour of 24 to 29 passes and
/// gives 86,400 seconds or more.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
unsafe fn hms(w: __m512i, pattern: u64, limits: u64) -> (__m512i, __m512i) {
    let x = _mm512_xor_si512(w, _mm512_set1_epi64(pattern as i64));
    let bad = _mm512_subs_epu8(x, _mm512_set1_epi64(limits as i64));
    // words: HH, 10 x the tens of MM, the units of MM, SS; then HH x 3600 + tens x 600, and units x 60 + SS
    let w8 = _mm512_set1_epi64(i64::from_le_bytes([10, 1, 0, 10, 1, 0, 10, 1]));
    let w16 = _mm512_set1_epi64(3600 | 60 << 16 | 60 << 32 | 1 << 48);
    let v = madd2(x, w8, w16);
    let secs = _mm512_and_si512(_mm512_add_epi64(v, _mm512_srli_epi64::<32>(v)), _mm512_set1_epi64(0xFFFF_FFFF));
    (secs, bad)
}

const HMS: u64 = u64::from_le_bytes(*b"00:00:00");
const HMS_MAX: u64 = u64::from_le_bytes([2, 9, 0, 5, 9, 0, 5, 9]);
const HM: u64 = u64::from_le_bytes(*b"00:00\0\0\0");
const HM_MAX: u64 = u64::from_le_bytes([2, 9, 0, 5, 9, 0, 0, 0]);
const DAY_SECS: i64 = 86_400;

/// The date of the timestamps whose bytes 0..8 are `w0` and 8..16 `w1`: the lanes whose date is in `dv`, and
/// for those the seconds from window_start to its midnight.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
unsafe fn date(dv: *const DayV, w0: __m512i, w1: __m512i) -> (__mmask8, __m512i) {
    // the day of the month from its two digits: bits 0..3 pick within a pair of registers, bit 4 picks the pair
    let slot = _mm512_maddubs_epi16(_mm512_and_si512(w1, _mm512_set1_epi64(0x0F0F)), _mm512_set1_epi64(0x010A));
    let high = _mm512_test_epi64_mask(slot, _mm512_set1_epi64(16));
    macro_rules! t {
        ($a:expr, $i:expr) => {
            _mm512_load_si512($a.add(8 * $i) as *const _)
        };
    }
    let (a, b) = ((*dv).ka.as_ptr(), (*dv).kb.as_ptr());
    let ka = _mm512_mask_blend_epi64(high, _mm512_permutex2var_epi64(t!(a, 0), slot, t!(a, 1)), _mm512_permutex2var_epi64(t!(a, 2), slot, t!(a, 3)));
    let kb = _mm512_mask_blend_epi64(high, _mm512_permutex2var_epi64(t!(b, 0), slot, t!(b, 1)), _mm512_permutex2var_epi64(t!(b, 2), slot, t!(b, 3)));
    // bytes 8..11 with bit 24 set against the low half of kb (0 where there is no date: bit 24 then differs)
    let y = _mm512_ternarylogic_epi64::<{ (TA & TB) | TC }>(w1, _mm512_set1_epi64(0xFF_FFFF), _mm512_set1_epi64(1 << 24));
    let hit = _mm512_mask_testn_epi64_mask(_mm512_cmpeq_epi64_mask(w0, ka), _mm512_xor_si512(y, kb), _mm512_set1_epi64(0xFFFF_FFFF));
    (hit, _mm512_srai_epi64::<32>(kb))
}

/// Eight `ts` fields, `w0..w3` their bytes 0..32, `kf1`/`kf2`/`kf3` the lanes of 20, 25 and 13 bytes (what F1,
/// F2 and an F3 of today's length have): the lanes that hold a timestamp of a kind this takes (F1 or F2 of a
/// date in `dv`, or 13 digits), for those its ms from window_start `ws`, and the lanes whose date is in `dv`.
/// A lane it does not take may still hold a timestamp: the caller sends that line to the one-line parser.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn stamp(dv: *const DayV, ws: i64, w0: __m512i, w1: __m512i, w2: __m512i, w3: __m512i, kf1: __mmask8, kf2: __mmask8, kf3: __mmask8) -> (__mmask8, __m512i, __mmask8) {
    let day_secs = k!(DAY_SECS);
    // F1 or F2: the date (bytes 0..11), HH:MM:SS (11..19), Z or a sign (19), HH:MM (20..25)
    let (kdate, rel) = date(dv, w0, w1);
    let (secs, bad_t) = hms(_mm512_or_si512(_mm512_srli_epi64::<24>(w1), _mm512_slli_epi64::<40>(w2)), HMS, HMS_MAX);
    let sign = _mm512_and_si512(_mm512_srli_epi64::<24>(w2), k!(0xFF));
    let (off, bad_o) = hms(_mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64::<32>(w2), _mm512_slli_epi64::<32>(w3)), k!(0xFF_FFFF_FFFFu64)), HM, HM_MAX);
    // a date seen before and a time of day; F1 ends in Z; F2 in + or - and an offset below 24:00
    let k12 = _mm512_mask_cmplt_epu64_mask(_mm512_mask_testn_epi64_mask(kdate, bad_t, bad_t), secs, day_secs);
    let kz = _mm512_mask_cmpeq_epi64_mask(kf1 & k12, sign, k!(b'Z'));
    let (kplus, kminus) = (_mm512_mask_cmpeq_epi64_mask(kf2, sign, k!(b'+')), _mm512_mask_cmpeq_epi64_mask(kf2, sign, k!(b'-')));
    let koff = _mm512_mask_cmplt_epu64_mask(_mm512_mask_testn_epi64_mask(kplus | kminus, bad_o, bad_o), off, day_secs);
    // seconds from window_start, the offset taken off (+) or put on (-)
    let tsec = _mm512_add_epi64(rel, secs);
    let tsec = _mm512_mask_sub_epi64(tsec, kplus, tsec, off);
    let tsec = _mm512_mask_add_epi64(tsec, kminus, tsec, off);
    let t12 = _mm512_mul_epi32(tsec, k!(1000));
    // F3 of 13 digits: the first five behind three zeros, and the last eight; each four digits a half
    let zeros = k!(0x3030_3030_3030_3030u64);
    let nines = k!(0x0909_0909_0909_0909u64);
    let x0 = _mm512_xor_si512(w0, zeros);
    let x1 = _mm512_and_si512(_mm512_xor_si512(w1, zeros), k!(0xFF_FFFF_FFFFu64));
    let bad3 = _mm512_or_si512(_mm512_subs_epu8(x0, nines), _mm512_subs_epu8(x1, nines));
    let w8 = _mm512_set1_epi16(0x010A);
    let w16 = _mm512_set1_epi32(0x0001_0064);
    let va = madd2(_mm512_slli_epi64::<24>(x0), w8, w16);
    let vb = madd2(_mm512_or_si512(_mm512_srli_epi64::<40>(x0), _mm512_slli_epi64::<24>(x1)), w8, w16);
    let e4 = k!(10_000);
    let na = _mm512_add_epi64(_mm512_mul_epu32(va, e4), _mm512_srli_epi64::<32>(va));
    let nb = _mm512_add_epi64(_mm512_mul_epu32(vb, e4), _mm512_srli_epi64::<32>(vb));
    let t3 = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mul_epu32(na, k!(100_000_000)), nb), k!(ws));
    let kts = kz | (k12 & koff) | _mm512_mask_testn_epi64_mask(kf3, bad3, bad3);
    (kts, _mm512_mask_blend_epi64(kf3, t12, t3), kdate)
}

/// The keys (`pack_at`) of eight IDs of 8 or 9 bytes, each byte 1..=127: `lo` their first eight bytes, `hi` the
/// eight after, `nine` the lanes of 9 bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
unsafe fn pack8(lo: __m512i, hi: __m512i, nine: __mmask8) -> __m512i {
    // first byte highest, then the seven low bits of the bytes squeezed together: pairs of 7, of 14, of 28
    let swap = _mm512_set4_epi64(0x0809_0A0B_0C0D_0E0F, 0x0001_0203_0405_0607, 0x0809_0A0B_0C0D_0E0F, 0x0001_0203_0405_0607);
    let w = _mm512_shuffle_epi8(lo, swap);
    let r = _mm512_ternarylogic_epi64::<{ (TC & TA) | (!TC & TB & 0xFF) }>(w, _mm512_srli_epi16::<1>(w), _mm512_set1_epi16(0x007F));
    let r = _mm512_ternarylogic_epi64::<{ (TC & TA) | (!TC & TB & 0xFF) }>(r, _mm512_srli_epi32::<2>(r), _mm512_set1_epi32(0x3FFF));
    let r = _mm512_ternarylogic_epi64::<{ (TC & TA) | (!TC & TB & 0xFF) }>(r, _mm512_srli_epi64::<4>(r), _mm512_set1_epi64(0x0FFF_FFFF));
    _mm512_or_si512(_mm512_slli_epi64::<7>(r), _mm512_maskz_and_epi64(nine, hi, _mm512_set1_epi64(0xFF)))
}

/// Where the lines of eight users start in the ID table, in bytes: 16 x `IdTable::home`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
unsafe fn homes(cx: &Ctx, ukey: __m512i) -> __m512i {
    const { assert!(std::mem::size_of::<Slot>() == 16) };
    _mm512_and_si512(_mm512_srl_epi64(_mm512_mullo_epi64(ukey, k!(0x9E37_79B9_7F4A_7C15u64)), _mm_cvtsi64_si128(60 - cx.ids.bits as i64)), k!(!63u64))
}

/// The permutes that take word i of every line out of the copies of a field: 16 bytes a line (two registers,
/// `EVEN` its first word, `ODD` its second), or 32 bytes a line (four registers: `IX_A` of two gives words 0
/// and 1 of four lines, `IX_B` words 2 and 3; `IX_LO`/`IX_HI` of two such put one word of all eight together).
const EVEN: [i64; 8] = [0, 2, 4, 6, 8, 10, 12, 14];
const ODD: [i64; 8] = [1, 3, 5, 7, 9, 11, 13, 15];
const IX_A: [i64; 8] = [0, 4, 8, 12, 1, 5, 9, 13];
const IX_B: [i64; 8] = [2, 6, 10, 14, 3, 7, 11, 15];
const IX_LO: [i64; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
const IX_HI: [i64; 8] = [4, 5, 6, 7, 12, 13, 14, 15];
macro_rules! ix {
    ($a:expr) => {
        _mm512_loadu_si512($a.as_ptr() as *const _)
    };
}
/// Words 0..4 of eight 32-byte copies.
macro_rules! words4 {
    ($a:expr) => {{
        let (a01, a23) = (_mm512_permutex2var_epi64(z!($a, 0), ix!(IX_A), z!($a, 1)), _mm512_permutex2var_epi64(z!($a, 2), ix!(IX_A), z!($a, 3)));
        let (b01, b23) = (_mm512_permutex2var_epi64(z!($a, 0), ix!(IX_B), z!($a, 1)), _mm512_permutex2var_epi64(z!($a, 2), ix!(IX_B), z!($a, 3)));
        (
            _mm512_permutex2var_epi64(a01, ix!(IX_LO), a23),
            _mm512_permutex2var_epi64(a01, ix!(IX_HI), a23),
            _mm512_permutex2var_epi64(b01, ix!(IX_LO), b23),
            _mm512_permutex2var_epi64(b01, ix!(IX_HI), b23),
        )
    }};
}
/// Words 0 and 1 of eight 16-byte copies.
macro_rules! words2 {
    ($a:expr) => {
        (_mm512_permutex2var_epi64(z!($a, 0), ix!(EVEN), z!($a, 1)), _mm512_permutex2var_epi64(z!($a, 0), ix!(ODD), z!($a, 1)))
    };
}

/// The 16 bytes at the starts `$a` (buffer offsets, a register just stored to `$a`) of four lines, from `$i` on,
/// as one register: loaded straight into it (no copy through memory: a register read back from four stores
/// waits for them to be written)
macro_rules! lanes16 {
    ($buf:expr, $a:expr, $i:expr) => {{
        let p = |j: usize| $buf.add(std::ptr::read_volatile($a.as_ptr().add($i + j)) as usize) as *const __m128i;
        let x = _mm512_inserti64x2::<1>(_mm512_castsi128_si512(_mm_loadu_si128(p(0))), _mm_loadu_si128(p(1)));
        _mm512_inserti64x2::<3>(_mm512_inserti64x2::<2>(x, _mm_loadu_si128(p(2))), _mm_loadu_si128(p(3)))
    }};
}
/// Words 0 and 1 of the 16 bytes at the eight starts `$a`, as `words2!` of their copies.
macro_rules! words2at {
    ($buf:expr, $a:expr) => {{
        let (z0, z1) = (lanes16!($buf, $a, 0), lanes16!($buf, $a, 4));
        (_mm512_permutex2var_epi64(z0, ix!(EVEN), z1), _mm512_permutex2var_epi64(z0, ix!(ODD), z1))
    }};
}
/// Words 0..4 of the 32 bytes at the eight starts `$a`, as `words4!` of their copies.
macro_rules! words4at {
    ($buf:expr, $a:expr) => {{
        let p = |j: usize| $buf.add(std::ptr::read_volatile($a.as_ptr().add(j)) as usize) as *const __m256i;
        let z = |j: usize| _mm512_inserti64x4::<1>(_mm512_castsi256_si512(_mm256_loadu_si256(p(j))), _mm256_loadu_si256(p(j + 1)));
        let (z0, z1, z2, z3) = (z(0), z(2), z(4), z(6));
        let (a01, a23) = (_mm512_permutex2var_epi64(z0, ix!(IX_A), z1), _mm512_permutex2var_epi64(z2, ix!(IX_A), z3));
        let (b01, b23) = (_mm512_permutex2var_epi64(z0, ix!(IX_B), z1), _mm512_permutex2var_epi64(z2, ix!(IX_B), z3));
        (
            _mm512_permutex2var_epi64(a01, ix!(IX_LO), a23),
            _mm512_permutex2var_epi64(a01, ix!(IX_HI), a23),
            _mm512_permutex2var_epi64(b01, ix!(IX_LO), b23),
            _mm512_permutex2var_epi64(b01, ix!(IX_HI), b23),
        )
    }};
}
/// A of event lane `$j` of the lanes `$l` (`Lanes`): the masks of the 128 bytes where its walk stands; the
/// lane's bit when its user_id looks long (longer than 11 bytes: v5's 33). `$k` = the bytes , " \n and 0x20 in
/// every byte of a register. SAFETY: a walk stands at len + 1 at most, and 176 bytes are readable from there.
macro_rules! lane_a {
    ($l:expr, $buf:expr, $j:expr, $pf:expr, $k:expr) => {{
        let (cm, qt, nv, lowest) = $k;
        let line = $buf.add(*$l.cur.get_unchecked($j) as usize);
        prefetch(line.wrapping_add($pf));
        prefetch(line.wrapping_add($pf + 64));
        let v0 = _mm512_loadu_si512(line as *const _);
        let v1 = _mm512_loadu_si512(line.add(64) as *const _);
        let c0 = _mm512_cmpeq_epi8_mask(v0, cm);
        *$l.c0.get_unchecked_mut($j) = c0;
        *$l.q0.get_unchecked_mut($j) = _mm512_cmpeq_epi8_mask(v0, qt);
        *$l.c1.get_unchecked_mut($j) = _mm512_cmpeq_epi8_mask(v1, cm);
        *$l.q1.get_unchecked_mut($j) = _mm512_cmpeq_epi8_mask(v1, qt);
        // as signed bytes, those above 0x7F are below 0x20 too
        *$l.odd.get_unchecked_mut($j) = _mm512_cmplt_epi8_mask(v0, lowest);
        *$l.odd1.get_unchecked_mut($j) = _mm512_cmplt_epi8_mask(v1, lowest);
        *$l.nl1.get_unchecked_mut($j) = _mm512_cmpeq_epi8_mask(v1, nv);
        (((tz(c0 & c0.wrapping_sub(1)) - tz(c0) > 12) as u32) << $j)
    }};
}
/// The long-ID table's slots of the user_ids of the lanes `$lu` of `$l` (their A done) asked for.
macro_rules! user_slots {
    ($cx:expr, $l:expr, $buf:expr, $lu:expr) => {{
        let mut lu: u32 = $lu;
        while lu != 0 {
            let j = lu.trailing_zeros() as usize;
            lu &= lu - 1;
            let line = $buf.add(*$l.cur.get_unchecked(j) as usize);
            let s1 = tz(*$l.c0.get_unchecked(j)) + 1;
            $cx.names.prefetch_long(line.add(s1 + (*line.add(s1) == b'"') as usize + 1));
        }
    }};
}
/// The four byte patterns `lane_a!` compares with.
macro_rules! lane_k {
    () => {
        (_mm512_set1_epi8(b',' as i8), _mm512_set1_epi8(b'"' as i8), _mm512_set1_epi8(b'\n' as i8), _mm512_set1_epi8(0x20))
    };
}

// ---------------------------------------------------------------- events

/// Eight event lines on their way through `batch`: every field is 64 bytes, a register's worth.
#[repr(C, align(64))]
struct Lanes {
    // the eight walks: where each lane's next line starts in the buffer, and where its part of the piece ends
    cur: [u64; 8],
    lim: [u64; 8],
    // A: per line
    c0: [u64; 8],  // commas of its bytes 0..64
    q0: [u64; 8],  // quotes
    c1: [u64; 8],  // commas of its bytes 64..128
    q1: [u64; 8],  // quotes
    odd: [u64; 8],  // bytes of its first 64 below 0x20 or above 0x7F
    odd1: [u64; 8], // of its bytes 64..128
    nl1: [u64; 8],  // newlines of its bytes 64..128
    // the commas of the lines `batch` works on (c0, c1 are already those of the next eight then)
    pc0: [u64; 8],
    pc1: [u64; 8],
    // B: the line as `batch` sees it (`unlong`): its newline is byte 64 + e2 (64: none in bytes 64..128); the
    // bytes cut out before user_id (24 or 0) and before ts (0, 24 or 48)
    e2: [u64; 8],
    sh1: [u64; 8],
    sh2: [u64; 8],
    // B: where to copy from (buffer offsets)
    ae: [u64; 8],
    au: [u64; 8],
    at: [u64; 8],
    ai: [u64; 8],
    // B: product_id and price_cents (start in the line, length), read for a purchase
    ps: [u64; 8],
    pl: [u64; 8],
    cs: [u64; 8],
    cl: [u64; 8],
    // C: the fields' bytes, line after line
    se: [u8; 128], // event_id: 16 bytes a line
    su: [u8; 128], // user_id
    st: [u8; 256], // ts: 32 bytes a line
    si: [u8; 256], // ingest_ts, and behind it the first 8 bytes of event_type
    // D: what the lines give
    ek: [u64; 8],
    uk: [u64; 8],
    ing: [u64; 8],
    tx: [u64; 8],
    home: [u64; 8], // the byte offset of the user's line in the ID table
}

/// Parses the event lines that start in [start, end) of the `len` bytes at `buf` into `st`, as `parse_events_v3`
/// does; `buf` must be readable for 144 bytes past `len`, and buf[len] must be `\n` (`read_span`).
///
/// A, per lane, has no branch: the masks are taken wherever the lane's walk stands (`walks`), and `batch` knows
/// which lanes stand on a line of theirs and which of those lines it can take.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
pub unsafe fn parse_events(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    if let Some((cur, lim)) = walks(buf, start, end, len) {
        let st: *mut Shard = st;
        let mut l: Lanes = std::mem::zeroed();
        (l.cur, l.lim) = (cur, lim);
        let pf = *PF.get_or_init(|| env_num("SESS_PF").unwrap_or(256));
        // A of the first eight lines; `batch` does A of the next eight itself, while its own lines are converted
        let k = lane_k!();
        let mut lu = 0u32;
        for j in 0..8 {
            lu |= lane_a!(l, buf, j, pf, k);
        }
        user_slots!(cx, l, buf, lu);
        while batch(cx, buf, len, st, &mut l, pf) {}
        flush::<true>(cx, buf, len, &mut *st);
    }
    st.drain::<true>(cx);
    sfence();
}

/// The masks of eight lines (c0|c1 commas, q0|q1 quotes, o0|o1 odd bytes of each line's bytes 0..128; n1 the newlines
/// of bytes 64..128) as those of the lines with the bytes 9..33 of a long ID (v5: 33 bytes, quotes left out) as
/// event_id or user_id cut out: the commas, quotes and odd bytes of their first 64 bytes and 64..128, the newline
/// of bytes 64..128 (64: none), and the bytes cut before user_id (24 or 0) and before ts (0, 24 or 48). A line
/// with no such ID keeps its masks. A line whose bytes past 128 are needed for its cut masks sees zeros there (no
/// comma, no newline): it is not usual, and `event_line` takes it.
///
/// The cut bytes are seen by no check here: the long-ID table (and `ev_hash`) take only 32 lowercase hex digits
/// after a 1-byte prefix (no comma, quote, newline or odd byte among them), and a lane whose ID they refuse goes to
/// `event_line`. A cut is made only where the comma after it shows a field of 33 bytes, quotes left out; any line
/// that is usual afterwards has its event_id in bytes 0..36, so the quote bits read here are exact for it.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
#[allow(clippy::type_complexity)]
unsafe fn unlong(c0: __m512i, c1: __m512i, q0: __m512i, q1: __m512i, o0: __m512i, o1: __m512i, n1: __m512i) -> (__m512i, __m512i, __m512i, __m512i, __m512i, __m512i, __m512i, __m512i) {
    let (m1, one) = (k!(-1), k!(1));
    // event_id from byte qe, its comma p1; user_id from s1 + qu, its comma p2 (64 + the first comma of c1 when c0
    // has one comma)
    let qe = _mm512_and_si512(q0, one);
    let c0m = _mm512_add_epi64(c0, m1);
    let p1 = tzv!(c0, c0m);
    let s1 = _mm512_add_epi64(p1, one);
    let qu = qbit!(q0, s1);
    let t1 = _mm512_and_si512(c0, c0m);
    let p2 = _mm512_mask_blend_epi64(_mm512_test_epi64_mask(t1, t1), _mm512_add_epi64(tzv!(c1, _mm512_add_epi64(c1, m1)), k!(64)), tzv!(t1, _mm512_add_epi64(t1, m1)));
    let l0 = _mm512_cmpeq_epi64_mask(p1, _mm512_add_epi64(k!(33), _mm512_add_epi64(qe, qe)));
    let l1 = _mm512_cmpeq_epi64_mask(p2, _mm512_add_epi64(_mm512_add_epi64(s1, k!(33)), _mm512_add_epi64(qu, qu)));
    let d1 = _mm512_maskz_mov_epi64(l0, k!(24));
    let d2 = _mm512_add_epi64(d1, _mm512_maskz_mov_epi64(l1, k!(24)));
    // 24 bits out of each 128-bit mask from bit `cut` on (192: none): the later cut first, so the earlier one's
    // place is the same before and after it
    let (mut c0, mut c1, mut q0, mut q1, mut o0, mut o1, mut n1) = (c0, c1, q0, q1, o0, o1, n1);
    for cut in [_mm512_mask_blend_epi64(l1, k!(192), _mm512_add_epi64(_mm512_add_epi64(s1, qu), k!(9))), _mm512_mask_blend_epi64(l0, k!(192), _mm512_add_epi64(qe, k!(9)))] {
        // the bits below the cut in the low and the high word
        let lo = _mm512_andnot_si512(_mm512_sllv_epi64(m1, cut), m1);
        let hi = _mm512_andnot_si512(_mm512_sllv_epi64(m1, _mm512_max_epi64(_mm512_sub_epi64(cut, k!(64)), _mm512_setzero_si512())), m1);
        macro_rules! cut {
            ($w0:expr, $w1:expr) => {{
                let s0 = _mm512_ternarylogic_epi64::<{ TA | TB }>(_mm512_srli_epi64::<24>($w0), _mm512_slli_epi64::<40>($w1), $w1);
                (_mm512_ternarylogic_epi64::<{ (TA & TB) | (!TA & TC) }>(lo, $w0, s0), _mm512_ternarylogic_epi64::<{ (TA & TB) | (!TA & TC) }>(hi, $w1, _mm512_srli_epi64::<24>($w1)))
            }};
        }
        (c0, c1) = cut!(c0, c1);
        (q0, q1) = cut!(q0, q1);
        (o0, o1) = cut!(o0, o1);
        n1 = _mm512_ternarylogic_epi64::<{ (TA & TB) | (!TA & TC) }>(hi, n1, _mm512_srli_epi64::<24>(n1));
    }
    let _ = o1;
    (c0, c1, q0, q1, o0, tzv!(n1, _mm512_add_epi64(n1, m1)), d1, d2)
}

/// The lines the walks of `l` stand on (A filled in), each walk moved past its line: the usual ones into the
/// batch or, when the event is not kept, into the dedup records alone; the others through `event_line`. False
/// when no walk stood on a line of its part: the piece is done.
///
/// The usual line: its first 64 bytes are 0x20..=0x7F, its third comma is among the first 63 of them and its
/// newline within 64 bytes of the fourth field's start; seven fields, well quoted; both IDs 1 to 9 bytes; ts is
/// F1 or F2 of a date seen before, or 13 digits, and valid; ingest_ts is a valid F1 of a date seen before;
/// event_type is `view`, `cart` or `purchase`. For such a line this does what `event_fields` does.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
unsafe fn batch(cx: &Ctx, buf: *const u8, len: usize, st: *mut Shard, l: &mut Lanes, pf: usize) -> bool {
    let (m1, one, c64) = (k!(-1), k!(1), k!(64));
    // room for eight more in the batch, made before anything is in a register (a call would spill it all)
    let bt: *mut Batch = &mut (*st).bt;
    if (*bt).n > BATCH - 8 {
        (*st).drain::<true>(cx);
    }

    // ---- B, first part: where the four fields to copy start
    let pv = ld!(l.cur);
    let (rc0, rc1) = (ld!(l.c0), ld!(l.c1));
    sv!(l.pc0, rc0);
    sv!(l.pc1, rc1);
    let (c0, c1, q0, q1, odd, e2, sh1, sh2) = unlong(rc0, rc1, ld!(l.q0), ld!(l.q1), ld!(l.odd), ld!(l.odd1), ld!(l.nl1));
    sv!(l.e2, e2);
    sv!(l.sh1, sh1);
    sv!(l.sh2, sh2);
    // the walks that stand on a line of their part; each goes past its line (`event_line` below puts the walk
    // of a line that is not usual where that line really ends)
    let live = _mm512_cmplt_epu64_mask(pv, ld!(l.lim));
    if live == 0 {
        return false;
    }
    sv!(l.cur, _mm512_mask_add_epi64(pv, live, pv, _mm512_add_epi64(_mm512_add_epi64(e2, k!(65)), sh2)));
    // the first three commas: p1, p2, p3
    let c0m = _mm512_add_epi64(c0, m1);
    let t1 = _mm512_and_si512(c0, c0m);
    let t1m = _mm512_add_epi64(t1, m1);
    let t2 = _mm512_and_si512(t1, t1m);
    let t2m = _mm512_add_epi64(t2, m1);
    let (p1, p2, p3) = (tzv!(c0, c0m), tzv!(t1, t1m), tzv!(t2, t2m));
    // a field's value starts after its opening quote and is two shorter for the pair
    let qe = _mm512_and_si512(q0, one);
    let s1 = _mm512_add_epi64(p1, one);
    let qu = qbit!(q0, s1);
    let us = _mm512_add_epi64(s1, qu);
    let s2 = _mm512_add_epi64(p2, one);
    let qt = qbit!(q0, s2);
    let tp = _mm512_add_epi64(s2, qt);
    // the second window: 64 bytes from the fourth field's start `o`, through the newline
    let o = _mm512_add_epi64(p3, one);
    let thru = _mm512_add_epi64(_mm512_sllv_epi64(k!(2), e2), m1);
    let bhi = _mm512_or_si512(_mm512_and_si512(c1, thru), _mm512_sllv_epi64(one, e2));
    let qhi = _mm512_and_si512(q1, thru);
    let sh = _mm512_sub_epi64(c64, o);
    let b2 = _mm512_or_si512(_mm512_srlv_epi64(c0, o), _mm512_sllv_epi64(bhi, sh));
    let q2 = _mm512_or_si512(_mm512_srlv_epi64(q0, o), _mm512_sllv_epi64(qhi, sh));
    let b2m = _mm512_add_epi64(b2, m1);
    let r1 = tzv!(b2, b2m);
    let qi = _mm512_and_si512(q2, one);
    let ip = _mm512_add_epi64(o, qi);
    let s4 = _mm512_add_epi64(r1, one);
    let qk = qbit!(q2, s4);
    let el = _mm512_sub_epi64(p1, _mm512_add_epi64(qe, qe));
    let ul = _mm512_sub_epi64(_mm512_sub_epi64(p2, s1), _mm512_add_epi64(qu, qu));
    let tl = _mm512_sub_epi64(_mm512_sub_epi64(p3, s2), _mm512_add_epi64(qt, qt));
    let il = _mm512_sub_epi64(r1, _mm512_add_epi64(qi, qi));
    // (each compare takes the lanes the one before left: one instruction, where an AND of two masks is two)
    // three commas in the first 63 bytes, the newline inside the second window (and so in bytes 64..128); both
    // IDs 1 to 9 bytes, ingest_ts 20, ts 13, 20 or 25: every start is
    // then within the first 66 bytes
    let k_w = _mm512_mask_cmple_epu64_mask(_mm512_mask_cmplt_epu64_mask(_mm512_test_epi64_mask(t2, t2), e2, o), p3, k!(62));
    let k_len = _mm512_mask_cmple_epu64_mask(_mm512_mask_cmple_epu64_mask(k_w, _mm512_sub_epi64(el, one), k!(8)), _mm512_sub_epi64(ul, one), k!(8));
    let k_len = _mm512_mask_cmpeq_epi64_mask(k_len, il, k!(20));
    let (kf1, kf2, kf3) = (_mm512_mask_cmpeq_epi64_mask(k_len, tl, k!(20)), _mm512_mask_cmpeq_epi64_mask(k_len, tl, k!(25)), _mm512_mask_cmpeq_epi64_mask(k_len, tl, k!(13)));
    let k_pos = kf1 | kf2 | kf3;
    // any other lane copies from its line's start
    sv!(l.ae, _mm512_mask_add_epi64(pv, k_pos, pv, qe));
    sv!(l.au, _mm512_mask_add_epi64(pv, k_pos, pv, _mm512_add_epi64(us, sh1)));
    sv!(l.at, _mm512_mask_add_epi64(pv, k_pos, pv, _mm512_add_epi64(tp, sh2)));
    sv!(l.ai, _mm512_mask_add_epi64(pv, k_pos, pv, _mm512_add_epi64(ip, sh2)));

    // ---- C: the fields' bytes to their fixed places
    // SAFETY: a start is at most 66 + 48 bytes into its line and 32 bytes are read: within the PAD readable ones
    for i in 0..8 {
        copy16!(l.se, buf, l.ae, i);
        copy16!(l.su, buf, l.au, i);
        copy32!(l.st, buf, l.at, i);
        copy32!(l.si, buf, l.ai, i);
    }

    // ---- A of the next eight lines (every walk is past its line already): their loads go out while this batch
    // is converted. A lane whose line here is not usual is walked again at the end, from where that line ends.
    let k = lane_k!();
    let mut lu = 0u32;
    for j in 0..8 {
        lu |= lane_a!(l, buf, j, pf, k);
    }
    user_slots!(cx, l, buf, lu);

    // ---- B, second part (it does not wait for the copies): the quotes, the other three boundaries
    let ma = _mm512_xor_si512(t2, t2m); // through the third comma
    let bad1 = misquoted!(_mm512_and_si512(c0, ma), _mm512_and_si512(q0, ma));
    let x1 = _mm512_and_si512(b2, b2m);
    let x1m = _mm512_add_epi64(x1, m1);
    let r2 = tzv!(x1, x1m);
    let x2 = _mm512_and_si512(x1, x1m);
    let x2m = _mm512_add_epi64(x2, m1);
    let r3 = tzv!(x2, x2m);
    let x3 = _mm512_and_si512(x2, x2m);
    let x3m = _mm512_add_epi64(x3, m1);
    let r4 = tzv!(x3, x3m);
    let x4 = _mm512_and_si512(x3, x3m);
    let bad = _mm512_or_si512(bad1, misquoted!(b2, q2));
    // well quoted, and four boundaries from the third comma through the newline: seven fields; no byte of the
    // first 64 that is a newline or does not pack
    let bad = _mm512_or_si512(bad, odd);
    let k_shape = _mm512_mask_testn_epi64_mask(_mm512_mask_testn_epi64_mask(_mm512_mask_test_epi64_mask(k_pos, x3, x3), x4, x4), bad, bad);
    let kl = _mm512_sub_epi64(_mm512_sub_epi64(r2, s4), _mm512_add_epi64(qk, qk));
    let s5 = _mm512_add_epi64(r2, one);
    let qp = qbit!(q2, s5);
    let pl = _mm512_sub_epi64(_mm512_sub_epi64(r3, s5), _mm512_add_epi64(qp, qp));
    let s6 = _mm512_add_epi64(r3, one);
    let qc = qbit!(q2, s6);
    sv!(l.ps, _mm512_add_epi64(_mm512_add_epi64(_mm512_add_epi64(o, s5), qp), sh2));
    sv!(l.pl, pl);
    sv!(l.cs, _mm512_add_epi64(_mm512_add_epi64(_mm512_add_epi64(o, s6), qc), sh2));
    sv!(l.cl, _mm512_sub_epi64(_mm512_sub_epi64(r4, s6), _mm512_add_epi64(qc, qc)));

    // ---- D: the copies, a word of every line to a register
    let (elo, ehi) = words2!(l.se);
    let (ulo, uhi) = words2!(l.su);
    // an ID of 1 to 7 bytes: the bytes after it (its comma, ...) cleared first (8 and 9: a shift by 64 or more is 0)
    let below = |n: __m512i| _mm512_andnot_si512(_mm512_sllv_epi64(m1, _mm512_slli_epi64::<3>(n)), m1);
    let ekey = pack8(_mm512_and_si512(elo, below(el)), ehi, _mm512_cmpeq_epi64_mask(el, k!(9)));
    let ukey = pack8(_mm512_and_si512(ulo, below(ul)), uhi, _mm512_cmpeq_epi64_mask(ul, k!(9)));
    let (w0, w1, w2, w3) = words4!(l.st);
    let (i0, i1, i2, i3) = words4!(l.si);
    let dv: *const DayV = &(*st).dv;
    let (kts, trel, kdate) = stamp(dv, cx.ws, w0, w1, w2, w3, kf1, kf2, kf3);

    // ingest_ts: F1
    let (kidate, irel) = date(dv, i0, i1);
    let (isecs, bad_i) = hms(_mm512_or_si512(_mm512_srli_epi64::<24>(i1), _mm512_slli_epi64::<40>(i2)), HMS, HMS_MAX);
    let king = _mm512_mask_cmpeq_epi64_mask(
        _mm512_mask_cmplt_epu64_mask(_mm512_mask_testn_epi64_mask(kidate, bad_i, bad_i), isecs, k!(DAY_SECS)),
        _mm512_and_si512(_mm512_srli_epi64::<24>(i2), k!(0xFF)),
        k!(b'Z'),
    );
    let irel = _mm512_mul_epi32(_mm512_add_epi64(irel, isecs), k!(1000));

    // event_type: its first 8 bytes are in the ingest_ts copy, after the 20 bytes of ingest_ts, its closing quote
    // if it has one, the comma and the type's opening quote if it has one: from byte 21, 22 or 23, in words 2 and 3
    let q8 = _mm512_slli_epi64::<3>(_mm512_add_epi64(qi, qk));
    let ty = _mm512_or_si512(_mm512_srlv_epi64(i2, _mm512_add_epi64(q8, k!(40))), _mm512_sllv_epi64(i3, _mm512_sub_epi64(k!(24), q8)));
    let ty4 = _mm512_and_si512(ty, k!(0xFFFF_FFFFu64));
    let k4 = _mm512_cmpeq_epi64_mask(kl, k!(4));
    let k_vc = _mm512_mask_cmpeq_epi64_mask(k4, ty4, k!(u32::from_le_bytes(*b"view"))) | _mm512_mask_cmpeq_epi64_mask(k4, ty4, k!(u32::from_le_bytes(*b"cart")));
    let k_buy = _mm512_mask_cmpeq_epi64_mask(_mm512_cmpeq_epi64_mask(kl, k!(8)), ty, k!(u64::from_le_bytes(*b"purchase")));

    // what becomes of each line
    let mut usual = k_shape & kts & king & (k_vc | k_buy) & live;
    // a long ID: its key from the long-ID table; one the table does not take sends the line to `event_line`
    let (mut ekey, mut ukey) = (ekey, ukey);
    let zero = _mm512_setzero_si512();
    let (kl0, kl1) = (_mm512_mask_cmpneq_epi64_mask(usual, sh1, zero), _mm512_mask_cmpneq_epi64_mask(usual, sh2, sh1));
    if (kl0 | kl1) != 0 {
        sv!(l.ek, ekey);
        sv!(l.uk, ukey);
        let mut m = kl0 as u32;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            let p = buf.add(l.ae[i] as usize);
            match if cx.evhash { ev_hash(p) } else { None } {
                Some((k, r)) => {
                    (*st).evh_push(k, r);
                    l.ek[i] = k;
                }
                None => match cx.names.long(p, 33) {
                    0 => usual &= !(1 << i),
                    k => l.ek[i] = k,
                },
            }
        }
        let mut m = kl1 as u32;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            match cx.names.long(buf.add(l.au[i] as usize), 33) {
                0 => usual &= !(1 << i),
                k => l.uk[i] = k,
            }
        }
        (ekey, ukey) = (ld!(l.ek), ld!(l.uk));
    }
    // rules.md "Events" 3 and 4: product_id not empty, ts in the window, ingested no more than 24 hours after it
    let ok = _mm512_mask_cmple_epi64_mask(
        _mm512_mask_cmplt_epu64_mask(_mm512_mask_cmpgt_epi64_mask(usual, pl, _mm512_setzero_si512()), trel, k!(cx.we - cx.ws)),
        _mm512_sub_epi64(irel, trel),
        k!(DAY),
    );
    let (keep, buy, gone, other) = (ok & k_vc, ok & k_buy, usual & !ok, live & !usual);
    #[cfg(test)]
    USUAL.fetch_add(usual.count_ones() as usize, Relaxed);

    let ingest = _mm512_add_epi64(irel, k!(cx.ws));
    let tx = _mm512_slli_epi64::<32>(_mm512_add_epi64(trel, k!(cx.ws - cx.base)));
    let home = homes(cx, ukey);
    sv!(l.ek, ekey);
    sv!(l.uk, ukey);
    sv!(l.ing, ingest);
    sv!(l.tx, tx);
    sv!(l.home, home);
    // the ID table's lines of all eight users, on their way (a lane that is not kept asks for a line in vain)
    let slots = cx.ids.slots.as_ptr() as *const u8;
    for i in 0..8 {
        prefetch(slots.add(std::ptr::read_volatile(l.home.as_ptr().add(i)) as usize));
    }
    // the kept views and carts: into the batch, next to each other
    let at = (*bt).n;
    _mm512_storeu_si512((*bt).a.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(keep, ekey));
    _mm512_storeu_si512((*bt).b.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(keep, ingest));
    _mm512_storeu_si512((*bt).ukey.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(keep, ukey));
    _mm512_storeu_si512((*bt).tx.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(keep, tx));
    _mm256_storeu_si256((*bt).home.as_mut_ptr().add(at) as *mut _, _mm256_maskz_compress_epi32(keep, _mm512_cvtepi64_epi32(_mm512_srli_epi64::<4>(home))));
    (*bt).n = at + keep.count_ones() as usize;
    if (*bt).n >= BATCH {
        (*st).drain::<true>(cx);
    }
    if (buy | gone | other) == 0 {
        return true;
    }
    // the purchases: price and category one at a time
    let mut m = buy as u32;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        // where the line starts: its walk went 65 + e2 + sh2 past it
        let line = buf.add((l.cur[i] - 65 - l.e2[i] - l.sh2[i]) as usize);
        match parse_price(std::slice::from_raw_parts(line.add(l.cs[i] as usize), l.cl[i] as usize)) {
            None => (*st).log(cx, l.ek[i], l.ing[i] as i64, 0),
            Some(price) => {
                let cat = cx.products.get_at(line.add(l.ps[i] as usize), l.pl[i] as usize).unwrap_or(cx.unknown_cat);
                let at = (*bt).n;
                *(*bt).a.get_unchecked_mut(at) = l.ek[i];
                *(*bt).b.get_unchecked_mut(at) = l.ing[i] as i64;
                *(*bt).ukey.get_unchecked_mut(at) = l.uk[i];
                *(*bt).tx.get_unchecked_mut(at) = l.tx[i] | ((cat as u64) << 24 | price as u64);
                *(*bt).home.get_unchecked_mut(at) = (l.home[i] >> 4) as u32;
                (*bt).n = at + 1;
                if at + 1 == BATCH {
                    (*st).drain::<true>(cx);
                }
            }
        }
    }
    // the events that are not kept: their dedup records alone
    let mut m = gone as u32;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        (*st).log(cx, l.ek[i], l.ing[i] as i64, 0);
    }
    // every other line as a whole; a date it brings is remembered for the lines to come
    let mut m = other as u32;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        if (k_shape >> i) & 1 != 0 {
            if ((kf1 | kf2) & !kdate) >> i & 1 != 0 {
                day_fill(cx.ws, buf.add(l.at[i] as usize), &mut (*st).dv);
            }
            if (!kidate >> i) & 1 != 0 {
                day_fill(cx.ws, buf.add(l.ai[i] as usize), &mut (*st).dv);
            }
        }
        l.cur[i] = defer::<true>(cx, buf, (l.cur[i] - 65 - l.e2[i] - l.sh2[i]) as usize, len, l.pc0[i], l.pc1[i], &mut *st) as u64;
        let lu = lane_a!(l, buf, i, pf, k);
        user_slots!(cx, l, buf, lu);
    }
    true
}

/// The lines `defer` left waiting, through `event_line` (`EVENTS`) or `click_line`.
#[inline(never)]
unsafe fn flush<const EVENTS: bool>(cx: &Ctx, buf: *const u8, len: usize, st: &mut Shard) {
    while st.pn != 0 {
        let p = st.pend[st.ph];
        (st.ph, st.pn) = ((st.ph + 1) % PEND, st.pn - 1);
        if EVENTS {
            event_line(cx, buf, p, len, st);
        } else {
            click_line(cx, buf, p, len, st);
        }
    }
}

/// Leaves the event (`EVENTS`) or click line at `buf + p` (`c0`, `c1` the commas of its bytes 0..64, 64..128; c1 may
/// be 0) to `event_line` or `click_line` a few lines later,
/// after asking for the long-ID table's slots of its event_id and user_id when they look like long IDs (33 bytes,
/// quoted or not), and runs `event_line` on the line that waited longest when `Shard::pend` is full. Returns
/// where the next line starts. (A wrong guess only asks for a slot in vain: `event_line` does the parse.)
#[inline(never)]
unsafe fn defer<const EVENTS: bool>(cx: &Ctx, buf: *const u8, p: usize, len: usize, c0: u64, c1: u64, st: &mut Shard) -> usize {
    let line = buf.add(p);
    let c = c0 as u128 | (c1 as u128) << 64;
    if c != 0 {
        let p1 = c.trailing_zeros() as usize;
        let q = (*line == b'"') as usize;
        // (a click ID is not looked up: `Shard::click_key`)
        if EVENTS && !cx.evhash && p1 == 33 + 2 * q {
            cx.names.prefetch_long(line.add(q + 1));
        }
        let c2 = c & (c - 1);
        if c2 != 0 {
            let (s, p2) = (p1 + 1, c2.trailing_zeros() as usize);
            let q = (*line.add(s) == b'"') as usize;
            if p2 == s + 33 + 2 * q {
                cx.names.prefetch_long(line.add(s + q + 1));
            }
        }
    }
    // SAFETY: buf[len] is a newline
    let e = p + memchr::memchr(b'\n', std::slice::from_raw_parts(line, len + 1 - p)).unwrap_or(len - p);
    if st.pn == PEND {
        let q = st.pend[st.ph];
        st.pend[st.ph] = p;
        st.ph = (st.ph + 1) % PEND;
        if EVENTS {
            event_line(cx, buf, q, len, st);
        } else {
            click_line(cx, buf, q, len, st);
        }
    } else {
        st.pend[(st.ph + st.pn) % PEND] = p;
        st.pn += 1;
    }
    e + 1
}

// ---------------------------------------------------------------- clicks

/// Eight click lines on their way through `click_batch`.
#[repr(C, align(64))]
struct ClickLanes {
    cur: [u64; 8],
    lim: [u64; 8],
    // A: per line, the masks cut at its newline
    c0: [u64; 8], // commas of its bytes 0..64
    q0: [u64; 8], // quotes
    c1: [u64; 8], // commas of its bytes 64..128
    q1: [u64; 8], // quotes
    o0: [u64; 8], // bytes of its first 64 below 0x20 or above 0x7F
    o1: [u64; 8], // of its bytes 64..128
    n0: [u64; 8], // newlines of its first 64 bytes
    n1: [u64; 8], // of its bytes 64..128
    // B: where its newline is, when that is in its first 128 bytes and no byte before it is below 0x20 or above
    // 0x7F; else 128
    e: [u64; 8],
    // B: where to copy from (buffer offsets)
    ac: [u64; 8],
    au: [u64; 8],
    ak: [u64; 8],
    at: [u64; 8],
    // B: the IDs' lengths
    cl: [u64; 8],
    ul: [u64; 8],
    // D
    key: [u64; 8],  // the campaign's length
    camp: [u64; 8], // its number, for the lanes whose campaign is in campaigns.csv
    home: [u64; 8],
    ck: [u64; 8], // the click ID packed (8 or 9 bytes)
    uk: [u64; 8], // the user ID packed
    tx: [u64; 8],
    // kept clicks with a user ID that is not 8 or 9 bytes (a long one: its slot asked for): their click keys,
    // user IDs (buffer offset, length) and `Batch::tx`, into the batch at the next `click_batch` (`click_late`)
    pk: [u64; LATE], // the click key (`Shard::click_key`)
    pu: [u64; LATE],
    pul: [u64; LATE],
    ptx: [u64; LATE],
    ph: usize, // the oldest
    pn: usize,
}

/// Room for clicks waiting in `ClickLanes`; `click_batch` lets no more than LATE / 2 wait (some LATE / 4 batches
/// of eight lines at one long ID in four lines), so a slot asked for has that long to arrive.
const LATE: usize = 32;

/// Parses the click lines that start in [start, end) of the `len` bytes at `buf` into `st`, as `parse_clicks_v3`
/// does; `buf` must be readable for 144 bytes past `len`, and buf[len] must be `\n` (`read_span`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2,bmi1,bmi2,popcnt")]
pub unsafe fn parse_clicks(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    if let Some((cur, lim)) = walks(buf, start, end, len) {
        let st: *mut Shard = st;
        let mut l: ClickLanes = std::mem::zeroed();
        (l.cur, l.lim) = (cur, lim);
        let (cm, qt, nv, lowest) = (_mm512_set1_epi8(b',' as i8), _mm512_set1_epi8(b'"' as i8), _mm512_set1_epi8(b'\n' as i8), _mm512_set1_epi8(0x20));
        let pf = *PF.get_or_init(|| env_num("SESS_PF").unwrap_or(256));
        // SAFETY: as in `parse_events`: a walk stands at len + 1 at most, and 144 bytes are readable from there
        macro_rules! lane {
            ($j:expr) => {{
                let line = buf.add(*l.cur.get_unchecked($j) as usize);
                prefetch(line.wrapping_add(pf));
                let v0 = _mm512_loadu_si512(line as *const _);
                let v1 = _mm512_loadu_si512(line.add(64) as *const _);
                // the masks as they are; `click_batch` finds the newline and cuts the masks there for all eight
                // (as signed bytes, those above 0x7F are below 0x20 too)
                *l.c0.get_unchecked_mut($j) = _mm512_cmpeq_epi8_mask(v0, cm);
                *l.c1.get_unchecked_mut($j) = _mm512_cmpeq_epi8_mask(v1, cm);
                *l.q0.get_unchecked_mut($j) = _mm512_cmpeq_epi8_mask(v0, qt);
                *l.q1.get_unchecked_mut($j) = _mm512_cmpeq_epi8_mask(v1, qt);
                *l.o0.get_unchecked_mut($j) = _mm512_cmplt_epi8_mask(v0, lowest);
                *l.o1.get_unchecked_mut($j) = _mm512_cmplt_epi8_mask(v1, lowest);
                *l.n0.get_unchecked_mut($j) = _mm512_cmpeq_epi8_mask(v0, nv);
                *l.n1.get_unchecked_mut($j) = _mm512_cmpeq_epi8_mask(v1, nv);
            }};
        }
        loop {
            lane!(0);
            lane!(1);
            lane!(2);
            lane!(3);
            lane!(4);
            lane!(5);
            lane!(6);
            lane!(7);
            if !click_batch(cx, buf, len, st, &mut l) {
                break;
            }
        }
        click_late(cx, buf, &mut *st, &mut l, 0);
        flush::<false>(cx, buf, len, &mut *st);
    }
    st.drain::<false>(cx);
    sfence();
}

/// The lanes whose click ID (length `cl`) is 8 or 9 bytes: packed in `click_batch`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2,bmi1,bmi2,popcnt")]
unsafe fn short_c(cl: __m512i) -> __mmask8 {
    _mm512_cmple_epu64_mask(_mm512_sub_epi64(cl, k!(8)), k!(1))
}

/// The clicks `click_batch` left waiting in `l` into the batch, the oldest first, until `keep` wait.
#[inline(never)]
unsafe fn click_late(cx: &Ctx, buf: *const u8, st: &mut Shard, l: &mut ClickLanes, keep: usize) {
    while l.pn > keep {
        let j = l.ph % LATE;
        (l.ph, l.pn) = (j + 1, l.pn - 1);
        // SAFETY: an ID of the line, which is in the buffer: 9 bytes readable from its start (PAD)
        let uk = cx.names.key_at(buf.add(l.pu[j] as usize), l.pul[j] as usize);
        let n = st.bt.n;
        // SAFETY: n < BATCH between calls: a full batch is drained at once
        *st.bt.a.get_unchecked_mut(n) = l.pk[j];
        *st.bt.ukey.get_unchecked_mut(n) = uk;
        *st.bt.tx.get_unchecked_mut(n) = l.ptx[j];
        *st.bt.home.get_unchecked_mut(n) = cx.ids.aim(uk);
        st.bt.n = n + 1;
        if n + 1 == BATCH {
            st.drain::<false>(cx);
        }
    }
}

/// The lines the walks of `l` stand on (A filled in), each walk moved past its line: the usual ones into the
/// batch when the click is kept; the others through `click_line`. False when the piece is done.
///
/// The usual line: it ends in its first 120 bytes and every byte before that is 0x20..=0x7F; four fields, each
/// quoted whole or not at all, and no other quote; IDs of 1 to 48 bytes; campaign_id at most 7 bytes; ts is F1 or
/// F2 of a date seen before, or 13 digits, and valid. For such a line this does what `click_fields` does. IDs of 8
/// or 9 bytes are packed here; a kept click with any other ID (a long one) waits in `l` for the next batch, its
/// slot of the long-ID table asked for, and gets its keys then (`Shard::click_key`, `Interner::key_at`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2,bmi1,bmi2,popcnt")]
#[allow(unused_assignments)] // the third `comma!` clears a bit no one reads
unsafe fn click_batch(cx: &Ctx, buf: *const u8, len: usize, st: *mut Shard, l: &mut ClickLanes) -> bool {
    let (m1, one, c64) = (k!(-1), k!(1), k!(64));
    if l.pn > LATE - 16 {
        click_late(cx, buf, &mut *st, l, LATE - 16);
    }
    let bt: *mut Batch = &mut (*st).bt;
    if (*bt).n > BATCH - 8 {
        (*st).drain::<false>(cx);
    }

    // ---- B: where the fields are
    let pv = ld!(l.cur);
    let live = _mm512_cmplt_epu64_mask(pv, ld!(l.lim));
    if live == 0 {
        return false;
    }
    // the first byte below 0x20 or above 0x7F must be the line's newline: both IDs are then bytes that pack. (No
    // newline and no such byte in the 128: both places are 128, and that is no place)
    macro_rules! tz128 {
        ($a:expr, $b:expr) => {{
            let (a, b) = ($a, $b);
            _mm512_mask_blend_epi64(_mm512_test_epi64_mask(a, a), _mm512_add_epi64(tzv!(b, _mm512_add_epi64(b, m1)), c64), tzv!(a, _mm512_add_epi64(a, m1)))
        }};
    }
    let w = tz128!(ld!(l.n0), ld!(l.n1));
    let e = _mm512_mask_blend_epi64(_mm512_cmpeq_epi64_mask(tz128!(ld!(l.o0), ld!(l.o1)), w), k!(128), w);
    sv!(l.e, e);
    sv!(l.cur, _mm512_mask_add_epi64(pv, live, pv, _mm512_add_epi64(e, one)));
    // the commas and quotes before the newline: the bits below e of the low word and below e - 64 of the high one
    // (a shift by 64 gives 0: all bits kept)
    let (b0, b1) = (
        _mm512_andnot_si512(_mm512_sllv_epi64(m1, _mm512_min_epu64(e, c64)), m1),
        _mm512_andnot_si512(_mm512_sllv_epi64(m1, _mm512_max_epi64(_mm512_sub_epi64(e, c64), _mm512_setzero_si512())), m1),
    );
    let (q0, q1) = (_mm512_and_si512(ld!(l.q0), b0), _mm512_and_si512(ld!(l.q1), b1));
    // the quotes counted: per byte from nibbles, then the bytes of each word summed
    let nq = {
        let lut = _mm512_set4_epi32(0x0403_0302, 0x0302_0201, 0x0302_0201, 0x0201_0100);
        let nib = _mm512_set1_epi8(0x0F);
        let cnt = |x: __m512i| _mm512_add_epi8(_mm512_shuffle_epi8(lut, _mm512_and_si512(x, nib)), _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(x), nib)));
        _mm512_sad_epu8(_mm512_add_epi8(cnt(q0), cnt(q1)), _mm512_setzero_si512())
    };
    // the first three commas of the 128-bit masks (lo, hi): from lo while it has one, then from hi
    let (mut lo, mut hi) = (_mm512_and_si512(ld!(l.c0), b0), _mm512_and_si512(ld!(l.c1), b1));
    macro_rules! comma {
        () => {{
            let klo = _mm512_test_epi64_mask(lo, lo);
            let (lm, hm) = (_mm512_add_epi64(lo, m1), _mm512_add_epi64(hi, m1));
            let p = _mm512_mask_blend_epi64(klo, _mm512_add_epi64(tzv!(hi, hm), c64), tzv!(lo, lm));
            lo = _mm512_mask_and_epi64(lo, klo, lo, lm);
            hi = _mm512_mask_and_epi64(hi, !klo, hi, hm);
            p
        }};
    }
    let p1 = comma!();
    let p2 = comma!();
    let p3 = comma!();
    // no fourth comma before the newline (with fewer than three, p3 is 128: ts then has no room)
    let k3 = _mm512_testn_epi64_mask(_mm512_or_si512(lo, hi), _mm512_or_si512(lo, hi));
    // the quote bit at a place 0..128 (0 for any other place: a shift by 64 or more gives 0)
    macro_rules! qb {
        ($at:expr) => {{
            let at = $at;
            _mm512_and_si512(_mm512_or_si512(_mm512_srlv_epi64(q0, at), _mm512_srlv_epi64(q1, _mm512_sub_epi64(at, c64))), one)
        }};
    }
    let (s1, s2, s3) = (_mm512_add_epi64(p1, one), _mm512_add_epi64(p2, one), _mm512_add_epi64(p3, one));
    let (qc, qu, qk, qt) = (_mm512_and_si512(q0, one), qb!(s1), qb!(s2), qb!(s3));
    // each field quoted at both ends or neither, and no quote but those: as many quotes as the quoted fields have
    // ends. (A field that is one quote counts it twice: too many)
    let ends = _mm512_or_si512(
        _mm512_or_si512(_mm512_xor_si512(qc, qb!(_mm512_add_epi64(p1, m1))), _mm512_xor_si512(qu, qb!(_mm512_add_epi64(p2, m1)))),
        _mm512_or_si512(_mm512_xor_si512(qk, qb!(_mm512_add_epi64(p3, m1))), _mm512_xor_si512(qt, qb!(_mm512_add_epi64(e, m1)))),
    );
    let quoted = _mm512_add_epi64(_mm512_add_epi64(qc, qu), _mm512_add_epi64(qk, qt));
    let cl = _mm512_sub_epi64(p1, _mm512_add_epi64(qc, qc));
    let ul = _mm512_sub_epi64(_mm512_sub_epi64(p2, s1), _mm512_add_epi64(qu, qu));
    let kl = _mm512_sub_epi64(_mm512_sub_epi64(p3, s2), _mm512_add_epi64(qk, qk));
    let tl = _mm512_sub_epi64(_mm512_sub_epi64(e, s3), _mm512_add_epi64(qt, qt));
    // a newline in the first 120 bytes after exactly three commas, well quoted; both IDs 1 to 48 bytes, the
    // campaign 0 to 7 (an empty one is in no file, as `get_short` finds out), ts 13, 20 or 25
    let k_w = _mm512_mask_cmplt_epu64_mask(_mm512_mask_cmpeq_epi64_mask(_mm512_testn_epi64_mask(ends, ends) & live & k3, nq, _mm512_add_epi64(quoted, quoted)), e, k!(120));
    let k_len = _mm512_mask_cmplt_epu64_mask(_mm512_mask_cmplt_epu64_mask(k_w, _mm512_add_epi64(cl, m1), k!(48)), _mm512_add_epi64(ul, m1), k!(48));
    let k_len = _mm512_mask_cmple_epu64_mask(k_len, kl, k!(7));
    let (kf1, kf2, kf3) = (_mm512_mask_cmpeq_epi64_mask(k_len, tl, k!(20)), _mm512_mask_cmpeq_epi64_mask(k_len, tl, k!(25)), _mm512_mask_cmpeq_epi64_mask(k_len, tl, k!(13)));
    let k_shape = kf1 | kf2 | kf3;
    // any other lane copies from its line's start
    let (ac, au) = (_mm512_mask_add_epi64(pv, k_shape, pv, qc), _mm512_mask_add_epi64(pv, k_shape, pv, _mm512_add_epi64(s1, qu)));
    sv!(l.ac, ac);
    sv!(l.au, au);
    sv!(l.ak, _mm512_mask_add_epi64(pv, k_shape, pv, _mm512_add_epi64(s2, qk)));
    sv!(l.at, _mm512_mask_add_epi64(pv, k_shape, pv, _mm512_add_epi64(s3, qt)));

    // ---- C, D
    // the campaign: its number comes from a table, one lane at a time, its key (`short_key`) read from the line;
    // the numbers are read again as a register at the end, when those stores are long done. `IdMap::get_short`
    sv!(l.key, kl);
    let (rkeys, rvals, rmask, rshift) = (cx.campaigns.rkeys.as_ptr(), cx.campaigns.rvals.as_ptr(), cx.campaigns.rkeys.len() - 1, cx.campaigns.rshift);
    let mut found = 0u8;
    let mut m = (k_shape & live) as u32;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        // SAFETY: a campaign starts within its line's first 120 bytes: 8 bytes from there are readable
        let want = short_key((buf.add(l.ak[i] as usize) as *const u64).read_unaligned(), l.key[i] as usize);
        let mut at = (spread(want) >> rshift) as usize;
        // SAFETY: the slot is below the table's size (the shift leaves as many bits as the mask has)
        loop {
            let have = *rkeys.add(at);
            if have == want {
                l.camp[i] = *rvals.add(at) as u64;
                found |= 1 << i;
                break;
            }
            if have == 0 {
                break;
            }
            at = (at + 1) & rmask;
        }
    }

    // IDs of 8 or 9 bytes packed here (for any other lane the key is of no use). SAFETY: a start is within the
    // line's first 120 bytes and at most 32 bytes are read from there, and a ts starts 13 bytes before the
    // newline at the latest: within the 144 readable ones
    let (c9, u9) = (_mm512_cmpeq_epi64_mask(cl, k!(9)), _mm512_cmpeq_epi64_mask(ul, k!(9)));
    let short = _mm512_mask_cmple_epu64_mask(_mm512_cmple_epu64_mask(_mm512_sub_epi64(cl, k!(8)), one), _mm512_sub_epi64(ul, k!(8)), one);
    let (clo, chi) = words2at!(buf, l.ac);
    let (ulo, uhi) = words2at!(buf, l.au);
    let ckey = pack8(clo, chi, c9);
    let ukey = pack8(ulo, uhi, u9);
    let (w0, w1, w2, w3) = words4at!(buf, l.at);
    let (kts, trel, kdate) = stamp(&(*st).dv, cx.ws, w0, w1, w2, w3, kf1, kf2, kf3);
    // ms from the start of the clicks' window, seven days before window_start
    let tw = _mm512_add_epi64(trel, k!(cx.ws - cx.base));
    let usual = k_shape & kts & live;
    // rules.md "Clicks" 3: ts in the click window, the campaign in campaigns.csv
    let keep = _mm512_mask_cmplt_epu64_mask(usual & found, tw, k!(cx.we - cx.base));
    #[cfg(test)]
    USUAL.fetch_add(usual.count_ones() as usize, Relaxed);

    let now = keep & short;
    // a kept click with a long click ID and a user ID of 8 or 9 bytes: into the batch now too (a click ID is not
    // looked up: `Shard::click_key`), after the others
    let ushort = _mm512_cmple_epu64_mask(_mm512_sub_epi64(ul, k!(8)), one);
    let cnow = keep & ushort & !short;
    let home = homes(cx, ukey);
    // the ID table's line of every lane taken now asked for (a lane not taken asks for the first line): the
    // same eight every time, no branch on which lanes
    sv!(l.home, _mm512_maskz_mov_epi64(now | cnow, home));
    let slots = cx.ids.slots.as_ptr() as *const u8;
    for i in 0..8 {
        prefetch(slots.add(std::ptr::read_volatile(l.home.as_ptr().add(i)) as usize));
    }
    let tx = _mm512_or_si512(_mm512_slli_epi64::<32>(tw), ld!(l.camp));
    let at = (*bt).n;
    _mm512_storeu_si512((*bt).a.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(now, ckey));
    _mm512_storeu_si512((*bt).ukey.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(now, ukey));
    _mm512_storeu_si512((*bt).tx.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(now, tx));
    _mm256_storeu_si256((*bt).home.as_mut_ptr().add(at) as *mut _, _mm256_maskz_compress_epi32(now, _mm512_cvtepi64_epi32(_mm512_srli_epi64::<4>(home))));
    (*bt).n = at + now.count_ones() as usize;
    if (*bt).n >= BATCH {
        (*st).drain::<false>(cx);
    }
    let late = keep & !ushort;
    if cnow | late != 0 {
        sv!(l.cl, cl);
        sv!(l.ul, ul);
        sv!(l.tx, tx);
        sv!(l.ck, ckey);
        sv!(l.uk, ukey);
        // SAFETY: a click ID starts within its line's first 120 bytes, so 48 bytes from its start are readable
        // (PAD); n < BATCH between pushes: a full batch is drained at once
        let mut m = cnow as u32;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            let ck = (*st).click_key48(cx, buf.add(l.ac[i] as usize), l.cl[i] as usize);
            let n = (*bt).n;
            *(*bt).a.get_unchecked_mut(n) = ck;
            *(*bt).ukey.get_unchecked_mut(n) = l.uk[i];
            *(*bt).tx.get_unchecked_mut(n) = l.tx[i];
            *(*bt).home.get_unchecked_mut(n) = (l.home[i] >> 4) as u32;
            (*bt).n = n + 1;
            if n + 1 == BATCH {
                (*st).drain::<false>(cx);
            }
        }
        // a kept click with a long user ID: waits for the next batch, its slot asked for meanwhile
        let cs = short_c(cl);
        let mut m = late as u32;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            let ck = if (cs >> i) & 1 != 0 { l.ck[i] } else { (*st).click_key48(cx, buf.add(l.ac[i] as usize), l.cl[i] as usize) };
            let j = (l.ph + l.pn) % LATE; // pn < LATE: at most LATE - 16 wait before a batch, and it adds 8 at most
            (l.pk[j], l.pu[j], l.pul[j], l.ptx[j]) = (ck, l.au[i], l.ul[i], l.tx[i]);
            l.pn += 1;
            let (s, n) = (l.au[i], l.ul[i]);
            if (33..=40).contains(&n) {
                // SAFETY: the 32 bytes before the ID's end are in the line
                cx.names.prefetch_long(buf.add((s + n - 32) as usize));
            }
        }
    }
    // every other line as a whole; a date it brings is remembered for the lines to come
    let mut m = (live & !usual) as u32;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        if ((k_shape & (kf1 | kf2) & !kdate) >> i) & 1 != 0 {
            day_fill(cx.ws, buf.add(l.at[i] as usize), &mut (*st).dv);
        }
        // where the line starts: its walk went e + 1 past it
        l.cur[i] = defer::<false>(cx, buf, (l.cur[i] - 1 - l.e[i]) as usize, len, l.c0[i], l.c1[i], &mut *st) as u64;
    }
    true
}
