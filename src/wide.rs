//! Events eight lines at a time, on a CPU with AVX-512 (F, BW, VL, DQ, CD: every scoring machine has them).
//!
//! `parse_events` in main.rs works one line at a time: some 250 instructions a line, and the core runs them about
//! as fast as it can decode them, so only fewer instructions make it faster. Here a line costs its two 64-byte
//! loads and six compares into mask registers (stored as they are); everything after that is done for eight lines
//! by one instruction: each line is a 64-bit lane of a 512-bit register.
//!
//!   A. per line: the masks of its commas and quotes, where it starts, where its newline is
//!   B. eight lines: where the fields are (count-leading-zeros gives the place of a comma) and whether the
//!      quotes are in place
//!   C. per line: the bytes of five fields copied to a fixed place (a gather would do this in one instruction,
//!      but on the scorer's CPU it costs 13 ns: the microcode fix for CVE-2022-40982)
//!   D. eight lines: both IDs packed, both timestamps converted, the type compared, the window and the 24 hours
//!      checked; the kept events go to the batch in one compressed store each
//!
//! A lane that is not the usual shape (see `batch`) goes to `event_line`, which is exact on any line. What a
//! usual lane gives is what `event_fields` gives: the tests compare the two on random lines.
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

/// True when this CPU has what `parse_events` here needs (and SESS_NARROW is not set).
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

/// Lines the eight-line path took itself (the tests check that it is most of them).
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

/// Eight event lines on their way through `batch`: every field is 64 bytes, a register's worth.
#[repr(C, align(64))]
struct Lanes {
    // A: per line
    p: [u64; 8],  // where the line starts in the buffer
    c0: [u64; 8], // commas of its bytes 0..64
    q0: [u64; 8], // quotes
    c1: [u64; 8], // commas of its bytes 64..128
    q1: [u64; 8], // quotes
    e2: [u64; 8], // its newline is byte 64 + e2
    // B: where to copy from (buffer offsets)
    ae: [u64; 8],
    au: [u64; 8],
    at: [u64; 8],
    ai: [u64; 8],
    ak: [u64; 8],
    // B: product_id and price_cents (start in the line, length), read for a purchase
    ps: [u64; 8],
    pl: [u64; 8],
    cs: [u64; 8],
    cl: [u64; 8],
    // C: the fields' bytes, line after line
    se: [u8; 128], // event_id: 16 bytes a line
    su: [u8; 128], // user_id
    st: [u8; 256], // ts: 32 bytes a line
    si: [u8; 256], // ingest_ts
    sk: [u8; 128], // event_type: 16 bytes a line (its first 8 are used; 8 a line would be eight scalar loads, which
                   // the compiler turns into a gather)
    // D: what the lines give
    ek: [u64; 8],
    uk: [u64; 8],
    ing: [u64; 8],
    tx: [u64; 8],
    home: [u64; 8],
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

/// The date of the timestamps whose bytes 0..8 are `w0` and 8..16 `w1`: the lanes whose date is in `dv`, and
/// for those the seconds from window_start to its midnight.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
unsafe fn date(dv: *const DayV, w0: __m512i, w1: __m512i) -> (__mmask8, __m512i) {
    // the day of the month from its two digits: bits 0..3 pick within a pair of registers, bit 4 picks the pair
    let d = _mm512_and_si512(w1, _mm512_set1_epi64(0x0F0F));
    let slot = _mm512_add_epi64(_mm512_add_epi64(_mm512_slli_epi64::<3>(d), _mm512_slli_epi64::<1>(d)), _mm512_srli_epi64::<8>(d));
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
    let hit = _mm512_cmpeq_epi64_mask(w0, ka) & _mm512_testn_epi64_mask(_mm512_xor_si512(y, kb), _mm512_set1_epi64(0xFFFF_FFFF));
    (hit, _mm512_srai_epi64::<32>(kb))
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

/// Parses the event lines that start in [start, end) of the `len` bytes at `buf` into `st`, as `parse_events_v3`
/// does; `buf` must be readable for 144 bytes past `len`, and buf[len] must be `\n` (`read_span`).
///
/// The piece is cut in eight at line starts and each lane walks its own part: where a line ends is known only
/// from its newline (load, compare, mask to register, count: some 16 cycles), and with one walk the eight lines
/// of a batch would wait for one another, 130 cycles in which the core has little else to do. Eight walks wait
/// side by side. The parts end within a few lines of each other; a lane whose part is done sits out.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
pub unsafe fn parse_events(cx: &Ctx, buf: *const u8, start: usize, end: usize, len: usize, st: &mut Shard) {
    let file = std::slice::from_raw_parts(buf, len);
    let first = start.saturating_sub(1);
    if let Some(nl) = memchr::memchr(b'\n', &file[first..]) {
        let p0 = first + nl + 1;
        let stop = end.min(len);
        // lane j takes the lines that start in [cut j, cut j + 1): it starts at the first line start from its cut
        // on, which is right after the first newline at or after the byte before the cut
        let (mut cur, mut lim) = ([len; 8], [stop; 8]);
        cur[0] = p0;
        for j in 1..8 {
            let cut = p0 + stop.saturating_sub(p0) * j / 8;
            lim[j - 1] = cut;
            if let Some(i) = memchr::memchr(b'\n', &file[cut - 1..]) {
                cur[j] = cut + i;
            }
        }
        let st: *mut Shard = st;
        let mut l: Lanes = std::mem::zeroed();
        let (cm, qt, nv, lowest) = (_mm512_set1_epi8(b',' as i8), _mm512_set1_epi8(b'"' as i8), _mm512_set1_epi8(b'\n' as i8), _mm512_set1_epi8(0x20));
        loop {
            let mut live = 0u8;
            for j in 0..8 {
                let mut p = cur[j];
                while p < lim[j] {
                    // SAFETY: 144 bytes are readable from every line start
                    let line = buf.add(p);
                    let v0 = _mm512_loadu_si512(line as *const _);
                    let v1 = _mm512_loadu_si512(line.add(64) as *const _);
                    let n1 = _mm512_cmpeq_epi8_mask(v1, nv);
                    // a byte of the first 64 below 0x20 or above 0x7F (as a signed byte, it is below 0x20): no
                    // newline there, and both IDs are bytes that pack
                    let odd = _mm512_cmplt_epi8_mask(v0, lowest);
                    if (n1 == 0) | (odd != 0) {
                        p = event_line(cx, buf, p, len, &mut *st);
                        continue;
                    }
                    l.p[j] = p as u64;
                    l.c0[j] = _mm512_cmpeq_epi8_mask(v0, cm);
                    l.q0[j] = _mm512_cmpeq_epi8_mask(v0, qt);
                    l.c1[j] = _mm512_cmpeq_epi8_mask(v1, cm);
                    l.q1[j] = _mm512_cmpeq_epi8_mask(v1, qt);
                    let e2 = tz(n1);
                    l.e2[j] = e2 as u64;
                    p += 65 + e2;
                    live |= 1 << j;
                    break;
                }
                cur[j] = p;
            }
            if live == 0 {
                break;
            }
            // a lane that sits out still holds its last line (or zeros): `batch` reads it and leaves it out
            batch(cx, buf, len, st, &mut l, live);
        }
    }
    st.drain::<true>(cx);
    sfence();
}

/// The lines in the lanes `live` of `l` (A filled in): the usual ones into the batch or, when the event is not
/// kept, into the dedup records alone; the others through `event_line`.
///
/// The usual line: its third comma is in its first 63 bytes and its newline within 64 bytes of the fourth
/// field's start; seven fields, well quoted; both IDs 8 or 9 bytes; ts is F1 or F2 of a date seen before, or 13
/// digits, and valid; ingest_ts is a valid F1 of a date seen before; event_type is `view`, `cart` or `purchase`.
/// For such a line this does what `event_fields` does.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2")]
unsafe fn batch(cx: &Ctx, buf: *const u8, len: usize, st: *mut Shard, l: &mut Lanes, live: u8) {
    macro_rules! ld {
        ($a:expr) => {
            _mm512_load_si512($a.as_ptr() as *const _)
        };
    }
    macro_rules! sv {
        ($a:expr, $v:expr) => {
            _mm512_store_si512($a.as_mut_ptr() as *mut _, $v)
        };
    }
    macro_rules! k {
        ($v:expr) => {
            _mm512_set1_epi64($v as i64)
        };
    }
    let (m1, one, c64) = (k!(-1), k!(1), k!(64));
    // where the lowest set bit of x is (64 for none), given x - 1
    macro_rules! tzv {
        ($x:expr, $xm:expr) => {
            _mm512_sub_epi64(c64, _mm512_lzcnt_epi64(_mm512_andnot_si512($x, $xm)))
        };
    }
    // the quote bit at place `at` of the mask `q`, as 0 or 1
    macro_rules! qbit {
        ($q:expr, $at:expr) => {
            _mm512_and_si512(_mm512_srlv_epi64($q, $at), one)
        };
    }
    // `window2`: nonzero where a quote of `q` is out of place among the boundaries `b`
    macro_rules! misquoted {
        ($b:expr, $q:expr) => {{
            let s = _mm512_or_si512(_mm512_slli_epi64::<1>($b), one);
            let z = _mm512_srli_epi64::<1>($b);
            let x = _mm512_and_si512($q, s);
            let yz = _mm512_and_si512($q, z);
            // a quote neither first nor last of a field, or both; an opening one without its closing one, or the reverse
            _mm512_or_si512(_mm512_andnot_si512(_mm512_xor_si512(s, z), $q), _mm512_and_si512(_mm512_sub_epi64(_mm512_slli_epi64::<1>(yz), x), $b))
        }};
    }

    // ---- B, first part: where the five fields to copy start
    let (c0, q0, c1, q1, e2, pv) = (ld!(l.c0), ld!(l.q0), ld!(l.c1), ld!(l.q1), ld!(l.e2), ld!(l.p));
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
    let kp = _mm512_add_epi64(_mm512_add_epi64(o, s4), qk);
    let el = _mm512_sub_epi64(p1, _mm512_add_epi64(qe, qe));
    let ul = _mm512_sub_epi64(_mm512_sub_epi64(p2, s1), _mm512_add_epi64(qu, qu));
    let tl = _mm512_sub_epi64(_mm512_sub_epi64(p3, s2), _mm512_add_epi64(qt, qt));
    let il = _mm512_sub_epi64(r1, _mm512_add_epi64(qi, qi));
    let (kf1, kf2, kf3) = (_mm512_cmpeq_epi64_mask(tl, k!(20)), _mm512_cmpeq_epi64_mask(tl, k!(25)), _mm512_cmpeq_epi64_mask(tl, k!(13)));
    // three commas in the first 63 bytes, the newline inside the second window; both IDs 8 or 9 bytes (the two
    // lengths less 8 are both 0 or 1), ingest_ts 20, ts 13, 20 or 25: every start is then within the first 88 bytes
    let k_w = _mm512_test_epi64_mask(t2, t2) & _mm512_cmplt_epu64_mask(e2, o) & _mm512_cmple_epu64_mask(p3, k!(62));
    let k_len = _mm512_cmple_epu64_mask(_mm512_or_si512(_mm512_sub_epi64(el, k!(8)), _mm512_sub_epi64(ul, k!(8))), one) & _mm512_cmpeq_epi64_mask(il, k!(20)) & (kf1 | kf2 | kf3);
    let k_pos = k_w & k_len;
    // any other lane copies from its line's start
    sv!(l.ae, _mm512_mask_add_epi64(pv, k_pos, pv, qe));
    sv!(l.au, _mm512_mask_add_epi64(pv, k_pos, pv, us));
    sv!(l.at, _mm512_mask_add_epi64(pv, k_pos, pv, tp));
    sv!(l.ai, _mm512_mask_add_epi64(pv, k_pos, pv, ip));
    sv!(l.ak, _mm512_mask_add_epi64(pv, k_pos, pv, kp));

    // ---- C: the fields' bytes to their fixed places
    // SAFETY: a start is at most 88 bytes into its line and 32 bytes are read: within the 144 readable ones
    // (the starts are read back from memory: taken out of the registers they would cost the two ports that the
    // 512-bit instructions have)
    macro_rules! at {
        ($a:expr, $i:expr) => {
            buf.add(std::ptr::read_volatile($a.as_ptr().add($i)) as usize)
        };
    }
    for i in 0..8 {
        _mm_storeu_si128(l.se.as_mut_ptr().add(16 * i) as *mut _, _mm_loadu_si128(at!(l.ae, i) as *const _));
        _mm_storeu_si128(l.su.as_mut_ptr().add(16 * i) as *mut _, _mm_loadu_si128(at!(l.au, i) as *const _));
        _mm256_storeu_si256(l.st.as_mut_ptr().add(32 * i) as *mut _, _mm256_loadu_si256(at!(l.at, i) as *const _));
        _mm256_storeu_si256(l.si.as_mut_ptr().add(32 * i) as *mut _, _mm256_loadu_si256(at!(l.ai, i) as *const _));
        _mm_storeu_si128(l.sk.as_mut_ptr().add(16 * i) as *mut _, _mm_loadu_si128(at!(l.ak, i) as *const _));
    }

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
    // well quoted, and four boundaries from the third comma through the newline: seven fields
    let k_shape = k_pos & _mm512_test_epi64_mask(x3, x3) & _mm512_testn_epi64_mask(x4, x4) & _mm512_testn_epi64_mask(bad, bad);
    let kl = _mm512_sub_epi64(_mm512_sub_epi64(r2, s4), _mm512_add_epi64(qk, qk));
    let s5 = _mm512_add_epi64(r2, one);
    let qp = qbit!(q2, s5);
    let pl = _mm512_sub_epi64(_mm512_sub_epi64(r3, s5), _mm512_add_epi64(qp, qp));
    let s6 = _mm512_add_epi64(r3, one);
    let qc = qbit!(q2, s6);
    sv!(l.ps, _mm512_add_epi64(_mm512_add_epi64(o, s5), qp));
    sv!(l.pl, pl);
    sv!(l.cs, _mm512_add_epi64(_mm512_add_epi64(o, s6), qc));
    sv!(l.cl, _mm512_sub_epi64(_mm512_sub_epi64(r4, s6), _mm512_add_epi64(qc, qc)));

    // ---- D: the copies, a word of every line to a register
    macro_rules! z {
        ($a:expr, $i:expr) => {
            _mm512_load_si512($a.as_ptr().add(64 * $i) as *const _)
        };
    }
    let (even, odd) = (_mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15));
    let (ix_a, ix_b) = (_mm512_setr_epi64(0, 4, 8, 12, 1, 5, 9, 13), _mm512_setr_epi64(2, 6, 10, 14, 3, 7, 11, 15));
    let (ix_lo, ix_hi) = (_mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11), _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15));
    let (e0, e1) = (z!(l.se, 0), z!(l.se, 1));
    let (u0, u1) = (z!(l.su, 0), z!(l.su, 1));
    let ekey = pack8(_mm512_permutex2var_epi64(e0, even, e1), _mm512_permutex2var_epi64(e0, odd, e1), _mm512_cmpeq_epi64_mask(el, k!(9)));
    let ukey = pack8(_mm512_permutex2var_epi64(u0, even, u1), _mm512_permutex2var_epi64(u0, odd, u1), _mm512_cmpeq_epi64_mask(ul, k!(9)));
    // words 0..4 of the eight ts fields, words 0..3 of the ingest_ts fields
    let (a01, a23) = (_mm512_permutex2var_epi64(z!(l.st, 0), ix_a, z!(l.st, 1)), _mm512_permutex2var_epi64(z!(l.st, 2), ix_a, z!(l.st, 3)));
    let (b01, b23) = (_mm512_permutex2var_epi64(z!(l.st, 0), ix_b, z!(l.st, 1)), _mm512_permutex2var_epi64(z!(l.st, 2), ix_b, z!(l.st, 3)));
    let (w0, w1) = (_mm512_permutex2var_epi64(a01, ix_lo, a23), _mm512_permutex2var_epi64(a01, ix_hi, a23));
    let (w2, w3) = (_mm512_permutex2var_epi64(b01, ix_lo, b23), _mm512_permutex2var_epi64(b01, ix_hi, b23));
    let (a01, a23) = (_mm512_permutex2var_epi64(z!(l.si, 0), ix_a, z!(l.si, 1)), _mm512_permutex2var_epi64(z!(l.si, 2), ix_a, z!(l.si, 3)));
    let (b01, b23) = (_mm512_permutex2var_epi64(z!(l.si, 0), ix_b, z!(l.si, 1)), _mm512_permutex2var_epi64(z!(l.si, 2), ix_b, z!(l.si, 3)));
    let (i0, i1) = (_mm512_permutex2var_epi64(a01, ix_lo, a23), _mm512_permutex2var_epi64(a01, ix_hi, a23));
    let i2 = _mm512_permutex2var_epi64(b01, ix_lo, b23);

    const HMS: u64 = u64::from_le_bytes(*b"00:00:00");
    const HMS_MAX: u64 = u64::from_le_bytes([2, 9, 0, 5, 9, 0, 5, 9]);
    const HM: u64 = u64::from_le_bytes(*b"00:00\0\0\0");
    const HM_MAX: u64 = u64::from_le_bytes([2, 9, 0, 5, 9, 0, 0, 0]);
    let dv: *const DayV = &(*st).dv;
    let day_secs = k!(86_400);
    let byte = k!(0xFF);

    // ts as F1 or F2: the date (bytes 0..11), HH:MM:SS (11..19), Z or a sign (19), HH:MM (20..25)
    let (kdate, rel) = date(dv, w0, w1);
    let (secs, bad_t) = hms(_mm512_or_si512(_mm512_srli_epi64::<24>(w1), _mm512_slli_epi64::<40>(w2)), HMS, HMS_MAX);
    let sign = _mm512_and_si512(_mm512_srli_epi64::<24>(w2), byte);
    let (kz, kplus, kminus) = (_mm512_cmpeq_epi64_mask(sign, k!(b'Z')), _mm512_cmpeq_epi64_mask(sign, k!(b'+')), _mm512_cmpeq_epi64_mask(sign, k!(b'-')));
    let (off, bad_o) = hms(_mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64::<32>(w2), _mm512_slli_epi64::<32>(w3)), k!(0xFF_FFFF_FFFFu64)), HM, HM_MAX);
    let k12 = kdate & _mm512_testn_epi64_mask(bad_t, bad_t) & _mm512_cmplt_epu64_mask(secs, day_secs);
    let koff = _mm512_testn_epi64_mask(bad_o, bad_o) & _mm512_cmplt_epu64_mask(off, day_secs) & (kplus | kminus);
    // seconds from window_start, the offset taken off (+) or put on (-)
    let tsec = _mm512_add_epi64(rel, secs);
    let tsec = _mm512_mask_sub_epi64(tsec, kf2 & kplus, tsec, off);
    let tsec = _mm512_mask_add_epi64(tsec, kf2 & kminus, tsec, off);
    let t12 = _mm512_mul_epi32(tsec, k!(1000));
    // ts as F3 of 13 digits: the first five behind three zeros, and the last eight; each four digits a half
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
    let t3 = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mul_epu32(na, k!(100_000_000)), nb), k!(cx.ws));
    // ms from window_start; the lanes whose ts is a timestamp
    let trel = _mm512_mask_blend_epi64(kf3, t12, t3);
    let kts = (kf1 & k12 & kz) | (kf2 & k12 & koff) | (kf3 & _mm512_testn_epi64_mask(bad3, bad3));

    // ingest_ts: F1
    let (kidate, irel) = date(dv, i0, i1);
    let (isecs, bad_i) = hms(_mm512_or_si512(_mm512_srli_epi64::<24>(i1), _mm512_slli_epi64::<40>(i2)), HMS, HMS_MAX);
    let king = kidate & _mm512_testn_epi64_mask(bad_i, bad_i) & _mm512_cmplt_epu64_mask(isecs, day_secs) & _mm512_cmpeq_epi64_mask(_mm512_and_si512(_mm512_srli_epi64::<24>(i2), byte), k!(b'Z'));
    let irel = _mm512_mul_epi32(_mm512_add_epi64(irel, isecs), k!(1000));

    // event_type
    let ty = _mm512_permutex2var_epi64(z!(l.sk, 0), even, z!(l.sk, 1));
    let ty4 = _mm512_and_si512(ty, k!(0xFFFF_FFFFu64));
    let k_vc = _mm512_cmpeq_epi64_mask(kl, k!(4)) & (_mm512_cmpeq_epi64_mask(ty4, k!(u32::from_le_bytes(*b"view"))) | _mm512_cmpeq_epi64_mask(ty4, k!(u32::from_le_bytes(*b"cart"))));
    let k_buy = _mm512_cmpeq_epi64_mask(kl, k!(8)) & _mm512_cmpeq_epi64_mask(ty, k!(u64::from_le_bytes(*b"purchase")));

    // what becomes of each line
    let usual = k_shape & kts & king & (k_vc | k_buy) & live;
    // rules.md "Events" 3 and 4: product_id not empty, ts in the window, ingested no more than 24 hours after it
    let ok = usual
        & _mm512_cmpgt_epi64_mask(pl, _mm512_setzero_si512())
        & _mm512_cmplt_epu64_mask(trel, k!(cx.we - cx.ws))
        & _mm512_cmple_epi64_mask(_mm512_sub_epi64(irel, trel), k!(DAY));
    let (keep, buy, gone, other) = (ok & k_vc, ok & k_buy, usual & !ok, live & !usual);
    #[cfg(test)]
    USUAL.fetch_add(usual.count_ones() as usize, Relaxed);

    let ingest = _mm512_add_epi64(irel, k!(cx.ws));
    let tx = _mm512_slli_epi64::<32>(_mm512_add_epi64(trel, k!(cx.ws - cx.base)));
    let home = _mm512_and_si512(_mm512_srl_epi64(_mm512_mullo_epi64(ukey, k!(0x9E37_79B9_7F4A_7C15u64)), _mm_cvtsi64_si128(64 - cx.ids.bits as i64)), k!(!3u64));
    sv!(l.ek, ekey);
    sv!(l.uk, ukey);
    sv!(l.ing, ingest);
    sv!(l.tx, tx);
    sv!(l.home, home);
    // the ID table's lines of all eight users, on their way (a lane that is not kept asks for a line in vain)
    let slots = cx.ids.slots.as_ptr();
    for i in 0..8 {
        prefetch(slots.add(std::ptr::read_volatile(l.home.as_ptr().add(i)) as usize));
    }
    // the kept views and carts: into the batch, next to each other
    let bt: *mut Batch = &mut (*st).bt;
    if (*bt).n > BATCH - 8 {
        (*st).drain::<true>(cx);
    }
    let at = (*bt).n;
    _mm512_storeu_si512((*bt).a.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(keep, ekey));
    _mm512_storeu_si512((*bt).b.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(keep, ingest));
    _mm512_storeu_si512((*bt).ukey.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(keep, ukey));
    _mm512_storeu_si512((*bt).tx.as_mut_ptr().add(at) as *mut _, _mm512_maskz_compress_epi64(keep, tx));
    _mm256_storeu_si256((*bt).home.as_mut_ptr().add(at) as *mut _, _mm256_maskz_compress_epi32(keep, _mm512_cvtepi64_epi32(home)));
    (*bt).n = at + keep.count_ones() as usize;
    if (*bt).n >= BATCH {
        (*st).drain::<true>(cx);
    }
    if (buy | gone | other) == 0 {
        return;
    }
    // the purchases: price and category one at a time
    let mut m = buy as u32;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        let line = buf.add(l.p[i] as usize);
        match parse_price(std::slice::from_raw_parts(line.add(l.cs[i] as usize), l.cl[i] as usize)) {
            None => (*st).log(cx, l.ek[i], l.ing[i] as i64, 0),
            Some(price) => {
                let cat = cx.products.get_at(line.add(l.ps[i] as usize), l.pl[i] as usize).unwrap_or(cx.unknown_cat);
                let at = (*bt).n;
                *(*bt).a.get_unchecked_mut(at) = l.ek[i];
                *(*bt).b.get_unchecked_mut(at) = l.ing[i] as i64;
                *(*bt).ukey.get_unchecked_mut(at) = l.uk[i];
                *(*bt).tx.get_unchecked_mut(at) = l.tx[i] | ((cat as u64) << 24 | price as u64);
                *(*bt).home.get_unchecked_mut(at) = l.home[i] as u32;
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
        event_line(cx, buf, l.p[i] as usize, len, &mut *st);
    }
}
