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

#[inline]
fn num(b: &[u8]) -> Option<i64> {
    let mut v = 0i64;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (c - b'0') as i64;
    }
    Some(v)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
    }
}

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
    if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let y = num(&b[0..4])?;
    let mo = num(&b[5..7])?;
    let d = num(&b[8..10])?;
    let h = num(&b[11..13])?;
    let mi = num(&b[14..16])?;
    let s = num(&b[17..19])?;
    if !(1..=12).contains(&mo) || d < 1 || d > days_in_month(y, mo) || h > 23 || mi > 59 || s > 59 {
        return None;
    }
    Some(days_from_civil(y, mo, d) * DAY + h * 3_600_000 + mi * 60_000 + s * 1000)
}

/// F1 `...Z`, F2 `...+HH:MM`, F3 digits = epoch ms. None = bad timestamp.
#[inline]
fn parse_ts(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    if b.iter().all(|c| c.is_ascii_digit()) {
        let mut v = 0i64;
        for &c in b {
            v = v.checked_mul(10)?.checked_add((c - b'0') as i64)?;
        }
        return Some(v);
    }
    match b.len() {
        20 if b[19] == b'Z' => civil(b),
        25 if (b[19] == b'+' || b[19] == b'-') && b[22] == b':' => {
            let oh = num(&b[20..22])?;
            let om = num(&b[23..25])?;
            if oh > 23 || om > 59 {
                return None;
            }
            let off = (oh * 60 + om) * 60_000;
            let t = civil(b)?;
            Some(if b[19] == b'+' { t - off } else { t + off })
        }
        _ => None,
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

/// One parsed (well-formed, non-empty event_id) event line.
#[derive(Clone, Copy)]
struct RawEvent {
    h: u64,
    ingest: i64,
    ts: i64,
    eid: Bytes,
    uid: Bytes,
    price: u32,
    cat: u16,
    kind: u8,
    keep: bool, // valid and not late
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
    links: BytesMap<Bytes>,
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

    // 1. links
    let links = build_links(&csv_files(&data.join("links")), threads);
    lap(&format!("links: {} linked ids", links.len()));

    let ctx = Ctx { ws, we, day0, ndays, ncamp, links, products, campaigns, unknown_cat, cat_rank };

    // 2. events: parse into event_id buckets
    let ev_files = csv_files(&data.join("events"));
    let ev_chunks = chunks(&ev_files);
    let parsed = par(
        threads,
        ev_chunks.len(),
        || (0..EB).map(|_| Vec::<RawEvent>::new()).collect::<Vec<_>>(),
        |buckets, i| {
            let (f, s, e) = ev_chunks[i];
            for_lines(f, s, e, |l| {
                if let Some(r) = parse_event(&ctx, l) {
                    buckets[(r.h >> 56) as usize % EB].push(r);
                }
            });
        },
    );
    lap("events parsed");

    // 3. dedup per event bucket, route survivors to person buckets
    let routed = par(
        threads,
        EB,
        || (0..PB).map(|_| Vec::<PEvent>::new()).collect::<Vec<_>>(),
        |out, b| {
            let mut v: Vec<RawEvent> = Vec::with_capacity(parsed.iter().map(|t| t[b].len()).sum());
            for t in &parsed {
                v.extend_from_slice(&t[b]);
            }
            v.sort_unstable_by_key(|r| (r.h, r.ingest));
            let mut seen: Vec<Bytes> = Vec::new();
            let mut i = 0;
            while i < v.len() {
                let mut j = i + 1;
                while j < v.len() && v[j].h == v[i].h {
                    j += 1;
                }
                // sorted by ingest, so the first copy of each distinct event_id is the kept copy
                seen.clear();
                for r in &v[i..j] {
                    if seen.contains(&r.eid) {
                        continue;
                    }
                    seen.push(r.eid);
                    if r.keep {
                        let name = ctx.links.get(r.uid).copied().unwrap_or(r.uid);
                        let nh = hash_bytes(name);
                        out[person_bucket(nh)].push(PEvent { nh, name, ts: r.ts, price: r.price, cat: r.cat, kind: r.kind });
                    }
                }
                i = j;
            }
        },
    );
    drop(parsed);
    lap("events deduped");

    // 4. clicks
    let ck_files = csv_files(&data.join("clicks"));
    let ck_chunks = chunks(&ck_files);
    let clicks = par(
        threads,
        ck_chunks.len(),
        || (0..PB).map(|_| Vec::<PClick>::new()).collect::<Vec<_>>(),
        |out, i| {
            let (f, s, e) = ck_chunks[i];
            for_lines(f, s, e, |l| {
                if let Some(c) = parse_click(&ctx, l) {
                    out[person_bucket(c.nh)].push(c);
                }
            });
        },
    );
    lap("clicks parsed");

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

fn build_links(files: &[Bytes], threads: usize) -> BytesMap<Bytes> {
    let ch = chunks(files);
    let parts = par(
        threads,
        ch.len(),
        Vec::<(Bytes, Bytes)>::new,
        |v, i| {
            let (f, s, e) = ch[i];
            for_lines(f, s, e, |l| {
                if let Some([a, b]) = split::<2>(l) {
                    if !a.is_empty() && !b.is_empty() {
                        v.push((a, b));
                    }
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
    let mut map: BytesMap<Bytes> = BytesMap::default();
    map.reserve(n);
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        map.insert(names[i], names[best[r] as usize]);
    }
    map
}

#[inline]
fn parse_event(ctx: &Ctx, line: Bytes) -> Option<RawEvent> {
    let [eid, uid, ts, ingest, etype, pid, price] = split::<7>(line)?;
    if eid.is_empty() {
        return None;
    }
    let ingest = parse_ts(ingest)?; // always valid F1 per rules.md
    let mut r = RawEvent { h: hash_bytes(eid), ingest, ts: 0, eid, uid, price: 0, cat: 0, kind: 0, keep: false };
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
fn parse_click(ctx: &Ctx, line: Bytes) -> Option<PClick> {
    let [cid, uid, camp, ts] = split::<4>(line)?;
    if cid.is_empty() || uid.is_empty() {
        return None;
    }
    let t = parse_ts(ts)?;
    if t < ctx.ws - WEEK || t >= ctx.we {
        return None;
    }
    let camp = *ctx.campaigns.get(camp)?;
    let name = ctx.links.get(uid).copied().unwrap_or(uid);
    Some(PClick { nh: hash_bytes(name), name, ts: t, cid, camp })
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
