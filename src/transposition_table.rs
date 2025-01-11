use crate::GameMove;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// transposition table parameters
const TT_BUCKET_SIZE: usize = 5; // how many HashEntries for each HashTable slot
                                 // TT_BUCKET_SIZE *
const TT_DISCARD_AGE: u8 = 5; // adjusts which HashEntries are considered outdated
const TT_DEPTH_SHIFT: u8 = 2; // how depth is weighted compared to ply

// transposition table
// implemented as hashtable with buckets
//
// key is computed with low bytes of zobrist hash of position
// for each key, there is a bucket with TT_BUCKET_SIZE entries
//
// when a position is put in, an entry to replace is heuristically selected
// from the bucket
pub struct HashTable {
    size: usize,
    buckets: Vec<HashBucket>,
}

impl HashTable {
    pub fn new(size: usize) -> Self {
        Self {
            size: size / TT_BUCKET_SIZE,
            buckets: vec![HashBucket::new(); size / TT_BUCKET_SIZE],
        }
    }

    pub fn clear(&self) {
        for i in 0..self.size {
            for e_i in 0..TT_BUCKET_SIZE {
                self.buckets[i].entries[e_i].clear();
                // self.buckets[i].entries[e_i].ply = 0;
                // self.buckets[i].entries[e_i].depth_flags &= !DEPTH_MASK;
            }
        }
    }

    pub fn occupancy(&self) -> usize {
        let mut hits = 0;
        for i in 0..1000 {
            let bucket = &self.buckets[i];
            for e_i in 0..TT_BUCKET_SIZE {
                if bucket.entries[e_i].ply() > 0 {
                    hits += 1;
                }
            }
        }
        hits = hits / TT_BUCKET_SIZE;
        hits
    }

    #[inline]
    pub fn put(&self, hash: u64, entry: HashEntry) {
        if TT_BUCKET_SIZE == 2 {
            self.put_2way(hash, entry);
        } else {
            self.put_any(hash, entry);
        }
    }

    // put for 2 way hash table (bucket size == 2)
    fn put_2way(&self, hash: u64, entry: HashEntry) {
        let bucket = &self.buckets[hash as usize % self.size];
        let depth_entry: HashEntry = (&bucket.entries[0]).into();
        if entry.depth() > depth_entry.depth()
            || entry.depth() + entry.ply > depth_entry.depth() + depth_entry.ply + TT_DISCARD_AGE
        {
            bucket.entries[0].update(entry);
        } else {
            bucket.entries[1].update(entry);
        }
    }

    // put for arbitrary bucket size
    fn put_any(&self, hash: u64, entry: HashEntry) {
        let slot: usize = hash as usize % self.size;
        let bucket = &self.buckets[slot];

        let mut worst_idx = 0;
        let mut worst_score = 10000;
        for i in 0..TT_BUCKET_SIZE {
            let cur_entry: HashEntry = (&bucket.entries[i]).into();
            // we never want 2 entries of same position!!!
            if cur_entry.check_hash(hash) {
                if entry.depth() >= cur_entry.depth() {
                    bucket.entries[i].update(entry);
                }
                return;
            }
            let score = ((cur_entry.depth() as usize) << TT_DEPTH_SHIFT) + (cur_entry.ply as usize);
            if score < worst_score {
                worst_score = score;
                worst_idx = i;
            }
        }
        bucket.entries[worst_idx].update(entry);
    }

    #[inline]
    pub fn get(&self, hash: &u64) -> Option<HashEntry> {
        let slot: usize = *hash as usize % self.size;
        for i in 0..TT_BUCKET_SIZE {
            let entry: HashEntry = (&self.buckets[slot].entries[i]).into();
            if entry.check_hash(*hash) {
                return Some(entry);
            }
        }
        None
    }
}

#[derive(Clone)]
#[repr(align(32))]
struct HashBucket {
    entries: [ConcurrentEntry; TT_BUCKET_SIZE],
}

impl HashBucket {
    const fn new() -> Self {
        let entries = [
            ConcurrentEntry::empty(),
            ConcurrentEntry::empty(),
            ConcurrentEntry::empty(),
            ConcurrentEntry::empty(),
            ConcurrentEntry::empty(),
        ];
        Self { entries }
    }
}

/// Because the implementation is spread across 2 values, it is not truly atomic.
/// This implementation should prevent major concurrency bugs, however
struct ConcurrentEntry {
    hash_and_move: AtomicU64,
    score_flags: AtomicU32,
}

impl ConcurrentEntry {
    // fn new(hash_and_move: AtomicU64, score_flags: AtomicU32) -> Self {
    //     Self {
    //         hash_and_move,
    //         score_flags,
    //     }
    // }
    const fn empty() -> Self {
        Self {
            hash_and_move: AtomicU64::new(0),
            score_flags: AtomicU32::new((ALPHA_FLAG as u32) << 8),
        }
    }
    fn update(&self, e: HashEntry) {
        let mut first = (e.hash_hi as u64) << 32;
        first |= e.game_move.raw() as u64;
        self.hash_and_move.store(first, Ordering::Relaxed);
        let mut second = (e.score_val as u32) << 16;
        second |= (e.depth_flags as u32) << 8;
        second |= e.ply as u32;
        self.score_flags.store(second, Ordering::Relaxed);
    }
    fn clear(&self) {
        const MASK: u32 = 0xFFFF_0000 | (!(DEPTH_MASK as u32) << 8);
        self.score_flags.fetch_and(MASK, Ordering::Relaxed);
    }
    fn ply(&self) -> u32 {
        self.score_flags.load(Ordering::Relaxed) & 0xFF
    }
}

impl Clone for ConcurrentEntry {
    fn clone(&self) -> Self {
        Self {
            hash_and_move: AtomicU64::new(self.hash_and_move.load(Ordering::Relaxed)),
            score_flags: AtomicU32::new(self.score_flags.load(Ordering::Relaxed)),
        }
    }
}

impl From<HashEntry> for ConcurrentEntry {
    fn from(e: HashEntry) -> Self {
        let mut first = (e.hash_hi as u64) << 32;
        first |= e.game_move.raw() as u64;
        let hash_and_move = AtomicU64::new(first);
        let mut second = (e.score_val as u32) << 16;
        second |= (e.depth_flags as u32) << 8;
        second |= e.ply as u32;
        let score_flags = AtomicU32::new(second);
        Self {
            hash_and_move,
            score_flags,
        }
    }
}

impl From<&ConcurrentEntry> for HashEntry {
    fn from(e: &ConcurrentEntry) -> Self {
        let val = e.hash_and_move.load(Ordering::Relaxed);
        let hash_hi = (val >> 32) as u32;
        let game_move = GameMove::from_raw((val & 0xFFFF_FFFF) as u32);
        let val2 = e.score_flags.load(Ordering::Relaxed);
        let score_val = val2 as i32 >> 16;
        let depth_flags = ((val2 & 0xFF00) >> 8) as u8;
        let ply = (val2 & 0xFF) as u8;
        HashEntry {
            hash_hi,
            game_move,
            score_val,
            depth_flags,
            ply,
        }
    }
}

// Right now this totals to 16 bytes.
// This means not only less memory usage, but is also much faster because of
// cache utilization. each slot (2 entries) is loaded with a single cache line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HashEntry {
    hash_hi: u32,            // 4 bytes (~3 low bytes are encoded in the HashTable idx)
    pub game_move: GameMove, // 4 bytes
    score_val: i32,          // 4 bytes
    depth_flags: u8,         // 1 byte
    ply: u8,                 // 1 byte
}

const ALPHA_FLAG: u8 = 0b10000000;
const BETA_FLAG: u8 = 0b01000000;
const DEPTH_MASK: u8 = 0b00111111;

impl HashEntry {
    // const fn empty() -> Self {
    //     Self::new(1, GameMove::null_move(), ScoreCutoff::Alpha(0), 0, 0)
    // }

    pub const fn new(
        hash: u64,
        game_move: GameMove,
        score: ScoreCutoff,
        depth: usize,
        ply: usize,
    ) -> Self {
        let depth_flags: u8;
        let score_val: i32;
        match score {
            ScoreCutoff::Alpha(s) => {
                score_val = s;
                depth_flags = ALPHA_FLAG | (depth as u8);
            }
            ScoreCutoff::Beta(s) => {
                score_val = s;
                depth_flags = BETA_FLAG | (depth as u8);
            }
            ScoreCutoff::Exact(s) => {
                score_val = s;
                depth_flags = depth as u8;
            }
        }
        Self {
            hash_hi: (hash >> 32) as u32,
            game_move,
            score_val,
            depth_flags,
            ply: ply as u8,
        }
    }

    pub fn depth(&self) -> u8 {
        self.depth_flags & DEPTH_MASK
    }

    pub fn score(&self) -> ScoreCutoff {
        if (self.depth_flags & BETA_FLAG) != 0 {
            return ScoreCutoff::Beta(self.score_val);
        }
        if (self.depth_flags & ALPHA_FLAG) != 0 {
            return ScoreCutoff::Alpha(self.score_val);
        }
        return ScoreCutoff::Exact(self.score_val);
    }

    #[inline]
    pub fn check_hash(&self, hash: u64) -> bool {
        self.hash_hi == (hash >> 32) as u32
    }
}
#[derive(Clone, Copy)]
pub enum ScoreCutoff {
    Alpha(i32),
    Beta(i32),
    Exact(i32),
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn check_size() {
        assert_eq!(std::mem::size_of::<HashEntry>(), 16);
    }
    #[test]
    fn concurrent() {
        let mut m = HashEntry::new(
            0xDEADBEEF_6969,
            GameMove::from_placement(crate::Piece::WhiteCap, 15),
            ScoreCutoff::Beta(-10000),
            6,
            16,
        );
        let x: ConcurrentEntry = m.into();
        let m2: HashEntry = (&x).into();
        assert_eq!(m, m2);
        x.clear();
        m.ply = 0;
        m.depth_flags &= !DEPTH_MASK;
        let m2: HashEntry = (&x).into();
        assert_eq!(m, m2);
        assert_eq!(HashEntry::empty(), (&ConcurrentEntry::empty()).into());
    }
}
