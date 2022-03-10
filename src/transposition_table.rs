use crate::GameMove;

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

    pub fn clear(&mut self) {
        for i in 0..self.size {
            for e_i in 0..TT_BUCKET_SIZE {
                self.buckets[i].entries[e_i].ply = 0;
                self.buckets[i].entries[e_i].depth_flags &= !DEPTH_MASK;
            }
        }
    }

    pub fn occupancy(&self) -> usize {
        let mut hits = 0;
        for i in 0..1000 {
            let bucket = &self.buckets[i];
            for e_i in 0..TT_BUCKET_SIZE {
                if bucket.entries[e_i].ply > 0 {
                    hits += 1;
                }
            }
        }
        hits = hits / TT_BUCKET_SIZE;
        hits
    }

    #[inline]
    pub fn put(&mut self, hash: u64, entry: HashEntry) {
        if TT_BUCKET_SIZE == 2 {
            self.put_2way(hash, entry);
        } else {
            self.put_any(hash, entry);
        }
    }

    // put for 2 way hash table (bucket size == 2)
    fn put_2way(&mut self, hash: u64, entry: HashEntry) {
        let bucket = &mut self.buckets[hash as usize % self.size];
        let depth_entry = bucket.entries[0];
        if entry.depth() > depth_entry.depth()
            || entry.depth() + entry.ply > depth_entry.depth() + depth_entry.ply + TT_DISCARD_AGE
        {
            bucket.entries[0] = entry;
        } else {
            bucket.entries[1] = entry;
        }
    }

    // put for arbitrary bucket size
    fn put_any(&mut self, hash: u64, entry: HashEntry) {
        let slot: usize = hash as usize % self.size;
        let bucket = &mut self.buckets[slot];

        let mut worst_idx = 0;
        let mut worst_score = 10000;
        for i in 0..TT_BUCKET_SIZE {
            let cur_entry = bucket.entries[i];
            // we never want 2 entries of same position!!!
            if cur_entry.check_hash(hash) {
                if entry.depth() >= cur_entry.depth() {
                    bucket.entries[i] = entry;
                }
                return;
            }
            let score = ((cur_entry.depth() as usize) << TT_DEPTH_SHIFT) + (cur_entry.ply as usize);
            if score < worst_score {
                worst_score = score;
                worst_idx = i;
            }
        }
        bucket.entries[worst_idx] = entry;
    }

    #[inline]
    pub fn get(&self, hash: &u64) -> Option<&HashEntry> {
        let slot: usize = *hash as usize % self.size;
        for i in 0..TT_BUCKET_SIZE {
            if self.buckets[slot].entries[i].check_hash(*hash) {
                return Some(&self.buckets[slot].entries[i]);
            }
        }
        None
    }
}

#[derive(Clone, Copy)]
#[repr(align(32))]
struct HashBucket {
    entries: [HashEntry; TT_BUCKET_SIZE],
}

impl HashBucket {
    fn new() -> Self {
        Self {
            entries: [HashEntry::empty(); TT_BUCKET_SIZE],
        }
    }
}

// right now this totals to 16 bytes, extended to 24 bytes for 8 byte memory alignment.
// it would be nice to get this into 16 bytes!
// this would be not only mean less memory usage, but also be much faster because of
// cache utilization. each slot (2 entries) would be loaded with a single cache line.
#[derive(Clone, Copy)]
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
    fn empty() -> Self {
        Self::new(1, GameMove::null_move(), ScoreCutoff::Alpha(0), 0, 0)
    }

    pub fn new(
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
}
