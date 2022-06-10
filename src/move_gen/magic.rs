use crate::{board::Bitboard6, Bitboard, BitboardStorage, TakBoard};

use super::{find_dir_limit, MoveLimits};

// const FILE_TABLE: [[u8; 4]; 6] = [[0u8; 4]; 6];

// Bits which are orthogonal to the indexing square for a 6x6 Bitboard
const ORTH_MASK6: [u64; 36] = [
    0x2020202027e00,
    0x4040404047e00,
    0x8080808087e00,
    0x10101010107e00,
    0x20202020207e00,
    0x40404040407e00,
    0x20202027e0200,
    0x40404047e0400,
    0x80808087e0800,
    0x101010107e1000,
    0x202020207e2000,
    0x404040407e4000,
    0x202027e020200,
    0x404047e040400,
    0x808087e080800,
    0x1010107e101000,
    0x2020207e202000,
    0x4040407e404000,
    0x2027e02020200,
    0x4047e04040400,
    0x8087e08080800,
    0x10107e10101000,
    0x20207e20202000,
    0x40407e40404000,
    0x27e0202020200,
    0x47e0404040400,
    0x87e0808080800,
    0x107e1010101000,
    0x207e2020202000,
    0x407e4040404000,
    0x7e020202020200,
    0x7e040404040400,
    0x7e080808080800,
    0x7e101010101000,
    0x7e202020202000,
    0x7e404040404000,
];

// Masks for magic numbers
const MOVE_MAGIC6: [u64; 36] = [
    2207646891008,
    4415293765632,
    8830587515904,
    17661175016448,
    35322350017536,
    70644700036096,
    2207650676736,
    4415297159168,
    8830590386176,
    17661176840192,
    35322349748224,
    70644699758592,
    2208619954176,
    4416166166528,
    8831325700096,
    17661644767232,
    35322282901504,
    70644632911872,
    2456754978816,
    4638632050688,
    9019566063616,
    17781434089472,
    35305170141184,
    70627520151552,
    65979321286656,
    61589898395648,
    57209099124736,
    48447500582912,
    30924303499264,
    66246653509632,
    16890706249515008,
    15767013989548032,
    14645529376456704,
    12402560150274048,
    7916621697908736,
    16959143302660096,
];

#[rustfmt::skip]
// Number of bits necessary for the magic bitboard for a 6x6 Board
const MAGIC_COUNT6: [usize; 36] = [
    8, 7, 7, 7, 7, 8,
    7, 6, 6, 6, 6, 7,
    7, 6, 6, 6, 6, 7,
    7, 6, 6, 6, 6, 7,
    7, 6, 6, 6, 6, 7,
    8, 7, 7, 7, 7, 8,
];

struct MagicMoveArray<const N: usize, const FULL: usize> {
    storage: [MoveGenValue; FULL],
    offsets: [Offset; N],
}

impl<const N: usize, const FULL: usize> MagicMoveArray<N, FULL> {
    fn lookup<T: TakBoard>(&self, board: &T, idx: usize) -> MoveLimits {
        let shift = 64 - MAGIC_COUNT6[idx];
        let real_idx = self.offsets[idx].lookup(board.bits().wall.raw_bits(), shift);
        todo!()
    }
}

#[derive(Clone)]
struct Offset {
    offset: usize,
    magic: u64,
}

impl Offset {
    fn new(offset: usize, magic: u64) -> Self {
        Self { offset, magic }
    }
    fn lookup(&self, key: u64, shift: usize) -> usize {
        (self.magic.wrapping_mul(key) >> shift) as usize + self.offset
    }
}

#[derive(Default)]
struct Jagged2DBuilder<V> {
    storage: Vec<Vec<V>>,
    offsets: Vec<Offset>,
}

impl<V> Jagged2DBuilder<V> {
    pub fn push(&mut self, magic: u64, data: Vec<V>) {
        let offset = self.full_len();
        self.storage.push(data);
        self.offsets.push(Offset::new(offset, magic));
    }
    pub fn full_len(&self) -> usize {
        self.offsets.last().map(|x| x.offset).unwrap_or(0)
            + self.storage.last().map(|x| x.len()).unwrap_or(0)
    }
}

pub fn get_move_limits<T: TakBoard>(board: &T, sq_idx: usize) -> MoveLimits {
    todo!()
}

struct MoveGenValue {
    steps: [u8; 4],
    north: u8,
    east: u8,
    south: u8,
    west: u8,
}

impl MoveGenValue {
    fn get<T: TakBoard>(&self, board: &T, idx: usize) -> MoveLimits {
        let capstone = board.board()[idx].has_cap();
        if capstone {
            let mut can_crush = [false, false, false, false];
            // If the index would be off the board, index to the square itself
            // The answer must be "no" because the piece in question is a capstone
            can_crush[0] = board.board()[self.north as usize].has_wall();
            can_crush[1] = board.board()[self.east as usize].has_wall();
            can_crush[2] = board.board()[self.south as usize].has_wall();
            can_crush[3] = board.board()[self.west as usize].has_wall();
            MoveLimits::new(self.steps, can_crush)
        } else {
            MoveLimits::new(self.steps, [false, false, false, false])
        }
    }
}

#[derive(Clone, Default)]
struct LookupTempKey {
    steps: [u8; 4],
    set: bool,
}

impl LookupTempKey {
    fn try_set(&mut self, limits: &MoveLimits) -> Option<()> {
        if self.set {
            return None;
        }
        self.steps = limits.steps.clone();
        self.set = true;
        Some(())
    }
    fn reset(&mut self) {
        self.set = false;
    }
}

#[cfg(feature = "random")]
pub fn generate_move_magic6<R: rand_core::RngCore>(rng: &mut R) {
    let mut builder = Jagged2DBuilder::default();
    for idx in 0..36 {
        let mut magic = 0;
        let bits: Vec<_> = get_iter6(idx).collect();
        let limits: Vec<_> = bits
            .iter()
            .copied()
            .map(|x| move_limits::<Bitboard6>(&BitboardStorage::from_walls(x), idx))
            .collect();
        let shift_up = MAGIC_COUNT6[idx];
        let shift_down = 64 - MAGIC_COUNT6[idx];
        let mut lookup = vec![LookupTempKey::default(); 1 << shift_up];
        'gen_magic: loop {
            for val in lookup.iter_mut() {
                val.reset();
            }
            magic = gen_magic(rng);
            for (bit, limit) in bits.iter().copied().zip(limits.iter()) {
                let key = (bit * magic) >> shift_down;
                if lookup[key as usize].try_set(&limit).is_none() {
                    // Hash collision, try again
                    break 'gen_magic;
                }
            }
        }
        builder.push(magic, lookup);
        dbg!(magic);
    }
    let offsets: Vec<_> = builder.offsets.iter().cloned().map(|x| x.offset).collect();
    // dbg!(&offsets[..]);
    // dbg!(offsets.len());
    // dbg!(builder.full_len());
}

#[cfg(feature = "random")]
fn gen_magic<R: rand_core::RngCore>(rng: &mut R) -> u64 {
    rng.next_u64() & rng.next_u64() & rng.next_u64()
}

/// Return an iterator over all possible relevant blocker configurations for a given square
#[cfg(feature = "random")]
fn get_iter6(idx: usize) -> impl Iterator<Item = u64> {
    use bitintr::Pdep;
    let orth_mask = MOVE_MAGIC6[idx];
    let bits_needed = MAGIC_COUNT6[idx];
    assert!(orth_mask.count_ones() == bits_needed as u32);
    let max = 1 << bits_needed;
    (0..max).map(move |x| x.pdep(orth_mask))
}

fn gen_mask6() -> Vec<u64> {
    // Mask out the squares where blockers are irrelevant
    let mut vec = Vec::new();
    for idx in 0..36 {
        let b = Bitboard6::index_to_bit(idx);
        let f = |x: Bitboard6| {
            if b & x == Bitboard6::ZERO {
                x
            } else {
                Bitboard6::ZERO
            }
        };
        let bad_mask = Bitboard6::index_to_bit(idx)
            | f(Bitboard6::top())
            | f(Bitboard6::bottom())
            | f(Bitboard6::left())
            | f(Bitboard6::right());
        // Todo this is wrong!
        let orth_mask = ORTH_MASK6[idx] & !bad_mask.raw_bits();
        vec.push(orth_mask);
    }
    vec
}

fn move_limits<T: Bitboard>(bits: &BitboardStorage<T>, st_index: usize) -> MoveLimits {
    let mut limits = MoveLimits::default();
    find_dir_limit(bits, st_index, 0, &mut limits, T::north);
    find_dir_limit(bits, st_index, 1, &mut limits, T::east);
    find_dir_limit(bits, st_index, 2, &mut limits, T::south);
    find_dir_limit(bits, st_index, 3, &mut limits, T::west);
    limits
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::board::Bitboard6;
    #[test]
    fn orth() {
        let vec = generate_orth::<Bitboard6>();
        assert_eq!(vec[0], Bitboard6::new(0x2020202027e00));
        assert_eq!(vec[15], Bitboard6::new(0x1010107e101000));
        assert!(ORTH_MASK6
            .iter()
            .zip(vec.into_iter())
            .all(|(x, y)| *x == y.raw_bits()));
        assert_eq!(256, get_iter6(0).count());
        assert_eq!(64, get_iter6(7).count());
        let mut iter = get_iter6(7);
        // dbg!(gen_mask6());
        while let Some(x) = iter.next() {
            println!("{}", x);
        }
        // assert!(false);
    }
    fn generate_orth<B: Bitboard>() -> Vec<B> {
        let mut table = vec![B::ZERO; B::size() * B::size()];

        for idx in 0..table.len() {
            let bit = B::index_to_bit(idx);
            table[idx] = ray_north(bit) | ray_south(bit) | ray_east(bit) | ray_west(bit);
        }
        table
    }
    fn ray_north<B: Bitboard>(mut bits: B) -> B {
        let mut last_bits = B::ZERO;
        while bits != last_bits {
            last_bits = bits;
            bits = bits | bits.north();
        }
        bits
    }
    fn ray_south<B: Bitboard>(mut bits: B) -> B {
        let mut last_bits = B::ZERO;
        while bits != last_bits {
            last_bits = bits;
            bits = bits | bits.south();
        }
        bits
    }
    fn ray_east<B: Bitboard>(mut bits: B) -> B {
        let mut last_bits = B::ZERO;
        while bits != last_bits {
            last_bits = bits;
            bits = bits | bits.east();
        }
        bits
    }
    fn ray_west<B: Bitboard>(mut bits: B) -> B {
        let mut last_bits = B::ZERO;
        while bits != last_bits {
            last_bits = bits;
            bits = bits | bits.west();
        }
        bits
    }
}
