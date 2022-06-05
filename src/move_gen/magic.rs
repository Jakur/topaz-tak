use crate::{board::Bitboard6, Bitboard};
use bitintr::Pdep;

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

/// Return an iterator over all possible relevant blocker configurations for a given square
fn get_iter6(idx: usize) -> impl Iterator<Item = u64> {
    // Mask out the squares where blockers are irrelevant
    let bad_mask = Bitboard6::index_to_bit(idx)
        | Bitboard6::top()
        | Bitboard6::bottom()
        | Bitboard6::left()
        | Bitboard6::right();
    let orth_mask = ORTH_MASK6[idx] & !bad_mask.raw_bits();
    let bits_needed = MAGIC_COUNT6[idx];
    let max = 1 << bits_needed;
    (0..max).map(move |x| x.pdep(orth_mask))
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
        // while let Some(x) = iter.next() {
        //     println!("{}", x);
        // }
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
