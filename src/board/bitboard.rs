use super::{Board6, Piece, Stack};
use board_game_traits::Color;

#[derive(Default, PartialEq, Clone)]
pub struct BitboardStorage<T> {
    pub white: T,
    pub black: T,
    pub flat: T,
    pub wall: T,
    pub cap: T,
}

impl<T> BitboardStorage<T>
where
    T: Bitboard,
{
    pub fn iter_stacks(&self, color: Color) -> BitIndexIterator<T> {
        let bits = match color {
            Color::White => self.white,
            Color::Black => self.black,
        };
        BitIndexIterator { bits }
    }
    pub fn flat_score(&self, player: Color) -> u32 {
        match player {
            Color::White => (self.white & self.flat).pop_count(),
            Color::Black => (self.black & self.flat).pop_count(),
        }
    }
    pub fn board_fill(&self) -> bool {
        (self.white | self.black).all_ones()
    }
    pub fn iter_empty(&self) -> BitIndexIterator<T> {
        let bits = !(self.white | self.black);
        BitIndexIterator { bits }
    }
    pub fn check_road(&self, color: Color) -> bool {
        let road_pieces = match color {
            Color::White => (self.flat | self.cap) & self.white,
            Color::Black => (self.flat | self.cap) & self.black,
        };
        road_pieces.check_road()
    }
    pub fn build_6(board: &[Stack; 36]) -> Self {
        let mut storage = Self::default();
        for (idx, stack) in board.iter().enumerate() {
            if let Some(piece) = stack.last() {
                let bit_pattern = T::index_to_bit(idx);
                match piece {
                    Piece::WhiteFlat => {
                        storage.white |= bit_pattern;
                        storage.flat |= bit_pattern;
                    }
                    Piece::BlackFlat => {
                        storage.black |= bit_pattern;
                        storage.flat |= bit_pattern;
                    }
                    Piece::WhiteWall => {
                        storage.white |= bit_pattern;
                        storage.wall |= bit_pattern;
                    }
                    Piece::BlackWall => {
                        storage.black |= bit_pattern;
                        storage.wall |= bit_pattern;
                    }
                    Piece::WhiteCap => {
                        storage.white |= bit_pattern;
                        storage.cap |= bit_pattern;
                    }
                    Piece::BlackCap => {
                        storage.black |= bit_pattern;
                        storage.cap |= bit_pattern;
                    }
                }
            }
        }
        storage
    }
}

// 09 10 11 12 13 14
// 16 17 18 19 20 21

pub trait Bitboard:
    Copy
    + Default
    + std::ops::BitOrAssign
    + std::ops::BitOr<Output = Self>
    + std::ops::BitAnd<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::SubAssign
    + std::ops::Not<Output = Self>
{
    fn adjacent(self) -> Self;
    fn check_road(self) -> bool;
    fn pop_lowest(&mut self) -> Self;
    fn nonzero(&self) -> bool;
    fn pop_count(self) -> u32;
    fn all_ones(self) -> bool;
    fn lowest_index(self) -> usize;
    fn index_to_bit(index: usize) -> Self;
    fn size() -> usize;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bitboard6(u64);

impl Default for Bitboard6 {
    fn default() -> Self {
        Self(0)
    }
}

impl std::ops::BitAnd for Bitboard6 {
    type Output = Self;

    fn bitand(self, Self(rhs): Self) -> Self::Output {
        Bitboard6::new(self.0 & rhs)
    }
}

impl std::ops::BitOr for Bitboard6 {
    type Output = Self;

    fn bitor(self, Self(rhs): Self) -> Self::Output {
        Bitboard6::new(self.0 | rhs)
    }
}

impl std::ops::Not for Bitboard6 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Bitboard6::new(!self.0)
    }
}

impl std::ops::BitOrAssign for Bitboard6 {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl std::ops::BitAndAssign for Bitboard6 {
    fn bitand_assign(&mut self, other: Self) {
        self.0 &= other.0;
    }
}

impl std::ops::Sub for Bitboard6 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::SubAssign for Bitboard6 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Bitboard6 {
    const BIT_TO_INDEX: [usize; 64] = Self::build_bit_to_index_table();
    const INDEX_TO_BIT: [u64; 36] = Self::build_index_to_bit_table();
    pub const fn new(data: u64) -> Self {
        const INNER: u64 = 0x7e7e7e7e7e7e00; // 6x6 Board
        Self(data & INNER)
    }
    #[rustfmt::skip]
    const fn build_bit_to_index_table() -> [usize; 64] {
        const EMPTY: usize = 100;
        let arr: [usize; 64] = [
            EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,
            EMPTY, 00, 01, 02, 03, 04, 05, EMPTY,
            EMPTY, 06, 07, 08, 09, 10, 11, EMPTY,
            EMPTY, 12, 13, 14, 15, 16, 17, EMPTY,
            EMPTY, 18, 19, 20, 21, 22, 23, EMPTY,
            EMPTY, 24, 25, 26, 27, 28, 29, EMPTY,
            EMPTY, 30, 31, 32, 33, 34, 35, EMPTY,
            EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY
        ];
        arr
    }
    #[rustfmt::skip]
    const fn build_index_to_bit_table() -> [u64; 36] {
        let arr: [u64; 36] = [
            09,	10,	11,	12,	13,	14,
            17,	18,	19,	20,	21,	22,
            25,	26,	27,	28,	29,	30,
            33,	34,	35,	36,	37,	38,
            41,	42,	43,	44,	45,	46,
            49,	50,	51,	52,	53,	54,
        ];
        arr
    }
}

impl Bitboard for Bitboard6 {
    fn adjacent(self) -> Self {
        let Bitboard6(data) = self;
        let up = data >> 8;
        let down = data << 8;
        let left = data >> 1;
        let right = data << 1;
        Bitboard6::new(up | down | left | right)
    }
    fn check_road(self) -> bool {
        const LEFT_TOP: Bitboard6 = Bitboard6::new(0x2020202027e00);
        const TOP: Bitboard6 = Bitboard6::new(0x7e00);
        const BOTTOM: Bitboard6 = Bitboard6::new(0x7e000000000000);
        const LEFT: Bitboard6 = Bitboard6::new(0x2020202020200);
        const RIGHT: Bitboard6 = Bitboard6::new(0x40404040404000);
        let mut unchecked = self & LEFT_TOP; // Only need to start from two edges
        while unchecked.nonzero() {
            // Find all connected components
            let mut component = unchecked.pop_lowest();
            let mut prev = Bitboard6::new(0);
            // While component is still being updated
            while component != prev {
                prev = component;
                component |= component.adjacent() & self
            }
            // If component has squares on two opposite edges there is a road
            if (component & TOP).nonzero() && (component & BOTTOM).nonzero() {
                return true;
            }
            if (component & LEFT).nonzero() && (component & RIGHT).nonzero() {
                return true;
            }
            unchecked = unchecked & !component;
        }
        false
    }
    fn pop_lowest(&mut self) -> Self {
        let highest_index = self.0.trailing_zeros();
        if highest_index == 64 {
            Self::new(0)
        } else {
            let value = 1 << highest_index;
            self.0 ^= value;
            Self::new(value)
        }
    }
    fn nonzero(&self) -> bool {
        self.0 != 0
    }
    fn lowest_index(self) -> usize {
        let raw_index = self.0.trailing_zeros();
        Self::BIT_TO_INDEX[raw_index as usize]
    }
    fn index_to_bit(idx: usize) -> Self {
        let bit_idx = Self::INDEX_TO_BIT[idx];
        let bit_pattern = Self(1 << bit_idx);
        bit_pattern
    }
    fn size() -> usize {
        6
    }
    fn all_ones(self) -> bool {
        self.0 == u64::MAX
    }
    fn pop_count(self) -> u32 {
        (self.0).count_ones()
    }
}

pub struct BitIndexIterator<T> {
    bits: T,
}

impl<T> Iterator for BitIndexIterator<T>
where
    T: Bitboard,
{
    type Item = usize;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.bits.nonzero() {
            let lowest = self.bits.pop_lowest();
            Some(lowest.lowest_index())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    pub fn pop_lowest() {
        let mut x = Bitboard6::new(0x20103c007e00);
        let y = x.pop_lowest();
        assert_eq!(x, Bitboard6::new(0x20103c007c00));
        assert_eq!(y, Bitboard6::new(0x200));
    }
    #[test]
    pub fn bitboard_road() {
        // Positive
        assert!(Bitboard6::new(0x20103c407e00).check_road());
        assert!(Bitboard6::new(0x2020303c446e00).check_road());
        assert!(Bitboard6::new(0xffffffffffffffff).check_road());
        // Negative
        assert!(!Bitboard6::new(0x20103c406e00).check_road());
        assert!(!Bitboard6::new(0x42243c34446200).check_road());
        assert!(!Bitboard6::new(0).check_road());
    }
    #[test]
    pub fn bitboard_creation() {
        use board_game_traits::Position;
        let tps =
            "2,x4,1/2,2,x2,1,x/2,212C,x,1,1,x/2,1,x,2S,12S,x/12,12221C,x,12,1,1/1S,12,x,1,1,x 1 22";
        let board = Board6::try_from_tps(tps).unwrap();
        let bitboards = BitboardStorage::<Bitboard6>::build_6(&board.board);
        let stacks1: Vec<_> = bitboards.iter_stacks(board.side_to_move()).collect();
        let stacks2 = board.scan_active_stacks(board.side_to_move());
        assert_eq!(stacks1, stacks2);
        let stacks3: Vec<_> = bitboards.iter_stacks(!board.side_to_move()).collect();
        let stacks4 = board.scan_active_stacks(!board.side_to_move());
        assert_eq!(stacks3, stacks4);
    }
}
