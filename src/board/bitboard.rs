use super::{Piece, Stack};
use crate::board::zobrist::TABLE;
use board_game_traits::Color;
use crate::TakBoard;

#[derive(Default, PartialEq, Clone)]
pub struct BitboardStorage<T> {
    pub white: T,
    pub black: T,
    pub flat: T,
    pub wall: T,
    pub cap: T,
    zobrist: u64,
}

impl<T> BitboardStorage<T>
where
    T: Bitboard,
{
    pub fn zobrist_top(&mut self, piece: Piece, sq_index: usize) {
        self.zobrist ^= TABLE.top_hash(piece, sq_index);
    }
    pub fn zobrist_middle(&mut self, piece: Piece, sq_index: usize, stack_index: usize) {
        self.zobrist ^= TABLE.stack_hash(piece, sq_index, stack_index);
    }
    pub fn zobrist_color(&mut self, color: Color) {
        self.zobrist ^= TABLE.color_hash(color); // Hash in new color
        self.zobrist ^= TABLE.color_hash(!color); // Hash out old color
    }
    pub fn set_zobrist(&mut self, hash: u64) {
        self.zobrist = hash;
    }
    pub fn zobrist(&self) -> u64 {
        self.zobrist
    }
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
    pub fn empty(&self) -> T {
        !(self.white | self.black)
    }
    pub fn road_pieces(&self, color: Color) -> T {
        let road_pieces = match color {
            Color::White => (self.flat | self.cap) & self.white,
            Color::Black => (self.flat | self.cap) & self.black,
        };
        road_pieces
    }
    pub fn blocker_pieces(&self, color: Color) -> T {
        let blocker_pieces = match color {
            Color::White => (self.wall | self.cap) & self.white,
            Color::Black => (self.wall | self.cap) & self.black,
        };
        blocker_pieces
    }
    pub fn all_pieces(&self, color: Color) -> T {
        match color {
            Color::White => self.white,
            Color::Black => self.black,
        }
    }
    pub fn check_road(&self, color: Color) -> bool {
        self.road_pieces(color).check_road()
    }
    pub fn build<B: TakBoard>(board: &[Stack]) -> Self {
        assert_eq!(board.len(), B::SIZE * B::SIZE);
        // Todo compare to manual build hash
        let mut storage = Self::default();
        storage.set_zobrist(TABLE.color_hash(Color::White));
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
    + PartialEq
    + std::ops::BitOrAssign
    + std::ops::BitOr<Output = Self>
    + std::ops::BitAnd<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::SubAssign
    + std::ops::Not<Output = Self>
    + std::ops::BitXor<Output = Self>
{
    const ZERO: Self;
    /// Returns all bits adjacent to a bitboard by sliding the entire bitboard in
    /// all 4 directions. 
    /// 
    /// Note: when only a single bit is included, the adjacent bits will not include
    /// the provided bit itself. However, because of the interactions between multiple bits, 
    /// many of the initial values may remain set depending on the input bit configuration. 
    fn adjacent(self) -> Self;

    /// Returns all bits adjacent to a bitboard by sliding the entire bitboard in 
    /// all 8 directions.
    fn loose_adjacent(self) -> Self;
    /// Flood fills and sets bits where the left + right or top + bottom edges 
    /// are one square away from meeting. 
    /// 
    /// In other words, when set with the appropriate bits, this can detect which
    /// squares would win the game for a player if filled in with a flat placement.
    /// This is used to quickly detect a subset of positions where a player is in Tak. 
    /// If complete tak threat detection is necessary, there must be a slower method 
    /// to fall back on which detects stack movement roads. 
    /// 
    /// The bits returned are not interpretable if a position is passed in where
    /// two edges of the board already meet. As such, it is not reasonable to use
    /// all white pieces as the input: walls should be excluded. However, more 
    /// creative uses of this function could be attempted to get a cheap, rough assessment
    /// of the danger of a position. 
    fn critical_squares(self) -> Self;
    /// Performs a flood fill to reveal all bits connected to a provided edge 
    fn flood(self, edge: Self) -> Self;
    /// Performs a flood fill including diagonal directions
    fn loose_flood(self, edge: Self) -> Self;
    /// Returns true if two opposite edges are connected by the provided bits
    fn check_road(self) -> bool;
    /// Returns the lowest bit index in the bitboard without modifying self
    fn lowest(&self) -> Self;
    /// Pops the lowest bit index in the bitboard.
    /// 
    /// This unsets the bit in the bitboard and returns a new bitboard where only
    /// that single bit is set. 
    fn pop_lowest(&mut self) -> Self;
    /// Returns true if any of the bits in the bitboard are set.
    fn nonzero(&self) -> bool;
    /// Returns the number of ones set in the bitboard
    fn pop_count(self) -> u32;
    /// Returns true if all the bits in the bitboard's intended range are set
    fn all_ones(self) -> bool;
    /// Finds the lowest set index in the bitboard and returns it as a [TakBoard] square index
    fn lowest_index(self) -> usize;
    /// Converts a provided [TakBoard] square index into a single set bit in a bitboard
    fn index_to_bit(index: usize) -> Self;
    fn north(self) -> Self;
    fn east(self) -> Self;
    fn south(self) -> Self;
    fn west(self) -> Self;
    fn top() -> Self;
    fn bottom() -> Self;
    fn left() -> Self;
    fn right() -> Self;
    /// Returns the board size that this bitboard corresponds to
    fn size() -> usize;
    fn simple_road_est(self) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bitboard5(u64);

impl Bitboard5 {
    const BIT_TO_INDEX: [usize; 64] = Self::build_bit_to_index_table();
    const INDEX_TO_BIT: [u64; 25] = Self::build_index_to_bit_table();
    const TOP: Bitboard5 = Bitboard5::new(0x3e0000);
    const BOTTOM: Bitboard5 = Bitboard5::new(0x3e000000000000);
    const LEFT: Bitboard5 = Bitboard5::new(0x2020202020000);
    const RIGHT: Bitboard5 = Bitboard5::new(0x20202020200000);
    const LEFT_TOP: Bitboard5 = Bitboard5::new(Self::LEFT.0 | Self::TOP.0);
    const INNER: u64 = 0x3e3e3e3e3e0000; // 5x5 Board
    const NS: [Bitboard5; 5] = [
        Bitboard5(Bitboard5::TOP.0),
        Bitboard5(Bitboard5::TOP.0 << 8),
        Bitboard5(Bitboard5::TOP.0 << 16),
        Bitboard5(Bitboard5::TOP.0 << 24),
        Bitboard5(Bitboard5::TOP.0 << 32),
    ];
    const EW: [Bitboard5; 5] = [
        Bitboard5(Bitboard5::LEFT.0),
        Bitboard5(Bitboard5::LEFT.0 << 1),
        Bitboard5(Bitboard5::LEFT.0 << 2),
        Bitboard5(Bitboard5::LEFT.0 << 3),
        Bitboard5(Bitboard5::LEFT.0 << 4),
    ];
    pub const fn new(data: u64) -> Self {
        Self(data & Self::INNER)
    }
    #[allow(clippy::zero_prefixed_literal)]
    #[rustfmt::skip]
    const fn build_bit_to_index_table() -> [usize; 64] {
        const EMPTY: usize = 100;
        let arr: [usize; 64] = [
            EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,
            EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,
            EMPTY, 00, 01, 02, 03, 04, EMPTY, EMPTY,
            EMPTY, 05, 06, 07, 08, 09, EMPTY, EMPTY,
            EMPTY, 10, 11, 12, 13, 14, EMPTY, EMPTY,
            EMPTY, 15, 16, 17, 18, 19, EMPTY, EMPTY,
            EMPTY, 20, 21, 22, 23, 24, EMPTY, EMPTY,
            EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY
        ];
        arr
    }
    #[allow(clippy::zero_prefixed_literal)]
    #[rustfmt::skip]
    const fn build_index_to_bit_table() -> [u64; 25] {
        let arr: [u64; 25] = [
            17,	18,	19,	20,	21,
            25,	26,	27,	28,	29,
            33,	34,	35,	36,	37,
            41,	42,	43,	44,	45,	
            49,	50,	51,	52,	53,
        ];
        arr
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bitboard6(u64);

impl Bitboard6 {
    const BIT_TO_INDEX: [usize; 64] = Self::build_bit_to_index_table();
    const INDEX_TO_BIT: [u64; 36] = Self::build_index_to_bit_table();
    const TOP: Bitboard6 = Bitboard6::new(0x7e00);
    const BOTTOM: Bitboard6 = Bitboard6::new(0x7e000000000000);
    const LEFT: Bitboard6 = Bitboard6::new(0x2020202020200);
    const RIGHT: Bitboard6 = Bitboard6::new(0x40404040404000);
    const LEFT_TOP: Bitboard6 = Bitboard6::new(Self::LEFT.0 | Self::TOP.0);
    const INNER: u64 = 0x7e7e7e7e7e7e00; // 6x6 Board
    const NS: [Bitboard6; 6] = [
        Bitboard6(Bitboard6::TOP.0),
        Bitboard6(Bitboard6::TOP.0 << 8),
        Bitboard6(Bitboard6::TOP.0 << 16),
        Bitboard6(Bitboard6::TOP.0 << 24),
        Bitboard6(Bitboard6::TOP.0 << 32),
        Bitboard6(Bitboard6::TOP.0 << 40)
    ];
    const EW: [Bitboard6; 6] = [
        Bitboard6(Bitboard6::LEFT.0),
        Bitboard6(Bitboard6::LEFT.0 << 1),
        Bitboard6(Bitboard6::LEFT.0 << 2),
        Bitboard6(Bitboard6::LEFT.0 << 3),
        Bitboard6(Bitboard6::LEFT.0 << 4),
        Bitboard6(Bitboard6::LEFT.0 << 5),
    ];
    pub const fn new(data: u64) -> Self {
        Self(data & Self::INNER)
    }
    #[allow(clippy::zero_prefixed_literal)]
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
    #[allow(clippy::zero_prefixed_literal)]
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bitboard7(u64);

impl Bitboard7 {
    const BIT_TO_INDEX: [usize; 64] = Self::build_bit_to_index_table();
    const INDEX_TO_BIT: [u64; 49] = Self::build_index_to_bit_table();
    const TOP: Bitboard7 = Bitboard7::new(0x7f00);
    const BOTTOM: Bitboard7 = Bitboard7::new(0x7f00000000000000);
    const LEFT: Bitboard7 = Bitboard7::new(0x101010101010100);
    const RIGHT: Bitboard7 = Bitboard7::new(0x4040404040404000);
    const LEFT_TOP: Bitboard7 = Bitboard7::new(Self::LEFT.0 | Self::TOP.0);
    const INNER: u64 = 0x7f7f7f7f7f7f7f00; // 7x7 Board
    const NS: [Bitboard7; 7] = [
        Bitboard7(Bitboard7::TOP.0),
        Bitboard7(Bitboard7::TOP.0 << 8),
        Bitboard7(Bitboard7::TOP.0 << 16),
        Bitboard7(Bitboard7::TOP.0 << 24),
        Bitboard7(Bitboard7::TOP.0 << 32),
        Bitboard7(Bitboard7::TOP.0 << 40),
        Bitboard7(Bitboard7::TOP.0 << 48),
    ];
    const EW: [Bitboard7; 7] = [
        Bitboard7(Bitboard7::LEFT.0),
        Bitboard7(Bitboard7::LEFT.0 << 1),
        Bitboard7(Bitboard7::LEFT.0 << 2),
        Bitboard7(Bitboard7::LEFT.0 << 3),
        Bitboard7(Bitboard7::LEFT.0 << 4),
        Bitboard7(Bitboard7::LEFT.0 << 5),
        Bitboard7(Bitboard7::LEFT.0 << 6),
    ];
    pub const fn new(data: u64) -> Self {
        Self(data & Self::INNER)
    }
    #[allow(clippy::zero_prefixed_literal)]
    #[rustfmt::skip]
    const fn build_bit_to_index_table() -> [usize; 64] {
        const EMPTY: usize = 100;
        let arr: [usize; 64] = [
            EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,
            00, 01, 02, 03, 04, 05, 06, EMPTY,
            07, 08, 09, 10, 11, 12, 13, EMPTY,
            14, 15, 16, 17, 18, 19, 20, EMPTY,
            21, 22, 23, 24, 25, 26, 27, EMPTY,
            28, 29, 30, 31, 32, 33, 34, EMPTY,
            35, 36, 37, 38, 39, 40, 41, EMPTY,
            42, 43, 44, 45, 46, 47, 48, EMPTY
        ];
        arr
    }
    #[allow(clippy::zero_prefixed_literal)]
    #[rustfmt::skip]
    const fn build_index_to_bit_table() -> [u64; 49] {
        let arr: [u64; 49] = [
            08, 09,	10,	11,	12,	13,	14,
            16, 17,	18,	19,	20,	21,	22,
            24, 25,	26,	27,	28,	29,	30,
            32, 33,	34,	35,	36,	37,	38,
            40, 41,	42,	43,	44,	45,	46,
            48, 49,	50,	51,	52,	53,	54,
            56, 57, 58, 59, 60, 61, 62,
        ];
        arr
    }
}

macro_rules! bitboard_impl {
    ($t: ty, $sz: expr) => {
        impl Default for $t {
            fn default() -> Self {
                Self(0)
            }
        }

        impl std::ops::BitAnd for $t {
            type Output = Self;

            fn bitand(self, Self(rhs): Self) -> Self::Output {
                Self::new(self.0 & rhs)
            }
        }

        impl std::ops::BitOr for $t {
            type Output = Self;

            fn bitor(self, Self(rhs): Self) -> Self::Output {
                Self::new(self.0 | rhs)
            }
        }

        impl std::ops::Not for $t {
            type Output = Self;

            fn not(self) -> Self::Output {
                Self::new(!self.0)
            }
        }

        impl std::ops::BitOrAssign for $t {
            fn bitor_assign(&mut self, other: Self) {
                self.0 |= other.0;
            }
        }

        impl std::ops::BitAndAssign for $t {
            fn bitand_assign(&mut self, other: Self) {
                self.0 &= other.0;
            }
        }

        impl std::ops::Sub for $t {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self::Output {
                Self(self.0 - rhs.0)
            }
        }

        impl std::ops::SubAssign for $t {
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl std::ops::BitXor for $t {
            type Output = Self;
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self(self.0 ^ rhs.0)
            }
        }

        impl Bitboard for $t {
            const ZERO: Self = Self::new(0);
            fn adjacent(self) -> Self {
                let data = self.0;
                let up = data >> 8;
                let down = data << 8;
                let left = data >> 1;
                let right = data << 1;
                Self::new(up | down | left | right)
            }
            fn loose_adjacent(self) -> Self {
                let data = self.0;
                let up = data >> 8;
                let down = data << 8;
                let left = data >> 1;
                let right = data << 1;
                let up_left = data >> 9;
                let up_right = data >> 7;
                let down_left = data << 7;
                let down_right = data << 9;
                Self::new(up | down | left | right | up_left | up_right | down_left | down_right)
            }
            fn critical_squares(self) -> Self {
                let left = self.flood(Self::LEFT);
                let right = self.flood(Self::RIGHT);
                let mut out = (left.adjacent() | Self::LEFT) & (right.adjacent() | Self::RIGHT);
                let top = self.flood(Self::TOP);
                let bottom = self.flood(Self::BOTTOM);
                out |= (top.adjacent() | Self::TOP) & (bottom.adjacent() | Self::BOTTOM);
                out
            }
            fn flood(self, edge: Self) -> Self {
                let mut prev = Self::new(0);
                let mut edge_connected = self & edge;
                while edge_connected != prev {
                    prev = edge_connected;
                    edge_connected |= edge_connected.adjacent() & self
                }
                edge_connected
            }
            fn loose_flood(self, edge: Self) -> Self {
                let mut prev = Self::new(0);
                let mut edge_connected = self & edge;
                while edge_connected != prev {
                    prev = edge_connected;
                    edge_connected |= edge_connected.loose_adjacent() & self
                }
                edge_connected
            }
            fn check_road(self) -> bool {
                let mut unchecked = self & Self::LEFT_TOP; // Only need to start from two edges
                while unchecked.nonzero() {
                    // Find all connected components
                    let mut component = unchecked.pop_lowest();
                    let mut prev = Self::new(0);
                    // While component is still being updated
                    while component != prev {
                        prev = component;
                        component |= component.adjacent() & self
                    }
                    // If component has squares on two opposite edges there is a road
                    if (component & Self::TOP).nonzero() && (component & Self::BOTTOM).nonzero() {
                        return true;
                    }
                    if (component & Self::LEFT).nonzero() && (component & Self::RIGHT).nonzero() {
                        return true;
                    }
                    unchecked = unchecked & !component;
                }
                false
            }
            fn lowest(&self) -> Self {
                let highest_index = self.0.trailing_zeros();
                if highest_index == 64 {
                    Self::new(0)
                } else {
                    let value = 1 << highest_index;
                    Self::new(value)
                }
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
                $sz
            }
            fn all_ones(self) -> bool {
                self.0 == Self::INNER
            }
            fn pop_count(self) -> u32 {
                (self.0).count_ones()
            }
            fn north(self) -> Self {
                Self::new(self.0 >> 8)
            }
            fn east(self) -> Self {
                Self::new(self.0 << 1)
            }
            fn south(self) -> Self {
                Self::new(self.0 << 8)
            }
            fn west(self) -> Self {
                Self::new(self.0 >> 1)
            }
            fn top() -> Self {
                Self::TOP
            }
            fn bottom() -> Self {
                Self::BOTTOM
            }
            fn left() -> Self {
                Self::LEFT
            }
            fn right() -> Self {
                Self::RIGHT
            }
            fn simple_road_est(self) -> i32 {
                let mut ew_count = 0;
                for b in Self::EW {
                    if (self & b).0 > 0 {
                        ew_count += 1;
                    }
                }
                let mut ns_count = 0;
                for b in Self::NS {
                    if (self & b).0 > 0 {
                        ns_count += 1;
                    }
                }
                if ew_count > ns_count {
                    ew_count
                } else {
                    ns_count
                }
            }
        }
    };
}

bitboard_impl![Bitboard5, 5];
bitboard_impl![Bitboard6, 6];
bitboard_impl![Bitboard7, 7];

pub struct BitIndexIterator<T> {
    bits: T,
}

impl<T> BitIndexIterator<T> {
    pub fn new(bits: T) -> Self { Self { bits } }
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
    use crate::board::Board6;
    use crate::board::zobrist;
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
        use crate::Position;
        use crate::board::Board6;
        let tps =
            "2,x4,1/2,2,x2,1,x/2,212C,x,1,1,x/2,1,x,2S,12S,x/12,12221C,x,12,1,1/1S,12,x,1,1,x 1 22";
        let board = Board6::try_from_tps(tps).unwrap();
        let bitboards = BitboardStorage::<Bitboard6>::build::<Board6>(&board.board);
        let stacks1: Vec<_> = bitboards.iter_stacks(board.side_to_move()).collect();
        let stacks2 = board.scan_active_stacks(board.side_to_move());
        assert_eq!(stacks1, stacks2);
        let stacks3: Vec<_> = bitboards.iter_stacks(!board.side_to_move()).collect();
        let stacks4 = board.scan_active_stacks(!board.side_to_move());
        assert_eq!(stacks3, stacks4);
    }
    #[test]
    pub fn test_critical_squares() {
        const TOP: Bitboard6 = Bitboard6::new(0x7e00);
        const BOTTOM: Bitboard6 = Bitboard6::new(0x7e000000000000);
        const LEFT: Bitboard6 = Bitboard6::new(0x2020202020200);
        const RIGHT: Bitboard6 = Bitboard6::new(0x40404040404000);
        let bb = Bitboard6::new(18144415765381120);
        let top_flood = bb.flood(TOP);
        let bottom_flood = bb.flood(BOTTOM);
        let left_flood = bb.flood(LEFT);
        let right_flood = bb.flood(RIGHT);
        assert_eq!(top_flood.pop_count(), 2);
        assert_eq!(bottom_flood.pop_count(), 5);
        assert_eq!(left_flood.pop_count(), 3);
        assert_eq!(right_flood.pop_count(), 7);
        assert_eq!(
            (top_flood.adjacent() & bottom_flood.adjacent()).pop_count(),
            1
        );
        assert_eq!(
            (left_flood.adjacent() & right_flood.adjacent()).pop_count(),
            1
        );
        let critical = bb.critical_squares();
        dbg!(critical.lowest_index());
        assert_eq!(critical.pop_count(), 2);
    }

    #[test]
    pub fn storage_zobrist() {
        let board = Board6::new();
        assert_eq!(board.zobrist(), zobrist::TABLE.manual_build_hash(&board))
    }
}
