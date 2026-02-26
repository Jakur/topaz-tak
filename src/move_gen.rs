use super::Piece;
use crate::board::{BitIndexIterator, TakBoard};
use crate::Color;
use crate::{Bitboard, BitboardStorage};
use std::fmt;

mod move_order;
use bytemuck::{Pod, Zeroable};
pub(crate) use move_order::ScoredMove;
pub use move_order::{
    CaptureHistory, CorrHist, CounterMoves, EvalHist, HistoryMoves, KillerMoves, PlaceHistory,
    SimpleMoveList, SmartMoveBuffer,
};
// #[allow(dead_code)]
// pub mod magic;
pub mod ptn;

pub trait MoveBuffer {
    fn add_move(&mut self, mv: GameMove);
    fn add_scored(&mut self, mv: GameMove, score: i16);
}

impl MoveBuffer for Vec<GameMove> {
    fn add_move(&mut self, mv: GameMove) {
        self.push(mv);
    }

    fn add_scored(&mut self, mv: GameMove, _score: i16) {
        self.push(mv);
    }
}

impl MoveBuffer for Vec<ScoredMove> {
    fn add_move(&mut self, mv: GameMove) {
        self.push(ScoredMove {
            mv,
            score: 0,
            is_tak: false,
            fcd: 0,
        });
    }

    fn add_scored(&mut self, mv: GameMove, score: i16) {
        self.push(ScoredMove {
            mv,
            score,
            is_tak: false,
            fcd: 0,
        });
    }
}

/// Generates all legal moves in a position, filling the provided buffer
pub fn generate_all_moves<T: TakBoard, B: MoveBuffer>(board: &T, moves: &mut B) {
    generate_all_place_moves(board, moves);
    if board.move_num() >= 2 {
        generate_all_stack_moves(board, moves);
    }
}

/// Generates all legal placements of flats, walls, and caps for the active player.
pub fn generate_all_place_moves<T: TakBoard, B: MoveBuffer>(board: &T, moves: &mut B) {
    let side_to_move = board.side_to_move();
    let start_locs = board.empty_tiles();
    if board.move_num() == 1 {
        // Force flats. Handle swapping of the pieces in the do_move function
        let piece = match side_to_move {
            Color::White => Piece::WhiteFlat,
            Color::Black => Piece::BlackFlat,
        };
        for index in start_locs {
            moves.add_move(GameMove::from_placement(piece, index));
        }
        return;
    }
    let (flat, wall, cap) = match side_to_move {
        Color::White => (Piece::WhiteFlat, Piece::WhiteWall, Piece::WhiteCap),
        Color::Black => (Piece::BlackFlat, Piece::BlackWall, Piece::BlackCap),
    };
    if board.caps_reserve(side_to_move) > 0 {
        for index in start_locs {
            moves.add_move(GameMove::from_placement(cap, index));
        }
    }
    if board.pieces_reserve(side_to_move) > 0 {
        for index in board.empty_tiles() {
            moves.add_move(GameMove::from_placement(flat, index));
            moves.add_move(GameMove::from_placement(wall, index));
        }
    }
}

/// Generates all cap or flat placement moves. Used for Tinue checking
pub fn generate_aggressive_place_moves<T: TakBoard, B: MoveBuffer>(board: &T, moves: &mut B) {
    let side_to_move = board.side_to_move();
    let start_locs = board.empty_tiles();
    let (flat, cap) = match side_to_move {
        Color::White => (Piece::WhiteFlat, Piece::WhiteCap),
        Color::Black => (Piece::BlackFlat, Piece::BlackCap),
    };
    if board.caps_reserve(side_to_move) > 0 {
        for index in start_locs {
            moves.add_move(GameMove::from_placement(cap, index));
        }
    }
    if board.pieces_reserve(side_to_move) > 0 {
        for index in board.empty_tiles() {
            moves.add_move(GameMove::from_placement(flat, index));
        }
    }
}

/// Generates all cap or flat placement moves. Used for Tinue checking
pub fn generate_passive_place_moves<T: TakBoard, B: MoveBuffer>(board: &T, moves: &mut B) {
    let side_to_move = board.side_to_move();
    if board.pieces_reserve(side_to_move) < 1 {
        return;
    }
    let wall = match side_to_move {
        Color::White => Piece::WhiteWall,
        Color::Black => Piece::BlackWall,
    };
    for index in board.empty_tiles() {
        moves.add_move(GameMove::from_placement(wall, index));
    }
}

/// Generates all legal sliding movements for the active player's stacks.
pub fn generate_all_stack_moves<T: TakBoard, B: MoveBuffer>(board: &T, moves: &mut B) {
    let start_locs = board.active_stacks(board.side_to_move());
    generate_masked_stack_moves(board, moves, start_locs);
}

/// Generates all legal sliding movements for only the provided stacks
pub fn generate_masked_stack_moves<T: TakBoard, B: MoveBuffer>(
    board: &T,
    moves: &mut B,
    stacks: BitIndexIterator<T::Bits>,
) {
    for index in stacks {
        let stack_height = board.index(index).len();
        let start_move = GameMove(index as u32);
        let limits = find_move_limits(board, index);
        for dir in 0..4 {
            let dir_move = start_move.set_direction(dir as u32);
            let max_steps = limits.steps[dir];
            let max_pieces = std::cmp::min(T::SIZE, stack_height);
            if limits.can_crush[dir] {
                // directional_crush_moves_old(moves, dir_move, max_steps, max_pieces - 1);
                directional_crush_moves(moves, dir_move, max_steps, max_pieces);
            }
            if max_steps == 0 {
                continue;
            }
            directional_stack_moves(moves, dir_move, max_steps, max_pieces);
        }
        // moves.add_limit(limits);
    }
}

/// Generates all legal sliding movements for only the provided stacks
#[cfg(test)]
pub fn generate_masked_stack_moves_old<T: TakBoard, B: MoveBuffer>(
    board: &T,
    moves: &mut B,
    stacks: BitIndexIterator<T::Bits>,
) {
    for index in stacks {
        let stack_height = board.index(index).len();
        let start_move = GameMove(index as u32);
        let limits = find_move_limits(board, index);
        for dir in 0..4 {
            let dir_move = start_move.set_direction(dir as u32);
            let max_steps = limits.steps[dir];
            let max_pieces = std::cmp::min(T::SIZE, stack_height);
            if limits.can_crush[dir] {
                directional_crush_moves_old(moves, dir_move, max_steps, max_pieces - 1);
                // directional_crush_moves(moves, dir_move, max_steps, max_pieces);
            }
            if max_steps == 0 {
                continue;
            }
            directional_stack_moves_old(moves, dir_move, max_steps, max_pieces);
        }
        // moves.add_limit(limits);
    }
}

/// A struct containing the necessary data to reverse a [GameMove].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RevGameMove {
    pub game_move: GameMove,
    pub dest_sq: usize,
    pub fifty_count: u8,
}

impl RevGameMove {
    pub fn new(game_move: GameMove, dest_sq: usize, fifty_count: u8) -> Self {
        Self {
            game_move,
            dest_sq,
            fifty_count,
        }
    }
}

/// A struct tightly packing all information necessary to make a Tak move.
///
/// Bit shifts and bit masks are used to set / retrieve the necessary move info.
/// There is enough space in this configuration to represent any move up to an 8x8 Tak Board.
/// The current data configuration is as follows:
/// 0011 1111 Src Tile
/// 1100 0000 Direction
/// 0000F00 Number Picked Up
/// 01FF000 Spread
/// 0800000 Wall Smash
/// F000000 Placement Piece
#[derive(Clone, Copy, Eq, PartialEq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct GameMove(u32);

impl GameMove {
    const PLACEMENT_BITS: u32 = 0xF000000;
    const LONG_SLIDE_TABLE: [u32; 256] = gen_long_slide_table();
    /// Creates a placement move from the provided piece and board index
    pub fn from_placement(piece: Piece, index: usize) -> Self {
        let bits = ((piece as u32) << 24) | index as u32;
        Self(bits)
    }
    /// Creates a null move which can be used to pass without changing the game state
    ///
    /// Technically this move does not follow the expected data layout and is thus illegal.
    /// There is no way for a null move to result from normal move creation rules,
    /// so it should only appear when explicitly made for low level data structures or advanced
    /// search heuristics, i.e. the [null move heuristic](https://www.chessprogramming.org/Null_Move_Pruning) in alpha beta search.
    /// In situations where a null move might arise, it must be checked for explicitly.
    pub const fn null_move() -> Self {
        Self(0)
    }
    /// Returns true if this is a placement move, i.e. not a stack move or null move.
    pub fn is_place_move(self) -> bool {
        (self.0 & Self::PLACEMENT_BITS) > 0
    }
    /// Returns the bits corresponding to the movement of the stack
    ///
    /// This compact representation can be thought of as a state machine. A value of
    /// 0 indicates that the number of pieces dropped should be incremented by 1. A
    /// value of 1 indicates that the number of pieces dropped should be incremented by
    /// 1 and then the "write head" should be moved to the next square in the slide.
    /// To avoid underflows, the consumer should check if the slide is complete
    /// before moving the "write head", although underflows should not impact correctness.
    pub const fn slide_bits(self) -> u32 {
        let bits = self.0 & 0x1FF000;
        bits >> 12
    }
    pub fn set_slide_bits(self, slide: u32) -> Self {
        Self(self.0 | slide << 12)
    }
    /// Counts the number of forward steps taken by a sliding move
    pub fn count_steps(self) -> u8 {
        self.slide_bits().count_ones() as u8
    }
    /// Determines the destination square of a sliding move
    pub fn dest_sq(self, board_size: usize) -> usize {
        let sq = self.src_index();
        let steps = self.count_steps() as usize;
        match self.direction() {
            0 => sq - board_size * steps,
            1 => sq + steps,
            2 => sq + board_size * steps,
            3 => sq - steps,
            _ => unimplemented!(),
        }
    }
    /// Determines the unique identifier from its slide bits and direction. Note this is only unique for this one square.
    pub fn unique_slide_idx(self) -> usize {
        let out = (self.slide_bits() << 2) + self.direction();
        out as usize
    }
    /// Returns the piece being placed.
    ///
    /// # Panics
    ///
    /// If this is not a placement move
    pub fn place_piece(self) -> Piece {
        Piece::from_index((self.0 >> 24) & 0xF)
    }
    /// Returns true if the move is a stack (sliding) move
    pub fn is_stack_move(self) -> bool {
        (self.0 & Self::PLACEMENT_BITS) == 0
    }
    /// Returns either the index where the piece is being placed, or the stack move origin square index.
    pub fn src_index(self) -> usize {
        self.0 as usize & 0x3F
    }
    /// Returns the direction of the movement. 0 &harr; N, 1 &harr; E, 2 &harr; S, 3 &harr; W
    pub fn direction(self) -> u32 {
        (self.0 & 0xC0) >> 6
    }
    /// Sets square index bits for the placement or stack move
    fn set_index(self, index: u32) -> Self {
        Self(self.0 | index)
    }
    /// North, East, South, West respectively
    fn set_direction(self, dir: u32) -> Self {
        GameMove(self.0 | (dir << 6))
    }
    /// Sets the number of pieces being picked up by a stack move
    fn set_number(self, num: u32) -> Self {
        GameMove(self.0 | (num << 8))
    }
    /// Returns the number of pieces being picked up by a stack move
    pub fn number(self) -> u32 {
        (self.0 & 0xF00) >> 8
    }
    /// Returns an iterator that returns the quantity and board index for each stack step
    pub fn quantity_iter(self, board_size: usize) -> QuantityMoveIterator {
        QuantityMoveIterator::new(self, board_size)
    }
    /// Returns a move with count pieces dropped on the next forward tile.
    ///
    /// This should only be called on incomplete moves, and thus is only useful for
    /// move generation.
    #[must_use = "Forgot to assign next move step!"]
    fn set_next_tile(self, count: u32) -> Self {
        const INIT: u32 = 31 - (0x800u32).leading_zeros();
        let sliding = self.0 & 0x1FF000;
        let start = if sliding == 0 {
            INIT
        } else {
            31 - sliding.leading_zeros()
        };
        Self(self.0 | 1 << (start + count))
    }
    /// Returns a move with a crush if the provided boolean is true, else returns self.
    fn chain_crush(self, cond: bool) -> Self {
        if cond {
            self.set_crush()
        } else {
            self
        }
    }
    /// Sets the crush bit according to the given value
    pub fn set_crush(self) -> Self {
        Self(self.0 | 0x800000)
    }
    /// Returns true if move is a stack move that crushes a wall
    ///
    /// This is necessary to properly implement [RevGameMove].
    pub fn crush(&self) -> bool {
        (self.0 & 0x800000) > 0
    }
    /// Returns a slide representation where 4 bits indicate the number of pieces dropped.
    ///
    /// There is one nibble reserved each new location that the stack could move to
    /// where the value indicates how many pieces were dropped there. This requires 7 nibbles at maximum,
    /// since even on an 8x8 board a stack cannot advance more than 7 squares. The
    /// lower value bits correspond to piece drops closer to the origin. By the rules
    /// of Tak, once a single nibble is 0, all higher bits of the slide must also be 0.
    pub fn sparse_slide_bits(&self) -> u32 {
        let slide_bits = self.slide_bits();
        let bits = Self::LONG_SLIDE_TABLE[slide_bits as usize];
        let mask = match slide_bits.count_ones() {
            1 => 0xF,
            2 => 0xFF,
            3 => 0xFFF,
            4 => 0xFFFF,
            5 => 0xF_FFFF,
            6 => 0xFF_FFFF,
            7 => 0xFFF_FFFF,
            _ => unimplemented!(),
        };
        bits & mask
    }
    /// A quick but non-exhaustive check that a move is valid
    pub fn is_valid(&self) -> bool {
        if *self == Self::null_move() {
            return false;
        } else if self.is_place_move() {
            !self.crush() && !self.is_stack_move()
        } else {
            let slide_bits_count = self.slide_bits().count_ones();
            let number = self.number();
            slide_bits_count != 0 && slide_bits_count < 8 && number != 0 && number <= 8
        }
    }
    /// Returns all symmetric versions of a move corresponding to symmetries of the board
    pub fn symmetries<T: TakBoard>(&self) -> Vec<GameMove> {
        vec![
            *self,
            self.flip_ns::<T>(),
            self.flip_ew::<T>(),
            self.rotate::<T>(),
            self.rotate::<T>().rotate::<T>(),
            self.rotate::<T>().rotate::<T>().rotate::<T>(),
            self.rotate::<T>().flip_ns::<T>(),
            self.rotate::<T>().flip_ew::<T>(),
        ]
    }
    pub fn reverse_symmetry<T: TakBoard>(self, symmetry: usize) -> GameMove {
        match symmetry {
            0 => self,
            1 => self.flip_ns::<T>(),
            2 => self.flip_ew::<T>(),
            3 => self.rotate::<T>().rotate::<T>().rotate::<T>(),
            4 => self.rotate::<T>().rotate::<T>(),
            5 => self.rotate::<T>(),
            6 => self
                .flip_ns::<T>()
                .rotate::<T>()
                .rotate::<T>()
                .rotate::<T>(),
            7 => self
                .flip_ew::<T>()
                .rotate::<T>()
                .rotate::<T>()
                .rotate::<T>(),
            _ => unimplemented!(),
        }
    }
    fn rotate<T: TakBoard>(&self) -> Self {
        assert!(self.is_valid());
        let (row, col) = T::row_col_static(self.src_index());
        let dest_idx = T::index_static(T::SIZE - 1 - col, row);
        let out = Self(self.0 & 0xFFFF_FF00).set_index(dest_idx as u32);
        if self.is_place_move() {
            out
        } else {
            let dir = match self.direction() {
                0 => 3,
                1 => 0,
                2 => 1,
                3 => 2,
                _ => unimplemented!(),
            };
            out.set_direction(dir)
        }
    }
    fn flip_ns<T: TakBoard>(&self) -> Self {
        assert!(self.is_valid());
        let (row, col) = T::row_col_static(self.src_index());
        let dest_idx = T::index_static(T::SIZE - 1 - row, col);
        let out = Self(self.0 & 0xFFFF_FF00).set_index(dest_idx as u32);
        if self.is_place_move() {
            out
        } else {
            let dir = match self.direction() {
                0 => 2,
                2 => 0,
                d => d,
            };
            out.set_direction(dir)
        }
    }
    fn flip_ew<T: TakBoard>(&self) -> Self {
        assert!(self.is_valid());
        let (row, col) = T::row_col_static(self.src_index());
        let dest_idx = T::index_static(row, T::SIZE - 1 - col);
        let out = Self(self.0 & 0xFFFF_FF00).set_index(dest_idx as u32);
        if self.is_place_move() {
            out
        } else {
            let dir = match self.direction() {
                1 => 3,
                3 => 1,
                d => d,
            };
            out.set_direction(dir)
        }
    }
    pub(crate) fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
    pub(crate) fn raw(&self) -> u32 {
        self.0
    }
}

impl std::default::Default for GameMove {
    fn default() -> Self {
        GameMove::null_move()
    }
}

impl fmt::Debug for GameMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_stack_move() {
            write!(f, "StackMovement({:#X})", self.0)
        } else {
            write!(f, "Placement({:#X})", self.0)
        }
    }
}
pub struct QuantityMoveIterator {
    slide_bits: u32,
    direction: u8,
    index: usize,
    board_size: usize,
}

impl QuantityMoveIterator {
    fn new(game_move: GameMove, board_size: usize) -> Self {
        let slide_bits = game_move.sparse_slide_bits();
        let src_index = game_move.src_index();
        let direction = game_move.direction() as u8;
        let index = src_index;
        Self {
            slide_bits,
            direction,
            index,
            board_size,
        }
    }
}

impl Iterator for QuantityMoveIterator {
    type Item = QuantityStep;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.slide_bits == 0 {
            return None;
        }
        let quantity = self.slide_bits & 0xF;
        self.slide_bits = self.slide_bits >> 4;
        match self.direction {
            0 => self.index -= self.board_size,
            1 => self.index += 1,
            2 => self.index += self.board_size,
            3 => self.index -= 1,
            _ => unimplemented!(),
        }
        let step = QuantityStep {
            index: self.index,
            quantity,
        };
        Some(step)
    }
}

pub struct QuantityStep {
    pub index: usize,
    pub quantity: u32,
}

#[derive(Debug, Default)]
pub struct MoveLimits {
    steps: [u8; 4],
    can_crush: [bool; 4],
}

impl MoveLimits {
    pub fn new(steps: [u8; 4], can_crush: [bool; 4]) -> Self {
        Self { steps, can_crush }
    }
    pub fn cap_north(&self) -> u8 {
        self.steps[0] + self.can_crush[0] as u8
    }
    pub fn cap_east(&self) -> u8 {
        self.steps[1] + self.can_crush[1] as u8
    }
    pub fn cap_south(&self) -> u8 {
        self.steps[2] + self.can_crush[2] as u8
    }
    pub fn cap_west(&self) -> u8 {
        self.steps[3] + self.can_crush[3] as u8
    }
}

/// Calculates how far a stack can move in all four directions, respecting the
/// edges of the board as well as the obstacle pieces: walls and capstones. Currently,
/// the size of the stack is not considered (it's unclear if this is faster). If a
/// wall is detected and the top stack piece is a capstone then a flag is set for
/// the given direction to show that wall crush moves can be generated on the
/// steps[dir] + 1 tile. If a fast bitwise method for wall detection is found,
/// the empty board limits can be precomputed to replace this function.
pub fn find_move_limits<T: TakBoard>(board: &T, st_index: usize) -> MoveLimits {
    let mut limits = MoveLimits::default();
    let bits = board.bits();
    find_dir_limit(bits, st_index, 0, &mut limits, T::Bits::north);
    find_dir_limit(bits, st_index, 1, &mut limits, T::Bits::east);
    find_dir_limit(bits, st_index, 2, &mut limits, T::Bits::south);
    find_dir_limit(bits, st_index, 3, &mut limits, T::Bits::west);
    limits
}

pub fn legal_stack_move<T: TakBoard>(board: &T, game_move: GameMove) -> bool {
    if !game_move.is_valid() {
        return false;
    }
    // Assume the game_move is a legal move on some board.
    // Meaning, e.g. it obeys wall crushing stack movement rules
    let src_index = game_move.src_index();
    let origin = &board.board()[src_index];
    let is_cap = match origin.top() {
        Some(piece) => {
            if piece.owner() != board.side_to_move() {
                return false;
            }
            piece.is_cap()
        }
        None => return false,
    };
    if game_move.number() as usize > origin.len() {
        return false;
    }
    let mut limits = MoveLimits::default();
    let dir = game_move.direction() as usize;
    let f = match dir {
        0 => T::Bits::north,
        1 => T::Bits::east,
        2 => T::Bits::south,
        _ => T::Bits::west,
    };
    find_dir_limit(board.bits(), src_index, dir, &mut limits, f);
    let steps_moved = game_move.count_steps();
    if game_move.crush() {
        if !is_cap || !limits.can_crush[dir] {
            return false;
        }
        if steps_moved != (limits.steps[dir] + 1) {
            return false;
        }
    } else if steps_moved > limits.steps[dir] {
        return false;
    }
    true
}

/// Finds the movement limit in a single direction, updating the index to be
/// searched according to the step_fn. See [find_move_limits].
fn find_dir_limit<B, F>(
    board: &BitboardStorage<B>,
    st_idx: usize,
    dir: usize,
    limits: &mut MoveLimits,
    step_fn: F,
) where
    B: Bitboard,
    F: Fn(B) -> B,
{
    let mut counter = 0;
    let st_bit = B::index_to_bit(st_idx);
    let cap_stack = (st_bit & board.cap) != B::ZERO;
    let mut bit = step_fn(st_bit);
    while bit != B::ZERO {
        if (bit & board.cap) != B::ZERO {
            break;
        } else if (bit & board.wall) != B::ZERO {
            if cap_stack {
                limits.can_crush[dir] = true;
            }
            break;
        }
        bit = step_fn(bit);
        counter += 1;
    }
    limits.steps[dir] = counter;
}

// struct SlidePatternMap {
//     data: [SlidePattern; 255],
//     starts: [usize; 8],
// }

// impl SlidePatternMap {
//     // fn foo(&self, pop_count: usize) -> &[SlidePattern] {
//     //     let idx = pop_count - 1;
//     //     if pop_count == 8 {
//     //         // todo what is this const number above?
//     //         &self.data[self.starts[idx]..]
//     //     } else {
//     //         let next_idx = idx + 1;
//     //         &self.data[self.starts[idx]..self.starts[next_idx]]
//     //     }
//     // }
//     fn iterate_slides<'a>(&'a self, pop_count: usize) -> impl Iterator<Item = &'a SlidePattern> {
//         let idx = pop_count - 1;
//         let slice = if pop_count == 8 {
//             &self.data[self.starts[idx]..]
//         } else {
//             let next_idx = idx + 1;
//             &self.data[self.starts[idx]..self.starts[next_idx]]
//         };
//         slice.iter()
//     }
// }

// #[derive(Debug, Clone, Copy)]
// struct SlidePattern {
//     slide_bits: u16,
//     pop_count: u8,
//     pickup_pieces: u8,
// }

// impl SlidePattern {
//     const fn new(slide_bits: u16, pop_count: u8, pickup_pieces: u8) -> Self {
//         Self {
//             slide_bits,
//             pop_count,
//             pickup_pieces,
//         }
//     }
// }

// const fn generate_stack_sliders() -> SlidePatternMap {
//     let mut idx = 0;
//     let mut data = [SlidePattern::new(0, 255, 0); 255];
//     let mut starts = [0; 8];
//     let mut target_count = 1;
//     // Radix "Sort"
//     while target_count <= 8 {
//         let mut x = 1u32;
//         starts[target_count as usize - 1] = idx;
//         while x < 1 << 8 {
//             let pop_count = x.count_ones();
//             if pop_count == target_count {
//                 data[idx] = SlidePattern::new(x as u16, pop_count as u8, x.ilog2() as u8 + 1);
//                 idx += 1;
//             }
//             x += 1;
//         }
//         target_count += 1;
//     }
//     SlidePatternMap { data, starts }
// }

// /// Find all stack moves for a single stack in one direction. Calls the recursive
// /// function [recursive_stack_moves] 1..=pieces_available times.
// pub fn directional_stack_moves2<B: MoveBuffer>(
//     moves: &mut B,
//     init_move: GameMove,
//     max_move: u8,
//     pieces_available: usize,
// ) {
//     const MAP: SlidePatternMap = generate_stack_sliders();
//     for pop_count in 1..=max_move {
//         for mv in MAP.iterate_slides(pop_count as usize) {
//             if mv.slide_bits >= 1 << pieces_available {
//                 break;
//             }
//             moves.add_move(
//                 init_move
//                     .set_number(mv.pickup_pieces as u32)
//                     .set_slide_bits(mv.slide_bits as u32),
//             )
//         }
//     }
// }

/// Find all stack moves for a single stack in one direction. Calls the recursive
/// function [recursive_stack_moves] 1..=pieces_available times.
pub fn directional_stack_moves_old<B: MoveBuffer>(
    moves: &mut B,
    init_move: GameMove,
    max_move: u8,
    pieces_available: usize,
) {
    for take_pieces in 1..pieces_available as u32 + 1 {
        recursive_stack_moves(
            moves,
            init_move.set_number(take_pieces),
            max_move,
            take_pieces,
        );
    }
}

/// Find all stack moves for a single stack in one direction. Calls the recursive
/// function [recursive_stack_moves] 1..=pieces_available times.
pub fn directional_stack_moves<B: MoveBuffer>(
    moves: &mut B,
    init_move: GameMove,
    max_move: u8,
    pieces_available: usize,
) {
    for x in 1u32..1 << pieces_available {
        let pop_count = x.count_ones();
        if pop_count > max_move as u32 {
            continue;
        }
        moves.add_move(init_move.set_number(x.ilog2() + 1).set_slide_bits(x));
    }
}

/// Find all crush moves for a single stack in one direction. The crush tile has
/// index max_move + 1 and only a lone capstone can crush a wall, so we set
/// the corresponding last tile and wall crush bits before recursing. Calls the
/// recursive function [recursive_crush_moves] for all legal initial stack sizes.
/// This is more restrictive than the condition on [directional_stack_moves] because
/// one flat must be placed for tile traversed.
pub fn directional_crush_moves<B: MoveBuffer>(
    moves: &mut B,
    init_move: GameMove,
    max_steps: u8,
    pieces_available: usize,
) {
    let init_move = init_move.set_crush();
    // let init_move = init_move.set_crush().set_tile(max_steps as usize + 1, 1);
    if max_steps == 0 {
        moves.add_move(init_move.set_number(1).set_next_tile(1));
        return;
    }
    let start_slide: u32 = 1;
    // let start_slide: u32 = (1 << max_steps) | (1 << max_steps - 1);
    for x in start_slide..1 << pieces_available {
        let log = x.ilog2();
        let mask = 1 << log | (1 << (log - 1));
        if mask & x != mask {
            continue;
        }
        let pop_count = x.count_ones();
        if pop_count != (max_steps + 1) as u32 {
            continue;
        }
        moves.add_move(init_move.set_number(log + 1).set_slide_bits(x));
    }
    // Todo figure out upper bound for number of pieces to take to save some cycles?
    // for take_pieces in max_steps as u32..pieces_available as u32 + 1 {
    //     recursive_crush_moves(
    //         moves,
    //         init_move.set_number(take_pieces + 1),
    //         max_steps,
    //         take_pieces,
    //     );
    // }
}
/// Find all crush moves for a single stack in one direction. The crush tile has
/// index max_move + 1 and only a lone capstone can crush a wall, so we set
/// the corresponding last tile and wall crush bits before recursing. Calls the
/// recursive function [recursive_crush_moves] for all legal initial stack sizes.
/// This is more restrictive than the condition on [directional_stack_moves] because
/// one flat must be placed for tile traversed.
pub fn directional_crush_moves_old<B: MoveBuffer>(
    moves: &mut B,
    init_move: GameMove,
    max_steps: u8,
    pieces_available: usize,
) {
    let init_move = init_move.set_crush();
    // let init_move = init_move.set_crush().set_tile(max_steps as usize + 1, 1);
    if max_steps == 0 {
        moves.add_move(init_move.set_number(1).set_next_tile(1));
        return;
    }
    // Todo figure out upper bound for number of pieces to take to save some cycles?
    for take_pieces in max_steps as u32..pieces_available as u32 + 1 {
        recursive_crush_moves(
            moves,
            init_move.set_number(take_pieces + 1),
            max_steps,
            take_pieces,
        );
    }
}

pub fn recursive_stack_moves<B: MoveBuffer>(
    moves: &mut B,
    in_progress: GameMove,
    moves_left: u8,
    pieces_left: u32,
) {
    if moves_left == 1 || pieces_left == 1 {
        let last_move = in_progress.set_next_tile(pieces_left);
        moves.add_move(last_move);
        return;
    }
    for piece_count in 1..pieces_left {
        recursive_stack_moves(
            moves,
            in_progress.set_next_tile(piece_count),
            moves_left - 1,
            pieces_left - piece_count,
        );
    }
    moves.add_move(in_progress.set_next_tile(pieces_left));
}

pub fn recursive_crush_moves<B: MoveBuffer>(
    moves: &mut B,
    in_progress: GameMove,
    moves_left: u8,
    pieces_left: u32,
) {
    if moves_left == 1 && pieces_left > 0 {
        let last_move = in_progress.set_next_tile(pieces_left).set_next_tile(1);
        // let last_move = in_progress.set_tile(tile_num, pieces_left);
        moves.add_move(last_move);
        return;
    }
    // Todo figure out max placement to save some cycles?
    for piece_count in 1..pieces_left {
        recursive_crush_moves(
            moves,
            in_progress.set_next_tile(piece_count),
            moves_left - 1,
            pieces_left - piece_count,
        );
    }
}

const fn gen_long_slide_table() -> [u32; 256] {
    let mut arr: [u32; 256] = [0; 256];
    let mut init = 1;
    while init < 256 {
        let mv = GameMove(init << 12);
        arr[init as usize] = long_slide(mv);
        init += 1;
    }
    arr
}

pub const fn long_slide(mv: GameMove) -> u32 {
    let mut long_slide = 0;
    let mut slide_bits = mv.slide_bits();
    let mut count = 0;
    while slide_bits != 0 {
        let mut quantity = 1; // Account for the final 1 bit
        while 0 == slide_bits & 1 {
            quantity += 1;
            slide_bits = slide_bits >> 1;
        }
        slide_bits = slide_bits >> 1; // Remove the trailing 1
        long_slide |= quantity << count;
        count += 4;
    }
    long_slide
}

#[cfg(test)]
mod test {
    use crate::board::Board6;
    use crate::Position;
    use std::collections::{HashMap, HashSet};

    use super::*;

    #[test]
    pub fn aaa() {
        eprintln!("{}", 22u32.ilog2());
        let board = Board6::try_from_tps(
            "2,2,2,121S,x,2/21,1,2,x,121C,1/x,1,1,x,2S,1/x,1,2,11122C,1,x/x2,2,x,1,x/1,x3,1,x 2 22",
        )
        .unwrap();
        let mut moves: Vec<GameMove> = Vec::new();
        generate_all_moves(&board, &mut moves);
        let crushes: Vec<_> = moves
            .iter()
            .copied()
            .filter(|x| x.is_stack_move() && x.crush())
            .collect();
        for crush in crushes {
            eprintln!("{}, {}", crush.to_ptn::<Board6>(), crush.slide_bits());
        }
    }

    // #[test]
    // pub fn foofoobar() {
    //     let sliders_map = generate_stack_sliders();
    //     let sliders = sliders_map.data;
    //     for slider in &sliders[0..16] {
    //         eprintln!("{slider:?}")
    //     }
    //     dbg!(sliders.iter().position(|x| x.pop_count == 255));
    //     //        assert!(sliders.iter().find(|x| x.pop_count == 255).is_none())
    //     // sliders.iter().map(|x| x)
    //     dbg!(sliders.iter().position(|x| x.pop_count == 4));
    //     assert!(sliders
    //         .iter()
    //         .skip(1)
    //         .find(|x| x.pop_count == 255)
    //         .is_none())
    // }

    #[test]
    pub fn foobar() {
        let board = Board6::try_from_tps("2,x,x,x,x,11/x,x,x,x,x,221/x,2,2,2,x,221/2,1,12C,1,21C,2/2,x,2,x,x,2/x,2,2,2,x,121 1 25").unwrap();
        let mut moves: Vec<GameMove> = Vec::new();
        let mut moves2: Vec<GameMove> = Vec::new();
        generate_masked_stack_moves(
            &board,
            &mut moves,
            board.active_stacks(board.side_to_move()),
        );
        generate_masked_stack_moves_old(
            &board,
            &mut moves2,
            board.active_stacks(board.side_to_move()),
        );
        let moves: HashSet<_> = moves.into_iter().collect();
        let moves2: HashSet<_> = moves2.into_iter().collect();
        let false_pos = moves.difference(&moves2);
        let false_neg = moves2.difference(&moves);
        eprintln!("False Positive");
        for mv in false_pos {
            eprintln!("{mv:?}, {}", mv.to_ptn::<Board6>());
        }
        println!("False Negative");
        for mv in false_neg {
            eprintln!("{mv:?}, {}", mv.to_ptn::<Board6>());
        }
    }

    #[test]
    pub fn foo() {
        // bar("1212121S,x2,2,1,1/12S,12,121,2S,11C,x/1,112S,112C,1,x,12/1,2,2,1,1,1/2,2,2,2S,121S,x/2,2,1,2S,212,12S 1 40", 0, 3);
        let board = Board6::try_from_tps("2,2,21S,12122,221,x/2,1,2,12C,2,x/1,12222211C,21S,2,x2/2,1,2,1,1,1/x,2,12S,1,1,1/x,2,x3,1 1 37").unwrap();
        for sq in board.active_stacks(board.side_to_move()) {
            let stack = &board.board()[sq];
            if stack.has_cap() {
                continue;
            }
            let limit = find_move_limits(&board, sq);
            if sq == 21 {
                eprintln!("{limit:?}");
            }
            for dir in 0..4 {
                // eprintln!("{sq} --> {dir}");
                bar(&board, sq, dir);
            }
        }
    }

    fn bar(board: &Board6, start_idx: usize, dir: u32) {
        let mut moves: Vec<GameMove> = Vec::new();
        generate_all_stack_moves(board, &mut moves);
        moves.retain(|x| x.is_stack_move() && x.src_index() == start_idx && x.direction() == dir);
        let mut manual = HashSet::new();
        for mv in moves {
            // eprintln!("{} -- {:b}", mv.to_ptn::<Board6>(), mv.slide_bits());
            manual.insert(mv.slide_bits());
        }
        let limits = find_move_limits(board, start_idx);
        // eprintln!("{limits:?}");
        let mut iterative = HashSet::new();
        let max = board.board()[start_idx].len().min(Board6::SIZE);
        for x in 1..1 << max {
            if (x as u32).count_ones() > limits.steps[dir as usize] as u32 {
                continue;
            }
            iterative.insert(x);
        }
        assert_eq!(manual.len(), iterative.len());
        assert_eq!(manual, iterative);
    }

    // fn check() {

    // }

    #[test]
    pub fn index_move_by_bits() {
        let board = Board6::try_from_tps("1111111,x5/x6/x6/x6/x6/x6 1 10").unwrap();
        let mut moves: Vec<GameMove> = Vec::new();
        generate_all_stack_moves(&board, &mut moves);
        let mut set: HashMap<_, GameMove> = HashMap::new();
        for mv in moves.iter().copied() {
            match set.entry(mv.slide_bits()) {
                std::collections::hash_map::Entry::Occupied(occupied_entry) => {
                    if mv.direction() != occupied_entry.get().direction() {
                        continue;
                    }
                    assert_eq!(
                        mv.to_ptn::<Board6>(),
                        occupied_entry.get().to_ptn::<Board6>()
                    )
                }
                std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(mv);
                }
            }
        }
        assert_eq!(2 * set.len(), moves.len());
        let max_idx = set.into_iter().max_by_key(|x| x.1.slide_bits());
        dbg!(max_idx);
    }
    #[test]
    pub fn single_direction_move() {
        let mut moves: Vec<GameMove> = Vec::new();
        recursive_stack_moves(&mut moves, GameMove(0), 3, 3);
        assert_eq!(moves.len(), 4);
        recursive_stack_moves(&mut moves, GameMove(0), 3, 2);
        assert_eq!(moves.len(), 4 + 2);
        recursive_stack_moves(&mut moves, GameMove(0), 3, 1);
        assert_eq!(moves.len(), 4 + 2 + 1);

        for stack_size in 4..7 {
            moves.clear();
            directional_stack_moves(&mut moves, GameMove(0), stack_size, stack_size as usize);
            assert_eq!(moves.len(), 2usize.pow(stack_size as u32) - 1);
        }

        moves.clear();
        recursive_crush_moves(&mut moves, GameMove(0), 1, 2);
        assert_eq!(moves.len(), 1);
    }
    #[test]
    pub fn four_direction_move() {
        let tps = "x6/x6/x3,12,x2/x2,1,1,x2/x6/x6 2 3";
        let board = Board6::try_from_tps(tps).unwrap();
        let mut moves: Vec<GameMove> = Vec::new();
        directional_stack_moves(&mut moves, GameMove(0), 2, 2);
        let one_dir = moves.len();
        moves.clear();
        generate_all_stack_moves(&board, &mut moves);
        assert_eq!(moves.len(), one_dir * 4);

        let tps2 = "x6/x6/x2,2,12121,x2/x6/x6/x6 1 6";
        let board = Board6::try_from_tps(tps2).unwrap();
        moves.clear();
        generate_all_stack_moves(&board, &mut moves);
        let mut dir_count = [0; 4];
        for m in moves.iter() {
            dir_count[m.direction() as usize] += 1;
        }
        assert_eq!(dir_count[0], dir_count[1]);
        assert_eq!(dir_count[2], dir_count[3]);
        assert_eq!(moves.len(), 182 - 3 * 34);
    }
    fn all_moves_allocate(board: &Board6) -> Vec<GameMove> {
        let mut vec = Vec::new();
        generate_all_moves(board, &mut vec);
        vec
    }
    #[allow(dead_code)]
    fn compare_move_lists<T: TakBoard>(my_moves: Vec<GameMove>, source_file: &str) {
        use std::collections::HashSet;
        let file_data = std::fs::read_to_string(source_file).unwrap();
        let my_set: HashSet<_> = my_moves
            .iter()
            .map(|m| {
                let mut m = m.to_ptn::<T>();
                m.retain(|c| c != '*');
                m
            })
            .collect();
        let correct: HashSet<_> = file_data.lines().map(|s| s.to_string()).collect();
        println!("<<<");
        for k in my_set.iter() {
            if !correct.contains(k) {
                println!("{}", k);
            }
        }
        println!(">>>");
        for k in correct {
            if !my_set.contains(&k) {
                println!("{}", k);
            }
        }
    }
    #[test]
    pub fn all_non_crush_moves() {
        let tps = "x,1,2,x3/2,221,2,x3/2C,x,21C,x3/2,1,1,1,x2/x,2,x4/x,2,x3,1 1 11";
        let board = Board6::try_from_tps(tps).unwrap();
        let moves = all_moves_allocate(&board);
        assert_eq!(moves.len(), 91);
    }
    #[test]
    pub fn crush_moves() {
        let s = "1,1,2212S,x3/x2,1,x,1,x/2,211212C,11112,2,1S,22112S/221,x,221C,1,x,2/2,2,1,1,1,2/2,x,1,x2,2 2 35";
        let mut board = Board6::try_from_tps(s).unwrap();
        let moves = all_moves_allocate(&board);
        let crush_moves: Vec<_> = moves
            .iter()
            .copied()
            .filter_map(|m| {
                if m.crush() {
                    Some(m.to_ptn::<Board6>())
                } else {
                    None
                }
            })
            .collect();
        let real_crush_moves = [
            "3b4>111*", "4b4>121*", "4b4>211*", "5b4>131*", "5b4>221*", "5b4>311*", "6b4>141*",
            "6b4>231*", "6b4>321*", "6b4>411*",
        ];
        for c in crush_moves.into_iter() {
            assert!(real_crush_moves.iter().any(|&m| m == c));
        }
        for m in moves.iter().copied() {
            assert!(board.legal_move(m));
            let r = board.do_move(m);
            if m.crush() {
                assert!(!board.legal_move(m));
            }
            board.reverse_move(r);
        }
        assert_eq!(moves.len(), 228);
        let s2 = "2,1,1,1,1,2S/1,12,1,2,x,111121C/x,2,2,212,2C,11121/2,21122,x,x,1,x/x,x,x,1,1,x/x,x,2,21,x,112S 1 35";
        let mut board2 = Board6::try_from_tps(s2).unwrap();
        let moves = all_moves_allocate(&board2);
        for m in moves.iter().copied() {
            assert!(board2.legal_move(m));
            let r = board2.do_move(m);
            board2.reverse_move(r);
        }

        assert_eq!(moves.len(), 190);
    }

    #[test]
    pub fn quantity_move_test() {
        let ptn = "7a5>1231";
        let m = GameMove::try_from_ptn_m(ptn, 7, Color::White).unwrap();
        assert_eq!(m.count_steps(), 4);
        let mut iter = m.quantity_iter(7);
        let st = m.src_index();
        let mut counter = 1;
        let vals = &[0, 1, 2, 3, 1];
        while let Some(qstep) = iter.next() {
            assert_eq!(qstep.index, st + counter); // True because > direction
            assert_eq!(qstep.quantity, vals[counter]);
            counter += 1;
        }
        assert_eq!(counter, 5);
    }

    #[test]
    pub fn future_8s_compatibility() {
        let ptn = "8a1+1111112";
        let ptn2 = "8a1+2111111";
        let mv = GameMove::try_from_ptn_m(ptn, 8, Color::White).unwrap();
        let mv2 = GameMove::try_from_ptn_m(ptn2, 8, Color::White).unwrap();
        assert!(mv.slide_bits() < 256);
        assert!(mv2.slide_bits() < 256);
        // dbg!(format!("{:#X}", mv.slide_bits()));
        // dbg!(format!("{:#X}", mv2.slide_bits()));
    }

    #[test]
    pub fn move_rotation() {
        let ptn = "e3";
        let mv = GameMove::try_from_ptn_m(ptn, 6, Color::White).unwrap();
        let rotated = mv.reverse_symmetry::<Board6>(5);
        assert_eq!(rotated.to_ptn::<Board6>(), "d5");
    }
}
