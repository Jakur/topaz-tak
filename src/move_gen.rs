use super::Piece;
use crate::board::TakBoard;
use crate::Color;
use crate::{Bitboard, BitboardStorage};
use miniserde::{de::Visitor, make_place, Deserialize, Serialize};
use std::fmt;

mod move_order;
pub use move_order::{HistoryMoves, KillerMoves, SmartMoveBuffer};
pub mod magic;
pub mod ptn;

pub trait MoveBuffer {
    fn add_move(&mut self, mv: GameMove);
    fn add_limit(&mut self, limit: MoveLimits);
}

impl MoveBuffer for Vec<GameMove> {
    fn add_move(&mut self, mv: GameMove) {
        self.push(mv);
    }

    fn add_limit(&mut self, _limit: MoveLimits) {}
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
pub fn generate_aggressive_place_moves<T: TakBoard>(board: &T, moves: &mut Vec<GameMove>) {
    let side_to_move = board.side_to_move();
    let start_locs = board.empty_tiles();
    let (flat, cap) = match side_to_move {
        Color::White => (Piece::WhiteFlat, Piece::WhiteCap),
        Color::Black => (Piece::BlackFlat, Piece::BlackCap),
    };
    if board.caps_reserve(side_to_move) > 0 {
        for index in start_locs {
            moves.push(GameMove::from_placement(cap, index));
        }
    }
    if board.pieces_reserve(side_to_move) > 1 {
        // Todo should this condition be 0?
        for index in board.empty_tiles() {
            moves.push(GameMove::from_placement(flat, index));
        }
    }
}

/// Generates all legal sliding movements for the active player's stacks.
pub fn generate_all_stack_moves<T: TakBoard, B: MoveBuffer>(board: &T, moves: &mut B) {
    let start_locs = board.active_stacks(board.side_to_move());
    for index in start_locs {
        let stack_height = board.index(index).len();
        let start_move = GameMove(index as u32);
        let limits = find_move_limits(board, index);
        for dir in 0..4 {
            let dir_move = start_move.set_direction(dir as u32);
            let max_steps = limits.steps[dir];
            let max_pieces = std::cmp::min(T::SIZE, stack_height);
            if limits.can_crush[dir] {
                directional_crush_moves(moves, dir_move, max_steps, max_pieces - 1);
            }
            if max_steps == 0 {
                continue;
            }
            directional_stack_moves(moves, dir_move, max_steps, max_pieces);
        }
        moves.add_limit(limits);
    }
}

/// A struct containing the necessary data to reverse a [GameMove].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RevGameMove {
    pub game_move: GameMove,
    pub dest_sq: usize,
}

impl RevGameMove {
    pub fn new(game_move: GameMove, dest_sq: usize) -> Self {
        Self { game_move, dest_sq }
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
#[derive(Clone, Copy, Eq, PartialEq)]
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
    pub fn null_move() -> Self {
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
    /// Counts the number of forward steps taken by a sliding move
    pub fn count_steps(self) -> u8 {
        self.slide_bits().count_ones() as u8
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
}

make_place!(Place);

impl Serialize for GameMove {
    fn begin(&self) -> miniserde::ser::Fragment {
        miniserde::ser::Fragment::U64(self.0 as u64)
    }
}

impl Visitor for Place<GameMove> {
    fn nonnegative(&mut self, val: u64) -> miniserde::Result<()> {
        self.out = Some(GameMove(val as u32));
        Ok(())
    }
}

impl Deserialize for GameMove {
    fn begin(out: &mut Option<Self>) -> &mut dyn Visitor {
        // All Deserialize impls will look exactly like this. There is no
        // other correct implementation of Deserialize.
        Place::new(out)
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

#[derive(Debug)]
pub struct MoveLimits {
    steps: [u8; 4],
    can_crush: [bool; 4],
}

impl MoveLimits {
    pub fn new() -> Self {
        Self {
            steps: [0; 4],
            can_crush: [false; 4],
        }
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
    let mut limits = MoveLimits::new();
    let bits = board.bits();
    find_dir_limit(bits, st_index, 0, &mut limits, T::Bits::north);
    find_dir_limit(bits, st_index, 1, &mut limits, T::Bits::east);
    find_dir_limit(bits, st_index, 2, &mut limits, T::Bits::south);
    find_dir_limit(bits, st_index, 3, &mut limits, T::Bits::west);
    limits
}

pub fn legal_stack_move<T: TakBoard>(board: &T, game_move: GameMove) -> bool {
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
    let mut limits = MoveLimits::new();
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

/// Find all stack moves for a single stack in one direction. Calls the recursive
/// function [recursive_stack_moves] 1..=pieces_available times.
pub fn directional_stack_moves<B: MoveBuffer>(
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

    use super::*;
    #[test]
    pub fn single_direction_move() {
        let mut moves = Vec::new();
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
        let mut moves = Vec::new();
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
}
