use super::{Board6, Piece};
use anyhow::{bail, ensure, Result};
use board_game_traits::{Color, Position};
use std::fmt;

pub mod ptn;

pub fn generate_all_moves(board: &Board6, moves: &mut Vec<GameMove>) {
    generate_all_place_moves(board, moves);
    generate_all_stack_moves(board, moves);
}

/// Generates all legal placements of flats, walls, and caps for the active player.
pub fn generate_all_place_moves(board: &Board6, moves: &mut Vec<GameMove>) {
    let side_to_move = board.side_to_move();
    let start_locs = board.scan_empty_tiles();
    if board.move_num() == 1 {
        // Force flats. Handle swapping of the pieces in the do_move function
        let piece = match side_to_move {
            Color::White => Piece::WhiteFlat,
            Color::Black => Piece::BlackFlat,
        };
        for index in start_locs {
            moves.push(GameMove::from_placement(piece, index));
        }
        return;
    }
    let (flat, wall, cap) = match side_to_move {
        Color::White => (Piece::WhiteFlat, Piece::WhiteWall, Piece::WhiteCap),
        Color::Black => (Piece::BlackFlat, Piece::BlackWall, Piece::BlackCap),
    };
    if board.caps_reserve(side_to_move) > 0 {
        for index in start_locs.iter().copied() {
            moves.push(GameMove::from_placement(cap, index));
        }
    }
    for index in start_locs {
        moves.push(GameMove::from_placement(flat, index));
        moves.push(GameMove::from_placement(wall, index));
    }
}

/// Generates all legal sliding movements for the active player's stacks.
pub fn generate_all_stack_moves(board: &Board6, moves: &mut Vec<GameMove>) {
    let start_locs = board.scan_active_stacks(board.side_to_move());
    for index in start_locs {
        let stack_height = board.board[index].len();
        let start_move = GameMove(index as u64);
        let limits = find_move_limits(board, index);
        for dir in 0..4 {
            let dir_move = start_move.set_direction(dir as u64);
            let max_steps = limits.steps[dir];
            let max_pieces = std::cmp::min(board.board_size(), stack_height);
            if limits.can_crush[dir] {
                directional_crush_moves(moves, dir_move, max_steps, max_pieces - 1);
            }
            if max_steps == 0 {
                continue;
            }
            directional_stack_moves(moves, dir_move, max_steps, max_pieces);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RevGameMove {
    pub game_move: GameMove,
    pub dest_sq: usize,
}

impl RevGameMove {
    pub fn new(game_move: GameMove, dest_sq: usize) -> Self {
        Self { game_move, dest_sq }
    }
    pub fn rev_iter(self, board_size: usize) -> RevStackMoveIterator {
        RevStackMoveIterator::new(self, board_size)
    }
}

/// 0011 1111 Src Tile
/// 1100 0000 Direction
/// 000000000F00 Number Picked Up
/// 00000000F000 Tile 1
/// 0000000F0000 Tile 2
/// 000000F00000 Tile 3
/// 00000F000000 Tile 4
/// 0000F0000000 Tile 5
/// 000F00000000 Tile 6
/// 00F000000000 Tile 7
/// 010000000000 Wall Smash
/// F00000000000 Placement Piece
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GameMove(u64);

impl GameMove {
    const PLACEMENT_THRESHOLD: u64 = 0x20000000000;
    pub fn null_move() -> Self {
        Self(0xF000_0000_0000_0000)
    }
    fn from_placement(piece: Piece, index: usize) -> Self {
        let bits = ((piece as u64) << 44) | index as u64;
        Self(bits)
    }
    pub fn is_place_move(self) -> bool {
        self.0 >= Self::PLACEMENT_THRESHOLD
    }
    fn slide_bits(self) -> u64 {
        let bits = self.0 & 0xFFFFFFF000;
        bits >> 12
    }
    pub fn place_piece(self) -> Piece {
        Piece::from_index(self.0 >> 44)
    }
    fn is_stack_move(self) -> bool {
        self.0 < Self::PLACEMENT_THRESHOLD
    }
    pub fn src_index(self) -> usize {
        self.0 as usize & 0x3F
    }
    fn direction(self) -> u64 {
        (self.0 & 0xC0) >> 6
    }
    fn set_index(self, index: u64) -> Self {
        Self(self.0 | index)
    }
    /// North, East, South, West respectively
    fn set_direction(self, dir: u64) -> Self {
        GameMove(self.0 | (dir << 6))
    }
    fn set_number(self, num: u64) -> Self {
        GameMove(self.0 | (num << 8))
    }
    pub fn number(self) -> u64 {
        (self.0 & 0xF00) >> 8
    }
    pub fn forward_iter(self, board_size: usize) -> StackMoveIterator {
        StackMoveIterator::new(self, board_size)
    }
    fn set_tile(self, tile_num: usize, count: u64) -> Self {
        match tile_num {
            1 => self.set_tile1(count),
            2 => self.set_tile2(count),
            3 => self.set_tile3(count),
            4 => self.set_tile4(count),
            5 => self.set_tile5(count),
            6 => self.set_tile6(count),
            7 => self.set_tile7(count),
            _ => unimplemented!(),
        }
    }
    fn set_tile1(self, count: u64) -> Self {
        Self(self.0 | (count << 12))
    }
    fn set_tile2(self, count: u64) -> Self {
        Self(self.0 | (count << 16))
    }
    fn set_tile3(self, count: u64) -> Self {
        Self(self.0 | (count << 20))
    }
    fn set_tile4(self, count: u64) -> Self {
        Self(self.0 | (count << 24))
    }
    fn set_tile5(self, count: u64) -> Self {
        Self(self.0 | (count << 28))
    }
    fn set_tile6(self, count: u64) -> Self {
        Self(self.0 | (count << 32))
    }
    fn set_tile7(self, count: u64) -> Self {
        Self(self.0 | (count << 36))
    }
    fn chain_crush(self, cond: bool) -> Self {
        if cond {
            self.set_crush()
        } else {
            self
        }
    }
    pub fn set_crush(self) -> Self {
        Self(self.0 | 0x10000000000)
    }
    pub fn crush(&self) -> bool {
        (self.0 & 0x10000000000) > 0
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

pub struct StackMoveIterator {
    slide_bits: u64,
    direction: u8,
    index: usize,
    board_size: usize,
}

impl StackMoveIterator {
    fn new(game_move: GameMove, board_size: usize) -> Self {
        let slide_bits = game_move.slide_bits();
        let direction = game_move.direction() as u8;
        let mut index = game_move.src_index();
        // We want to start on the first dest square, not the origin square
        match direction {
            0 => index -= board_size,
            1 => index += 1,
            2 => index += board_size,
            3 => index -= 1,
            _ => unimplemented!(),
        }
        Self {
            slide_bits,
            direction,
            index,
            board_size,
        }
    }
}

impl Iterator for StackMoveIterator {
    type Item = usize;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.slide_bits == 0 {
            return None;
        }
        let bits = self.slide_bits & 0xF;
        if bits == 0 {
            self.slide_bits = self.slide_bits >> 4;
            match self.direction {
                0 => self.index -= self.board_size,
                1 => self.index += 1,
                2 => self.index += self.board_size,
                3 => self.index -= 1,
                _ => unimplemented!(),
            }
        }
        self.slide_bits -= 1;
        Some(self.index)
    }
}

pub struct RevStackMoveIterator {
    slide_bits: u64,
    direction: u8,
    index: usize,
    board_size: usize,
}

impl RevStackMoveIterator {
    fn new(rev_move: RevGameMove, board_size: usize) -> Self {
        let game_move = rev_move.game_move;
        let slide = game_move.slide_bits();
        let zeros = slide.leading_zeros();
        let offset = zeros % 4;
        let shift = zeros - offset;
        let slide_bits = slide << shift;
        let direction = game_move.direction() as u8;
        let index = rev_move.dest_sq;
        Self {
            slide_bits,
            direction,
            index,
            board_size,
        }
    }
}

impl Iterator for RevStackMoveIterator {
    type Item = usize;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.slide_bits == 0 {
            return None;
        }
        // Drop the high bits instead of the low bits
        let bits = self.slide_bits & 0xF000_0000_0000_0000;
        if bits == 0 {
            self.slide_bits = self.slide_bits << 4;
            // Reversed from the normal direction step, since we're going backwards
            match self.direction {
                0 => self.index += self.board_size,
                1 => self.index -= 1,
                2 => self.index -= self.board_size,
                3 => self.index += 1,
                _ => unimplemented!(),
            }
        }
        self.slide_bits -= 0x1000_0000_0000_0000;
        Some(self.index)
    }
}

#[derive(Debug)]
pub struct MoveLimits {
    steps: [usize; 4],
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
pub fn find_move_limits(board: &Board6, st_index: usize) -> MoveLimits {
    let (st_row, st_col) = board.row_col(st_index);
    let mut limits = MoveLimits::new();
    find_dir_limit(board, st_row, st_col, 0, &mut limits, step_north);
    find_dir_limit(board, st_row, st_col, 1, &mut limits, step_east);
    find_dir_limit(board, st_row, st_col, 2, &mut limits, step_south);
    find_dir_limit(board, st_row, st_col, 3, &mut limits, step_west);
    limits
}

/// Finds the movement limit in a single direction, updating the index to be
/// searched according to the step_fn. See [find_move_limits].
fn find_dir_limit<F>(
    board: &Board6,
    st_row: usize,
    st_col: usize,
    dir: usize,
    limits: &mut MoveLimits,
    step_fn: F,
) where
    F: Fn(usize, usize) -> (usize, usize),
{
    let mut row = st_row;
    let mut col = st_col;
    let mut counter = 0;
    let res = step_fn(row, col);
    row = res.0;
    col = res.1;
    while let Some(stack) = board.try_tile(row, col) {
        match stack.last() {
            Some(piece) if piece.is_blocker() => {
                if board.tile(st_row, st_col).last().unwrap().is_cap() && piece.is_wall() {
                    limits.can_crush[dir] = true;
                }
                break;
            }
            _ => {}
        }
        let res = step_fn(row, col);
        row = res.0;
        col = res.1;
        counter += 1;
    }
    limits.steps[dir] = counter;
}

fn step_north(row: usize, col: usize) -> (usize, usize) {
    (row.wrapping_sub(1), col)
}

fn step_south(row: usize, col: usize) -> (usize, usize) {
    (row + 1, col)
}

fn step_east(row: usize, col: usize) -> (usize, usize) {
    (row, col + 1)
}

fn step_west(row: usize, col: usize) -> (usize, usize) {
    (row, col.wrapping_sub(1))
}

/// Find all stack moves for a single stack in one direction. Calls the recursive
/// function [recursive_stack_moves] 1..=pieces_available times.
pub fn directional_stack_moves(
    moves: &mut Vec<GameMove>,
    init_move: GameMove,
    max_move: usize,
    pieces_available: usize,
) {
    for take_pieces in 1..pieces_available as u64 + 1 {
        recursive_stack_moves(
            moves,
            init_move.set_number(take_pieces),
            1,
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
pub fn directional_crush_moves(
    moves: &mut Vec<GameMove>,
    init_move: GameMove,
    max_steps: usize,
    pieces_available: usize,
) {
    let init_move = init_move.set_crush().set_tile(max_steps + 1, 1);
    if max_steps == 0 {
        moves.push(init_move.set_number(1));
        return;
    }
    // Todo figure out upper bound for number of pieces to take to save some cycles?
    for take_pieces in max_steps as u64..pieces_available as u64 + 1 {
        recursive_crush_moves(
            moves,
            init_move.set_number(take_pieces + 1),
            1,
            max_steps,
            take_pieces,
        );
    }
}

pub fn recursive_stack_moves(
    moves: &mut Vec<GameMove>,
    in_progress: GameMove,
    tile_num: usize,
    moves_left: usize,
    pieces_left: u64,
) {
    if moves_left == 1 || pieces_left == 1 {
        let last_move = in_progress.set_tile(tile_num, pieces_left);
        moves.push(last_move);
        return;
    }
    for piece_count in 1..pieces_left {
        recursive_stack_moves(
            moves,
            in_progress.set_tile(tile_num, piece_count),
            tile_num + 1,
            moves_left - 1,
            pieces_left - piece_count,
        );
    }
    moves.push(in_progress.set_tile(tile_num, pieces_left));
}

pub fn recursive_crush_moves(
    moves: &mut Vec<GameMove>,
    in_progress: GameMove,
    tile_num: usize,
    moves_left: usize,
    pieces_left: u64,
) {
    if moves_left == 1 && pieces_left > 0 {
        let last_move = in_progress.set_tile(tile_num, pieces_left);
        moves.push(last_move);
        return;
    }
    // Todo figure out max placement to save some cycles?
    for piece_count in 1..pieces_left {
        recursive_crush_moves(
            moves,
            in_progress.set_tile(tile_num, piece_count),
            tile_num + 1,
            moves_left - 1,
            pieces_left - piece_count,
        );
    }
}

#[cfg(test)]
mod test {
    use board_game_traits::Position;

    use super::*;
    #[test]
    pub fn single_direction_move() {
        let m = GameMove(0);
        assert_eq!(GameMove(0xF000), m.set_tile1(0xF));
        let mut moves = Vec::new();
        recursive_stack_moves(&mut moves, GameMove(0), 1, 3, 3);
        assert_eq!(moves.len(), 4);
        recursive_stack_moves(&mut moves, GameMove(0), 1, 3, 2);
        assert_eq!(moves.len(), 4 + 2);
        recursive_stack_moves(&mut moves, GameMove(0), 1, 3, 1);
        assert_eq!(moves.len(), 4 + 2 + 1);

        for stack_size in 4..7 {
            moves.clear();
            directional_stack_moves(&mut moves, GameMove(0), stack_size, stack_size);
            assert_eq!(moves.len(), 2usize.pow(stack_size as u32) - 1);
        }

        moves.clear();
        recursive_crush_moves(&mut moves, GameMove(0), 1, 1, 2);
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
    fn compare_move_lists(my_moves: Vec<GameMove>, source_file: &str) {
        use std::collections::HashSet;
        let file_data = std::fs::read_to_string(source_file).unwrap();
        let my_set: HashSet<_> = my_moves
            .iter()
            .map(|m| {
                let mut m = m.to_ptn();
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
            .filter_map(|m| if m.crush() { Some(m.to_ptn()) } else { None })
            .collect();
        let real_crush_moves = [
            "3b4>111*", "4b4>121*", "4b4>211*", "5b4>131*", "5b4>221*", "5b4>311*", "6b4>141*",
            "6b4>231*", "6b4>321*", "6b4>411*",
        ];
        for c in crush_moves.into_iter() {
            assert!(real_crush_moves.iter().find(|&&m| m == c).is_some());
        }
        for m in moves.iter().copied() {
            let r = board.do_move(m);
            board.reverse_move(r);
        }
        assert_eq!(moves.len(), 228);
        let s2 = "2,1,1,1,1,2S/1,12,1,2,x,111121C/x,2,2,212,2C,11121/2,21122,x,x,1,x/x,x,x,1,1,x/x,x,2,21,x,112S 1 35";
        let mut board2 = Board6::try_from_tps(s2).unwrap();
        let moves = all_moves_allocate(&board2);
        for m in moves.iter().copied() {
            let r = board2.do_move(m);
            board2.reverse_move(r);
        }

        assert_eq!(moves.len(), 190);
    }
}
