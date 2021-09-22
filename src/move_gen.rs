use super::{Board6, Piece};
use anyhow::{bail, ensure, Result};
use board_game_traits::Color;
use std::fmt;

pub mod ptn;

pub fn generate_all_moves(board: &Board6, moves: &mut Vec<GameMove>) {
    generate_all_place_moves(board, moves);
    generate_all_stack_moves(board, moves);
}

/// Generates all legal placements of flats, walls, and caps for the active player.
pub fn generate_all_place_moves(board: &Board6, moves: &mut Vec<GameMove>) {
    let start_locs = board.scan_empty_tiles();
    let (flat, wall, cap) = match board.active_player {
        Color::White => (Piece::WhiteFlat, Piece::WhiteWall, Piece::WhiteCap),
        Color::Black => (Piece::BlackFlat, Piece::BlackWall, Piece::BlackCap),
    };
    if board.caps_reserve(board.active_player) > 0 {
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
    let start_locs = board.scan_active_stacks(board.active_player);
    for index in start_locs {
        let stack_height = board.board[index].len();
        let start_move = GameMove(index as u64);
        let limits = find_move_limits(board, index);
        for dir in 0..4 {
            let dir_move = start_move.set_direction(dir as u64);
            let max_steps = limits.steps[dir];
            if max_steps == 0 {
                continue;
            }
            let max_pieces = std::cmp::min(board.board_size(), stack_height);
            directional_stack_moves(moves, dir_move, max_steps, max_pieces);

            if limits.can_crush[dir] {
                directional_crush_moves(moves, dir_move, max_steps, max_pieces);
            }
        }
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
    fn place_piece(self) -> Piece {
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
    fn set_crush(self) -> Self {
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

pub struct StackMoveTile {
    pub row: usize,
    pub col: usize,
    pub pieces: usize,
}

impl StackMoveTile {
    pub fn new(row: usize, col: usize, pieces: usize) -> Self {
        Self { row, col, pieces }
    }
}

pub struct StackMoveIterator {
    slide_bits: u64,
    direction: u8,
    row: usize,
    col: usize,
}

impl StackMoveIterator {
    fn new(game_move: GameMove, board_size: usize) -> Self {
        let slide_bits = game_move.slide_bits();
        let direction = game_move.direction() as u8;
        let index = game_move.src_index();
        let row = index / board_size;
        let col = index % board_size;
        Self {
            slide_bits,
            direction,
            row,
            col,
        }
    }
}

impl Iterator for StackMoveIterator {
    type Item = StackMoveTile;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        let bits = self.slide_bits & 0xF;
        if bits == 0 {
            return None;
        }
        self.slide_bits = self.slide_bits >> 4;
        match self.direction {
            0 => self.row -= 1,
            1 => self.col += 1,
            2 => self.row += 1,
            3 => self.col -= 1,
            _ => unimplemented!(),
        }
        Some(StackMoveTile::new(self.row, self.col, bits as usize))
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
    max_move: usize,
    pieces_available: usize,
) {
    let init_move = init_move.set_crush().set_tile(max_move + 1, 1);
    for take_pieces in max_move as u64..pieces_available as u64 + 1 {
        recursive_crush_moves(
            moves,
            init_move.set_number(take_pieces + 1),
            1,
            max_move,
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
    if moves_left == 1 || pieces_left == 1 {
        let last_move = in_progress.set_tile(tile_num, pieces_left);
        moves.push(last_move);
        return;
    }
    let max_placement = 1 + pieces_left - moves_left as u64; // Todo figure out if this is right
    for piece_count in 1..max_placement {
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
            // dbg!(m.to_ptn());
        }
        // dbg!(&dir_count);
        assert_eq!(dir_count[0], dir_count[1]);
        assert_eq!(dir_count[2], dir_count[3]);
        assert_eq!(moves.len(), 182 - 3 * 34);
    }
    fn all_moves_allocate(board: &Board6) -> Vec<GameMove> {
        let mut vec = Vec::new();
        generate_all_moves(board, &mut vec);
        vec
    }
    fn compare_move_lists(my_moves: Vec<String>, source_file: &str) {
        use std::collections::HashSet;
        let file_data = std::fs::read_to_string(source_file).unwrap();
        let my_set: HashSet<_> = my_moves.iter().map(|s| s.as_ref()).collect();
        let correct: HashSet<_> = file_data.lines().collect();
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
        let board = Board6::try_from_tps(s).unwrap();
        let moves = all_moves_allocate(&board);
        assert_eq!(moves.len(), 228);
    }
}
