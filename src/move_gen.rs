use super::{Board6, Piece};
use std::fmt;

pub fn generate_all_stack_moves(board: &Board6, moves: &mut Vec<StackMovement>) {
    let start_locs = board.scan_active_stacks(board.active_player);
    for index in start_locs {
        let stack_height = board.board[index].len();
        let start_move = StackMovement(index as u64);
        let limits = find_move_limits(board, index);
        for dir in 0..4 {
            let dir_move = start_move.set_direction(dir as u64);
            let max_steps = limits.steps[dir];
            directional_stack_moves(
                moves,
                dir_move,
                max_steps,
                std::cmp::min(board.size(), stack_height),
            );
            // Todo Capstone movement
        }
    }
}

// pub fn generate_four_dir_moves(board: &Board6, tile)

/// 0011 1111 Src Tile
/// 1100 0000 Direction
/// 00000000F00 Number Picked Up
/// 0000000F000 Tile 1
/// 000000F0000 Tile 2
/// 00000F00000 Tile 3
/// 0000F000000 Tile 4
/// 000F0000000 Tile 5
/// 00F00000000 Tile 6
/// 0F000000000 Tile 7
/// 10000000000 Wall Smash
#[derive(Clone, Copy, PartialEq)]
pub struct StackMovement(u64);

impl StackMovement {
    fn src_index(self) -> usize {
        self.0 as usize & 0x3F
    }
    fn direction(self) -> u64 {
        (self.0 & 0xC0) >> 6
    }
    /// North, East, South, West respectively
    fn set_direction(self, dir: u64) -> Self {
        StackMovement(self.0 | (dir << 6))
    }
    fn set_number(self, num: u64) -> Self {
        StackMovement(self.0 | (num << 8))
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
        Self(self.0 | (count << 8))
    }
    fn set_tile2(self, count: u64) -> Self {
        Self(self.0 | (count << 12))
    }
    fn set_tile3(self, count: u64) -> Self {
        Self(self.0 | (count << 16))
    }
    fn set_tile4(self, count: u64) -> Self {
        Self(self.0 | (count << 20))
    }
    fn set_tile5(self, count: u64) -> Self {
        Self(self.0 | (count << 24))
    }
    fn set_tile6(self, count: u64) -> Self {
        Self(self.0 | (count << 28))
    }
    fn set_tile7(self, count: u64) -> Self {
        Self(self.0 | (count << 32))
    }
    fn set_crush(self) -> Self {
        Self(self.0 | 0x10000000000)
    }
}

impl fmt::Debug for StackMovement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StackMovement({:#X})", self.0)
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

pub fn find_move_limits(board: &Board6, st_index: usize) -> MoveLimits {
    let (st_row, st_col) = board.row_col(st_index);
    let mut limits = MoveLimits::new();
    find_dir_limit(board, st_row, st_col, 0, &mut limits, step_north);
    find_dir_limit(board, st_row, st_col, 1, &mut limits, step_east);
    find_dir_limit(board, st_row, st_col, 2, &mut limits, step_south);
    find_dir_limit(board, st_row, st_col, 3, &mut limits, step_west);
    limits
}

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
    while let Some(stack) = board.try_tile(row, col) {
        let res = step_fn(row, col);
        row = res.0;
        col = res.1;
        match stack.last() {
            Some(piece) if piece.is_blocker() => {
                if board.tile(st_row, st_col).last().unwrap().is_cap() && piece.is_wall() {
                    limits.can_crush[dir] = true;
                }
                break;
            }
            _ => {}
        }
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

pub fn directional_stack_moves(
    moves: &mut Vec<StackMovement>,
    init_move: StackMovement,
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

pub fn directional_crush_moves(
    moves: &mut Vec<StackMovement>,
    init_move: StackMovement,
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
    moves: &mut Vec<StackMovement>,
    in_progress: StackMovement,
    tile_num: usize,
    moves_left: usize,
    pieces_left: u64,
) {
    // Todo Consider packing extra data into StackMovement?
    if moves_left == 0 || pieces_left == 1 {
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
    moves: &mut Vec<StackMovement>,
    in_progress: StackMovement,
    tile_num: usize,
    moves_left: usize,
    pieces_left: u64,
) {
    if moves_left == 0 || pieces_left == 1 {
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
        let m = StackMovement(0);
        assert_eq!(StackMovement(0xF00), m.set_tile1(0xF));
        let mut moves = Vec::new();
        recursive_stack_moves(&mut moves, StackMovement(0), 1, 3, 3);
        assert_eq!(moves.len(), 4);
        recursive_stack_moves(&mut moves, StackMovement(0), 1, 3, 2);
        assert_eq!(moves.len(), 4 + 2);
        recursive_stack_moves(&mut moves, StackMovement(0), 1, 3, 1);
        assert_eq!(moves.len(), 4 + 2 + 1);

        for stack_size in 4..7 {
            moves.clear();
            directional_stack_moves(&mut moves, StackMovement(0), stack_size, stack_size);
            assert_eq!(moves.len(), 2usize.pow(stack_size as u32) - 1);
        }

        moves.clear();
        recursive_crush_moves(&mut moves, StackMovement(0), 1, 1, 2);
        assert_eq!(moves.len(), 1);
    }
    #[test]
    pub fn four_direction_move() {
        let tps = "x6/x6/x3,12,x2/x2,1,1,x2/x6/x6 2 3";
        let board = Board6::try_from_tps(tps).unwrap();
        let mut moves = Vec::new();
        directional_stack_moves(&mut moves, StackMovement(0), 2, 2);
        let one_dir = moves.len();
        moves.clear();
        generate_all_stack_moves(&board, &mut moves);
        assert_eq!(moves.len(), one_dir * 4);
    }
}
