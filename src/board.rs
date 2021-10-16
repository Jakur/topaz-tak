use crate::move_gen::generate_all_moves;
use crate::{GameMove, RevGameMove};
use anyhow::{anyhow, bail, ensure, Result};
pub use bitboard::{BitIndexIterator, Bitboard, Bitboard6, BitboardStorage};
use board_game_traits::{Color, GameResult, Position};
use std::fmt;

mod bitboard;
mod piece;
mod stack;
mod zobrist;
pub use piece::*;
pub use stack::*;

#[derive(PartialEq, Clone)]
pub struct Board6 {
    pub board: [Stack; 36],
    active_player: Color,
    move_num: usize,
    flats_left: [usize; 2],
    caps_left: [usize; 2],
    pub bits: BitboardStorage<Bitboard6>,
}

impl Board6 {
    pub fn new() -> Self {
        const SIZE: usize = 36;
        const INIT: Stack = Stack::new();
        let mut board = [INIT; SIZE];
        for (idx, stack) in board.iter_mut().enumerate() {
            stack.init(idx);
        }
        let bits = BitboardStorage::build_6(&board);
        Self {
            board,
            active_player: Color::White,
            move_num: 1,
            flats_left: [30, 30],
            caps_left: [1, 1],
            bits,
        }
    }
    pub const fn size() -> usize {
        6
    }
    pub const fn board_size(&self) -> usize {
        Self::size()
    }
    pub fn try_from_tps(tps: &str) -> Result<Self> {
        let data: Vec<_> = tps.split_whitespace().collect();
        ensure!(data.len() == 3, "Malformed tps string!");
        let rows: Vec<_> = data[0].split("/").collect();
        ensure!(rows.len() == 6, "Wrong board size for tps");
        let mut board = Self::new();
        for (r_idx, row) in rows.into_iter().enumerate() {
            let mut col = 0;
            let tiles = row.split(",");
            for tile in tiles {
                if tile.starts_with("x") {
                    if let Some(c) = tile.chars().nth(1) {
                        col += c
                            .to_digit(10)
                            .ok_or_else(|| anyhow!("Failed to parse digit"))?;
                    } else {
                        col += 1;
                    }
                } else {
                    let stack = parse_tps_stack(tile)?;
                    ensure!(col < 6, "Too many columns for this board size");
                    board.board[r_idx * 6 + col as usize]
                        .extend(stack.into_iter(), &mut board.bits);
                    col += 1;
                }
            }
        }
        for stack in board.board.iter() {
            for piece in stack.iter() {
                match piece {
                    Piece::WhiteCap => board.caps_left[0] -= 1,
                    Piece::BlackCap => board.caps_left[1] -= 1,
                    Piece::WhiteFlat | Piece::WhiteWall => board.flats_left[0] -= 1,
                    Piece::BlackFlat | Piece::BlackWall => board.flats_left[1] -= 1,
                }
            }
        }
        let active_player = match data[1] {
            "1" => Color::White,
            "2" => Color::Black,
            _ => bail!("Unknown active player id"),
        };
        board.active_player = active_player;
        board.move_num = data[2].parse()?;
        let zobrist_hash = zobrist::TABLE.manual_build_hash(&board);
        board.bits.set_zobrist(zobrist_hash);
        Ok(board)
    }
    pub fn zobrist(&self) -> u64 {
        self.bits.zobrist()
    }
    pub fn row_col(&self, index: usize) -> (usize, usize) {
        Self::row_col_static(index)
    }
    pub fn active_player(&self) -> Color {
        self.active_player
    }
    pub fn swap_active_player(&mut self) {
        self.active_player = !self.active_player;
        self.bits.zobrist_color(self.active_player);
    }
    pub fn move_num(&self) -> usize {
        self.move_num
    }
    pub fn row_col_static(index: usize) -> (usize, usize) {
        (index / 6, index % 6)
    }
    pub fn try_tile(&self, row: usize, col: usize) -> Option<&Stack> {
        if row >= 6 || col >= 6 {
            None
        } else {
            Some(&self.board[row * 6 + col])
        }
    }
    pub fn tile(&self, row: usize, col: usize) -> &Stack {
        &self.board[row * 6 + col]
    }
    pub fn tile_mut(&mut self, row: usize, col: usize) -> &mut Stack {
        &mut self.board[row * 6 + col]
    }
    pub fn index(&self, i: usize) -> &Stack {
        &self.board[i]
    }
    pub fn active_stacks(&self, player: Color) -> BitIndexIterator<Bitboard6> {
        self.bits.iter_stacks(player)
    }
    pub fn empty_tiles(&self) -> BitIndexIterator<Bitboard6> {
        self.bits.iter_empty()
    }
    pub fn pieces_reserve(&self, player: Color) -> usize {
        self.flats_left[player as usize]
    }
    pub fn caps_reserve(&self, player: Color) -> usize {
        self.caps_left[player as usize]
    }
    fn board_fill(&self) -> bool {
        self.bits.board_fill()
    }
    fn flat_game(&self) -> GameResult {
        let white_score = self.bits.flat_score(Color::White);
        let black_score = self.bits.flat_score(Color::Black);
        if white_score > black_score {
            GameResult::WhiteWin
        } else if black_score > white_score {
            GameResult::BlackWin
        } else {
            GameResult::Draw
        }
    }
    pub fn road(&self, player: Color) -> bool {
        self.bits.check_road(player)
    }
    pub fn road_stack_throw(&self, road_pieces: Bitboard6, stack_move: GameMove) -> bool {
        let color = self.active_player();
        let src_sq = stack_move.src_index();
        let dir = stack_move.direction();
        let stack = &self.board[src_sq];
        let mut pickup = stack_move.number();
        let mut slide_bits = stack_move.slide_bits();
        let mut update = Bitboard6::ZERO;
        let mut index = src_sq;
        let mut mask = Bitboard6::index_to_bit(index);
        let val = stack
            .from_top(pickup as usize)
            .map(|p| p.road_piece(color))
            .unwrap_or(false); // Could have taken all pieces
        if val {
            update |= Bitboard6::index_to_bit(index);
        }
        while slide_bits != 0 {
            match dir {
                0 => index -= self.board_size(),
                1 => index += 1,
                2 => index += self.board_size(),
                3 => index -= 1,
                _ => unimplemented!(),
            }
            pickup -= slide_bits & 0xF;
            slide_bits = slide_bits >> 4;
            let bb = Bitboard6::index_to_bit(index);
            mask |= bb;
            let val = stack
                .from_top(pickup as usize)
                .map(|p| p.road_piece(color))
                .unwrap();
            if val {
                update |= bb;
            }
        }
        let updated = road_pieces ^ (road_pieces ^ update) & mask;
        updated.check_road()
    }
}

fn parse_tps_stack(tile: &str) -> Result<Vec<Piece>> {
    let mut vec = Vec::new();
    for c in tile.chars() {
        match c {
            '1' => vec.push(Piece::WhiteFlat),
            '2' => vec.push(Piece::BlackFlat),
            'S' => match vec.pop() {
                Some(Piece::WhiteFlat) => vec.push(Piece::WhiteWall),
                Some(Piece::BlackFlat) => vec.push(Piece::BlackWall),
                _ => bail!("Bad wall notation"),
            },
            'C' => match vec.pop() {
                Some(Piece::WhiteFlat) => vec.push(Piece::WhiteCap),
                Some(Piece::BlackFlat) => vec.push(Piece::BlackCap),
                _ => bail!("Bad capstone notation"),
            },
            _ => bail!("Unknown character encountered in tile"),
        }
    }
    Ok(vec)
}

impl Position for Board6 {
    type Move = GameMove;
    type ReverseMove = RevGameMove;
    fn start_position() -> Self {
        Self::new()
    }
    fn side_to_move(&self) -> Color {
        self.active_player
    }
    fn generate_moves(&self, moves: &mut Vec<<Self as Position>::Move>) {
        generate_all_moves(self, moves);
    }
    fn game_result(&self) -> Option<GameResult> {
        let prev_player = !self.side_to_move();
        let current_player = self.side_to_move();
        // If both players have a road the active player wins, but the eval will
        // be checked after the players switch, so we check the prev player first
        if self.road(prev_player) {
            return Some(GameResult::win_by(prev_player));
        } else if self.road(current_player) {
            return Some(GameResult::win_by(current_player));
        }
        if self.flats_left[0] == 0 || self.flats_left[1] == 0 || self.board_fill() {
            return Some(self.flat_game());
        }
        None
    }
    fn reverse_move(&mut self, rev_m: <Self as Position>::ReverseMove) {
        if let Color::White = self.active_player {
            self.move_num -= 1;
        }
        self.swap_active_player();
        let m = rev_m.game_move;
        let src_index = m.src_index();
        if m.is_place_move() {
            let piece = self.board[src_index].pop(&mut self.bits).unwrap();
            if piece.is_cap() {
                self.caps_left[piece.owner() as usize] += 1;
            } else {
                self.flats_left[piece.owner() as usize] += 1;
            }
        } else {
            if rev_m.game_move.crush() {
                // Stand the wall back up
                let dest_tile = &mut self.board[rev_m.dest_sq];
                dest_tile.uncrush_wall(&mut self.bits);
            }
            let iter = rev_m.rev_iter(self.board_size());
            // We need to get a mutable reference to multiple areas of the array. Hold on.
            let (origin, rest, offset): (&mut Stack, &mut [Stack], usize) = {
                if src_index > rev_m.dest_sq {
                    // Easy case, because all indices remain the same
                    let (split_left, split_right) = self.board.split_at_mut(src_index);
                    (&mut split_right[0], split_left, 0)
                } else {
                    // Should not go out of bounds since src index < dest_sq
                    let (split_left, split_right) = self.board.split_at_mut(src_index + 1);
                    (&mut split_left[src_index], split_right, src_index + 1)
                }
            };
            for idx in iter {
                let piece = rest[idx - offset].pop(&mut self.bits).unwrap();
                origin.push(piece, &mut self.bits); // This will need to be reversed later
            }
            let pieces_moved = rev_m.game_move.number() as usize;
            origin.reverse_top(pieces_moved, &mut self.bits);
        }
    }
    fn do_move(&mut self, m: GameMove) -> <Self as Position>::ReverseMove {
        let swap_pieces = self.move_num == 1;
        if let Color::Black = self.active_player {
            self.move_num += 1;
        }
        self.swap_active_player();
        let src_index = m.src_index();
        if m.is_place_move() {
            let mut piece = m.place_piece();
            if swap_pieces {
                piece = piece.swap_color();
            }
            self.board[src_index].push(piece, &mut self.bits);
            if piece.is_cap() {
                self.caps_left[piece.owner() as usize] -= 1;
            } else {
                self.flats_left[piece.owner() as usize] -= 1;
            }
            RevGameMove::new(m, src_index)
        } else {
            let num_pieces = m.number() as usize;
            let stack_move = m.forward_iter(self.board_size());
            let src_stack = &mut self.board[src_index];
            // Todo remove allocation with split_at_mut
            debug_assert!(src_stack.len() >= num_pieces);
            let take_stack: Vec<_> = src_stack.split_off(num_pieces, &mut self.bits);
            let mut last_idx = 0;
            for (piece, sq) in take_stack.into_iter().zip(stack_move) {
                debug_assert!(sq != src_index);
                self.board[sq].push(piece, &mut self.bits);
                last_idx = sq;
            }
            let last_square = &mut self.board[last_idx];
            let len = last_square.len();
            if len >= 2 {
                if last_square.try_crush_wall(&mut self.bits) {
                    return RevGameMove::new(m.set_crush(), last_idx);
                }
            }
            RevGameMove::new(m, last_idx)
        }
    }
}

impl fmt::Debug for Board6 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let mut board_string = String::new();
        for (i, stack) in self.board.iter().enumerate() {
            for piece in stack.iter() {
                let s = match piece {
                    Piece::WhiteCap => "1C",
                    Piece::BlackCap => "2C",
                    Piece::WhiteWall => "1S",
                    Piece::BlackWall => "2S",
                    Piece::WhiteFlat => "1",
                    Piece::BlackFlat => "2",
                };
                board_string.push_str(s);
            }
            if stack.is_empty() {
                board_string.push_str("x");
            }
            if i > 0 && i % 6 == 5 {
                board_string.push_str("/");
            } else {
                board_string.push_str(",")
            }
        }
        board_string.pop(); // Trailing comma or slash
        let side = if let Color::White = self.side_to_move() {
            1
        } else {
            2
        };

        write!(f, "{} {} {}", board_string, side, self.move_num)
    }
}

#[cfg(test)]
impl Board6 {
    pub fn scan_active_stacks(&self, player: Color) -> Vec<usize> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, vec)| match vec.last() {
                Some(piece) if piece.owner() == player => Some(i),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    pub fn test_read_tps() {
        let example_tps = "x6/x2,2,x3/x3,2C,x2/x2,211S,x2,2/x6/x,1,1,2,2,1 2 7";
        let board = Board6::try_from_tps(example_tps);
        assert!(board.is_ok());
        let mut board = board.unwrap();
        assert_eq!(board.active_player, Color::Black);
        assert_eq!(board.move_num, 7);
        let mut b = Board6::new();
        b.tile_mut(1, 2).push(Piece::BlackFlat, &mut board.bits);
        b.tile_mut(2, 3).push(Piece::BlackCap, &mut board.bits);
        let stack = vec![Piece::BlackFlat, Piece::WhiteFlat, Piece::WhiteWall];
        // b.tile_mut(3, 2) = &mut stack;
        let stack_dest = b.tile_mut(3, 2);
        stack_dest.extend(stack.into_iter(), &mut board.bits);
        b.tile_mut(3, 5).push(Piece::BlackFlat, &mut board.bits);

        b.tile_mut(5, 1).push(Piece::WhiteFlat, &mut board.bits);
        b.tile_mut(5, 2).push(Piece::WhiteFlat, &mut board.bits);
        b.tile_mut(5, 3).push(Piece::BlackFlat, &mut board.bits);
        b.tile_mut(5, 4).push(Piece::BlackFlat, &mut board.bits);
        b.tile_mut(5, 5).push(Piece::WhiteFlat, &mut board.bits);
        assert_eq!(board.board, b.board);

        assert_eq!(board.scan_active_stacks(Color::White).len(), 4);
        assert_eq!(board.scan_active_stacks(Color::Black).len(), 5);
        assert_eq!(board.flat_game(), GameResult::BlackWin);
    }
    #[test]
    pub fn test_forward_move() {
        let s = "2,2,2,21,12,x/x4,2,x/x4,2C,x/1,2,12,122211C,x2/x2,1S,1,12,1/x3,2S,1,1 1 19";
        let do_move = |ptn: &str, tps: &str| {
            let mut board = Board6::try_from_tps(s).unwrap();
            let parsed = GameMove::try_from_ptn(ptn, &board).unwrap();
            board.do_move(parsed);
            assert_eq!(board, Board6::try_from_tps(tps).unwrap());
        };
        do_move(
            "a2",
            "2,2,2,21,12,x/x4,2,x/x4,2C,x/1,2,12,122211C,x2/1,x,1S,1,12,1/x3,2S,1,1 2 19",
        );
        do_move(
            "c2+",
            "2,2,2,21,12,x/x4,2,x/x4,2C,x/1,2,121S,122211C,x2/x3,1,12,1/x3,2S,1,1 2 19",
        );
        do_move(
            "5d3<41",
            "2,2,2,21,12,x/x4,2,x/x4,2C,x/1,21C,122221,1,x2/x2,1S,1,12,1/x3,2S,1,1 2 19",
        );
        do_move(
            "5d3-41*",
            "2,2,2,21,12,x/x4,2,x/x4,2C,x/1,2,12,1,x2/x2,1S,12221,12,1/x3,21C,1,1 2 19",
        );
    }
    #[test]
    pub fn test_make_unmake_move() {
        let s = "2,2,2,21,12,x/x4,2,x/x4,2C,x/1,2,12,122211C,x2/x2,1S,1,12,1/x3,2S,1,1 1 19";
        let mut board = Board6::try_from_tps(s).unwrap();
        let check_move = |tps: &str, ptn: &str, board: &mut Board6| {
            let parsed = GameMove::try_from_ptn(ptn, board).unwrap();
            let rev_move = board.do_move(parsed);
            board.reverse_move(rev_move);
            assert_eq!(*board, Board6::try_from_tps(tps).unwrap())
        };
        let ptns = &[
            "a2", "c2+", "d2>", "f2<", "a3-", "5d3<41", "5d3-41*", "6d3+", "Sf6",
        ];
        for ptn in ptns {
            check_move(s, ptn, &mut board);
        }
        let s2 = concat!(
            "21,x,12211112C,2,1,1/1,2,22222221C,2,221,2/2,22221S,x2,1,",
            "1112S/1,2,1,x,212S,1/x,2,1,2,1,1/x,2,1,x2,1 2 44"
        );
        let mut board2 = Board6::try_from_tps(s2).unwrap();
        let ptns = &["3e3+12", "6c6<15", "a4+", "e1", "Sd4"];
        for ptn in ptns {
            check_move(s2, ptn, &mut board2)
        }
    }
    #[test]
    pub fn check_board_road() {
        let s = "2,1,1,1,1,2S/1,12,1,x2,111121C/x,2,2,212,2C,11121/2,2,1,1,12,2/x3,1,1,x/x2,2,21,x,112S 1 36";
        let board = Board6::try_from_tps(s).unwrap();
        assert!(!board.road(Color::White));
        assert!(board.road(Color::Black));
    }
    #[test]
    pub fn basic_perft() {
        let ptn_moves = &[
            "c2", "c3", "d3", "b3", "c4", "1c2+", "1d3<", "1b3>", "1c4-", "Cc2", "a1", "1c2+", "a2",
        ];
        let mut board = Board6::new();
        let res = crate::execute_moves_check_valid(&mut board, ptn_moves);
        assert!(res.is_ok());

        let p_res: Vec<_> = (0..3)
            .map(|depth| crate::perft(&mut board, depth as u16))
            .collect();
        assert_eq!(&p_res[..], &[1, 190, 20698]);
    }
}
