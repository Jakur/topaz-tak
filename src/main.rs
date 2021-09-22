use anyhow::{anyhow, bail, ensure, Result};
use bitboard::{Bitboard, Bitboard6, BitboardStorage};
use board_game_traits::{Color, GameResult, Position};
use move_gen::{generate_all_moves, GameMove};

mod bitboard;
mod move_gen;

fn main() {
    let x = Bitboard6::new(0x20103c407e00);
    println!("Road: {}", x.check_road());
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Piece {
    WhiteFlat = 1,
    WhiteWall = 2,
    WhiteCap = 3,
    BlackFlat = 4,
    BlackWall = 5,
    BlackCap = 6,
}

impl Piece {
    fn from_index(index: u64) -> Self {
        match index {
            1 => Piece::WhiteFlat,
            2 => Piece::WhiteWall,
            3 => Piece::WhiteCap,
            4 => Piece::BlackFlat,
            5 => Piece::BlackWall,
            6 => Piece::BlackCap,
            _ => unimplemented!(),
        }
    }
    fn owner(self) -> Color {
        match self {
            Piece::WhiteFlat | Piece::WhiteWall | Piece::WhiteCap => Color::White,
            Piece::BlackFlat | Piece::BlackWall | Piece::BlackCap => Color::Black,
        }
    }
    fn is_wall(self) -> bool {
        match self {
            Piece::WhiteWall | Piece::BlackWall => true,
            _ => false,
        }
    }
    fn is_cap(self) -> bool {
        match self {
            Piece::WhiteCap | Piece::BlackCap => true,
            _ => false,
        }
    }
    fn is_blocker(self) -> bool {
        match self {
            Piece::WhiteFlat | Piece::BlackFlat => false,
            _ => true,
        }
    }
    fn wall(color: Color) -> Self {
        match color {
            Color::White => Piece::WhiteWall,
            Color::Black => Piece::BlackWall,
        }
    }
    fn flat(color: Color) -> Self {
        match color {
            Color::White => Piece::WhiteFlat,
            Color::Black => Piece::BlackFlat,
        }
    }
    fn cap(color: Color) -> Self {
        match color {
            Color::White => Piece::WhiteCap,
            Color::Black => Piece::BlackFlat,
        }
    }
}

pub struct Board6 {
    board: [Vec<Piece>; 36],
    active_player: Color,
    move_num: usize,
    flats_left: [usize; 2],
    caps_left: [usize; 2],
    stack_cache: Vec<Piece>,
}

impl Board6 {
    fn new() -> Self {
        const SIZE: usize = 36;
        const INIT: Vec<Piece> = Vec::new();
        Self {
            board: [INIT; SIZE],
            active_player: Color::White,
            move_num: 1,
            flats_left: [30, 30],
            caps_left: [1, 1],
            stack_cache: Vec::with_capacity(Self::size()),
        }
    }
    const fn size() -> usize {
        6
    }
    const fn board_size(&self) -> usize {
        Self::size()
    }
    fn try_from_tps(tps: &str) -> Result<Self> {
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
                    board.board[r_idx * 6 + col as usize].extend(stack.into_iter());
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
        Ok(board)
    }
    fn row_col(&self, index: usize) -> (usize, usize) {
        Self::row_col_static(index)
    }
    fn row_col_static(index: usize) -> (usize, usize) {
        (index / 6, index % 6)
    }
    fn try_tile(&self, row: usize, col: usize) -> Option<&Vec<Piece>> {
        if row >= 6 || col >= 6 {
            None
        } else {
            Some(&self.board[row * 6 + col])
        }
    }
    fn tile(&self, row: usize, col: usize) -> &Vec<Piece> {
        &self.board[row * 6 + col]
    }
    fn tile_mut(&mut self, row: usize, col: usize) -> &mut Vec<Piece> {
        &mut self.board[row * 6 + col]
    }
    fn scan_active_stacks(&self, player: Color) -> Vec<usize> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, vec)| match vec.last() {
                Some(piece) if piece.owner() == player => Some(i),
                _ => None,
            })
            .collect()
    }
    fn scan_empty_tiles(&self) -> Vec<usize> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, vec)| match vec.last() {
                Some(_) => None,
                None => Some(i),
            })
            .collect()
    }
    fn pieces_reserve(&self, player: Color) -> usize {
        self.flats_left[player as usize]
    }
    fn caps_reserve(&self, player: Color) -> usize {
        self.caps_left[player as usize]
    }
    fn board_fill(&self) -> bool {
        self.board.iter().all(|stack| !stack.is_empty())
    }
    fn flat_game(&self) -> GameResult {
        let mut white_score = 0;
        let mut black_score = 0;
        for stack in self.board.iter() {
            match stack.last() {
                Some(Piece::WhiteFlat) => white_score += 1,
                Some(Piece::BlackFlat) => black_score += 1,
                _ => {}
            }
        }
        if white_score > black_score {
            GameResult::WhiteWin
        } else if black_score > white_score {
            GameResult::BlackWin
        } else {
            GameResult::Draw
        }
    }
    fn road(&self, player: Color) -> bool {
        let bitboards = BitboardStorage::<Bitboard6>::build_6(self);
        match player {
            Color::White => bitboards.white_flat_cap.check_road(),
            Color::Black => bitboards.black_flat_cap.check_road(),
        }
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
    type ReverseMove = GameMove;
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
    fn reverse_move(&mut self, _: <Self as Position>::ReverseMove) {
        todo!()
    }
    fn do_move(&mut self, m: GameMove) -> <Self as Position>::ReverseMove {
        if m.is_place_move() {
            todo!()
        } else {
            let iter = m.forward_iter(self.board_size());
            let src = m.src_index();
            let mut take = m.number() as usize;
            let src_tile = &mut self.board[src];
            self.stack_cache
                .extend(src_tile.drain(src_tile.len() - take..).rev());
            for data in iter {
                for _ in 0..data.pieces {
                    let last = self.stack_cache.pop().unwrap();
                    let tile = self.tile_mut(data.row, data.col);
                    tile.push(last);
                }
                take -= data.pieces;
                if m.crush() && take == 0 {
                    let tile = self.tile_mut(data.row, data.col);
                    let len = tile.len();
                    match tile[tile.len() - 2] {
                        Piece::WhiteWall => tile[len - 2] = Piece::WhiteFlat,
                        Piece::BlackWall => tile[len - 2] = Piece::BlackFlat,
                        _ => unimplemented!(),
                    }
                }
            }
            debug_assert!(self.stack_cache.is_empty());
            m
        }
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
        let board = board.unwrap();
        assert_eq!(board.active_player, Color::Black);
        assert_eq!(board.move_num, 7);
        let mut b = Board6::new();
        b.tile_mut(1, 2).push(Piece::BlackFlat);
        b.tile_mut(2, 3).push(Piece::BlackCap);
        let mut stack = vec![Piece::BlackFlat, Piece::WhiteFlat, Piece::WhiteWall];
        // b.tile_mut(3, 2) = &mut stack;
        std::mem::swap(b.tile_mut(3, 2), &mut stack);
        b.tile_mut(3, 5).push(Piece::BlackFlat);

        b.tile_mut(5, 1).push(Piece::WhiteFlat);
        b.tile_mut(5, 2).push(Piece::WhiteFlat);
        b.tile_mut(5, 3).push(Piece::BlackFlat);
        b.tile_mut(5, 4).push(Piece::BlackFlat);
        b.tile_mut(5, 5).push(Piece::WhiteFlat);
        assert_eq!(board.board, b.board);

        assert_eq!(board.scan_active_stacks(Color::White).len(), 4);
        assert_eq!(board.scan_active_stacks(Color::Black).len(), 5);
    }
}
