use crate::move_gen::{generate_all_moves, generate_masked_stack_moves};
use crate::{Color, GameMove, GameResult, Position, RevGameMove};
use anyhow::{anyhow, bail, ensure, Result};
pub use bitboard::*;
use std::fmt;

mod bitboard;
mod piece;
mod stack;
mod zobrist;
pub use piece::*;
pub use stack::*;

type Tile2D = (usize, usize);

pub trait TakBoard:
    Position<Move = GameMove, ReverseMove = RevGameMove> + std::fmt::Debug + Clone
{
    type Bits: Bitboard;
    // These consts break object safety
    const SIZE: usize;
    const FLATS: usize;
    const CAPS: usize;
    fn hash(&self) -> u64;
    fn legal_move(&self, game_move: GameMove) -> bool;
    fn ply(&self) -> usize;
    fn null_move(&mut self);
    fn rev_null_move(&mut self);
    fn get_tak_threats(
        &mut self,
        legal_moves: &Vec<GameMove>,
        hint: Option<&[GameMove]>,
    ) -> Vec<GameMove>;
    fn can_make_road(
        &self,
        storage: &mut Vec<GameMove>,
        hint: Option<&[GameMove]>,
    ) -> Option<GameMove>;
    fn move_num(&self) -> usize;
    fn flat_game(&self) -> Option<GameResult>;
    fn zobrist(&self) -> u64;
    fn row_col(&self, index: usize) -> (usize, usize);
    fn active_player(&self) -> Color;
    fn swap_active_player(&mut self);
    fn row_col_static(index: usize) -> (usize, usize);
    fn index_static(row: usize, col: usize) -> usize;
    fn try_tile(&self, row: usize, col: usize) -> Option<&Stack>;
    fn tile(&self, row: usize, col: usize) -> &Stack;
    fn tile_mut(&mut self, row: usize, col: usize) -> &mut Stack;
    fn index(&self, i: usize) -> &Stack;
    fn active_stacks(&self, player: Color) -> BitIndexIterator<Self::Bits>;
    fn empty_tiles(&self) -> BitIndexIterator<Self::Bits>;
    fn pieces_reserve(&self, player: Color) -> usize;
    fn caps_reserve(&self, player: Color) -> usize;
    fn road(&self, player: Color) -> bool;
    fn road_stack_throw(&self, road_pieces: <Self as TakBoard>::Bits, stack_move: GameMove)
        -> bool;
    fn make_ptn_moves(&mut self, moves: &[&str]) -> Option<()>;
    fn bits(&self) -> &BitboardStorage<Self::Bits>;
    fn board(&self) -> &[Stack];
    fn with_komi(self, half_flats: u8) -> Self
    where
        Self: Sized;
    fn komi(&self) -> u8;
    fn flat_diff(&self, player: Color) -> i32;
    /// Returns a cloned board rotated counter-clockwise in standard orientation
    fn rotate(&self) -> Self;
    // Returns a symmetry of the position given and index from 1 to 7 (inclusive)
    fn symmetry(&self, idx: usize) -> Option<Self>;
    /// Returns all 8 possible symmetrics of the board, including a copy of the board itself
    fn symmetries(&self) -> Vec<Self>;
    /// Reset all stack indices and the zobrist hash of the position
    fn reset_stacks(&mut self);
    fn try_from_tps(tps: &str) -> Result<Self>;
    fn fifty_move_rule(&self) -> usize;
}

macro_rules! board_impl {
    ($t: ty, $bits: ty, $sz: expr, $flats: expr, $caps: expr) => {
        impl $t {
            pub fn new() -> Self {
                const SIZE: usize = $sz * $sz;
                const INIT: Stack = Stack::new();
                let mut board = [INIT; SIZE];
                for (idx, stack) in board.iter_mut().enumerate() {
                    stack.init(idx);
                }
                let bits = BitboardStorage::build::<Self>(&board, Color::White);
                Self {
                    board,
                    active_player: Color::White,
                    move_num: 1,
                    flats_left: [Self::FLATS, Self::FLATS],
                    caps_left: [Self::CAPS, Self::CAPS],
                    bits,
                    komi: 0,
                    fifty_move: 0,
                }
            }
            fn flat_winner(&self) -> GameResult {
                let white_score = 2 * self.bits.flat_score(Color::White);
                let black_score = 2 * self.bits.flat_score(Color::Black) + self.komi as u32;
                if white_score > black_score {
                    GameResult::WhiteWin
                } else if black_score > white_score {
                    GameResult::BlackWin
                } else {
                    GameResult::Draw
                }
            }
            fn transform_board<T>(&self, f: T) -> Self
            where
                T: Fn(Tile2D) -> Tile2D,
            {
                let mut out = self.clone();
                for row in 0..Self::SIZE {
                    for col in 0..Self::SIZE {
                        let src = self.tile(row, col);
                        let (dest_row, dest_col) = f((row, col));
                        *out.tile_mut(dest_row, dest_col) = src.clone();
                    }
                }
                out.reset_stacks();
                out
            }
            fn flip_ns(&self) -> Self {
                self.transform_board(|(row, col)| (Self::SIZE - 1 - row, col))
            }
            fn flip_ew(&self) -> Self {
                self.transform_board(|(row, col)| (row, Self::SIZE - 1 - col))
            }
            pub(crate) fn set_move_number(&mut self, move_number: usize) {
                self.move_num = move_number;
            }
        }

        impl TakBoard for $t {
            type Bits = $bits;
            const SIZE: usize = $sz;
            const FLATS: usize = $flats;
            const CAPS: usize = $caps;
            fn hash(&self) -> u64 {
                self.zobrist()
            }
            fn legal_move(&self, game_move: GameMove) -> bool {
                if game_move.is_place_move() {
                    let p = game_move.place_piece();
                    if p.owner() != self.side_to_move() && self.move_num != 1 {
                        return false;
                    }
                    if p.is_cap() && self.caps_reserve(self.side_to_move()) == 0 {
                        return false;
                    }
                    if !p.is_cap() && self.pieces_reserve(self.side_to_move()) == 0 {
                        return false;
                    }
                    (self.bits().empty() & <$bits>::index_to_bit(game_move.src_index())).nonzero()
                } else {
                    crate::move_gen::legal_stack_move(self, game_move)
                }
            }
            fn ply(&self) -> usize {
                match self.side_to_move() {
                    Color::White => self.move_num() * 2,
                    Color::Black => self.move_num() * 2 + 1,
                }
            }
            fn null_move(&mut self) {
                if let Color::Black = self.side_to_move() {
                    self.move_num += 1;
                }
                self.fifty_move += 1;
                self.swap_active_player();
            }
            fn rev_null_move(&mut self) {
                if let Color::White = self.side_to_move() {
                    self.move_num -= 1
                }
                self.fifty_move -= 1;
                self.swap_active_player();
            }
            fn get_tak_threats(
                &mut self,
                legal_moves: &Vec<GameMove>,
                hint: Option<&[GameMove]>,
            ) -> Vec<GameMove> {
                let mut tak_threats = Vec::new();
                let mut stack_moves = Vec::new();
                for m in legal_moves.iter().copied() {
                    let rev = self.do_move(m);
                    self.null_move();
                    if let Some(_winning) = self.can_make_road(&mut stack_moves, hint) {
                        tak_threats.push(m);
                    }
                    self.rev_null_move();
                    self.reverse_move(rev);
                    stack_moves.clear();
                }
                tak_threats
            }
            fn can_make_road(
                &self,
                storage: &mut Vec<GameMove>,
                hint: Option<&[GameMove]>,
            ) -> Option<GameMove> {
                let player = self.active_player();
                let road_pieces = self.bits.road_pieces(player);
                // Check placement
                let place_road = find_placement_road(player, road_pieces, self.bits.empty());
                if place_road.is_some() {
                    return place_road;
                }
                // Check stack movements
                if let Some(suggestions) = hint {
                    for m in suggestions.iter().copied() {
                        if m.is_stack_move() && self.legal_move(m) {
                            if self.road_stack_throw(road_pieces, m) {
                                return Some(m);
                            }
                        }
                    }
                }
                let mut stacks = self.active_stacks(self.side_to_move());
                let cs = road_pieces.critical_squares() & !self.bits.cap;
                let cs_wall = cs.reachable_any();
                let cs_flat = (cs & !self.bits.wall).reachable_any();
                let mut bad_bits = Self::Bits::ZERO;
                for st_idx in stacks.clone() {
                    let (_, friendly) = self.board[st_idx].captive_friendly();
                    let bit = Self::Bits::index_to_bit(st_idx);
                    // Filter squares we don't need
                    match self.board[st_idx].top().unwrap() {
                        Piece::WhiteFlat | Piece::BlackFlat => {
                            if friendly == 0 && (bit & cs_flat) == Self::Bits::ZERO {
                                bad_bits |= bit;
                            }
                        }
                        Piece::WhiteWall | Piece::BlackWall => {
                            if friendly == 0
                                || (friendly == 1 && (bit & (cs | cs_flat)) == Self::Bits::ZERO)
                            {
                                bad_bits |= Self::Bits::index_to_bit(st_idx);
                            }
                        }
                        Piece::WhiteCap | Piece::BlackCap => {
                            if friendly == 0 && (bit & cs_wall) == Self::Bits::ZERO {
                                bad_bits |= Self::Bits::index_to_bit(st_idx);
                            }
                        }
                    }
                }
                stacks.mask(!bad_bits);
                generate_masked_stack_moves(self, storage, stacks);
                for m in storage.iter().copied() {
                    if self.road_stack_throw(road_pieces, m) {
                        return Some(m);
                    }
                }
                None
            }
            fn move_num(&self) -> usize {
                self.move_num
            }
            fn flat_game(&self) -> Option<GameResult> {
                if (self.flats_left[0] == 0 && self.caps_left[0] == 0)
                    || (self.flats_left[1] == 0 && self.caps_left[1] == 0)
                    || self.bits.board_fill()
                {
                    Some(self.flat_winner())
                } else {
                    None
                }
            }
            fn zobrist(&self) -> u64 {
                self.bits.zobrist()
            }
            fn row_col(&self, index: usize) -> (usize, usize) {
                Self::row_col_static(index)
            }
            fn active_player(&self) -> Color {
                self.active_player
            }
            fn swap_active_player(&mut self) {
                self.active_player = !self.active_player;
                self.bits.zobrist_color(self.active_player);
            }
            fn row_col_static(index: usize) -> (usize, usize) {
                (index / Self::SIZE, index % Self::SIZE)
            }
            fn index_static(row: usize, col: usize) -> usize {
                row * Self::SIZE + col
            }
            fn try_tile(&self, row: usize, col: usize) -> Option<&Stack> {
                if row >= Self::SIZE || col >= Self::SIZE {
                    None
                } else {
                    Some(&self.board[row * Self::SIZE + col])
                }
            }
            fn tile(&self, row: usize, col: usize) -> &Stack {
                &self.board[row * Self::SIZE + col]
            }
            fn tile_mut(&mut self, row: usize, col: usize) -> &mut Stack {
                &mut self.board[row * Self::SIZE + col]
            }
            fn index(&self, i: usize) -> &Stack {
                &self.board[i]
            }
            fn active_stacks(&self, player: Color) -> BitIndexIterator<<Self as TakBoard>::Bits> {
                self.bits.iter_stacks(player)
            }
            fn empty_tiles(&self) -> BitIndexIterator<<Self as TakBoard>::Bits> {
                self.bits.iter_empty()
            }
            fn pieces_reserve(&self, player: Color) -> usize {
                self.flats_left[player as usize]
            }
            fn caps_reserve(&self, player: Color) -> usize {
                self.caps_left[player as usize]
            }
            fn road(&self, player: Color) -> bool {
                self.bits.check_road(player)
            }
            fn road_stack_throw(
                &self,
                road_pieces: <Self as TakBoard>::Bits,
                stack_move: GameMove,
            ) -> bool {
                let color = self.active_player();
                let src_sq = stack_move.src_index();
                let stack = &self.board[src_sq];
                let mut pickup = stack_move.number() as u32;
                let mut update = <Self as TakBoard>::Bits::ZERO;
                let mut mask = <Self as TakBoard>::Bits::index_to_bit(src_sq);
                let val = stack
                    .from_top(pickup as usize)
                    .map(|p| p.road_piece(color))
                    .unwrap_or(false); // Could have taken all pieces
                if val {
                    update |= <Self as TakBoard>::Bits::index_to_bit(src_sq);
                }
                for qstep in stack_move.quantity_iter(Self::SIZE) {
                    pickup -= qstep.quantity;
                    let bb = <Self as TakBoard>::Bits::index_to_bit(qstep.index);
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
            fn make_ptn_moves(&mut self, moves: &[&str]) -> Option<()> {
                for s in moves {
                    let m = GameMove::try_from_ptn(s, self)?;
                    self.do_move(m);
                }
                Some(())
            }

            fn bits(&self) -> &BitboardStorage<Self::Bits> {
                &self.bits
            }

            fn board(&self) -> &[Stack] {
                &self.board
            }

            fn with_komi(mut self, half_flats: u8) -> Self {
                self.komi = half_flats;
                self
            }

            fn komi(&self) -> u8 {
                self.komi
            }

            fn flat_diff(&self, player: Color) -> i32 {
                let white = (self.bits.flat_score(Color::White) * 2) as i32;
                let black = (self.bits.flat_score(Color::Black) * 2) as i32 + self.komi as i32;

                match player {
                    Color::White => white - black,
                    Color::Black => black - white,
                }
            }

            fn rotate(&self) -> Self {
                self.transform_board(|(row, col)| (Self::SIZE - 1 - col, row))
            }

            fn symmetry(&self, idx: usize) -> Option<Self> {
                let state = match idx {
                    0 => self.clone(),
                    1 => self.flip_ns(),
                    2 => self.flip_ew(),
                    3 => self.rotate(),
                    4 => self.rotate().rotate(),
                    5 => self.rotate().rotate().rotate(),
                    6 => self.rotate().flip_ns(),
                    7 => self.rotate().flip_ew(),
                    _ => return None,
                };
                Some(state)
            }

            fn symmetries(&self) -> Vec<Self> {
                vec![
                    self.clone(),
                    self.flip_ns(),
                    self.flip_ew(),
                    self.rotate(),
                    self.rotate().rotate(),
                    self.rotate().rotate().rotate(),
                    self.rotate().flip_ns(),
                    self.rotate().flip_ew(),
                ]
            }

            fn reset_stacks(&mut self) {
                // Fix stack index
                let mut len = 0;
                for (idx, stack) in self.board.iter_mut().enumerate() {
                    stack.set_index(idx);
                    len += stack.len();
                }
                if len >= 2 && self.move_num < 2 {
                    self.move_num = 2;
                }
                // Set zobrist
                self.bits = BitboardStorage::build::<Self>(&self.board, self.side_to_move());
            }

            fn try_from_tps(tps: &str) -> Result<Self> {
                let mut komi = 0;
                let tps = match tps.split_once("!") {
                    Some((tps, rest)) => {
                        let mut split = rest.split(&['!', ' ']);
                        if split.by_ref().find(|x| *x == "komi").is_some() {
                            komi = split
                                .next()
                                .map(|x| x.parse())
                                .ok_or_else(|| anyhow!("Could not parse komi properly"))??;
                        }
                        tps
                    }
                    _ => tps,
                };
                let data: Vec<_> = tps.split_whitespace().collect();
                ensure!(data.len() == 3, "Malformed tps string!");
                let rows: Vec<_> = data[0].split("/").collect();
                ensure!(rows.len() == Self::SIZE, "Wrong board size for tps");
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
                            ensure!(
                                col < Self::SIZE as u32,
                                "Too many columns for this board size"
                            );
                            for piece in stack.into_iter() {
                                board.board[r_idx * Self::SIZE + col as usize]
                                    .push(piece, &mut board.bits);
                            }
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
                board.bits.set_zobrist(zobrist_hash.0, zobrist_hash.1);
                Ok(board.with_komi(komi))
            }
            fn fifty_move_rule(&self) -> usize {
                self.fifty_move as usize
            }
        }
        impl Position for $t {
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
                    Some(GameResult::win_by(prev_player))
                } else if self.road(current_player) {
                    Some(GameResult::win_by(current_player))
                } else {
                    if self.fifty_move >= 50 {
                        return Some(GameResult::Draw);
                    }
                    self.flat_game()
                }
            }
            fn reverse_move(&mut self, rev_m: <Self as Position>::ReverseMove) {
                self.fifty_move = rev_m.fifty_count;
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
                    let iter = rev_m.game_move.quantity_iter(Self::SIZE);
                    let mut build = stack::Pickup::default();
                    for q_step in iter {
                        build.append(
                            self.board[q_step.index]
                                .split_off(q_step.quantity as usize, &mut self.bits),
                        );
                    }
                    self.board[src_index].add_stack(
                        rev_m.game_move.number() as usize,
                        &mut build,
                        &mut self.bits,
                    );
                    if rev_m.game_move.crush() {
                        self.board[rev_m.dest_sq].uncrush_top(&mut self.bits);
                    }
                }
            }
            fn do_move(&mut self, m: GameMove) -> <Self as Position>::ReverseMove {
                let swap_pieces = self.move_num == 1;
                let curr_fifty_move = self.fifty_move;
                self.fifty_move += 1;
                if let Color::Black = self.active_player {
                    self.move_num += 1;
                }
                self.swap_active_player();
                let src_index = m.src_index();
                if m.is_place_move() {
                    self.fifty_move = 0;
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
                    RevGameMove::new(m, src_index, curr_fifty_move)
                } else {
                    let num_pieces = m.number() as usize;
                    // let stack_move = m.forward_iter(Self::SIZE);
                    let stack_move = m.quantity_iter(Self::SIZE);
                    let src_stack = &mut self.board[src_index];
                    debug_assert!(src_stack.len() >= num_pieces);
                    let mut take_stack = src_stack.split_off(num_pieces, &mut self.bits);
                    let mut last_idx = 0;
                    for q_step in stack_move.into_iter() {
                        debug_assert!(q_step.index != src_index);
                        self.board[q_step.index].add_stack(
                            q_step.quantity as usize,
                            &mut take_stack,
                            &mut self.bits,
                        );
                        last_idx = q_step.index;
                    }
                    if m.crush() {
                        self.fifty_move = 0;
                    }
                    RevGameMove::new(m, last_idx, curr_fifty_move)
                }
            }
        }

        impl fmt::Debug for $t {
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
                    if i > 0 && i % Self::SIZE == (Self::SIZE - 1) {
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
    };
}

#[derive(PartialEq, Clone)]
pub struct Board5 {
    pub board: [Stack; Self::SIZE * Self::SIZE],
    active_player: Color,
    move_num: usize,
    flats_left: [usize; 2],
    caps_left: [usize; 2],
    pub bits: BitboardStorage<<Self as TakBoard>::Bits>,
    komi: u8,
    fifty_move: u8,
}

#[derive(PartialEq, Clone)]
pub struct Board6 {
    pub board: [Stack; Self::SIZE * Self::SIZE],
    active_player: Color,
    move_num: usize,
    flats_left: [usize; 2],
    caps_left: [usize; 2],
    pub bits: BitboardStorage<<Self as TakBoard>::Bits>,
    komi: u8,
    fifty_move: u8,
}

#[derive(PartialEq, Clone)]
pub struct Board7 {
    pub board: [Stack; Self::SIZE * Self::SIZE],
    active_player: Color,
    move_num: usize,
    flats_left: [usize; 2],
    caps_left: [usize; 2],
    pub bits: BitboardStorage<<Self as TakBoard>::Bits>,
    komi: u8,
    fifty_move: u8,
}

board_impl![Board5, Bitboard5, 5, 21, 1];
board_impl![Board6, Bitboard6, 6, 30, 1];
board_impl![Board7, Bitboard7, 7, 40, 2];

pub fn find_placement_road<T>(player: Color, road_pieces: T, empty: T) -> Option<GameMove>
where
    T: Bitboard,
{
    let valid = road_pieces.critical_squares() & empty;
    if valid != T::ZERO {
        let sq_index = valid.lowest_index();
        Some(GameMove::from_placement(Piece::flat(player), sq_index))
    } else {
        None
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

// fn piece_transform(v: crate::eval::ValidPiece)

impl Board6 {
    pub fn build_from_pieces(
        caps: [u8; 2],
        pieces: [crate::eval::PieceSquare; 62],
        white_to_move: bool,
    ) -> Self {
        // Caps: white, black
        // Pieces: [top, next, next, bottom, top ...]
        let mut board = Self::new();
        for piece in pieces.into_iter().rev().skip_while(|x| x.square() >= 36) {
            let add = piece.topaz_piece();
            board.board[piece.square() as usize].push(add, &mut board.bits);
            board.flats_left[piece.piece().color_index()] -= 1;
        }
        if caps[0] <= 35 {
            let color = board.board[caps[0] as usize].promote_top(&mut board.bits);
            board.caps_left[color as usize] -= 1;
            board.flats_left[color as usize] += 1;
        }
        if caps[1] <= 35 {
            let color = board.board[caps[1] as usize].promote_top(&mut board.bits);
            board.caps_left[color as usize] -= 1;
            board.flats_left[color as usize] += 1;
        }
        if white_to_move {
            board.active_player = Color::White;
        } else {
            board.active_player = Color::Black;
        }
        board.move_num = 4; // Assume we are past the swap phase
        let zobrist_hash = zobrist::TABLE.manual_build_hash(&board);
        board.bits.set_zobrist(zobrist_hash.0, zobrist_hash.1);
        board
    }
    #[cfg(feature = "random")]
    pub fn do_random_move<R: rand_core::RngCore>(&mut self, rng: &mut R) {
        let mut moves = Vec::new();
        generate_all_moves(self, &mut moves);
        let idx = rng.next_u32() as usize % moves.len();
        self.do_move(moves[idx]);
    }
}

#[cfg(test)]
impl Board6 {
    pub fn scan_active_stacks(&self, player: Color) -> Vec<usize> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, vec)| match vec.top() {
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
    pub fn wall_reveal_tak_moves() {
        let tps = "1,112C,11,11,x,1/x,121C,x2,21,1/1,2,x,12,1S,x/x,2,2,1221S,x,2/x3,121,x2/2,2,2,1,2,x 1 26";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let mut moves = Vec::new();
        generate_all_moves(&board, &mut moves);
        let rev = board.do_move(GameMove::try_from_ptn("2d6-11", &board).unwrap());
        board.null_move();
        let mut storage = Vec::new();
        assert!(board.can_make_road(&mut storage, None).is_some());
        board.rev_null_move();
        board.reverse_move(rev);
        let tak: Vec<_> = board
            .get_tak_threats(&moves, None)
            .into_iter()
            .map(|x| x.to_ptn::<Board6>())
            .collect();
        assert!(tak.contains(&"2d6-11".to_string()));
        dbg!(&tak);
    }
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
        b.tile_mut(3, 2).push(Piece::BlackFlat, &mut board.bits);
        b.tile_mut(3, 2).push(Piece::WhiteFlat, &mut board.bits);
        b.tile_mut(3, 2).push(Piece::WhiteWall, &mut board.bits);
        b.tile_mut(3, 5).push(Piece::BlackFlat, &mut board.bits);

        b.tile_mut(5, 1).push(Piece::WhiteFlat, &mut board.bits);
        b.tile_mut(5, 2).push(Piece::WhiteFlat, &mut board.bits);
        b.tile_mut(5, 3).push(Piece::BlackFlat, &mut board.bits);
        b.tile_mut(5, 4).push(Piece::BlackFlat, &mut board.bits);
        b.tile_mut(5, 5).push(Piece::WhiteFlat, &mut board.bits);
        assert_eq!(board.board, b.board);

        assert_eq!(board.scan_active_stacks(Color::White).len(), 4);
        assert_eq!(board.scan_active_stacks(Color::Black).len(), 5);
        assert_eq!(board.flat_winner(), GameResult::BlackWin);
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
        // let ptns = &["5d3-41*"];
        // for ptn in ptns {
        //     check_move(s, ptn, &mut board);
        // }
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

    #[test]
    pub fn board_fill() {
        let tps = "1,2,1,1,1,x/2,2,2,1,21,1/2,2,112C,21S,2,2/2,1,221C,1,12,212/1,1,1,2,1,2/2,2,1,1,2,112S 2 30";
        let mut board = Board6::try_from_tps(tps).unwrap();
        let res = crate::execute_moves_check_valid(&mut board, &["f6"]);
        assert!(res.is_ok());
        assert_eq!(board.flat_game(), Some(GameResult::BlackWin));
    }

    #[test]
    pub fn komi_flat_game() {
        let tps = "x2,2,1112,2,2/12,222121C,x,1S,221S,x/x2,21211112,21,2,2/x,1,21S,2,2,1S/1,21,1,1,1112C,1/1,21,2,2221S,21,221 1 58";
        let mut board = Board6::try_from_tps(tps).unwrap().with_komi(5);
        let res = crate::execute_moves_check_valid(&mut board, &["b4"]);
        assert!(res.is_ok());
        assert_eq!(board.flat_game(), Some(GameResult::BlackWin));
    }

    #[test]
    pub fn rotation() {
        let s = "2,1,1,1,1,2S/1,12,1,x2,111121C/x,2,2,212,2C,11121/2,2,1,1,12,2/x3,1,1,x/x2,2,21,x,112S 1 36";
        let board = Board6::try_from_tps(s).unwrap();
        let mut hash = vec![board.zobrist()];
        let mut rotated = board.clone();
        for _ in 0..4 {
            rotated = rotated.rotate();
            dbg!(&rotated);
            hash.push(rotated.hash());
        }
        dbg!(&hash);
        assert_eq!(hash[0], *hash.last().unwrap());
        hash.pop(); // Get rid of duplicate of last rotation
        for h in hash.iter().copied() {
            // All unique
            assert_eq!(hash.iter().filter(|&&x| x == h).count(), 1)
        }
        for board in board.symmetries() {
            dbg!(board);
        }
    }
}
