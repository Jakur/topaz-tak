#![allow(clippy::style)]

use anyhow::{anyhow, ensure, Result};
pub use board::{Bitboard, BitboardStorage, Piece, Stack, TakBoard};
pub use move_gen::{generate_all_moves, GameMove, RevGameMove};

pub mod board;
#[cfg(feature = "evaluation")]
pub mod eval;
pub mod move_gen;
pub mod proof;
#[cfg(feature = "evaluation")]
pub mod search;
#[cfg(feature = "evaluation")]
pub mod transposition_table;

use crate::board::{Board5, Board6, Board7};

#[cfg(feature = "evaluation")]
pub struct GameInitializer {
    pub hash_size: usize,
    pub max_depth: usize,
    pub komi: u8,
    pub add_noise: bool,
    pub num_threads: usize,
    pub use_nn: bool,
}

#[cfg(feature = "evaluation")]
impl GameInitializer {
    pub fn new(hash_size: usize, max_depth: usize, komi: u8, add_noise: bool) -> Self {
        Self {
            hash_size,
            max_depth,
            komi,
            add_noise,
            num_threads: 1,
            use_nn: false,
        }
    }
    pub fn get_board<E: eval::Evaluator + Default>(&self) -> (E::Game, E) {
        (E::Game::start_position().with_komi(self.komi), E::default())
    }
    pub fn get_board_tps<E: eval::Evaluator + Default>(&self, tps: &str) -> (E::Game, E) {
        (
            E::Game::try_from_tps(tps).unwrap().with_komi(self.komi),
            E::default(),
        )
    }
    pub fn small_clone(&self) -> Self {
        Self { ..*self }
    }
    // pub fn various_search(&mut self, g: TakGame) {
    //     match g {
    //         TakGame::Standard5(board) => {
    //             if self.num_threads == 1 {
    //                 if self.use_nn {
    //                     unimplemented!()
    //                 } else {
    //                     todo!()
    //                 }
    //             } else {
    //                 todo!()
    //             }
    //         }
    //         TakGame::Standard6(board) => todo!(),
    //         TakGame::Standard7(board) => todo!(),
    //     }
    // }
}

#[cfg(feature = "evaluation")]
impl std::default::Default for GameInitializer {
    fn default() -> Self {
        Self {
            hash_size: 2 << 22,
            max_depth: 128,
            komi: 0,
            use_nn: false,
            num_threads: 1,
            add_noise: false,
        }
    }
}

/// Represents a player's color.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl std::ops::Not for Color {
    type Output = Color;

    #[inline]
    fn not(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        fmt.write_str(match *self {
            Color::White => "White",
            Color::Black => "Black",
        })
    }
}

/// The result of a game after it has finished.
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum GameResult {
    WhiteWin = 0,
    BlackWin = 1,
    Draw = 2,
}

impl GameResult {
    /// Returns WhiteWin for white, BlackWin for black
    #[inline]
    pub fn win_by(color: Color) -> Self {
        match color {
            Color::White => Self::WhiteWin,
            Color::Black => Self::BlackWin,
        }
    }
}

impl std::ops::Not for GameResult {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        match self {
            GameResult::WhiteWin => GameResult::BlackWin,
            GameResult::BlackWin => GameResult::WhiteWin,
            GameResult::Draw => GameResult::Draw,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TimeBank {
    pub goal_time: u64,
    pub max_time: u64,
}

impl TimeBank {
    pub fn init(total_res: u64, total_time: u64, _increment: u64) -> Self {
        let use_time = if total_res > 40 {
            total_time / 15
        } else if total_res > 20 {
            total_time / 10
        } else {
            total_time / 6
        };
        Self {
            goal_time: use_time,
            max_time: use_time,
        }
    }
    pub fn flat(use_time: u64) -> Self {
        Self {
            goal_time: use_time,
            max_time: use_time,
        }
    }
}

pub trait Position {
    type Move: Eq + Clone + std::fmt::Debug;
    type ReverseMove;
    fn start_position() -> Self
    where
        Self: Sized;
    fn side_to_move(&self) -> Color;
    fn generate_moves(&self, moves: &mut Vec<Self::Move>);
    fn do_move(&mut self, mv: Self::Move) -> Self::ReverseMove;
    fn reverse_move(&mut self, mv: Self::ReverseMove);
    fn game_result(&self) -> Option<GameResult>;
}

#[non_exhaustive]
pub enum TakGame {
    Standard5(Board5),
    Standard6(Board6),
    Standard7(Board7),
}

impl TakGame {
    pub fn try_from_tps(tps: &str) -> Result<Self> {
        let size = tps.chars().filter(|&c| c == '/').count() + 1;
        match size {
            5 => Ok(TakGame::Standard5(Board5::try_from_tps(tps)?)),
            6 => Ok(TakGame::Standard6(Board6::try_from_tps(tps)?)),
            7 => Ok(TakGame::Standard7(Board7::try_from_tps(tps)?)),
            _ => Err(anyhow!("Unknown game size: {}", size)),
        }
    }
    pub fn try_new(size: usize) -> Result<Self> {
        match size {
            5 => Ok(TakGame::Standard5(Board5::new())),
            6 => Ok(TakGame::Standard6(Board6::new())),
            7 => Ok(TakGame::Standard7(Board7::new())),
            _ => Err(anyhow!("Unknown game size: {}", size)),
        }
    }
    pub fn try_new_with_komi(size: usize, half_flats: u8) -> Result<Self> {
        match size {
            5 => Ok(TakGame::Standard5(Board5::new().with_komi(half_flats))),
            6 => Ok(TakGame::Standard6(Board6::new().with_komi(half_flats))),
            7 => Ok(TakGame::Standard7(Board7::new().with_komi(half_flats))),
            _ => Err(anyhow!("Unknown game size: {}", size)),
        }
    }
}

impl Position for TakGame {
    type Move = GameMove;

    type ReverseMove = RevGameMove;

    fn start_position() -> Self
    where
        Self: Sized,
    {
        TakGame::Standard6(Board6::new())
    }

    fn side_to_move(&self) -> Color {
        match self {
            TakGame::Standard5(board) => board.side_to_move(),
            TakGame::Standard6(board) => board.side_to_move(),
            TakGame::Standard7(board) => board.side_to_move(),
        }
    }

    fn generate_moves(&self, moves: &mut Vec<Self::Move>) {
        match self {
            TakGame::Standard5(board) => board.generate_moves(moves),
            TakGame::Standard6(board) => board.generate_moves(moves),
            TakGame::Standard7(board) => board.generate_moves(moves),
        }
    }

    fn do_move(&mut self, mv: Self::Move) -> Self::ReverseMove {
        match self {
            TakGame::Standard5(board) => board.do_move(mv),
            TakGame::Standard6(board) => board.do_move(mv),
            TakGame::Standard7(board) => board.do_move(mv),
        }
    }

    fn reverse_move(&mut self, mv: Self::ReverseMove) {
        match self {
            TakGame::Standard5(board) => board.reverse_move(mv),
            TakGame::Standard6(board) => board.reverse_move(mv),
            TakGame::Standard7(board) => board.reverse_move(mv),
        }
    }

    fn game_result(&self) -> Option<GameResult> {
        match self {
            TakGame::Standard5(board) => board.game_result(),
            TakGame::Standard6(board) => board.game_result(),
            TakGame::Standard7(board) => board.game_result(),
        }
    }
}

#[derive(Debug)]
pub enum TeiCommand {
    Stop,
    Quit,
    Go(String),
    Position(String),
    NewGame(usize),
}

pub fn execute_moves_check_valid(board: &mut Board6, ptn_slice: &[&str]) -> Result<Vec<GameMove>> {
    let mut moves = Vec::new();
    let mut made_moves = Vec::new();
    for m_str in ptn_slice {
        moves.clear();
        let m =
            GameMove::try_from_ptn(m_str, board).ok_or_else(|| anyhow!("Invalid ptn string"))?;
        generate_all_moves(board, &mut moves);
        ensure!(moves.iter().any(|&x| x == m), "Illegal move attempted");
        board.do_move(m);
        made_moves.push(m);
    }
    Ok(made_moves)
}

pub fn perft<P: Position>(board: &mut P, depth: u16) -> u64 {
    if depth == 0 {
        1
    } else {
        let mut moves = vec![];
        board.generate_moves(&mut moves);
        moves
            .into_iter()
            .map(|mv| {
                let reverse_move = board.do_move(mv);
                let num_moves = perft(board, depth - 1);
                board.reverse_move(reverse_move);
                num_moves
            })
            .sum()
    }
}
