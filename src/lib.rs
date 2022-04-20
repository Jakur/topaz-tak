#![allow(clippy::style)]

use anyhow::{anyhow, ensure, Result};
pub use board::{Bitboard, BitboardStorage, Piece, Stack, TakBoard};
pub use board_game_traits::{Color, GameResult};
pub use move_gen::{generate_all_moves, GameMove, RevGameMove};

pub mod board;
pub mod eval;
mod move_gen;
pub mod search;
pub mod transposition_table;

use crate::board::{Board5, Board6, Board7};

#[derive(Clone, Debug)]
pub struct TimeBank {
    pub goal_time: u64,
    pub max_time: u64,
}

impl TimeBank {
    pub fn init(total_res: u64, total_time: u64, increment: u64) -> Self {
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
        ensure!(
            moves.iter().find(|&&x| x == m).is_some(),
            "Illegal move attempted"
        );
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
