use super::{Bitboard, Board6, Piece, Stack};
use crate::{GameMove, RevGameMove};
use board_game_traits::{Color, GameResult, Position};

pub trait Evaluate: Position<Move = GameMove, ReverseMove = RevGameMove> {
    fn evaluate(&self) -> i32;
    fn hash(&self) -> u64;
    fn legal_move(&self, game_move: GameMove) -> bool;
    fn ply(&self) -> usize;
    fn null_move(&mut self);
    fn rev_null_move(&mut self);
}

fn win_color(res: GameResult) -> Option<Color> {
    match res {
        GameResult::WhiteWin => Some(Color::White),
        GameResult::BlackWin => Some(Color::Black),
        GameResult::Draw => None,
    }
}

pub const WIN_SCORE: i32 = 10_000;
pub const LOSE_SCORE: i32 = -1 * WIN_SCORE;

#[rustfmt::skip]
const LOCATION_WEIGHT: [i32; 36] = [
    00, 05, 05, 05, 05, 00,
    05, 10, 15, 15, 10, 05,
    05, 15, 20, 20, 15, 05,
    05, 15, 20, 20, 15, 05,
    05, 10, 15, 15, 10, 05,
    00, 05, 05, 05, 05, 00, 
];

fn piece_weight(p: Piece) -> i32 {
    match p {
        Piece::WhiteFlat | Piece::BlackFlat => 100,
        Piece::WhiteWall | Piece::BlackWall => 40,
        Piece::WhiteCap | Piece::BlackCap => 80,
    }
}

fn stack_top_multiplier(p: Piece) -> (i32, i32) {
    match p {
        Piece::WhiteFlat | Piece::BlackFlat => (-50, 60),
        Piece::WhiteWall | Piece::BlackWall => (-30, 70),
        Piece::WhiteCap | Piece::BlackCap => (-20, 90),
    }
}

impl Evaluate for Board6 {
    fn evaluate(&self) -> i32 {
        let mut score = 0;
        for (idx, stack) in self.board.iter().enumerate() {
            if stack.len() == 1 {
                let top = *stack.last().unwrap();
                let pw = piece_weight(top) + LOCATION_WEIGHT[idx];
                if let Color::White = top.owner() {
                    score += pw;
                } else {
                    score -= pw;
                }
            } else if stack.len() > 1 {
                let top = *stack.last().unwrap();
                let pw = piece_weight(top) + LOCATION_WEIGHT[idx];
                let (captive, friendly) = captive_friendly(&stack, top);
                let (c_mul, f_mul) = stack_top_multiplier(top);
                let stack_score = captive * c_mul + friendly * f_mul + pw;
                if let Color::White = top.owner() {
                    score += stack_score;
                } else {
                    score -= stack_score;
                }
            }
        }
        let white_connectivity = (self.bits.white.adjacent() & self.bits.white).pop_count();
        let black_connectivity = (self.bits.black.adjacent() & self.bits.black).pop_count();
        score += white_connectivity as i32 * 20;
        score -= black_connectivity as i32 * 20;
        if let Color::White = self.side_to_move() {
            score - 100
        } else {
            -1 * score
        }
    }
    fn hash(&self) -> u64 {
        self.zobrist()
    }
    fn legal_move(&self, game_move: GameMove) -> bool {
        let mut vec = Vec::new();
        self.generate_moves(&mut vec);
        vec.into_iter().find(|&m| m == game_move).is_some()
    }
    fn ply(&self) -> usize {
        match self.side_to_move() {
            Color::White => self.move_num() * 2,
            Color::Black => self.move_num() * 2 + 1,
        }
    }
    fn null_move(&mut self) {
        self.swap_active_player();
    }
    fn rev_null_move(&mut self) {
        self.swap_active_player();
    }
}

fn captive_friendly(stack: &Stack, top: Piece) -> (i32, i32) {
    let mut captive = 0;
    let mut friendly = 0;
    let controller = top.owner();
    for piece in stack.iter().rev().skip(1) {
        if piece.owner() == controller {
            friendly += 1;
        } else {
            captive += 1;
        }
    }
    (captive, friendly)
}
