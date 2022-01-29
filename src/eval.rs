use super::{Bitboard, Piece, Stack};
use crate::board::BitIndexIterator;
use crate::board::Board6;
use crate::board::TakBoard;
use board_game_traits::{Color, Position};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub trait Evaluator {
    type Game: TakBoard;
    fn evaluate(&self, game: &Self::Game, depth: usize) -> i32;
}

pub struct Evaluator6 {}

impl Evaluator6 {
    const TEMPO_OFFSET: i32 = 50;
    const CONNECTIVITY: i32 = 20;
    const fn piece_weight(p: Piece) -> i32 {
        match p {
            Piece::WhiteFlat | Piece::BlackFlat => 100,
            Piece::WhiteWall | Piece::BlackWall => 40,
            Piece::WhiteCap | Piece::BlackCap => 80,
        }
    }
    const fn stack_top_multiplier(p: Piece) -> (i32, i32) {
        match p {
            Piece::WhiteFlat | Piece::BlackFlat => (-50, 60),
            Piece::WhiteWall | Piece::BlackWall => (-30, 70),
            Piece::WhiteCap | Piece::BlackCap => (-20, 90),
        }
    }
}

impl Evaluator for Evaluator6 {
    type Game = Board6;
    fn evaluate(&self, game: &Self::Game, depth: usize) -> i32 {
        let mut score = 0;
        for (idx, stack) in game.board.iter().enumerate() {
            if stack.len() == 1 {
                let top = *stack.last().unwrap();
                let pw = Self::piece_weight(top) + LOCATION_WEIGHT[idx];
                if let Color::White = top.owner() {
                    score += pw;
                } else {
                    score -= pw;
                }
            } else if stack.len() > 1 {
                let top = *stack.last().unwrap();
                let pw = Self::piece_weight(top) + LOCATION_WEIGHT[idx];
                let (captive, friendly) = captive_friendly(&stack, top);
                let (c_mul, f_mul) = Self::stack_top_multiplier(top);
                let stack_score = captive * c_mul + friendly * f_mul + pw;
                if let Color::White = top.owner() {
                    score += stack_score;
                } else {
                    score -= stack_score;
                }
            }
        }
        let white_connectivity = (game.bits.white.adjacent() & game.bits.white).pop_count();
        let black_connectivity = (game.bits.black.adjacent() & game.bits.black).pop_count();
        score += white_connectivity as i32 * Self::CONNECTIVITY;
        score -= black_connectivity as i32 * Self::CONNECTIVITY;
        if let Color::White = game.side_to_move() {
            if depth % 2 == 0 {
                score
            } else {
                score - Self::TEMPO_OFFSET
            }
        } else {
            if depth % 2 == 0 {
                -1 * score
            } else {
                -1 * score + Self::TEMPO_OFFSET
            }
        }
    }
}

pub const WIN_SCORE: i32 = 10_000;
pub const LOSE_SCORE: i32 = -1 * WIN_SCORE;

pub struct Weights6 {
    location: [i32; 36],
    connectivity: i32,
    tempo_offset: i32,
    piece: [i32; 3],
    stack_top: [i32; 6],
}

impl Weights6 {
    pub fn add_noise(&mut self) {
        let mut seed: [u8; 32] = [0; 32];
        getrandom::getrandom(&mut seed).unwrap();
        let mut rng = Xoshiro256PlusPlus::from_seed(seed);
        for i in 0..self.location.len() {
            self.location[i] += (rng.next_u32() % 4) as i32;
        }
    }
}

impl Evaluator for Weights6 {
    type Game = Board6;
    fn evaluate(&self, game: &Self::Game, depth: usize) -> i32 {
        let mut score = 0;
        for (idx, stack) in game.board.iter().enumerate() {
            if stack.len() == 1 {
                let top = *stack.last().unwrap();
                let pw = self.piece_weight(top) + self.location[idx];
                if let Color::White = top.owner() {
                    score += pw;
                } else {
                    score -= pw;
                }
            } else if stack.len() > 1 {
                let top = *stack.last().unwrap();
                let pw = self.piece_weight(top) + self.location[idx];
                let (mut captive, mut friendly) = captive_friendly(&stack, top);
                let (c_mul, f_mul) = self.stack_top_multiplier(top);
                let mut mobility = 0;
                let mut safety = 0;
                match top {
                    Piece::WhiteFlat | Piece::BlackFlat => {}
                    Piece::WhiteWall | Piece::BlackWall => {
                        safety += 4;
                    }
                    Piece::WhiteCap | Piece::BlackCap => {
                        safety += 32;
                        mobility += 1;
                    }
                }
                let neighbors = <Board6 as TakBoard>::Bits::index_to_bit(idx).adjacent();
                for sq in BitIndexIterator::new(neighbors) {
                    if let Some(piece) = game.board[sq].last() {
                        if piece.owner() == top.owner() {
                            match piece {
                                Piece::WhiteFlat | Piece::BlackFlat => {
                                    safety += 1;
                                    mobility += 1;
                                }
                                Piece::WhiteWall | Piece::BlackWall => {
                                    safety += 3;
                                }
                                Piece::WhiteCap | Piece::BlackCap => {
                                    safety += 6;
                                    mobility += 1;
                                }
                            }
                        } else {
                            match piece {
                                Piece::WhiteFlat | Piece::BlackFlat => {
                                    mobility += 2;
                                }
                                Piece::WhiteWall | Piece::BlackWall => {
                                    safety -= 4;
                                }
                                Piece::WhiteCap | Piece::BlackCap => {
                                    safety -= 8;
                                }
                            }
                        }
                    } else {
                        mobility += 2;
                    }
                }
                if mobility < 2 && !top.is_blocker() {
                    friendly /= 2;
                }
                if safety < 0 {
                    captive *= 2;
                }
                let stack_score = captive * c_mul + friendly * f_mul + pw;
                if let Color::White = top.owner() {
                    score += stack_score;
                } else {
                    score -= stack_score;
                }
            }
        }
        // Danger FOR the associated color
        // const DANGER_MUL: i32 = 20; // 20
        // let white_danger = (game.bits.road_pieces(Color::Black).critical_squares()
        //     & !game.bits.blocker_pieces(Color::White))
        // .pop_count() as i32;
        // let black_danger = (game.bits.road_pieces(Color::White).critical_squares()
        //     & !game.bits.blocker_pieces(Color::Black))
        // .pop_count() as i32;
        let white_connectivity = (game.bits.white.adjacent() & game.bits.white).pop_count();
        let black_connectivity = (game.bits.black.adjacent() & game.bits.black).pop_count();
        score += white_connectivity as i32 * self.connectivity;
        score -= black_connectivity as i32 * self.connectivity;
        if let Color::White = game.side_to_move() {
            // score -= DANGER_MUL * white_danger;
            // score += DANGER_MUL * black_danger;
            if depth % 2 == 0 {
                score
            } else {
                score - self.tempo_offset
            }
        } else {
            // score += DANGER_MUL * white_danger;
            // score -= DANGER_MUL * black_danger;
            if depth % 2 == 0 {
                -1 * score
            } else {
                -1 * score + self.tempo_offset
            }
        }
    }
}

fn local_influence(top: Piece) -> i32 {
    match top {
        Piece::WhiteFlat => 1,
        Piece::WhiteWall => 16,
        Piece::WhiteCap => 512,
        Piece::BlackFlat => -1,
        Piece::BlackWall => -16,
        Piece::BlackCap => -512,
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

impl Weights6 {
    pub fn new(
        location: [i32; 36],
        connectivity: i32,
        tempo_offset: i32,
        piece: [i32; 3],
        stack_top: [i32; 6],
    ) -> Self {
        Self {
            location,
            connectivity,
            tempo_offset,
            piece,
            stack_top,
        }
    }
    fn piece_weight(&self, p: Piece) -> i32 {
        match p {
            Piece::WhiteFlat | Piece::BlackFlat => self.piece[0],
            Piece::WhiteWall | Piece::BlackWall => self.piece[1],
            Piece::WhiteCap | Piece::BlackCap => self.piece[2],
        }
    }
    fn stack_top_multiplier(&self, p: Piece) -> (i32, i32) {
        match p {
            Piece::WhiteFlat | Piece::BlackFlat => (self.stack_top[0], self.stack_top[1]),
            Piece::WhiteWall | Piece::BlackWall => (self.stack_top[2], self.stack_top[3]),
            Piece::WhiteCap | Piece::BlackCap => (self.stack_top[4], self.stack_top[5]),
        }
    }
}

impl Default for Weights6 {
    fn default() -> Self {
        let piece_arr = [
            Evaluator6::piece_weight(Piece::WhiteFlat),
            Evaluator6::piece_weight(Piece::WhiteWall),
            Evaluator6::piece_weight(Piece::WhiteCap),
        ];
        let st1 = Evaluator6::stack_top_multiplier(Piece::WhiteFlat);
        let st2 = Evaluator6::stack_top_multiplier(Piece::WhiteWall);
        let st3 = Evaluator6::stack_top_multiplier(Piece::WhiteCap);
        Self::new(
            LOCATION_WEIGHT,
            Evaluator6::CONNECTIVITY,
            Evaluator6::TEMPO_OFFSET,
            piece_arr,
            [st1.0, st1.1, st2.0, st2.1, st3.0, st3.1],
        )
    }
}

// corner, offcorner, edge, edge, offcorner, corner,
// offcorner, kosumi, ctouch, ctouch, kosumi, corner,
// edge, ctouch, center, center, ctouch, edge,
// edge, ctouch, center, center, ctouch, edge,
// offcorner, kosumi, ctouch, ctouch, kosumi, corner,
// corner, offcorner, edge, edge, offcorner, corner,

#[rustfmt::skip]
const LOCATION_WEIGHT: [i32; 36] = [
    00, 05, 05, 05, 05, 00,
    05, 10, 15, 15, 10, 05,
    05, 15, 20, 20, 15, 05,
    05, 15, 20, 20, 15, 05,
    05, 10, 15, 15, 10, 05,
    00, 05, 05, 05, 05, 00, 
];

#[cfg(test)]
mod test {
    use super::*;
    use crate::generate_all_moves;
    #[test]
    fn check_tak_threats() {
        let s = "x2,2,x2,1/x5,1/x,2,x,1,1,1/x,2,x2,1,x/x,2C,x4/x,2,x4 2 6";
        let mut board = Board6::try_from_tps(s).unwrap();
        let mut moves = Vec::new();
        generate_all_moves(&mut board, &mut moves);
        let tak_threats = board.get_tak_threats(&moves, None);
        for m in tak_threats.iter() {
            dbg!(m.to_ptn::<Board6>());
        }
        assert_eq!(tak_threats.len(), 5);
    }
}
