use super::{Bitboard, Piece, Stack};
use crate::board::BitIndexIterator;
use crate::board::Bitboard6;
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
    const TEMPO_OFFSET: i32 = 150;
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

impl Evaluator for Weights6 {
    type Game = Board6;
    #[inline(never)]
    fn evaluate(&self, game: &Self::Game, depth: usize) -> i32 {
        let mut score = 0;
        // let white_res = game.pieces_reserve(Color::White) as i32;
        // let black_res = game.pieces_reserve(Color::Black) as i32;
        // let mut flat_diff = (game.bits.flat & game.bits.white).pop_count() as i32
        //     - (game.bits.flat & game.bits.black).pop_count() as i32;
        // flat_diff *= 2;
        // flat_diff -= game.komi() as i32;
        // let white_res_lead = black_res - white_res;
        // let white_leading = white_res_lead >= 4 &&

        for (idx, stack) in game.board.iter().enumerate() {
            if stack.len() == 1 {
                let top = *stack.last().unwrap();
                let mut pw = self.piece_weight(top) + self.location[idx];
                if top.is_cap() {
                    pw += self.location[idx];
                }
                if let Color::White = top.owner() {
                    score += pw;
                } else {
                    score -= pw;
                }
            } else if stack.len() > 1 {
                let top = *stack.last().unwrap();
                let mut pw = self.piece_weight(top) + self.location[idx];
                let (mut captive, mut friendly) = captive_friendly(&stack, top);
                let (c_mul, f_mul) = self.stack_top_multiplier(top);
                let mut mobility = 0;
                let mut safety = 0;
                match top {
                    Piece::WhiteFlat | Piece::BlackFlat => {}
                    Piece::WhiteWall | Piece::BlackWall => {
                        safety += 16;
                    }
                    Piece::WhiteCap => {
                        safety += 64;
                        mobility += 1;
                        if let Some(Piece::WhiteFlat) = stack.from_top(1) {
                            pw += 30;
                        }
                    }
                    Piece::BlackCap => {
                        safety += 64;
                        mobility += 1;
                        if let Some(Piece::BlackFlat) = stack.from_top(1) {
                            pw += 30;
                        }
                    }
                }
                let neighbors = <Board6 as TakBoard>::Bits::index_to_bit(idx).adjacent();
                for sq in BitIndexIterator::new(neighbors) {
                    let stack = &game.board[sq];
                    if let Some(piece) = stack.last() {
                        if piece.owner() == top.owner() {
                            match piece {
                                Piece::WhiteFlat | Piece::BlackFlat => {
                                    safety += 1;
                                    mobility += 1;
                                }
                                Piece::WhiteWall | Piece::BlackWall => {
                                    if stack.len() < Self::Game::SIZE {
                                        safety += 4;
                                    }
                                }
                                Piece::WhiteCap | Piece::BlackCap => {
                                    if stack.len() < Self::Game::SIZE {
                                        safety += 32;
                                    }
                                    mobility += 1;
                                }
                            }
                        } else {
                            match piece {
                                Piece::WhiteFlat | Piece::BlackFlat => {
                                    mobility += 2;
                                }
                                Piece::WhiteWall | Piece::BlackWall => {
                                    if stack.len() < Self::Game::SIZE {
                                        safety -= 4;
                                    }
                                }
                                Piece::WhiteCap | Piece::BlackCap => {
                                    if stack.len() < Self::Game::SIZE {
                                        safety -= 32;
                                    }
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
                let friendly_score = friendly * f_mul;
                // let friendly_score = std::cmp::min(friendly * f_mul, 300);
                let stack_score = captive * c_mul + friendly_score + pw;
                if let Color::White = top.owner() {
                    score += stack_score;
                } else {
                    score -= stack_score;
                }
            }
        }
        let black_cap = game.bits.cap & game.bits.black;
        if black_cap != <Self::Game as TakBoard>::Bits::ZERO {
            // let black_c_lonely =
            //     black_cap.adjacent() & (game.bits.white & (game.bits.flat | game.bits.wall));
            let black_c_lonely = black_cap.adjacent() & (game.bits.flat | game.bits.wall);
            if black_c_lonely == <Self::Game as TakBoard>::Bits::ZERO {
                score += 30;
            }
        }
        let white_cap = game.bits.cap & game.bits.white;
        if white_cap != <Self::Game as TakBoard>::Bits::ZERO {
            // let white_c_lonely =
            //     white_cap.adjacent() & (game.bits.black & (game.bits.flat | game.bits.wall));
            let white_c_lonely = white_cap.adjacent() & (game.bits.flat | game.bits.wall);
            if white_c_lonely == <Self::Game as TakBoard>::Bits::ZERO {
                score -= 30;
            }
        }

        // // Danger FOR the associated color
        // const DANGER_MUL: i32 = 40; // 20
        // let white_danger = (game.bits.road_pieces(Color::Black).critical_squares()
        //     & !game.bits.empty())
        // .pop_count() >= 1;
        // let black_danger = (game.bits.road_pieces(Color::White).critical_squares()
        //     & !game.bits.empty())
        // .pop_count() >= 1;
        let white_comp =
            connected_components(game.bits.road_pieces(Color::White), Bitboard6::flood);
        let black_comp =
            connected_components(game.bits.road_pieces(Color::Black), Bitboard6::flood);
        // Punish more components?
        score -= white_comp.steps as i32 * self.connectivity;
        score += black_comp.steps as i32 * self.connectivity;

        // Examine the largest components
        // let mut white_road_pc = white_comp.bits.simple_road_est();
        // let mut black_road_pc = black_comp.bits.simple_road_est();
        let loose_white_comp =
            connected_components(game.bits.road_pieces(Color::White), Bitboard6::loose_flood);
        let loose_black_comp =
            connected_components(game.bits.road_pieces(Color::Black), Bitboard6::loose_flood);
        let loose_white_pc = loose_white_comp.bits.simple_road_est();
        let loose_black_pc = loose_black_comp.bits.simple_road_est();
        // score -= loose_white_comp.steps as i32 * self.connectivity;
        // score += loose_black_comp.steps as i32 * self.connectivity;
        // if loose_white > white_road_pc {
        //     white_road_pc += 1;
        // }
        // if loose_black > black_road_pc {
        //     black_road_pc += 1;
        // }
        let mut white_road_score = match loose_white_pc {
            6 => 50,
            5 => 35,
            4 => 20,
            3 => 10,
            _ => 0,
        };
        let mut black_road_score = match loose_black_pc {
            6 => 50,
            5 => 35,
            4 => 20,
            3 => 10,
            _ => 0,
        };
        if loose_white_pc > loose_black_pc {
            white_road_score *= 2;
        } else if loose_black_pc > loose_white_pc {
            black_road_score *= 2;
        } else {
            match game.side_to_move() {
                Color::White => white_road_score *= 2,
                Color::Black => black_road_score *= 2,
            }
        }
        let white_res = game.pieces_reserve(Color::White);
        let black_res = game.pieces_reserve(Color::Black);
        let pieces_left = white_res + black_res;
        let mul = (pieces_left as i32 * 100) / 60;

        score += (white_road_score * mul) / 100;
        score -= (black_road_score * mul) / 100;
        // let white_steps = simple_road_est::<Board6>(white_comp.bits);
        // let black_steps = simple_road_est::<Board6>(black_comp.bits);
        if white_res < 10 || black_res < 10 {
            // Half flats in white's favor
            let mut flat_diff = (game.bits.flat & game.bits.white).pop_count() as i32
                - (game.bits.flat & game.bits.black).pop_count() as i32;
            flat_diff *= 2;
            flat_diff -= game.komi() as i32;
            // TODO board fill considerations?
            if white_res < black_res {
                match game.side_to_move() {
                    Color::White => flat_diff += 2,
                    _ => {}
                }
            } else if black_res < white_res {
                match game.side_to_move() {
                    Color::Black => flat_diff -= 2,
                    _ => {}
                }
            }
            if flat_diff > 0 {
                score += (flat_diff * 100) / white_res as i32;
            } else if flat_diff < 0 {
                score += (flat_diff * 100) / black_res as i32;
            }
        }
        //     let res_adv = white_res - black_res;
        //     flat_diff += res_adv / 2;
        //     if res_adv > 0 {
        //         if flat_diff > 0 {
        //             if white_res < 5 {
        //                 score += flat_diff * 30;
        //             } else {
        //                 score += flat_diff * 20;
        //             }
        //         } else {
        //             score += flat_diff * 10;
        //         }
        //     } else {
        //         if flat_diff < 0 {
        //             if black_res < 5 {
        //                 score += flat_diff * 30;
        //             } else {
        //                 score += flat_diff * 20;
        //             }
        //         } else {
        //             score += flat_diff * 10;
        //         }
        //     }
        // }
        // let white_connectivity = (game.bits.white.adjacent() & game.bits.white).pop_count();
        // let black_connectivity = (game.bits.black.adjacent() & game.bits.black).pop_count();
        // score += white_connectivity as i32 * self.connectivity;
        // score -= black_connectivity as i32 * self.connectivity;
        if let Color::White = game.side_to_move() {
            // if white_danger {
            //     score += DANGER_MUL * 2;
            // }
            // if black_danger {
            //     score += DANGER_MUL;
            // }
            // if white_steps <= black_steps {
            //     score += 60;
            // } else {
            //     score -= 60;
            // }
            // score -= DANGER_MUL * white_danger;
            // score += DANGER_MUL * black_danger;
            if depth % 2 == 0 {
                score
            } else {
                score - self.tempo_offset
            }
        } else {
            // if white_danger {
            //     score += DANGER_MUL;
            // }
            // if black_danger {
            //     score += DANGER_MUL * 2;
            // }
            // if black_steps <= white_steps {
            //     score -= 60;
            // } else {
            //     score += 60;
            // }
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

struct BitOutcome<B> {
    bits: B,
    steps: usize,
}

impl<B> BitOutcome<B> {
    fn new(bits: B, steps: usize) -> Self {
        Self { bits, steps }
    }
}

// fn apply_bit_fn<B: Bitboard, F: Fn(B) -> B>(mut b: B, f: F) -> BitOutcome<B> {
//     let mut counter = 0;
//     let mut last = B::ZERO;
//     while b != last {
//         last = b;
//         b = f(b);
//         counter += 1;
//     }
//     BitOutcome::new(b, counter)
// }

fn repeat_slide<B: Bitboard, F: Fn(B) -> B>(mut b: B, f: F, end: B) -> BitOutcome<B> {
    let mut counter = 0;
    let mut last = B::ZERO;
    while b != last {
        if b & end != B::ZERO {
            break;
        }
        last = b;
        b = f(b);
        counter += 1;
    }
    BitOutcome::new(b, counter)
}

fn connected_components<B: Bitboard, F: Fn(B, B) -> B>(mut bits: B, flood_fn: F) -> BitOutcome<B> {
    let mut count = 0;
    let mut largest = B::ZERO;
    while bits != B::ZERO {
        let lowest = bits.lowest();
        let set = flood_fn(bits, lowest);
        if set.pop_count() > largest.pop_count() {
            largest = set;
        }
        bits = bits ^ set;
        count += 1;
    }
    BitOutcome::new(largest, count)
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

pub struct Weights6 {
    location: [i32; 36],
    connectivity: i32,
    tempo_offset: i32,
    piece: [i32; 3],
    stack_top: [i32; 6],
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
    pub fn add_noise(&mut self) {
        let mut seed: [u8; 32] = [0; 32];
        getrandom::getrandom(&mut seed).unwrap();
        let mut rng = Xoshiro256PlusPlus::from_seed(seed);
        let offset = rng.next_u32() % 11 + 10;
        let st_row = (rng.next_u32() % 6) as i32;
        let st_col = (rng.next_u32() % 6) as i32;
        // println!("{} @ ({}, {})", offset, st_col, st_row);
        for row in 0..6 {
            for col in 0..6 {
                let idx = row * 6 + col;
                let dist = (st_row - row).abs() + (st_col - col).abs();
                let delta = offset as i32 - dist * dist;
                if delta > 0 {
                    self.location[idx as usize] += delta;
                }
            }
        }
        // for r in 0..6 {
        //     println!("{:?}", &self.location[r * 6..(r + 1) * 6]);
        // }
        // for i in 0..self.location.len() {
        //     self.location[i] += (rng.next_u32() % 4) as i32;
        // }
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
    #[test]
    fn components() {
        let s = "2,x5/2,x,1,2,x2/x2,1,1,2C,x/x,2,2,1,x2/x3,1,x2/x5,1 1 7";
        let board = Board6::try_from_tps(s).unwrap();
        assert_eq!(
            2,
            connected_components(board.bits.road_pieces(Color::White), Bitboard6::flood).steps
        );
        assert_eq!(
            4,
            connected_components(board.bits.road_pieces(Color::Black), Bitboard6::flood).steps
        );
    }
}
