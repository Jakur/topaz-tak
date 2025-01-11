use super::{Bitboard, Piece, Stack};
use crate::board::BitIndexIterator;
use crate::board::TakBoard;
use crate::board::{Board5, Board6};
use crate::Color;
use crate::Position;

use incremental::BoardData;
use incremental::PieceSquare;
pub use incremental::NNUE6;
// static NN6: once_cell::sync::OnceCell<incremental::Weights> = once_cell::sync::OnceCell::new();

pub fn global_init_weights(f_name: &str) {
    // println!("Setting global weights!");
    // let weights = incremental::Weights::from_file(File::open(f_name).unwrap()).unwrap();
    // NN6.set(weights)
    //     .ok()
    //     .expect("Unable to override existing NN6 Weights!");
}

fn nn_file() -> Option<String> {
    let _ = dotenv::dotenv();
    let mut path = None;
    for (key, value) in std::env::vars() {
        if key == "NN" {
            path = Some(value);
        }
    }
    path
}

#[allow(dead_code)]
mod incremental;
mod smooth;
// pub use smooth::SmoothWeights6;

pub trait Evaluator {
    type Game: TakBoard + Send;
    fn evaluate(&mut self, game: &Self::Game, depth: usize) -> i32;
    fn is_quiet(&self, game: &Self::Game) -> bool;
}

impl Evaluator for NNUE6 {
    type Game = Board6;

    fn evaluate(&mut self, game: &Self::Game, depth: usize) -> i32 {
        const TEMPO_OFFSET: i32 = 0; // -100

        // Build TakBoard abstraction
        let mut caps = [255, 255];
        let mut data = [PieceSquare(255); 62];
        let mut idx = 0;
        let mut cap_idx = 0;
        for sq in game
            .active_stacks(Color::White)
            .chain(game.active_stacks(Color::Black))
        {
            let stack = &game.board()[sq];
            let top = stack.top().unwrap();
            if top.is_cap() {
                caps[cap_idx] = sq as u8;
                cap_idx += 1;
            }
            data[idx] = build_piece_square(sq, top);
            idx += 1;
            for i in 1..10 {
                if let Some(p) = stack.from_top(i) {
                    data[idx] = build_piece_square(sq, p);
                    idx += 1;
                } else {
                    break;
                }
            }
        }
        let takboard = BoardData::minimal(caps, data, game.side_to_move() == Color::White);
        let eval = self.incremental_eval(takboard);
        // Evaluate
        // let eval =
        //     self.incremental_eval(takboard) / 4 + self.classical.evaluate(game, depth) * 3 / 4;
        if depth % 2 == 0 {
            eval
        } else {
            eval - TEMPO_OFFSET
        }
    }

    fn is_quiet(&self, game: &Self::Game) -> bool {
        true
    }
}

fn build_piece_square(sq: usize, piece: Piece) -> PieceSquare {
    let p = match piece {
        Piece::WhiteFlat => incremental::WHITE_FLAT,
        Piece::WhiteWall => incremental::WHITE_WALL,
        Piece::WhiteCap => incremental::WHITE_FLAT, // This is intentional
        Piece::BlackFlat => incremental::BLACK_FLAT,
        Piece::BlackWall => incremental::BLACK_WALL,
        Piece::BlackCap => incremental::BLACK_FLAT, // This is intentional
    };
    PieceSquare::new(sq, p.0)
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

pub const WIN_SCORE: i32 = 10_000;
pub const LOSE_SCORE: i32 = -1 * WIN_SCORE;

fn endgame_calc<T: TakBoard>(game: &T) -> (i32, i32) {
    let empty_count = game.bits().empty().pop_count() as usize;
    let white_res = game.pieces_reserve(Color::White) + game.caps_reserve(Color::White);
    let black_res = game.pieces_reserve(Color::Black) + game.caps_reserve(Color::Black);
    // Half flats in white's favor
    let mut flat_diff = (game.bits().flat & game.bits().white).pop_count() as i32
        - (game.bits().flat & game.bits().black).pop_count() as i32;
    flat_diff *= 2;
    flat_diff -= game.komi() as i32;
    // TODO board fill considerations?
    let (white_fill, black_fill) = if empty_count % 2 == 0 {
        (empty_count / 2, empty_count / 2)
    } else {
        match game.side_to_move() {
            Color::White => (1 + empty_count / 2, empty_count / 2),
            Color::Black => (empty_count / 2, 1 + empty_count / 2),
        }
    };
    // Todo checkout this side to move business
    if white_res < black_res {
        if white_fill < white_res {
            if empty_count % 2 == 1 {
                flat_diff += 2;
            }
        } else {
            match game.side_to_move() {
                Color::White => flat_diff += 2,
                _ => {}
            }
        }
    } else if black_res < white_res {
        if black_fill < black_res {
            if empty_count % 2 == 1 {
                flat_diff -= 2;
            }
        } else {
            match game.side_to_move() {
                Color::Black => flat_diff -= 2,
                _ => {}
            }
        }
    } else {
        match game.side_to_move() {
            Color::White => {
                if white_fill < white_res {
                    if empty_count % 2 == 1 {
                        flat_diff += 2;
                    }
                } else {
                    flat_diff += 2;
                }
            }
            Color::Black => {
                if black_fill < black_res {
                    if empty_count % 2 == 1 {
                        flat_diff -= 2;
                    }
                } else {
                    flat_diff -= 2;
                }
            }
        }
    }
    let divisor = if flat_diff > 0 {
        std::cmp::min(white_res, white_fill + 1) as i32
    } else {
        std::cmp::min(black_res, black_fill + 1) as i32
    };
    return (flat_diff, divisor);
}

macro_rules! eval_impl {
    ($board: ty, $weights: ty) => {
        impl $weights {
            const SIZE: usize = <Self as Evaluator>::Game::SIZE;
            fn eval_stack(
                &self,
                game: &<Self as Evaluator>::Game,
                idx: usize,
                stack: &Stack,
            ) -> (i32, bool) {
                let top = stack.top().unwrap();
                let mut pw = self.piece_weight(top) + self.location[idx];
                // let (mut captive, mut friendly) = stack.captive_friendly();
                let (mut captive, mut friendly) = stack.limited_captive_friendly();
                // let (c_mul, f_mul) = self.stack_top_multiplier(top);
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
                        if let Some(Piece::WhiteFlat) = stack.under_top() {
                            pw += 30;
                        }
                    }
                    Piece::BlackCap => {
                        safety += 64;
                        mobility += 1;
                        if let Some(Piece::BlackFlat) = stack.under_top() {
                            pw += 30;
                        }
                    }
                }
                let neighbors =
                    <<Self as Evaluator>::Game as TakBoard>::Bits::index_to_bit(idx).adjacent();
                for sq in BitIndexIterator::new(neighbors) {
                    let stack = &game.board[sq];
                    if let Some(piece) = stack.top() {
                        if piece.owner() == top.owner() {
                            match piece {
                                Piece::WhiteFlat | Piece::BlackFlat => {
                                    safety += 1;
                                    mobility += 1;
                                }
                                Piece::WhiteWall | Piece::BlackWall => {
                                    if stack.len() < Self::SIZE {
                                        safety += 4;
                                    }
                                }
                                Piece::WhiteCap | Piece::BlackCap => {
                                    if stack.len() < Self::SIZE {
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
                                    if stack.len() < Self::SIZE {
                                        safety -= 4;
                                    }
                                }
                                Piece::WhiteCap | Piece::BlackCap => {
                                    if stack.len() < Self::SIZE {
                                        safety -= 32;
                                    }
                                }
                            }
                        }
                    } else {
                        mobility += 2;
                    }
                }
                // Todo reconsider these with new method?
                if mobility < 2 && !top.is_blocker() {
                    friendly /= 2;
                }
                if safety < 0 {
                    captive *= 2;
                }
                let mut stack_score =
                    pw + self.stack_eval.eval_cap_friendly(top, captive, friendly);
                // let friendly_flats = game.bits().flat & game.bits().all_pieces(top.owner());
                let enemy_flats = game.bits().flat & game.bits().all_pieces(!top.owner());
                let orth = <<Self as Evaluator>::Game as TakBoard>::Bits::orthogonal(idx);
                let count = (orth & enemy_flats).pop_count() as i32;
                let all_friendly = friendly + 1;
                if count >= all_friendly {
                    stack_score += std::cmp::min(75, 5 * all_friendly * all_friendly);
                }
                // let friendly_score = friendly * f_mul;
                // // let friendly_score = std::cmp::min(friendly * f_mul, 300);
                // let stack_score = captive * c_mul + friendly_score + pw;
                let has_weakness = captive > 0 && top.is_flat();
                if let Color::White = top.owner() {
                    return (stack_score, has_weakness);
                } else {
                    return (-stack_score, has_weakness);
                }
            }
        }
        impl Evaluator for $weights {
            type Game = $board;
            #[inline(never)]
            fn evaluate(&mut self, game: &Self::Game, depth: usize) -> i32 {
                let mut score = 0;
                let white_r = game.bits.road_pieces(Color::White);
                let black_r = game.bits.road_pieces(Color::Black);
                let white_cs = white_r.critical_squares();
                let black_cs = black_r.critical_squares();
                // let mut white_weakness = 0;
                // let mut black_weakness = 0;
                match game.side_to_move() {
                    Color::White => {
                        if (white_cs & game.bits.empty()).nonzero() {
                            return WIN_SCORE - depth as i32 - 1;
                        }
                    }
                    Color::Black => {
                        if (black_cs & game.bits.empty()).nonzero() {
                            return WIN_SCORE - depth as i32 - 1;
                        }
                    }
                }
                // let blockers =
                //     game.bits.blocker_pieces(Color::White) | game.bits.blocker_pieces(Color::Black);
                for (idx, stack) in game.board.iter().enumerate() {
                    // if stack.len() != 0 {
                    //     let orth_bits = <Self::Game as TakBoard>::Bits::orthogonal(idx);
                    //     let white_c = (white_r & orth_bits).pop_count() as i32;
                    //     let black_c = (black_r & orth_bits).pop_count() as i32;
                    //     let coherence_bonus = ((white_c * white_c) - (black_c * black_c))
                    //         / (1 + (blockers & orth_bits).pop_count() as i32);
                    //     score += coherence_bonus / 2;
                    // }
                    if stack.len() == 1 {
                        let top = stack.top().unwrap();
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
                        let (s, weakness) = self.eval_stack(game, idx, stack);
                        // let top = stack.top().unwrap();
                        // if top.owner() == Color::White {
                        //     white_weakness += weakness as i32;
                        // } else {
                        //     black_weakness += weakness as i32;
                        // }
                        score += s;
                    }
                }
                let black_cap = game.bits.cap & game.bits.black;
                if black_cap != <Self::Game as TakBoard>::Bits::ZERO {
                    let black_c_lonely =
                        black_cap.adjacent() & (game.bits.flat | game.bits.wall) & game.bits.white;
                    if black_c_lonely == <Self::Game as TakBoard>::Bits::ZERO {
                        score += 30;
                    }
                }
                let white_cap = game.bits.cap & game.bits.white;
                if white_cap != <Self::Game as TakBoard>::Bits::ZERO {
                    let white_c_lonely =
                        white_cap.adjacent() & (game.bits.flat | game.bits.wall) & game.bits.black;
                    if white_c_lonely == <Self::Game as TakBoard>::Bits::ZERO {
                        score -= 30;
                    }
                }
                // if white_weakness > black_weakness {
                //     score -= 40;
                // } else if black_weakness > white_weakness {
                //     score += 40;
                // }
                score += self.cs_threat * attackable_cs(Color::White, white_cs, game);
                score -= self.cs_threat * attackable_cs(Color::Black, black_cs, game);
                let (loose_white_pc, white_comp) =
                    flat_placement_road_short(white_r, game.bits.empty());
                let (loose_black_pc, black_comp) =
                    flat_placement_road_short(black_r, game.bits.empty());
                score -= white_comp as i32 * self.connectivity;
                score += black_comp as i32 * self.connectivity;

                let mut white_road_score = match loose_white_pc {
                    1 => self.flat_road[0],
                    2 => self.flat_road[1],
                    3 => self.flat_road[2],
                    4 => self.flat_road[3],
                    _ => 0,
                };
                let mut black_road_score = match loose_black_pc {
                    1 => self.flat_road[0],
                    2 => self.flat_road[1],
                    3 => self.flat_road[2],
                    4 => self.flat_road[3],
                    _ => 0,
                };

                if white_road_score > black_road_score {
                    white_road_score *= 2;
                } else if black_road_score > white_road_score {
                    black_road_score *= 2;
                } else {
                    match game.side_to_move() {
                        Color::White => white_road_score *= 2,
                        Color::Black => black_road_score *= 2,
                    }
                }
                let white_res = game.pieces_reserve(Color::White) + game.caps_reserve(Color::White);
                let black_res = game.pieces_reserve(Color::Black) + game.caps_reserve(Color::Black);
                let pieces_left = white_res + black_res;
                let mul = (pieces_left as i32 * 100) / 60;

                score += (white_road_score * mul) / 100;
                score -= (black_road_score * mul) / 100;
                // let empty_count = game.bits().empty().pop_count() as usize;
                let (mut flat_diff, divisor) = endgame_calc(game); // Note flat diff is in half-flats
                debug_assert!(flat_diff % 2 == 0);
                // Endgame fcd
                score += (flat_diff * 100) / divisor;
                // Early game fcd
                if let Color::Black = game.side_to_move() {
                    flat_diff *= -1;
                }
                let flat_diff_idx = (5 + (flat_diff / 2).clamp(-5, 5)) as usize;
                score += (self.flat_advantage[flat_diff_idx] * divisor) / 100;
                let white_res_adv = (white_res as i32 - black_res as i32).clamp(-5, 5);
                let reserve_idx = if let Color::White = game.side_to_move() {
                    (5 + white_res_adv) as usize
                } else {
                    (5 - white_res_adv) as usize
                };
                score += self.reserve_diff[reserve_idx];

                if let Color::White = game.side_to_move() {
                    if depth % 2 == 1 {
                        score + self.tempo_offset
                    } else {
                        score // self.tempo_offset
                    }
                } else {
                    if depth % 2 == 1 {
                        -1 * score + self.tempo_offset
                    } else {
                        -1 * score // + self.tempo_offset
                    }
                }
                // if Self::Game::SIZE == 10 && score.abs() < 350 {
                //     let eval_comp = self.eval_components(game);
                //     let pred_delta = self.nn.forward(SmallNN::prepare_data(eval_comp.data));
                //     score + pred_delta
                // } else {
                //     score
                // }
            }
            fn is_quiet(&self, game: &Self::Game) -> bool {
                true
            }
        }
    };
}

fn attackable_cs<B: TakBoard>(color: Color, cs: B::Bits, game: &B) -> i32 {
    let road_pieces = game.bits().road_pieces(color);
    let mut cs = cs & !game.bits().blocker_pieces(!color);
    let mut count = 0;
    while cs != B::Bits::ZERO {
        let lowest = cs.pop_lowest();
        if (lowest.adjacent() & road_pieces).pop_count() >= 3 {
            count += 1;
        }
    }
    count
}

// eval_impl![crate::Board7, Weights7];
eval_impl![Board6, Weights6];
eval_impl![Board5, Weights5];

#[cfg(test)]
struct BitOutcome<B> {
    bits: B,
    steps: usize,
}

#[cfg(test)]
impl<B> BitOutcome<B> {
    fn new(bits: B, steps: usize) -> Self {
        Self { bits, steps }
    }
}

/// A method that finds the most complete road among orthogonally connected components.
/// It allows one non-blocking piece to be overwritten for free, leading to decreased
/// concern for blockades, and extending roads that seem unlikely to ever be finished
#[cfg(test)]
fn one_gap_road_old<B: Bitboard + std::fmt::Debug>(road_bits: B, blocker_bits: B) -> (i32, usize) {
    let mut iter = ComponentIterator::new(road_bits);
    let mut best = 0;
    let mut count = 0;
    while let Some(comp) = iter.next() {
        count += 1;
        if comp.pop_count() >= 3 {
            let (score_a, score_b) = comp.simple_road_est();
            let score = std::cmp::max(score_a, score_b);
            if score > best {
                best = score;
            }
            let adj = comp.adjacent() & !comp & !blocker_bits;
            let rest = road_bits & !comp;
            let mut potential = adj & rest.adjacent();
            while potential != B::ZERO {
                let lowest = potential.pop_lowest();
                let check = (lowest | road_bits).flood(lowest);
                let (score_a, score_b) = check.simple_road_est();
                let score = std::cmp::max(score_a, score_b);
                if score > best {
                    best = score;
                }
            }
        }
    }
    (best, count)
}

/// A heuristic for road building that approximates the number of flat placements
/// needed to make a road. In a future version, we should link together disjoint
/// components during the search to make the counts more accurate. Compared to
/// earlier approaches, this more accurately values row blockades, but it lacks
/// a way to see threats for a single decisive capture
fn flat_placement_road_short<B: Bitboard + std::fmt::Debug>(
    road_bits: B,
    empty: B,
) -> (i32, usize) {
    let mut iter = ComponentIterator::new(road_bits);
    let mut best = 1_000;
    let mut count = 0;
    let north = road_bits.flood(B::top()) | B::top();
    let south = road_bits.flood(B::bottom()) | B::bottom();
    let east = road_bits.flood(B::right()) | B::right();
    let west = road_bits.flood(B::left()) | B::left();
    let safe = empty;
    while let Some(comp) = iter.next() {
        count += 1;
        if comp.pop_count() >= 2 {
            let mut steps = 0;
            let mut n_steps = 1_000;
            let mut s_steps = 1_000;
            let mut e_steps = 1_000;
            let mut w_steps = 1_000;
            let mut explored = comp;
            let mut prev = B::ZERO;
            let mut other_neighbor = (!comp & road_bits).adjacent() & empty;
            while prev != explored {
                if (explored & north) != B::ZERO {
                    n_steps = std::cmp::min(n_steps, steps);
                }
                if (explored & south) != B::ZERO {
                    s_steps = std::cmp::min(s_steps, steps);
                }
                if (explored & east) != B::ZERO {
                    e_steps = std::cmp::min(e_steps, steps);
                }
                if (explored & west) != B::ZERO {
                    w_steps = std::cmp::min(w_steps, steps);
                }
                prev = explored;
                explored = (explored.adjacent() | explored) & safe;
                steps += 1;
                if steps > 2 {
                    break;
                }
                if (other_neighbor & explored) != B::ZERO {
                    // Combine components
                    explored = (road_bits | explored).flood(explored);
                    other_neighbor = (!explored & road_bits).adjacent() & empty;
                    // other_neighbor = other_neighbor & !explored;
                }
            }
            let score = std::cmp::min(n_steps + s_steps, e_steps + w_steps);
            best = std::cmp::min(best, score);
        }
    }
    (best, count)
}

struct ComponentIterator<B> {
    bits: B,
}

impl<B> ComponentIterator<B> {
    fn new(bits: B) -> Self {
        Self { bits }
    }
}

impl<B> Iterator for ComponentIterator<B>
where
    B: Bitboard,
{
    type Item = B;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bits == B::ZERO {
            None
        } else {
            let lowest = self.bits.lowest();
            let set = B::flood(self.bits, lowest);
            self.bits = self.bits ^ set;
            Some(set)
        }
    }
}

// struct LooseComponentIterator<B> {
//     bits: B,
// }

// impl<B> LooseComponentIterator<B> {
//     fn new(bits: B) -> Self {
//         Self { bits }
//     }
// }

// impl<B> Iterator for LooseComponentIterator<B>
// where
//     B: Bitboard,
// {
//     type Item = B;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.bits == B::ZERO {
//             None
//         } else {
//             let lowest = self.bits.lowest();
//             let set = B::loose_flood(self.bits, lowest);
//             self.bits = self.bits ^ set;
//             Some(set)
//         }
//     }
// }

#[cfg(test)]
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

// #[derive(Default, Debug)]
// struct MomentumCounter {
//     white_road_diff: i32,
//     black_road_diff: i32,
// }

pub struct Weights5 {
    location: [i32; 25],
    connectivity: i32,
    tempo_offset: i32,
    piece: [i32; 3],
    stack_eval: StackEval,
    flat_road: [i32; 4],
    cs_threat: i32,
    flat_advantage: [i32; 11],
    reserve_diff: [i32; 11],
}

impl Weights5 {
    pub fn new(
        location: [i32; 25],
        connectivity: i32,
        tempo_offset: i32,
        piece: [i32; 3],
        stack_top: [i32; 6],
        flat_road: [i32; 4],
        cs_threat: i32,
    ) -> Self {
        Self {
            location,
            connectivity,
            tempo_offset,
            piece,
            stack_eval: StackEval::build_simple(stack_top),
            flat_road,
            cs_threat,
            flat_advantage: [0; 11],
            reserve_diff: [0; 11],
        }
    }
    fn piece_weight(&self, p: Piece) -> i32 {
        match p {
            Piece::WhiteFlat | Piece::BlackFlat => self.piece[0],
            Piece::WhiteWall | Piece::BlackWall => self.piece[1],
            Piece::WhiteCap | Piece::BlackCap => self.piece[2],
        }
    }
    // fn stack_top_multiplier(&self, p: Piece) -> (i32, i32) {
    //     match p {
    //         Piece::WhiteFlat | Piece::BlackFlat => (self.stack_top[0], self.stack_top[1]),
    //         Piece::WhiteWall | Piece::BlackWall => (self.stack_top[2], self.stack_top[3]),
    //         Piece::WhiteCap | Piece::BlackCap => (self.stack_top[4], self.stack_top[5]),
    //     }
    // }
}

impl Default for Weights5 {
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
            LOCATION_WEIGHT5,
            Evaluator6::CONNECTIVITY,
            Evaluator6::TEMPO_OFFSET,
            piece_arr,
            [st1.0, st1.1, st2.0, st2.1, st3.0, st3.1],
            [40, 30, 15, 5],
            50,
        )
    }
}

#[derive(Debug)]
pub struct StackEval {
    stack_top: [i32; 6],
    flat: [i32; 36],
    wall: [i32; 36],
    cap: [i32; 36],
}

impl StackEval {
    pub fn new(stack_top: [i32; 6], flat: [i32; 36], wall: [i32; 36], cap: [i32; 36]) -> Self {
        Self {
            stack_top,
            flat,
            wall,
            cap,
        }
    }
    /// Construct a new StackEval table by beginning with default valuess
    pub fn build_simple(stack_top: [i32; 6]) -> Self {
        let mut flat = [0; 36];
        let mut wall = [0; 36];
        let mut cap = [0; 36];
        Self::fill(&mut flat, stack_top[0], stack_top[1]);
        Self::fill(&mut wall, stack_top[2], stack_top[3]);
        Self::fill(&mut cap, stack_top[4], stack_top[5]);
        Self {
            stack_top,
            flat,
            wall,
            cap,
        }
    }
    /// Set a special case in the value table
    pub fn set_cap_friendly(&mut self, top: Piece, captive: usize, friendly: usize, value: i32) {
        let idx = captive * 6 + friendly;
        match top {
            Piece::WhiteFlat | Piece::BlackFlat => {
                self.flat[idx] = value;
            }
            Piece::WhiteWall | Piece::BlackWall => {
                self.wall[idx] = value;
            }
            Piece::WhiteCap | Piece::BlackCap => {
                self.cap[idx] = value;
            }
        }
    }
    fn fill(arr: &mut [i32; 36], cap_mul: i32, f_mul: i32) {
        for idx in 1..36 {
            let cap = idx / 6;
            let friendly = idx % 6;
            arr[idx] = cap_mul * (cap as i32) + f_mul * (friendly as i32);
        }
    }
    /// Evaluate the side-agnostic value of a stack according to its captive and friendly count
    pub fn eval_cap_friendly(&self, top: Piece, captive: i32, friendly: i32) -> i32 {
        if captive > 5 || friendly > 5 {
            if top.is_flat() {
                return self.stack_top[0] * captive + self.stack_top[1] * friendly;
            } else if top.is_wall() {
                return self.stack_top[2] * captive + self.stack_top[3] * friendly;
            } else {
                return self.stack_top[4] * captive + self.stack_top[5] * friendly;
            }
        } else {
            let captive = captive as usize;
            let friendly = friendly as usize;
            if top.is_flat() {
                return self.flat[captive * 6 + friendly];
            } else if top.is_wall() {
                return self.wall[captive * 6 + friendly];
            } else {
                return self.cap[captive * 6 + friendly];
            }
        }
    }
}

#[derive(Debug)]
pub struct Weights6 {
    location: [i32; 36],
    connectivity: i32,
    tempo_offset: i32,
    piece: [i32; 3],
    pub stack_eval: StackEval,
    // stack_top: [i32; 6],
    flat_road: [i32; 4],
    cs_threat: i32,
    flat_advantage: [i32; 11],
    reserve_diff: [i32; 11],
}

impl Weights6 {
    pub fn new(
        location: [i32; 36],
        connectivity: i32,
        tempo_offset: i32,
        piece: [i32; 3],
        stack_top: [i32; 6],
        flat_road: [i32; 4],
        cs_threat: i32,
        flat_advantage: [i32; 11],
        reserve_diff: [i32; 11],
    ) -> Self {
        Self {
            location,
            connectivity,
            tempo_offset,
            piece,
            stack_eval: StackEval::build_simple(stack_top),
            flat_road,
            cs_threat,
            flat_advantage,
            reserve_diff,
        }
    }

    pub fn get_genome(&self) -> Vec<i32> {
        let loc = &self.location;
        let mut out = vec![loc[0], loc[1], loc[2], loc[7], loc[8], loc[14]];
        out.push(self.connectivity);
        for x in self.piece.iter() {
            out.push(*x);
        }
        for x in self.stack_eval.stack_top.iter() {
            out.push(*x);
        }
        for top in [Piece::WhiteFlat, Piece::WhiteWall, Piece::WhiteCap].into_iter() {
            for cap in 1..=5 {
                for friendly in 0..=5 - cap {
                    out.push(self.stack_eval.eval_cap_friendly(top, cap, friendly));
                }
            }
        }
        for x in self.flat_road.iter() {
            out.push(*x);
        }
        out.push(self.cs_threat);
        out
    }

    #[cfg(feature = "random")]
    pub fn add_noise(&mut self) {
        use rand_core::{RngCore, SeedableRng};
        let mut seed: [u8; 32] = [0; 32];
        getrandom::getrandom(&mut seed).unwrap();
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::from_seed(seed);
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
    // fn stack_top_multiplier(&self, p: Piece) -> (i32, i32) {
    //     match p {
    //         Piece::WhiteFlat | Piece::BlackFlat => (self.stack_top[0], self.stack_top[1]),
    //         Piece::WhiteWall | Piece::BlackWall => (self.stack_top[2], self.stack_top[3]),
    //         Piece::WhiteCap | Piece::BlackCap => (self.stack_top[4], self.stack_top[5]),
    //     }
    // }
}

impl Default for Weights6 {
    fn default() -> Self {
        // let data = vec![
        //     -29, 9, 29, 55, 26, 84, 74, 84, 83, 39, 46, 33, -13, 188, 85, 178, 159, 68, 181, 161,
        //     198,
        // ];
        // let mut data = vec![
        //     38, 54, 72, 79, 93, 69, 29, 161, 54, 48, 137, 159, 90, 120, 84, 157, 156, 111, 65, 10,
        //     82, 118, 61, 94, 57, 128, 5, 30, 46, 74, 156,
        // ];
        // let mut data = vec![
        //     9, 53, 77, 76, 96, 86, 19, 162, 24, 54, 138, 163, 64, 161, 18, 103, 127, 92, 39, 8, 89,
        //     83, 25, 59, 81, 98, 11, 25, 86, 143, 108,
        // ];
        // for x in data.iter_mut() {
        //     *x = (*x) / 2;
        // }
        // let data = vec![
        //     0, 5, 9, 16, 21, 25, 11, 114, 57, 95, 69, 97, 35, 122, 22, 144, 50, 19, 8, 2, 58, 77,
        //     8, 95, 0, 5, 4, 2, 2, 0, 10,
        // ];
        let data = vec![
            0, 4, 9, 14, 19, 23, 11, 114, 58, 97, 68, 98, 34, 120, 23, 142, 49, 23, 9, 2, 61, 32,
            17, 72, 2, 84, 2, 4, 5, 6, 8,
        ];
        // let data = vec![
        //     145, 163, 163, 161, 135, 102, 126, 163, 0, 1, 160, 161, 38, 160, 9, 163, 90, 161, 150,
        //     138, 152, 13, 8, 66, 37, 48, 93, 0, 4, 15, 0,
        // ];
        build_weights(&data)
        // Weights6 {
        //     location: [
        //         -11, -5, -2, -2, -5, -11, -5, 4, 9, 9, 4, -11, -2, 9, 14, 14, 9, -2, -2, 9, 14, 14,
        //         9, -2, -5, 4, 9, 9, 4, -11, -11, -5, -2, -2, -5, -11,
        //     ],
        //     connectivity: 14,
        //     tempo_offset: 150,
        //     piece: [122, 57, 113],
        //     stack_eval: StackEval::build_simple([-61, 85, -29, 99, -20, 116]),
        //     flat_road: [40, 22, 13, 8],
        //     cs_threat: 64,
        //     mom: Default::default(),
        // }
    }
}

type WeightBuilder = Vec<i32>;

#[rustfmt::skip]
fn get_location(slice: [i32; 6]) -> [i32; 36] {
    let [corner, offcorner, edge, kosumi, ctouch, center] = slice;
    [
        corner, offcorner, edge, edge, offcorner, corner,
        offcorner, kosumi, ctouch, ctouch, kosumi, offcorner,
        edge, ctouch, center, center, ctouch, edge,
        edge, ctouch, center, center, ctouch, edge,
        offcorner, kosumi, ctouch, ctouch, kosumi, offcorner,
        corner, offcorner, edge, edge, offcorner, corner
    ]
}

fn build_weights(w: &WeightBuilder) -> Weights6 {
    let location = get_location([w[0], w[1], w[2], w[3], w[4], w[5]]);
    let connectivity = w[6];
    let tempo_offset = 150;
    let piece = [w[7], w[8], w[9]];
    let stack_top = [-w[10], w[11], -w[12], w[13], -w[14], w[15]];
    //     connectivity: 14,
    //     tempo_offset: 150,
    //     piece: [122, 57, 113],
    //     stack_eval: StackEval::build_simple([-61, 85, -29, 99, -20, 116]),
    //     flat_road: [40, 22, 13, 8],
    //     cs_threat: 64,
    let flat_road = [w[16], w[17], w[18], w[19]];
    let cs_threat = w[20];
    // 0 +1, +2, +3, +4, +5
    let f: Vec<_> = w[21..=25].iter().scan(0, |acc, x| Some(*acc + x)).collect();

    let flat_diff = [
        -f[4], -f[3], -f[2], -f[1], -f[0], 0, f[0], f[1], f[2], f[3], f[4],
    ];
    let r: Vec<_> = w[26..=30].iter().scan(0, |acc, x| Some(*acc + x)).collect();
    let reserve_diff = [
        -r[4], -r[3], -r[2], -r[1], -r[0], 0, r[0], r[1], r[2], r[3], r[4],
    ];
    let weights = Weights6::new(
        location,
        connectivity,
        tempo_offset,
        piece,
        stack_top,
        flat_road,
        cs_threat,
        flat_diff,
        reserve_diff,
    );
    // let mut idx = 16;
    // for top in [Piece::WhiteFlat, Piece::WhiteWall, Piece::WhiteCap].into_iter() {
    //     for cap in 1..=5 {
    //         for friendly in 0..=5 - cap {
    //             weights
    //                 .stack_eval
    //                 .set_cap_friendly(top, cap, friendly, w[idx]);
    //             idx += 1;
    //         }
    //     }
    // }
    weights
}

#[allow(dead_code)]
pub struct Weights7 {
    location: [i32; 49],
    connectivity: i32,
    tempo_offset: i32,
    piece: [i32; 3],
    pub stack_eval: StackEval,
    // stack_top: [i32; 6],
    flat_road: [i32; 4],
    cs_threat: i32,
}

impl Weights7 {
    fn new(
        location: [i32; 49],
        connectivity: i32,
        tempo_offset: i32,
        piece: [i32; 3],
        stack_eval: StackEval,
        flat_road: [i32; 4],
        cs_threat: i32,
    ) -> Self {
        Self {
            location,
            connectivity,
            tempo_offset,
            piece,
            stack_eval,
            flat_road,
            cs_threat,
        }
    }
    fn piece_weight(&self, p: Piece) -> i32 {
        match p {
            Piece::WhiteFlat | Piece::BlackFlat => self.piece[0],
            Piece::WhiteWall | Piece::BlackWall => self.piece[1],
            Piece::WhiteCap | Piece::BlackCap => self.piece[2],
        }
    }
}

impl Default for Weights7 {
    fn default() -> Self {
        Weights7 {
            location: LOCATION_WEIGHT7,
            connectivity: 14,
            tempo_offset: 150,
            piece: [122, 57, 113],
            stack_eval: StackEval::build_simple([-61, 85, -29, 99, -20, 116]),
            flat_road: [40, 22, 13, 8],
            cs_threat: 64,
        }
    }
}

// pub trait EvalComponents<'a> {
//     fn create(weights: &'a SmoothWeights6) -> Self
//     where
//         Self: 'a;
//     fn score(&self) -> i32;
//     fn mark(&mut self, index: usize);
//     fn set_phase(&mut self, phase: usize);
// }

#[allow(clippy::zero_prefixed_literal)]
#[rustfmt::skip]
const LOCATION_WEIGHT7: [i32; 49] = [
    00, 05, 05, 05, 05, 05, 00,
    05, 10, 15, 15, 15, 10, 05,
    05, 15, 20, 20, 20, 15, 05,
    05, 15, 20, 20, 20, 15, 05,
    05, 15, 20, 20, 20, 15, 05,
    05, 10, 15, 15, 15, 10, 05,
    00, 05, 05, 05, 05, 05, 00,
];

#[allow(clippy::zero_prefixed_literal)]
#[rustfmt::skip]
const LOCATION_WEIGHT5: [i32; 25] = [
    00, 05, 05, 05, 00,
    05, 10, 15, 10, 05,
    05, 15, 20, 15, 05,
    05, 10, 15, 10, 05,
    00, 05, 05, 05, 00,
];

#[cfg(test)]
mod test {
    use super::*;
    use crate::board::Bitboard6;
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
    #[test]
    fn placement_metric_test() {
        let s = "x,2,1,x2,1/x,2,2,x,1,1/x,21,21C,2,1,12C/x2,2,x,1,1/x,2,2,x2,1/x2,2,x3 2 13";
        let board = Board6::try_from_tps(s).unwrap();
        let (w_metric, _) =
            flat_placement_road_h(board.bits.road_pieces(Color::White), board.bits.empty());
        assert_eq!(w_metric, 1);
        let (b_metric, _) =
            flat_placement_road_h(board.bits.road_pieces(Color::Black), board.bits.empty());
        assert_eq!(b_metric, 2);
        let s = "1,2,1,1,1,x/2,12C,2,1,x2/2,1S,2,1,x2/1,2,21C,1,x2/x,2,x4/x,2,x4 2 12";
        let board = Board6::try_from_tps(s).unwrap();
        let (w_metric, _) =
            flat_placement_road_h(board.bits.road_pieces(Color::White), board.bits.empty());
        assert_eq!(w_metric, 2);
        let (b_metric, _) =
            flat_placement_road_h(board.bits.road_pieces(Color::Black), board.bits.empty());
        assert_eq!(b_metric, 5);
    }

    #[test]
    fn one_gap_test() {
        let s = "x3,2,x,1/x2,2,2,2,1/x,2,2,x,121,2/1,1,2221C,2S,1,1/x2,1,1,12,1112C/x2,2,x2,1 2 21";
        let board = Board6::try_from_tps(s).unwrap();
        let (white_s, white_c) = one_gap_road_old(
            board.bits.road_pieces(Color::White),
            board.bits.blocker_pieces(Color::Black),
        );
        assert_eq!(6, white_s);
        assert_eq!(4, white_c);

        let (black_s, black_c) = one_gap_road_old(
            board.bits.road_pieces(Color::Black),
            board.bits.blocker_pieces(Color::White),
        );
        assert_eq!(5, black_s);
        assert_eq!(4, black_c);

        let s = "x,1,x4/1,2,2,2,2,2/12C,1C,x4/1,1,x4/1,x5/1,x5 2 8";
        let board = Board6::try_from_tps(s).unwrap();
        let (white_s, white_c) = one_gap_road(board.bits.road_pieces(Color::White));
        assert_eq!(6, white_s);
        assert_eq!(3, white_c);

        let (black_s, black_c) = one_gap_road(board.bits.road_pieces(Color::Black));
        assert_eq!(6, black_s);
        assert_eq!(2, black_c);
    }

    // #[test]
    // fn test_nn() {
    //     const MARGIN: f32 = 0.01;
    //     let evals = [0.0450, -0.1636, -0.0117, -0.2939];
    //     let tps = [
    //         "2S,x,1,1,x,1S/x,x,x,2,1,2/x,2,x,x,1,1S/x,12,1C,x,x,x/1S,1,x,2,x,x/x,x,x,2C,2,x 2 10",
    //         "2S,x,1,1,x,1S/x,x,x,2,1,2/x,2,x,x,1,1S/x,12,1C,2,x,x/1S,1,x,2,x,x/x,x,x,2C,2,x 1 11",
    //         "2S,x,1,1,x,1S/x,x,x,2,1,2/x,2,x,1,1,1S/x,12,1C,2,x,x/1S,1,x,2,x,x/x,x,x,2C,2,x 2 11",
    //         "2S,x,1,1,x,1S/x,x,x,2,1,2/x,2,2,1,1,1S/x,12,1C,2,x,x/1S,1,x,2,x,x/x,x,x,2C,2,x 1 12",
    //     ];
    //     super::global_init_weights("/home/justin/Code/rust/topaz-eval/explore/vals_perp_w0.txt");
    //     let mut nn = NNUE6::new();
    //     for (ref_eval, t) in evals.into_iter().zip(tps.into_iter()) {
    //         let mut game = Board6::try_from_tps(t).unwrap().with_komi(4);
    //         let (a, b) = nn.get_states(&game);
    //         eprintln!("{:?}", &a);
    //         eprintln!("{:?}", &b);
    //         game.null_move();
    //         let (c, d) = nn.get_states(&game);
    //         assert_eq!(a, d);
    //         assert_eq!(b, c);
    //         game.rev_null_move();
    //         let (e, f) = nn.get_states(&game);
    //         assert_eq!(a, e);
    //         assert_eq!(b, f);
    //         let e = (nn.evaluate(&game, 0) as f32) / 500.0;
    //         let mut low = ref_eval - MARGIN;
    //         let mut high = ref_eval + MARGIN;
    //         if low > high {
    //             std::mem::swap(&mut low, &mut high);
    //         }
    //         eprintln!("Ref: [ {:.3}, {:.3} ] Pred: {}", low, high, e);
    //         assert!(e > low);
    //         assert!(e < high);
    //     }
    // }
}
