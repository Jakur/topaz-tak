use super::{Bitboard, Piece, Stack};
use crate::board::BitIndexIterator;
use crate::board::TakBoard;
use crate::board::{Board5, Board6};
use crate::Color;
use crate::Position;

use std::fs::File;

lazy_static::lazy_static!(
    pub static ref NN6: incremental::Weights = {
        let f_name = nn_file().expect("Could not read NN variable from env");
        incremental::Weights::from_file(File::open(f_name).unwrap()).unwrap()
    };
);

fn nn_file() -> Option<String> {
    dotenv::dotenv().ok()?;
    let mut path = None;
    for (key, value) in std::env::vars() {
        if key == "NN" {
            path = Some(value);
        }
    }
    path
}

mod incremental;

pub trait Evaluator {
    type Game: TakBoard;
    fn evaluate(&mut self, game: &Self::Game, depth: usize) -> i32;
    fn eval_stack(&self, game: &Self::Game, index: usize, stack: &Stack) -> i32;
    fn eval_components(&self, game: &Self::Game) -> EvalComponents;
}

pub struct NNUE6 {
    incremental_weights: incremental::Incremental,
    old: Vec<u16>,
}

impl Default for NNUE6 {
    fn default() -> Self {
        Self::new()
    }
}

impl NNUE6 {
    pub fn new() -> Self {
        let incremental_weights = NN6.build_incremental(&[]);
        let old = Vec::new();
        Self {
            incremental_weights,
            old,
        }
    }
}

impl Evaluator for NNUE6 {
    type Game = Board6;

    fn evaluate(&mut self, game: &Self::Game, depth: usize) -> i32 {
        const TEMPO_OFFSET: i32 = 150;
        let mut new = incremental::nn_repr(game);
        self.incremental_weights.update_diff(&self.old, &new, &NN6);
        let score = NN6.incremental_forward(
            &self.incremental_weights,
            game.side_to_move() == Color::White,
        );
        std::mem::swap(&mut new, &mut self.old);
        if depth % 2 == 1 {
            score + TEMPO_OFFSET
        } else {
            score
        }
    }

    fn eval_stack(&self, _game: &Self::Game, _index: usize, _stack: &Stack) -> i32 {
        unimplemented!()
    }

    fn eval_components(&self, _game: &Self::Game) -> EvalComponents {
        unimplemented!()
    }
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

pub struct Tinue5 {
    pub attacker: Color,
}

pub struct Tinue6 {
    pub attacker: Color,
}

pub struct Tinue7 {
    pub attacker: Color,
}

macro_rules! tinue_eval {
    ($board: ty, $weights: ty) => {
        impl Evaluator for $weights {
            type Game = $board;
            #[inline(never)]
            fn evaluate(&mut self, game: &Self::Game, _depth: usize) -> i32 {
                let attacker = self.attacker;
                let mut score = 100 - 5 * game.pieces_reserve(attacker) as i32;
                score -= 10 * game.bits().blocker_pieces(!attacker).pop_count() as i32;
                score += 2 * game
                    .bits()
                    .road_pieces(attacker)
                    .critical_squares()
                    .pop_count() as i32;
                if game.side_to_move() == attacker {
                    score
                } else {
                    -score
                }
            }
            fn eval_stack(&self, _game: &Self::Game, _index: usize, _stack: &Stack) -> i32 {
                unimplemented!()
            }
            fn eval_components(&self, _game: &Self::Game) -> EvalComponents {
                unimplemented!()
            }
        }
    };
}

tinue_eval![Board5, Tinue5];
tinue_eval![Board6, Tinue6];
tinue_eval![crate::Board7, Tinue7];

macro_rules! eval_impl {
    ($board: ty, $weights: ty) => {
        impl Evaluator for $weights {
            type Game = $board;
            #[inline(never)]
            fn evaluate(&mut self, game: &Self::Game, depth: usize) -> i32 {
                let mut score = 0;

                for (idx, stack) in game.board.iter().enumerate() {
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
                        score += self.eval_stack(game, idx, stack);
                    }
                }
                let black_cap = game.bits.cap & game.bits.black;
                if black_cap != <Self::Game as TakBoard>::Bits::ZERO {
                    let black_c_lonely = black_cap.adjacent() & (game.bits.flat | game.bits.wall);
                    if black_c_lonely == <Self::Game as TakBoard>::Bits::ZERO {
                        score += 30;
                    }
                }
                let white_cap = game.bits.cap & game.bits.white;
                if white_cap != <Self::Game as TakBoard>::Bits::ZERO {
                    let white_c_lonely = white_cap.adjacent() & (game.bits.flat | game.bits.wall);
                    if white_c_lonely == <Self::Game as TakBoard>::Bits::ZERO {
                        score -= 30;
                    }
                }
                score += self.cs_threat * attackable_cs(Color::White, game);
                score -= self.cs_threat * attackable_cs(Color::Black, game);
                let white_r = game.bits.road_pieces(Color::White);
                let black_r = game.bits.road_pieces(Color::Black);
                let (loose_white_pc, white_comp) =
                    flat_placement_road_h(white_r, game.bits.empty());
                let (loose_black_pc, black_comp) =
                    flat_placement_road_h(black_r, game.bits.empty());
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
                let empty_count = game.bits().empty().pop_count() as usize;
                if white_res < 10 || black_res < 10 || empty_count <= 4 {
                    // Half flats in white's favor
                    let mut flat_diff = (game.bits.flat & game.bits.white).pop_count() as i32
                        - (game.bits.flat & game.bits.black).pop_count() as i32;
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
                    if flat_diff > 0 {
                        score +=
                            (flat_diff * 100) / std::cmp::min(white_res, white_fill + 1) as i32;
                    } else if flat_diff < 0 {
                        score +=
                            (flat_diff * 100) / std::cmp::min(black_res, black_fill + 1) as i32;
                    }
                }
                if let Color::White = game.side_to_move() {
                    if depth % 2 == 1 {
                        score + self.tempo_offset
                    } else {
                        score
                    }
                } else {
                    if depth % 2 == 1 {
                        -1 * score + self.tempo_offset
                    } else {
                        -1 * score
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
            fn eval_stack(&self, game: &Self::Game, idx: usize, stack: &Stack) -> i32 {
                let top = stack.top().unwrap();
                let mut pw = self.piece_weight(top) + self.location[idx];
                let (mut captive, mut friendly) = stack.captive_friendly();
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
                let neighbors = <Self::Game as TakBoard>::Bits::index_to_bit(idx).adjacent();
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
                // Todo reconsider these with new method?
                if mobility < 2 && !top.is_blocker() {
                    friendly /= 2;
                }
                if safety < 0 {
                    captive *= 2;
                }
                let stack_score = pw + self.stack_eval.eval_cap_friendly(top, captive, friendly);
                // let friendly_score = friendly * f_mul;
                // // let friendly_score = std::cmp::min(friendly * f_mul, 300);
                // let stack_score = captive * c_mul + friendly_score + pw;
                if let Color::White = top.owner() {
                    return stack_score;
                } else {
                    return -stack_score;
                }
            }
            fn eval_components(&self, game: &Self::Game) -> EvalComponents {
                let player = game.side_to_move();
                let opp = !game.side_to_move();
                let empty = [game.bits().empty().pop_count() as i32];
                let mut loc_score = [0; 2];
                let mut pieces = [0; 6];
                let mut capstone_status = [0; 2];
                let mut capstone_height = [0; 2];
                let mut stack_score = [0; 2];
                let mut stack_count = [0; 2];
                let reserves = [
                    (game.pieces_reserve(player) + game.caps_reserve(player)) as i32,
                    (game.pieces_reserve(opp) + game.caps_reserve(opp)) as i32,
                ];
                let flat_placement = [
                    flat_placement_road_h(game.bits().road_pieces(player), game.bits().empty()),
                    flat_placement_road_h(game.bits().road_pieces(opp), game.bits().empty()),
                ];
                let comps = [flat_placement[0].1 as i32, flat_placement[1].1 as i32];
                let flat_placement = [flat_placement[0].0, flat_placement[1].0];
                let cs = [attackable_cs(player, game), attackable_cs(opp, game)];
                let one_gap = [
                    one_gap_road(game.bits().road_pieces(player)).0,
                    one_gap_road(game.bits().road_pieces(opp)).0,
                ];
                // let pieces_left = white_res + black_res;
                // let time_mul = (pieces_left as i32 * 100) / 60;
                for (idx, stack) in game.board().iter().enumerate() {
                    if stack.len() == 0 {
                        continue;
                    }
                    let top = stack.top().unwrap();
                    let color = (top.owner() != player) as usize; // Friendly 0, Enemy 1
                    let mul = if top.is_cap() {
                        capstone_height[color] = stack.len() as i32;
                        if let Some(piece) = stack.from_top(1) {
                            // Todo check this
                            if piece.owner() == top.owner() {
                                capstone_status[color] = 1;
                            } else {
                                capstone_status[color] = -1;
                            }
                        }
                        2
                    } else {
                        1
                    };
                    loc_score[color] += mul * self.location[idx];
                    if stack.len() == 1 {
                        let piece_idx = if let Color::White = player {
                            top as usize - 1
                        } else {
                            top.swap_color() as usize - 1
                        };
                        pieces[piece_idx] += 1;
                    } else if stack.len() > 1 {
                        if let Color::White = top.owner() {
                            stack_score[color] += self.eval_stack(game, idx, stack);
                        } else {
                            stack_score[color] -= self.eval_stack(game, idx, stack);
                        }
                        stack_count[color] += 1;
                    }
                }
                let vec = vec![
                    empty.as_slice(),
                    &loc_score,
                    &pieces,
                    &capstone_status,
                    &capstone_height,
                    &stack_score,
                    &stack_count,
                    &reserves,
                    &comps,
                    &flat_placement,
                    &one_gap,
                    &cs,
                ];
                EvalComponents::from_arrays(vec)
            }
        }
    };
}

fn attackable_cs<B: TakBoard>(color: Color, game: &B) -> i32 {
    let road_pieces = game.bits().road_pieces(color);
    let mut cs = road_pieces.critical_squares() & !game.bits().blocker_pieces(!color);
    let mut count = 0;
    while cs != B::Bits::ZERO {
        let lowest = cs.pop_lowest();
        if (lowest.adjacent() & road_pieces).pop_count() >= 3 {
            count += 1;
        }
    }
    count
}

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
            let score = comp.simple_road_est();
            if score > best {
                best = score;
            }
            let adj = comp.adjacent() & !comp & !blocker_bits;
            let rest = road_bits & !comp;
            let mut potential = adj & rest.adjacent();
            while potential != B::ZERO {
                let lowest = potential.pop_lowest();
                let check = (lowest | road_bits).flood(lowest);
                let score = check.simple_road_est();
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
fn flat_placement_road_h<B: Bitboard + std::fmt::Debug>(road_bits: B, empty: B) -> (i32, usize) {
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

/// A variant of the original one gap road, this one instead allows a diagonal extension
/// from a diagonally adjacent piece, trying to simulate its capture. This outperforms
/// the original version, but the road play still seems uninspired, extending threats
/// that seem unlikely to be finished, while ignoring future threats simply because
/// they haven't yet reached fruition.
fn one_gap_road<B: Bitboard + std::fmt::Debug>(road_bits: B) -> (i32, usize) {
    let mut iter = ComponentIterator::new(road_bits);
    let mut best = 0;
    let mut count = 0;
    while let Some(comp) = iter.next() {
        count += 1;
        if comp.pop_count() >= 3 {
            let score = comp.simple_road_est();
            if score > best {
                best = score;
            }
            let rest = road_bits & !comp;
            let mut potential = comp.loose_adjacent() & rest;
            while potential != B::ZERO {
                let lowest = potential.pop_lowest();
                let bend = (lowest.loose_adjacent() | lowest) & road_bits;
                let fl = road_bits.flood(bend);
                let score = fl.simple_road_est();
                if score > best {
                    best = score;
                }
            }
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

pub struct Weights5 {
    location: [i32; 25],
    connectivity: i32,
    tempo_offset: i32,
    piece: [i32; 3],
    stack_eval: StackEval,
    flat_road: [i32; 4],
    cs_threat: i32,
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
    ) -> Self {
        Self {
            location,
            connectivity,
            tempo_offset,
            piece,
            stack_eval: StackEval::build_simple(stack_top),
            flat_road,
            cs_threat,
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
        Weights6 {
            location: [
                -11, -5, -2, -2, -5, -11, -5, 4, 9, 9, 4, -11, -2, 9, 14, 14, 9, -2, -2, 9, 14, 14,
                9, -2, -5, 4, 9, 9, 4, -11, -11, -5, -2, -2, -5, -11,
            ],
            connectivity: 14,
            tempo_offset: 150,
            piece: [122, 57, 113],
            stack_eval: StackEval::build_simple([-61, 85, -29, 99, -20, 116]),
            flat_road: [40, 22, 13, 8],
            cs_threat: 64,
        }
    }
}

pub struct EvalComponents {
    data: [i32; 27],
}

impl EvalComponents {
    pub fn from_arrays(vals: Vec<&[i32]>) -> Self {
        let mut data = [0; 27];
        let mut idx = 0;
        for slice in vals {
            for x in slice {
                data[idx] = *x;
                idx += 1;
            }
        }
        assert_eq!(idx, data.len());
        Self { data }
    }
    pub fn write_data<W: std::fmt::Write>(&self, writer: &mut W) -> anyhow::Result<()> {
        for x in self.data.iter() {
            write!(writer, "{},", *x)?;
        }
        Ok(())
    }
    pub fn labels() -> Vec<String> {
        let mut out = [
            "empty",
            "loc_friendly",
            "loc_enemy",
            "flat_friendly",
            "wall_friendly",
            "cap_friendly",
            "flat_enemy",
            "wall_enemy",
            "cap_enemy",
        ]
        .iter()
        .map(|x| format!("{}", x))
        .collect();
        for name in [
            "cap_status",
            "cap_height",
            "stack_score",
            "stack_count",
            "reserves",
            "comps",
            "flat_placement_r",
            "one_gap_r",
            "cs",
        ]
        .iter()
        {
            Self::add_names(name, &mut out);
        }
        out
    }
    fn add_names(name: &str, vec: &mut Vec<String>) {
        vec.push(format!("{}_friendly", name));
        vec.push(format!("{}_enemy", name));
    }
}

// corner, offcorner, edge, edge, offcorner, corner,
// offcorner, kosumi, ctouch, ctouch, kosumi, corner,
// edge, ctouch, center, center, ctouch, edge,
// edge, ctouch, center, center, ctouch, edge,
// offcorner, kosumi, ctouch, ctouch, kosumi, corner,
// corner, offcorner, edge, edge, offcorner, corner,

#[allow(clippy::zero_prefixed_literal)]
#[rustfmt::skip]
const LOCATION_WEIGHT: [i32; 36] = [
    00, 05, 05, 05, 05, 00,
    05, 10, 15, 15, 10, 05,
    05, 15, 20, 20, 15, 05,
    05, 15, 20, 20, 15, 05,
    05, 10, 15, 15, 10, 05,
    00, 05, 05, 05, 05, 00,
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
}
