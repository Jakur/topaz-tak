use self::incremental::NN_REPR;

use super::{Bitboard, Piece, Stack};
use crate::board;
use crate::board::BitIndexIterator;
use crate::board::TakBoard;
use crate::board::{Board5, Board6};
use crate::eval::incremental::Incremental;
use crate::eval::incremental::NN_REPR_SIZE;
use crate::Color;
use crate::Position;

pub use incremental::nn_repr;

use std::collections::HashMap;
use std::fs::File;

static NN6: once_cell::sync::OnceCell<incremental::Weights> = once_cell::sync::OnceCell::new();

pub fn global_init_weights(f_name: &str) {
    println!("Setting global weights!");
    let weights = incremental::Weights::from_file(File::open(f_name).unwrap()).unwrap();
    NN6.set(weights)
        .ok()
        .expect("Unable to override existing NN6 Weights!");
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

mod incremental;

pub trait Evaluator {
    type Game: TakBoard + Send;
    fn evaluate(&mut self, game: &Self::Game, depth: usize) -> i32;
    fn eval_stack(&self, game: &Self::Game, index: usize, stack: &Stack) -> i32;
    fn eval_components(&self, game: &Self::Game) -> EvalComponents;
    fn is_quiet(&self, game: &Self::Game) -> bool;
}

pub struct NNUE6 {
    incremental_white: incremental::Incremental,
    incremental_black: incremental::Incremental,
    old_white: NN_REPR,
    old_black: NN_REPR,
    classical: Weights6,
}

impl Default for NNUE6 {
    fn default() -> Self {
        Self::new()
    }
}

impl NNUE6 {
    pub fn new() -> Self {
        let nn = NN6.get_or_init(|| {
            let f_name = nn_file().expect("Could not read NN variable from env");
            incremental::Weights::from_file(File::open(f_name).unwrap()).unwrap()
        });
        let classical = Weights6::default();
        Self {
            incremental_white: nn.build_incremental(&[0; NN_REPR_SIZE]),
            incremental_black: nn.build_incremental(&[0; NN_REPR_SIZE]),
            old_white: [0; NN_REPR_SIZE],
            old_black: [0; NN_REPR_SIZE],
            classical,
        }
    }
    pub fn get_states(&mut self, game: &Board6) -> (NN_REPR, NN_REPR) {
        // const OFFSETS: [u16; 39] = [
        //     0, 156, 312, 468, 624, 780, 936, 1092, 1248, 1404, 1560, 1716, 1872, 2028, 2184, 2340,
        //     2496, 2652, 2808, 2964, 3120, 3276, 3432, 3588, 3744, 3900, 4056, 4212, 4368, 4524,
        //     4680, 4836, 4992, 5148, 5304, 5460, 5616, 5652, 5688,
        // ];
        // const OFFSETS: [i16; NN_REPR_SIZE] = [
        //     0, 257, 514, 771, 1028, 1285, 1542, 1799, 2056, 2313, 2570, 2827, 3084, 3341, 3598,
        //     3855, 4112, 4369, 4626, 4883, 5140, 5397, 5654, 5911, 6168, 6425, 6682, 6939, 7196,
        //     7453, 7710, 7967, 8224, 8481, 8738, 8995, 9252, 9273, 9294, 9315, 9336, 9357, 9378,
        //     9399, 9420, 9441, 9462, 9483,
        // ];
        // const OFFSETS: [i16; NN_REPR_SIZE] = [
        //     0, 71, 142, 213, 284, 355, 426, 497, 568, 639, 710, 781, 852, 923, 994, 1065, 1136,
        //     1207, 1278, 1349, 1420, 1491, 1562, 1633, 1704, 1775, 1846, 1917, 1988, 2059, 2130,
        //     2201, 2272, 2343, 2414, 2485, 2556, 2577, 2598, 2619, 2640, 2661, 2682, 2703, 2724,
        //     2745, 2766, 2787,
        // ];
        const OFFSETS: [i16; NN_REPR_SIZE] = [
            0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376,
            5760, 6144, 6528, 6912, 7296, 7680, 8064, 8448, 8832, 9216, 9600, 9984, 10368, 10752,
            11136, 11520, 11904, 12288, 12672, 13056, 13440, 13824, 13845, 13866, 13887, 13908,
            13929, 13950, 13971, 13992, 14013, 14034, 14055,
        ];
        let mut side = nn_repr(game);
        let mut opp = side.clone();
        for (idx, val) in side.into_iter().take(36).enumerate() {
            if val != 0 {
                // 196
                if val >= 192 {
                    opp[idx] -= 192;
                } else {
                    opp[idx] += 192;
                }
                let stack = val % 64;
                let new_stack = incremental::flip_stack_repr(stack);
                // Move out old stack
                opp[idx] -= stack;
                // Move in new stack
                opp[idx] += new_stack;
                // if val >= 196 {
                //     opp[idx] -= 147;
                // } else {
                //     opp[idx] += 147;
                // }
            }
        }
        // if side[38] == 0 {
        //     opp[38] = 1;
        // } else {
        //     opp[38] = 0;
        // }
        // Refine
        for (idx, (val_a, val_b)) in side.iter_mut().zip(opp.iter_mut()).enumerate().take(36) {
            *val_a = OFFSETS[idx] + *val_a; // + incremental::STACK_LOOKUP[*val_a as usize];
            *val_b = OFFSETS[idx] + *val_b; // + incremental::STACK_LOOKUP[*val_b as usize];
        }
        for i in (36..=42).step_by(2) {
            opp.swap(i, i + 1);
            // std::mem::swap(&mut opp[i], &mut opp[i + 1]);
            // opp[i] = side[i + 1];
            // opp[i + 1] = side[i];
        }
        for i in 44..OFFSETS.len() {
            opp[i] = -side[i];
        }
        for i in 36..OFFSETS.len() {
            side[i] += OFFSETS[i];
            opp[i] += OFFSETS[i];
        }
        (side, opp)
    }
}

impl Evaluator for NNUE6 {
    type Game = Board6;

    fn evaluate(&mut self, game: &Self::Game, depth: usize) -> i32 {
        const TEMPO_OFFSET: i32 = 0;
        let (mut side, mut opp) = self.get_states(game);
        let nn = NN6.get().unwrap(); // Could probably unchecked due to init in new
                                     // eprintln!("{:?}", side);
                                     // eprintln!("{:?}", opp);
                                     // let white_cap_penalty = (game.caps_reserve(Color::White) * 5 * game.ply()) as i32;
                                     // let black_cap_penalty = (game.caps_reserve(Color::Black) * 5 * game.ply()) as i32;
        let score = if let Color::White = game.side_to_move() {
            // let side_weights = nn.build_incremental(&s);
            // if depth == 100 {
            //     dbg!(&side_weights);
            // }
            // let opp_weights = nn.build_incremental(&o);
            self.incremental_white
                .update_diff(&self.old_white, &side, &nn);
            self.incremental_black
                .update_diff(&self.old_black, &opp, &nn);
            std::mem::swap(&mut side, &mut self.old_white);
            std::mem::swap(&mut opp, &mut self.old_black);
            // nn.incremental_forward(&side_weights, &opp_weights)
            nn.incremental_forward(&self.incremental_white, &self.incremental_black)
        } else {
            self.incremental_white
                .update_diff(&self.old_white, &opp, &nn);
            self.incremental_black
                .update_diff(&self.old_black, &side, &nn);
            std::mem::swap(&mut opp, &mut self.old_white);
            std::mem::swap(&mut side, &mut self.old_black);
            nn.incremental_forward(&self.incremental_black, &self.incremental_white)
        };
        // let mut score = if let Color::White = game.side_to_move() {
        //     self.incremental_white
        //         .update_diff(&self.old_white, &new, nn);
        //     // let rand_seed = Some(self.pos_seed ^ game.hash());
        //     let rand_seed = None;
        //     let cap_diff =
        //         (game.caps_reserve(Color::White) as i32) - (game.caps_reserve(Color::Black) as i32);

        //     std::mem::swap(&mut new, &mut self.old_white);
        //     -cap_diff * (5 * game.ply() as i32)
        //         + nn.incremental_forward(&self.incremental_white, rand_seed)
        // } else {
        //     self.incremental_black
        //         .update_diff(&self.old_black, &new, nn);
        //     // let rand_seed = Some(self.pos_seed ^ game.hash());
        //     let rand_seed = None;
        //     std::mem::swap(&mut new, &mut self.old_black);
        //     let cap_diff =
        //         (game.caps_reserve(Color::Black) as i32) - (game.caps_reserve(Color::White) as i32);
        //     -cap_diff * (5 * game.ply() as i32)
        //         + nn.incremental_forward(&self.incremental_black, rand_seed)
        // };
        // let least_pieces = std::cmp::min(
        //     game.pieces_reserve(Color::Black),
        //     game.pieces_reserve(Color::White),
        // );

        // if score > 400 || score < -400 || (game.komi() != 0 && least_pieces <= 8) {
        //     score += self.classical.evaluate(game, depth) / 4;
        // }
        // eprintln!("{}", score);
        // 500?
        let score = (score * 500.0) as i32;
        // let score = (libm::tanh(score.into()) * 250.0) as i32;
        if depth % 2 == 1 {
            score + TEMPO_OFFSET
        } else {
            score
        }
        // score
    }

    fn eval_stack(&self, _game: &Self::Game, _index: usize, _stack: &Stack) -> i32 {
        unimplemented!()
    }

    fn eval_components(&self, _game: &Self::Game) -> EvalComponents {
        unimplemented!()
    }

    fn is_quiet(&self, game: &Self::Game) -> bool {
        true
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

#[derive(Debug)]
struct Embedding<const N: usize> {
    lookup: [f32; N],
}

impl<const N: usize> Embedding<N> {
    fn forward(&self, x: &[i32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(N);
        for val in x.into_iter() {
            out.push(self.lookup[*val as usize])
        }
        out
    }
    fn forward_scalar(&self, x: i32) -> f32 {
        self.lookup[x as usize]
    }
    fn from_string(s: &str) -> Self {
        let mut lookup = [0.0; N];
        for (idx, val) in s.split(",").enumerate() {
            let num = val.parse().unwrap();
            lookup[idx] = num;
        }
        Self { lookup }
    }
}

struct Linear<const N: usize> {
    weights: [f32; N],
}

impl<const N: usize> Linear<N> {
    fn forward(&self, x: &[f32]) -> f32 {
        self.weights
            .iter()
            .copied()
            .zip(x.into_iter().copied())
            .map(|(x, y)| x * y)
            .sum()
    }
    fn from_string(s: &str) -> Self {
        let mut weights = [0.0; N];
        for (idx, val) in s.split(",").enumerate() {
            let num = val.parse().unwrap();
            weights[idx] = num;
        }
        Self { weights }
    }
}

pub struct SmoothWeights6 {
    top_pst1: Embedding<8>,
    top_pst2: Embedding<8>,
    top_pst3: Embedding<8>,
    // captive_pst: Embedding<7>,
    // friendly_pst: Embedding<7>,
    square_pst: Linear<36>,
    mobility: Embedding<8>,
    safety: Embedding<8>,
    small: Embedding<20>,
    road_dist: Embedding<36>,
    reserve: Embedding<13>,
    endgame_weight: f32,
    color: f32,
}

impl SmoothWeights6 {
    pub fn empty() -> Self {
        static DATA: &'static str = include_str!("eval/classic.table");
        let out = Self::parse(DATA);
        out
    }
    pub fn parse(data: &str) -> Self {
        let mut map = HashMap::new();
        for line in data.lines() {
            let mut split = line.split(":");
            let name = split.next().unwrap();
            let data = split.next().unwrap();
            map.insert(name, data);
        }
        Self {
            top_pst1: Embedding::<8>::from_string(map.get("top_pst1").unwrap()),
            top_pst2: Embedding::<8>::from_string(map.get("top_pst2").unwrap()),
            top_pst3: Embedding::<8>::from_string(map.get("top_pst3").unwrap()),
            // captive_pst: Embedding::<7>::from_string(map.get("captive_pst").unwrap()),
            // friendly_pst: Embedding::<7>::from_string(map.get("friendly_pst").unwrap()),
            square_pst: Linear::<36>::from_string(map.get("square_pst").unwrap()),
            mobility: Embedding::<8>::from_string(map.get("mobility").unwrap()),
            safety: Embedding::<8>::from_string(map.get("safety").unwrap()),
            small: Embedding::<20>::from_string(map.get("small").unwrap()),
            road_dist: Embedding::<36>::from_string(map.get("road_dist").unwrap()),
            reserve: Embedding::<13>::from_string(map.get("reserve").unwrap()),
            endgame_weight: map
                .get("endgame_weight")
                .map(|x| x.parse().unwrap())
                .unwrap(),
            color: map.get("color").map(|x| x.parse().unwrap()).unwrap(),
        }
    }
    // def adjacent_conv(self, x, embedding):
    // x = embedding(x).view(-1, 1, self.board_size, self.board_size)
    // x = F.conv2d(x, self.adjacent.to(x.device), padding=1).squeeze().view(-1, self.board_size * self.board_size)
    // return x
    fn adjacent_conv_sq(&self, xs: &[i32], embedding: &Embedding<8>, idx: usize) -> f32 {
        const MAIN_SCALAR: f32 = 2.0;
        const SECONDARY_SCALAR: f32 = 1.0;
        let x = xs[idx];
        let mut val = MAIN_SCALAR * embedding.forward_scalar(x);
        let neighbors =
            <<SmoothWeights6 as Evaluator>::Game as TakBoard>::Bits::index_to_bit(idx).adjacent();
        for sq in BitIndexIterator::new(neighbors) {
            val += SECONDARY_SCALAR * embedding.forward_scalar(xs[sq])
        }
        val
    }
}

impl std::default::Default for SmoothWeights6 {
    fn default() -> Self {
        Self::empty()
    }
}

impl Evaluator for SmoothWeights6 {
    type Game = Board6;

    fn evaluate(&mut self, game: &Self::Game, depth: usize) -> i32 {
        let comps = self.eval_components(game);
        let x = comps.data;
        let mut score = 0.0;
        let sqs = Self::Game::SIZE * Self::Game::SIZE;
        let offset1 = sqs * 3;
        let tops = &x[0..sqs];
        let captives = &x[sqs..sqs * 2];
        let friendly = &x[sqs * 2..sqs * 3];
        // let captives: Vec<_> = x[sqs..sqs * 2].into_iter().map(|&x| x as f32).collect();
        // let friendly: Vec<_> = x[sqs * 2..sqs * 3].into_iter().map(|&x| x as f32).collect();
        // let captives = self.captive_pst.forward(&x[sqs..sqs * 2]);
        // let friendly = self.friendly_pst.forward(&x[sqs * 2..sqs * 3]);
        let top_pst = self.top_pst1.forward(tops);
        score += self.square_pst.forward(&top_pst);

        // let safety = self.adjacent_conv(game, tops, &self.safety);
        // let mut mobility = self.adjacent_conv(game, tops, &self.mobility);
        // for (i, m) in mobility.iter_mut().enumerate() {
        //     *m /= 1.0 + captives[i];
        // }
        for idx in 0..sqs {
            if captives[idx] != 0 {
                let captive_top = self.top_pst2.forward_scalar(tops[idx]);
                let conv = self.adjacent_conv_sq(&tops, &self.safety, idx);
                score += (captives[idx] as f32) * captive_top * conv;
            }
            if friendly[idx] != 0 {
                let friendly_top = self.top_pst3.forward_scalar(tops[idx]);
                let conv = self.adjacent_conv_sq(&tops, &self.mobility, idx);
                let mobility_offset = 1 + captives[idx];
                score += (friendly[idx] as f32) * friendly_top * conv / (mobility_offset as f32)
            }
        }
        // let captive_top = self.top_pst2.forward(&tops);
        // let friendly_top = self.top_pst3.forward(&tops);
        // let captives = Self::mul_clamp_sum(&captives, &safety, &captive_top, -10.0, 10.0);
        // let friendly = Self::mul_clamp_sum(&friendly, &mobility, &friendly_top, -10.0, 10.0);
        // score += captives;
        // score += friendly;
        score += self
            .small
            .forward(&x[offset1..offset1 + 3])
            .into_iter()
            .sum::<f32>();
        let road_dist_adv = self.road_dist.forward_scalar(x[offset1 + 3]);
        // let reserves_sum = (x[offset1 + 4] + x[offset1 + 5]) as f32 / 60.0;
        let reserve_adv = 6 + (x[offset1 + 5] - x[offset1 + 4]).clamp(-6, 6);
        score += self.reserve.forward_scalar(reserve_adv);
        score += road_dist_adv; // reserves_sum;
        let len = x.len();
        score += self.endgame_weight * (x[len - 3] as f32) / (x[len - 2] as f32);
        score += self.color * x[len - 1] as f32;
        if depth % 2 == 0 {
            (score * 500.0).clamp(-2500.0, 2500.0) as i32
        } else {
            (score * 500.0).clamp(-2500.0, 2500.0) as i32
        }
    }

    fn eval_stack(&self, game: &Self::Game, idx: usize, stack: &Stack) -> i32 {
        unimplemented!()
    }

    fn eval_components(&self, game: &Self::Game) -> EvalComponents {
        // [36*Tops, 36*Captives, 36*Friendly, cap_lonely, cs_threat, components, road_dist_adv, reserves, fill, half flat_adv]
        let mut tops = [1; 36];
        let mut captive = [0; 36];
        let mut friendly = [0; 36];
        let side = game.side_to_move();
        let opp = !side;
        for (idx, stack) in game.board.iter().enumerate() {
            let piece = stack
                .top()
                .map(|x| {
                    if game.side_to_move() == Color::White {
                        x as i32 + 1
                    } else {
                        x.swap_color() as i32 + 1
                    }
                })
                .unwrap_or(1);
            tops[idx] = piece;
            let (cap, fr) = stack.captive_friendly();
            captive[idx] = std::cmp::min(cap, Self::Game::SIZE as i32);
            friendly[idx] = std::cmp::min(fr, Self::Game::SIZE as i32);
        }
        let mut small = [0; 3];
        let mut cap_lonely = 0;
        for s in [side, opp].into_iter() {
            let cap = game.bits.cap & game.bits.all_pieces(s);
            if cap != <Self::Game as TakBoard>::Bits::ZERO {
                let lonely =
                    cap.adjacent() & (game.bits.flat | game.bits.wall) & game.bits.all_pieces(opp);
                if lonely == <Self::Game as TakBoard>::Bits::ZERO {
                    cap_lonely += 1;
                }
            }
            cap_lonely = -cap_lonely;
        }
        small[0] = cap_lonely + 1; // Max 2
        let cs = 3 + 4 + std::cmp::min(4, attackable_cs(side, game))
            - std::cmp::min(4, attackable_cs(opp, game)); // Max 3 + 4 + 4 = 11
        small[1] = cs;
        let side_r = game.bits.road_pieces(side);
        let opp_r = game.bits.road_pieces(opp);
        let (loose_side_pc, side_comp) = flat_placement_road_h(side_r, game.bits.empty());
        let (loose_opp_pc, opp_comp) = flat_placement_road_h(opp_r, game.bits.empty());
        let comp_diff = 11 + 4 + std::cmp::min(side_comp, 4) - std::cmp::min(opp_comp, 4); // 11 + 4 + 4 = 19
        small[2] = comp_diff as i32;
        let mut last = [0; 6];
        let road_idx = 6 * (loose_side_pc.clamp(1, 5)) + (loose_opp_pc.clamp(1, 5));
        last[0] = road_idx;
        last[1] = (game.pieces_reserve(side) + game.caps_reserve(side)) as i32;
        last[2] = (game.pieces_reserve(opp) + game.caps_reserve(opp)) as i32;
        let (flat_diff, divisor) = endgame_calc(game);
        if side == Color::White {
            last[3] = flat_diff;
        } else {
            last[3] = -flat_diff;
        }
        last[4] = divisor;
        last[5] = if side == Color::White { 1 } else { -1 };
        EvalComponents::from_arrays(vec![&tops, &captive, &friendly, &small, &last])
    }
    fn is_quiet(&self, game: &Self::Game) -> bool {
        true
    }
}

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
        impl Evaluator for $weights {
            type Game = $board;
            #[inline(never)]
            fn evaluate(&mut self, game: &Self::Game, depth: usize) -> i32 {
                let mut score = 0;
                let white_r = game.bits.road_pieces(Color::White);
                let black_r = game.bits.road_pieces(Color::Black);
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
                        score += self.eval_stack(game, idx, stack);
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
                score += self.cs_threat * attackable_cs(Color::White, game);
                score -= self.cs_threat * attackable_cs(Color::Black, game);
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
            fn is_quiet(&self, game: &Self::Game) -> bool {
                true
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

// eval_impl![crate::Board7, Weights7];
eval_impl![Board6, Weights6];
eval_impl![Board5, Weights5];

struct BitOutcome<B> {
    bits: B,
    steps: usize,
}

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

#[derive(Default, Debug)]
struct MomentumCounter {
    white_road_diff: i32,
    black_road_diff: i32,
}

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
    mom: MomentumCounter,
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
            mom: Default::default(),
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
    mom: MomentumCounter,
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
            mom: Default::default(),
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

pub struct Weights7 {
    location: [i32; 49],
    connectivity: i32,
    tempo_offset: i32,
    piece: [i32; 3],
    pub stack_eval: StackEval,
    // stack_top: [i32; 6],
    flat_road: [i32; 4],
    cs_threat: i32,
    mom: MomentumCounter,
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
            mom: Default::default(),
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
            mom: Default::default(),
        }
    }
}

#[derive(Debug)]
pub struct EvalComponents {
    data: Vec<i32>,
}

impl EvalComponents {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn from_arrays(vals: Vec<&[i32]>) -> Self {
        let mut data = vec![0; 117];
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
    pub fn write_io<W: std::io::Write>(&self, writer: &mut W) -> anyhow::Result<()> {
        for x in self.data[0..self.data.len() - 1].iter() {
            write!(writer, "{},", *x)?;
        }
        write!(writer, "{}", *self.data.last().unwrap())?;
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

    #[test]
    fn test_nn() {
        const MARGIN: f32 = 0.01;
        let evals = [0.0450, -0.1636, -0.0117, -0.2939];
        let tps = [
            "2S,x,1,1,x,1S/x,x,x,2,1,2/x,2,x,x,1,1S/x,12,1C,x,x,x/1S,1,x,2,x,x/x,x,x,2C,2,x 2 10",
            "2S,x,1,1,x,1S/x,x,x,2,1,2/x,2,x,x,1,1S/x,12,1C,2,x,x/1S,1,x,2,x,x/x,x,x,2C,2,x 1 11",
            "2S,x,1,1,x,1S/x,x,x,2,1,2/x,2,x,1,1,1S/x,12,1C,2,x,x/1S,1,x,2,x,x/x,x,x,2C,2,x 2 11",
            "2S,x,1,1,x,1S/x,x,x,2,1,2/x,2,2,1,1,1S/x,12,1C,2,x,x/1S,1,x,2,x,x/x,x,x,2C,2,x 1 12",
        ];
        super::global_init_weights("/home/justin/Code/rust/topaz-eval/explore/vals_perp_w0.txt");
        let mut nn = NNUE6::new();
        for (ref_eval, t) in evals.into_iter().zip(tps.into_iter()) {
            let mut game = Board6::try_from_tps(t).unwrap().with_komi(4);
            let (a, b) = nn.get_states(&game);
            eprintln!("{:?}", &a);
            eprintln!("{:?}", &b);
            game.null_move();
            let (c, d) = nn.get_states(&game);
            assert_eq!(a, d);
            assert_eq!(b, c);
            game.rev_null_move();
            let (e, f) = nn.get_states(&game);
            assert_eq!(a, e);
            assert_eq!(b, f);
            let e = (nn.evaluate(&game, 0) as f32) / 500.0;
            let mut low = ref_eval - MARGIN;
            let mut high = ref_eval + MARGIN;
            if low > high {
                std::mem::swap(&mut low, &mut high);
            }
            eprintln!("Ref: [ {:.3}, {:.3} ] Pred: {}", low, high, e);
            assert!(e > low);
            assert!(e < high);
        }
    }
}
