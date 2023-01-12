use crate::eval::{attackable_cs, connected_components};
use crate::Position;
use crate::{board::Bitboard6, Bitboard, BitboardStorage, Piece};
use crate::{board::TakBoard, Board6, Color};
use bitintr::Pdep;
use rand_core::{RngCore, SeedableRng};
use std::str::FromStr;

// const DIM1: usize = 14016 + 64;
const DIM1: usize = 4160 + 78 * 36;
const DIM2: usize = 128;
const DIM3: usize = 64;

const SCALE_INT: i16 = 500;
const SCALE_FLOAT: f32 = SCALE_INT as f32;

const FLAT_OFFSET: u16 = 512 * 8;
// const OFFSET: u16 = FLAT_OFFSET + 36 * 4;
const UNDER_OFFSET: u16 = FLAT_OFFSET + 64;
const STACK_LOOKUP: [u16; 7 * 7 * 7] = include!("stack_lookup.table");

pub struct Weights {
    outer: Box<[[i16; DIM2]]>,
    bias1: [i16; DIM2],
    inner1: [[f32; DIM2]; DIM3], // Todo see if more precision is needed
    bias2: [f32; DIM3],
    white: [f32; DIM3],
    black: [f32; DIM3],
    white_bias: f32,
    black_bias: f32,
}

impl Weights {
    pub fn from_file<R: std::io::Read>(mut r: R) -> Option<Self> {
        let mut s = String::new();
        r.read_to_string(&mut s).unwrap();
        let mut lines = s.lines();
        Some(Self {
            outer: array_2d_boxed::<DIM2, DIM1, i16>(lines.next()?),
            bias1: array_1d(lines.next()?),
            inner1: array_2d(lines.next()?),
            bias2: array_1d(lines.next()?),
            white: array_1d(lines.next()?),
            black: array_1d(lines.next()?),
            white_bias: lines.next().map(|x| x.parse().ok())??,
            black_bias: lines.next().map(|x| x.parse().ok())??,
        })
    }
    #[cfg(test)]
    fn ones() -> Self {
        let init = [1; DIM2];
        Self {
            outer: vec![init; DIM1].into_boxed_slice(),
            bias1: [1; DIM2],
            inner1: [[1.0; DIM2]; DIM3], // Todo see if more precision is needed
            bias2: [1.0; DIM3],
            white: [1.0; DIM3],
            black: [1.0; DIM3],
            white_bias: 1.0,
            black_bias: 1.0,
        }
    }
    pub fn build_incremental(&self, nonzero_input: &[usize]) -> Incremental {
        let mut out = Incremental {
            storage: self.bias1.clone(),
        };
        for idx in nonzero_input.iter().copied() {
            out.move_in(idx, self);
        }
        out
    }
    pub fn incremental_forward(&self, inc: &Incremental, rand_seed: Option<u64>) -> i32 {
        let mut input = [0.0; DIM2];
        for i in 0..input.len() {
            input[i] = (relu6_int(inc.storage[i]) as f32) / SCALE_FLOAT;
        }
        if let Some(seed) = rand_seed {
            let mut rng = rand_xoshiro::Xoroshiro64Star::seed_from_u64(seed);
            for x in input.iter_mut() {
                if rng.next_u32() & 0x110_0000 == 0x110_0000 {
                    *x = 0.0;
                }
            }
        }
        let mut middle = self.bias2.clone();
        for i in 0..DIM2 {
            for j in 0..DIM3 {
                middle[j] += input[i] * self.inner1[j][i];
            }
        }
        let (mut out, inner2) = (self.white_bias, &self.white);
        // let (mut out, inner2) = if is_white {
        //     (self.white_bias, &self.white)
        // } else {
        //     (self.black_bias, &self.black)
        // };

        for i in 0..middle.len() {
            out += relu6(middle[i]) * inner2[i];
        }
        // dbg!(out);
        ((out as f64).tanh() * SCALE_FLOAT as f64).trunc() as i32
    }
}

pub fn nn_repr(board: &Board6) -> Vec<u16> {
    const OFFSET: usize = 4240;
    let mut arr1 = make_array(board);
    debug_assert!(arr1.iter().all(|&x| x < (OFFSET + 64) as u16));
    arr1.extend(make_under_array(board).into_iter());
    arr1
}

fn make_array(board: &Board6) -> Vec<u16> {
    use bitintr::Pext;
    let mut out = Vec::with_capacity(26);
    // let mut bits = Vec::new();
    // for color in [board.bits().white, board.bits().black] {
    //     for piece in [board.bits().flat, board.bits().wall, board.bits().cap] {
    //         bits.push(color & piece);
    //     }
    // }
    // let mut counter = 0;
    // for mask in [0xe0e0e00, 0x70707000, 0x70707000000000, 0xe0e0e00000000] {
    //     for b in [
    //         board.bits().white & board.bits().flat,
    //         board.bits().black & board.bits().flat,
    //     ] {
    //         let val = b.raw_bits().pext(mask);
    //         out.push(counter * 512 + val as u16);
    //         counter += 1;
    //     }
    // }
    // for x in (board.bits().white & board.bits().cap).index_iter() {
    //     let val = FLAT_OFFSET + x as u16;
    //     out.push(val);
    // }
    // for x in (board.bits().black & board.bits().cap).index_iter() {
    //     let val = FLAT_OFFSET + 36 + x as u16;
    //     out.push(val);
    // }
    // for x in (board.bits().white & board.bits().wall).index_iter() {
    //     let val = FLAT_OFFSET + 72 + x as u16;
    //     out.push(val);
    // }
    // for x in (board.bits().black & board.bits().wall).index_iter() {
    //     let val = FLAT_OFFSET + 108 + x as u16;
    //     out.push(val);
    // }
    let friendly = board.side_to_move();
    let opp = !friendly;
    let white_res = board.caps_reserve(friendly) as u16 + board.pieces_reserve(friendly) as u16;
    let black_res = board.caps_reserve(opp) as u16 + board.pieces_reserve(opp) as u16;
    // Reserve advantage?
    let white_res_adv = (white_res as i16 - black_res as i16).clamp(-8, 8) + 8;
    out.push(64 + white_res_adv as u16);
    // Flat Advantage
    let white_flat_adv = board.flat_diff(friendly).clamp(-6, 6) + 6;
    out.push(82 + white_flat_adv as u16);
    // Empty squares
    let empty = std::cmp::min(board.bits().empty().pop_count(), 8);
    out.push(96 + empty as u16);
    let comp_white = std::cmp::min(
        8,
        connected_components(board.bits.road_pieces(friendly), Bitboard6::flood).steps,
    );
    let comp_black = std::cmp::min(
        8,
        connected_components(board.bits.road_pieces(opp), Bitboard6::flood).steps,
    );
    out.push(132 + comp_white as u16);
    out.push(142 + comp_black as u16);
    let attack_white = std::cmp::min(attackable_cs(friendly, board), 4);
    let attack_black = std::cmp::min(attackable_cs(opp, board), 4);
    out.push(152 + attack_white as u16);
    out.push(158 + attack_black as u16);
    // Side to move
    out.push(166 + (friendly == Color::Black) as u16);
    out
}

fn make_under_array(board: &Board6) -> Vec<u16> {
    let mut out = Vec::with_capacity(36);
    for (outer_idx, stack) in board.board().iter().enumerate() {
        let stack_piece = stack
            .top()
            .map(|x| {
                if board.side_to_move() == Color::White {
                    x as usize
                } else {
                    x.swap_color() as usize
                }
            })
            .unwrap_or(0);
        let (cap, friendly) = stack.captive_friendly();
        let cap_friendly_offset = std::cmp::min(cap, 6) * 7 + std::cmp::min(friendly, 6);
        let idx = stack_piece * 49 + cap_friendly_offset as usize;
        out.push(UNDER_OFFSET + outer_idx as u16 * 78 + STACK_LOOKUP[idx]);
        // let len = std::cmp::min(stack.len(), 17);
        // for pos in 1..len {
        //     let p = stack.from_top(pos).unwrap();
        //     let c = (p.owner() == Color::White) as usize;
        //     debug_assert!(p.is_flat());
        //     let idx = c + 2 * pos - 1 + outer_idx * 32;
        //     out.push(UNDER_OFFSET + idx as u16);
        // }
    }
    out
}

trait Scalar: FromStr + Copy {
    const ZERO: Self;
}

impl Scalar for i16 {
    const ZERO: Self = 0;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
}

fn array_2d_boxed<const A: usize, const B: usize, T: Scalar>(line: &str) -> Box<[[T; A]]> {
    let init = [T::ZERO; A];
    let mut out = vec![init; B];
    // let mut out = Box::new([[T::ZERO; A]; B]);
    let mut count = 0;
    for (i, sp) in line.split(";").enumerate() {
        for (j, val) in sp
            .split(",")
            .filter_map(|x| x.parse::<T>().ok())
            .enumerate()
        {
            out[i][j] = val;
            count += 1;
        }
    }
    assert_eq!(count, A * B);
    out.into_boxed_slice()
}

fn array_2d<const A: usize, const B: usize, T: Scalar>(line: &str) -> [[T; A]; B] {
    let mut out = [[T::ZERO; A]; B];
    let mut count = 0;
    for (i, sp) in line.split(";").enumerate() {
        for (j, val) in sp
            .split(",")
            .filter_map(|x| x.parse::<T>().ok())
            .enumerate()
        {
            out[i][j] = val;
            count += 1;
        }
    }
    assert_eq!(count, A * B);
    out
}

fn array_1d<const A: usize, T: Scalar>(line: &str) -> [T; A] {
    // dbg!(&line);
    let mut out = [T::ZERO; A];
    let mut count = 0;
    for (i, val) in line
        .split(",")
        .filter_map(|x| x.parse::<T>().ok())
        .enumerate()
    {
        out[i] = val;
        count += 1;
    }
    assert_eq!(count, A);
    out
}

fn relu6_int(x: i16) -> i16 {
    const MAX: i16 = 6 * SCALE_INT;
    if x < 0 {
        return 0;
    }
    std::cmp::min(MAX, x)
}

fn relu6(x: f32) -> f32 {
    if x < 0.0 {
        return 0.0;
    }
    f32::min(6.0, x)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Incremental {
    storage: [i16; DIM2],
}

impl Default for Incremental {
    fn default() -> Self {
        Self::new()
    }
}

impl Incremental {
    pub fn new() -> Self {
        Incremental { storage: [0; DIM2] }
    }
    pub fn update_diff(&mut self, old: &Vec<u16>, new: &Vec<u16>, weights: &Weights) {
        let mut iter_old = old.iter().copied();
        let mut iter_new = new.iter().copied();
        let mut old_val = iter_old.next();
        let mut new_val = iter_new.next();
        loop {
            match (old_val, new_val) {
                (Some(old), Some(new)) => {
                    if old == new {
                        // Duplicate, advance both pointers
                        old_val = iter_old.next();
                        new_val = iter_new.next();
                    } else if old > new {
                        // hash in new
                        self.move_in(new as usize, weights);
                        // advance new
                        new_val = iter_new.next();
                    } else {
                        // hash out old
                        self.move_out(old as usize, weights);
                        // advance old
                        old_val = iter_old.next();
                    }
                }
                (Some(old), None) => {
                    // hash out old
                    self.move_out(old as usize, weights);
                    // advance old
                    old_val = iter_old.next();
                }
                (None, Some(new)) => {
                    // hash in new
                    self.move_in(new as usize, weights);
                    // advance new
                    new_val = iter_new.next();
                }
                (None, None) => {
                    break;
                }
            }
        }
    }
    pub fn move_in(&mut self, idx: usize, weights: &Weights) {
        for i in 0..DIM2 {
            self.storage[i] += weights.outer[idx][i];
        }
    }
    pub fn move_out(&mut self, idx: usize, weights: &Weights) {
        for i in 0..DIM2 {
            self.storage[i] -= weights.outer[idx][i];
        }
    }
}

// pub fn undo_nn_repr(nn: Vec<u16>) -> Board6 {
//     let mut flats = [30, 30];
//     let mut caps = [1, 1];
//     let mut stacks = vec![Vec::new(); 36];
//     let mut bits = vec![Bitboard6::ZERO; 2];
//     let mut st = nn.iter().take_while(|&&x| x < FLAT_OFFSET as u16);

//     for mask in [0xe0e0e00, 0x70707000, 0x70707000000000, 0xe0e0e00000000] {
//         for b in bits.iter_mut() {
//             let v = *st.next().unwrap() as u64 % 512;
//             *b |= Bitboard6::new(v.pdep(mask));
//         }
//     }
//     for (mut b, p) in bits
//         .into_iter()
//         .zip([Piece::WhiteFlat, Piece::BlackFlat].into_iter())
//     {
//         while b.nonzero() {
//             let idx = b.lowest_index();
//             stacks[idx].push(p);
//             b.pop_lowest();
//             flats[p.owner() as usize] -= 1;
//         }
//     }
//     for x in nn
//         .iter()
//         .copied()
//         .filter(|&x| x >= FLAT_OFFSET && x < FLAT_OFFSET)
//     {
//         let index = ((x - FLAT_OFFSET) % 36) as usize;
//         if stacks[index].len() == 0 {
//             continue;
//         }
//         let piece = match (x - FLAT_OFFSET) / 36 {
//             0 => {
//                 caps[0] -= 1;
//                 Piece::WhiteCap
//             }
//             1 => {
//                 caps[1] -= 1;
//                 Piece::BlackCap
//             }
//             2 => {
//                 flats[0] -= 1;
//                 Piece::WhiteWall
//             }
//             _ => {
//                 flats[1] -= 1;
//                 Piece::BlackWall
//             }
//         };
//         stacks[index].push(piece);
//     }

//     for under in nn.iter().copied().filter(|&x| x >= UNDER_OFFSET) {
//         let x = under - UNDER_OFFSET - 1;
//         let p = if x % 2 == 1 {
//             flats[0] -= 1;
//             Piece::WhiteFlat
//         } else {
//             flats[1] -= 1;
//             Piece::BlackFlat
//         };
//         let outer = (x / 32) as usize;
//         // assert!(stacks[outer].len() != 0);
//         stacks[outer as usize].push(p);
//     }
//     for s in stacks.iter_mut() {
//         s.reverse();
//     }
//     let mut out = Board6::new();
//     let mut storage = BitboardStorage::<Bitboard6>::default();
//     for (idx, st) in stacks.into_iter().enumerate() {
//         for p in st.into_iter() {
//             out.board[idx].push(p, &mut storage);
//         }
//     }
//     out.reset_stacks();
//     out
// }

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compute_diff() {
        let weights = &Weights::ones();
        let mut increment = Incremental::default();
        let old = vec![12, 56, 112, 1096];
        let new = vec![12, 40, 51, 52, 112, 1200];
        let mut iter_old = old.iter().copied();
        let mut iter_new = new.iter().copied();
        let mut old_val = iter_old.next();
        let mut new_val = iter_new.next();
        // Debug purposes only
        let mut hash_in = Vec::new();
        let mut hash_out = Vec::new();
        loop {
            match (old_val, new_val) {
                (Some(old), Some(new)) => {
                    if old == new {
                        // Duplicate, advance both pointers
                        old_val = iter_old.next();
                        new_val = iter_new.next();
                    } else if old > new {
                        // hash in new
                        increment.move_in(new, weights);
                        hash_in.push(new);
                        // advance new
                        new_val = iter_new.next();
                    } else {
                        // hash out old
                        increment.move_out(old, weights);
                        hash_out.push(old);
                        // advance old
                        old_val = iter_old.next();
                    }
                }
                (Some(old), None) => {
                    // hash out old
                    increment.move_out(old, weights);
                    hash_out.push(old);
                    // advance old
                    old_val = iter_old.next();
                }
                (None, Some(new)) => {
                    // hash in new
                    increment.move_in(new, weights);
                    hash_in.push(new);
                    // advance new
                    new_val = iter_new.next();
                }
                (None, None) => {
                    break;
                }
            }
        }
        assert_eq!(vec![56, 1096], hash_out);
        assert_eq!(vec![40, 51, 52, 1200], hash_in);
    }
    #[test]
    fn dummy() {
        // let v = vec![
        //     16, 512, 1032, 1604, 2305, 2560, 3078, 3652, 4110, 4403, 4817, 4819, 4851, 5010,
        // ];
        let s = "68 516 1025 1808 2112 2624 3073 3592 4594 4626";
        let v = s.split_whitespace().map(|x| x.parse().unwrap()).collect();

        let b = undo_nn_repr(v);
        dbg!(b);
        assert!(false);
    }
    #[test]
    fn test_undo_nn_repr() {
        let tps = "1,12,1,1,x2/x3,1,x2/x,2,2,1C,1,1/x2,1,12C,12,221/2,2,12,2,x,2/x3,1,x2 1 17";
        let board = crate::Board6::try_from_tps(tps).unwrap();
        // let mut out = Vec::with_capacity(26);
        // for color in [board.bits().white, board.bits().black] {
        //     for piece in [board.bits().flat, board.bits().wall, board.bits().cap] {
        //         bits.push(color & piece);
        //     }
        // }
        // let mut counter = 0;
        // for mask in [0xe0e0e00, 0x70707000, 0x70707000000000, 0xe0e0e00000000] {
        //     for b in bits.iter().copied() {
        //         let val = b.raw_bits().pext(mask);
        //         out.push(counter * 512 + val as u16);
        //         counter += 1;
        //     }
        // }
        let mut flats = [30, 30];
        let mut caps = [1, 1];
        let mut stacks = vec![Vec::new(); 36];
        let mut bits = vec![Bitboard6::ZERO; 2];
        let nn = nn_repr(&board);
        let mut st = nn.iter().take_while(|&&x| x < FLAT_OFFSET as u16);

        for mask in [0xe0e0e00, 0x70707000, 0x70707000000000, 0xe0e0e00000000] {
            for b in bits.iter_mut() {
                let v = *st.next().unwrap() as u64 % 512;
                *b |= Bitboard6::new(v.pdep(mask));
            }
        }
        for (mut b, p) in bits
            .into_iter()
            .zip([Piece::WhiteFlat, Piece::BlackFlat].into_iter())
        {
            while b.nonzero() {
                let idx = b.lowest_index();
                stacks[idx].push(p);
                b.pop_lowest();
                flats[p.owner() as usize] -= 1;
            }
        }
        for x in nn
            .iter()
            .copied()
            .filter(|&x| x >= FLAT_OFFSET && x < OFFSET)
        {
            let piece = match (x - FLAT_OFFSET) / 36 {
                0 => {
                    caps[0] -= 1;
                    Piece::WhiteCap
                }
                1 => {
                    caps[1] -= 1;
                    Piece::BlackCap
                }
                2 => {
                    flats[0] -= 1;
                    Piece::WhiteWall
                }
                _ => {
                    flats[1] -= 1;
                    Piece::BlackWall
                }
            };
            let index = (x - FLAT_OFFSET) % 36;
            stacks[index as usize].push(piece);
        }
        // Todo reserves / understacks
        let res: Vec<_> = nn
            .iter()
            .copied()
            .filter(|&x| x >= OFFSET && x < UNDER_OFFSET)
            .collect();
        assert_eq!(res.len(), 2);
        dbg!(res[0]);
        dbg!(res[1]);
        for under in nn.iter().copied().filter(|&x| x >= UNDER_OFFSET) {
            let x = under - UNDER_OFFSET - 1;
            dbg!(x);
            let p = if x % 2 == 1 {
                flats[0] -= 1;
                Piece::WhiteFlat
            } else {
                flats[1] -= 1;
                Piece::BlackFlat
            };
            dbg!(p);
            let outer = (x / 32) as usize;
            assert!(stacks[outer].len() != 0);
            // if stacks[outer].len() == 0 {
            //     continue;
            // }
            stacks[outer as usize].push(p);
        }
        for s in stacks.iter_mut() {
            s.reverse();
        }
        let mut out = Board6::new();
        let mut storage = BitboardStorage::<Bitboard6>::default();
        for (idx, st) in stacks.into_iter().enumerate() {
            for p in st.into_iter() {
                out.board[idx].push(p, &mut storage);
            }
        }
        out.reset_stacks();
        dbg!(out.pieces_reserve(Color::White) + out.caps_reserve(Color::White));
        dbg!(out.pieces_reserve(Color::Black) + out.caps_reserve(Color::Black));
        assert_eq!(
            format!("{:?}", board).split_whitespace().nth(0),
            format!("{:?}", out).split_whitespace().nth(0)
        );
    }
}
