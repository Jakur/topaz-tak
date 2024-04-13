use crate::eval::attackable_cs;
use crate::Bitboard;
use crate::Position;
use crate::{board::TakBoard, Board6, Color};
use std::str::FromStr;

use super::flat_placement_road_h;

// const DIM1: usize = 2976;
// pub(crate) const DIM1: usize = 5696;
pub const NN_REPR_SIZE: usize = 36 + 12;
pub type NN_REPR = [i16; NN_REPR_SIZE];
pub(crate) const DIM1: usize = 14080;
// pub(crate) const DIM1: usize = 2808;
const DIM2: usize = 128;
const DIM3: usize = 8;

const SCALE_INT: i16 = 500;
const SCALE_FLOAT: f32 = SCALE_INT as f32;

// const OFFSET: u16 = FLAT_OFFSET + 36 * 4;
// const UNDER_OFFSET: u16 = 2648;
const UNDER_OFFSET: u16 = 2680;
pub(crate) const STACK_LOOKUP: [i16; 384] = include!("stack_lookup.table");
const CONV_COMPRESS: [u16; 3usize.pow(9)] = include!("conv.table");

pub struct Weights {
    outer: Box<[[f32; DIM2]]>,
    inner: [[f32; DIM2]; DIM3],
    inner_bias: [f32; DIM3],
    side: [f32; DIM3],
    opp: [f32; DIM3],
}

impl Weights {
    pub fn from_file<R: std::io::Read>(mut r: R) -> Option<Self> {
        let mut s = String::new();
        r.read_to_string(&mut s).unwrap();
        let mut lines = s.lines();
        Some(Self {
            outer: array_2d_boxed::<DIM2, DIM1, f32>(lines.next()?),
            inner: array_2d(lines.next()?),
            inner_bias: array_1d(lines.next()?),
            side: array_1d(lines.next()?),
            opp: array_1d(lines.next()?),
        })
    }
    #[cfg(test)]
    fn ones() -> Self {
        let init = [1.0; DIM2];
        Self {
            outer: vec![init; DIM1].into_boxed_slice(),
            inner: [[1.0; DIM2]; DIM3],
            inner_bias: [1.0; DIM3],
            side: [1.0; DIM3],
            opp: [1.0; DIM3],
        }
    }
    pub fn build_incremental(&self, nonzero_input: &[usize]) -> Incremental {
        let mut out = Incremental {
            storage: [0.0; DIM2],
        };
        for idx in nonzero_input.iter().copied() {
            out.move_in(idx, self);
        }
        out
    }
    pub fn incremental_forward(&self, side: &Incremental, opp: &Incremental) -> f32 {
        let mut out = 0.0;
        let side_emb = self.get_embedding(side);
        let opp_emb = self.get_embedding(opp);
        for (val, w) in side_emb.into_iter().zip(self.side.into_iter()) {
            out += val * w;
        }
        for (val, w) in opp_emb.into_iter().zip(self.opp.into_iter()) {
            out += val * w;
        }
        out
    }
    fn get_embedding(&self, vals: &Incremental) -> [f32; DIM3] {
        let mut input = [0.0; DIM2];
        for i in 0..DIM2 {
            input[i] = square_clipped_relu(vals.storage[i]);
        }
        let mut middle = self.inner_bias.clone();
        for i in 0..DIM2 {
            for j in 0..DIM3 {
                middle[j] += input[i] * self.inner[j][i];
            }
        }
        for m in middle.iter_mut() {
            *m = float_relu(*m);
        }
        middle
    }
}

pub fn nn_repr(board: &Board6) -> NN_REPR {
    // const OFFSET: usize = 4240;
    let mut out = [0; NN_REPR_SIZE];
    let side = board.side_to_move();
    for idx in 0..36 {
        let stack = board.index(idx);
        // if let Some(top) = stack.top() {
        //     let (cap, friendly) = stack.captive_friendly();
        //     let adjusted_top = if board.side_to_move() == Color::White {
        //         top as i16
        //     } else {
        //         top.swap_color() as i16
        //     };
        //     let (c_score, f_score) = if top.owner() == board.side_to_move() {
        //         (cap as i16, friendly as i16)
        //     } else {
        //         (-cap as i16, -friendly as i16)
        //     };
        //     match top {
        //         crate::Piece::WhiteFlat | crate::Piece::BlackFlat => {
        //             out[39] += c_score;
        //             out[40] += f_score;
        //         }
        //         crate::Piece::WhiteWall | crate::Piece::BlackWall => {
        //             out[41] += c_score;
        //             out[42] += f_score;
        //         }
        //         crate::Piece::WhiteCap | crate::Piece::BlackCap => {
        //             out[43] += c_score;
        //             out[44] += f_score;
        //         }
        //     }
        //     // out[idx] = adjusted_top * 9
        //     //     + 3 * std::cmp::min(friendly as i16, 2)
        //     //     + std::cmp::min(cap as i16, 2);
        // } else {
        //     // out[idx] = 0;
        // }
        out[idx] = stack.interpret(side) as i16;
    }
    // Todo find out why index 37 does not match on side swap
    out[36] = (board.pieces_reserve(side) + board.caps_reserve(side)) as i16;
    out[37] = (board.pieces_reserve(!side) + board.caps_reserve(!side)) as i16;
    out[36] /= 2;
    out[37] /= 2;
    let empty = board.bits().empty();
    let (side_f, side_comp) = flat_placement_road_h(board.bits().road_pieces(side), empty);
    let (enemy_f, enemy_comp) = flat_placement_road_h(board.bits().road_pieces(!side), empty);
    out[38] = std::cmp::min(6, side_f as i16);
    out[39] = std::cmp::min(6, enemy_f as i16);
    out[40] = std::cmp::min(6, side_comp as i16);
    out[41] = std::cmp::min(6, enemy_comp as i16);
    out[42] = std::cmp::min(6, attackable_cs(side, board) as i16);
    out[43] = std::cmp::min(6, attackable_cs(!side, board) as i16);

    out[44] = if side == Color::White { 1 } else { -1 };
    out[45] = out[36] - out[37];
    out[46] = board.flat_diff(side) as i16;
    out[47] = board.caps_reserve(side) as i16 - (board.caps_reserve(!side) as i16);

    for i in 45..NN_REPR_SIZE {
        out[i] = out[i].clamp(-6, 6);
    }
    out
}

fn make_array(board: &Board6) -> Vec<u16> {
    use bitintr::Pext;
    let mut out = Vec::with_capacity(36 + 10);
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
    let friendly_res = board.caps_reserve(friendly) as u16 + board.pieces_reserve(friendly) as u16;
    let enemy_res = board.caps_reserve(opp) as u16 + board.pieces_reserve(opp) as u16;
    // Reserve advantage?
    let res_adv = (enemy_res as i16 - friendly_res as i16).clamp(-4, 4) + 4;
    // Flat Advantage
    let flat_adv = board.flat_diff(friendly).clamp(-4, 4) + 4;
    let res_flat_idx = friendly_res * 81 + (res_adv as u16) * 9 + (flat_adv as u16);
    out.push(res_flat_idx);
    // Offset 2592
    // Empty squares
    let empty_bits = board.bits().empty();
    let empty = std::cmp::min(empty_bits.pop_count(), 15);
    out.push(2592 + empty as u16);
    // let (road_est1, comp) = super::one_gap_road(board.bits.road_pieces(friendly));
    let (road_est2, comp) =
        super::flat_placement_road_h(board.bits.road_pieces(friendly), empty_bits);
    let comp_friendly = std::cmp::min(7, comp);
    // let (road_est3, comp) = super::one_gap_road(board.bits.road_pieces(opp));
    let (road_est4, comp) = super::flat_placement_road_h(board.bits.road_pieces(opp), empty_bits);
    let comp_black = std::cmp::min(7, comp);
    out.push(2608 + comp_friendly as u16);
    out.push(2616 + comp_black as u16);
    let attack_white = std::cmp::min(attackable_cs(friendly, board), 4);
    let attack_black = std::cmp::min(attackable_cs(opp, board), 4);
    out.push(2624 + attack_white as u16);
    out.push(2632 + attack_black as u16);
    // Side to move
    out.push(2640 + (friendly == Color::Black) as u16);
    // out.push(2648 + road_est1 as u16); // Max 6
    out.push(2656 + std::cmp::min(7, road_est2) as u16);
    // out.push(2664 + road_est3 as u16); // Max 6
    out.push(2672 + std::cmp::min(7, road_est4) as u16);
    out
}

// fn make_under_array2(board: &Board6) -> [u16; 36] {
//     let mut out = [0; 36];
//     for (outer_idx, stack) in board.board().iter().enumerate() {
//         if let Some(top) = stack.top() {
//             let top_idx = if board.side_to_move() == Color::White {
//                 top.owner() as u16
//             } else {
//                 !top.owner() as u16
//             };
//             let offset;
//             if top.is_flat() {
//                 let (cap, _) = stack.captive_friendly();
//                 offset = std::cmp::min(7, cap) as u16 + 8 * top_idx
//             } else if top.is_wall() {
//                 let (_, friendly) = stack.captive_friendly();
//                 // Wall (16)
//                 if friendly == 0 {
//                     offset = 16 + top_idx
//                 } else {
//                     offset = 16 + 2 + top_idx
//                 }
//             } else {
//                 // Capstone (20)
//                 if let Some(under) = stack.from_top(1) {
//                     if under.owner() == top.owner() {
//                         offset = 22 + top_idx
//                     } else {
//                         offset = 24 + top_idx
//                     }
//                 } else {
//                     offset = 20 + top_idx;
//                 }
//             }
//             // assert!(offset <= 25);
//             out[outer_idx] = UNDER_OFFSET + 27 * (outer_idx as u16) + offset;
//         } else {
//             out[outer_idx] = UNDER_OFFSET + 27 * (outer_idx as u16) + 26;
//         }
//     }
//     out
// }

// fn make_under_array(board: &Board6) -> [u16; 36] {
//     let mut out = [0; 36];
//     for (outer_idx, stack) in board.board().iter().enumerate() {
//         let stack_piece = stack
//             .top()
//             .map(|x| {
//                 if board.side_to_move() == Color::White {
//                     x as usize
//                 } else {
//                     x.swap_color() as usize
//                 }
//             })
//             .unwrap_or(0);
//         let (cap, friendly) = stack.captive_friendly();
//         let cap_friendly_offset = std::cmp::min(cap, 6) * 7 + std::cmp::min(friendly, 6);
//         let idx = stack_piece * 49 + cap_friendly_offset as usize;
//         out[outer_idx] = UNDER_OFFSET + outer_idx as u16 * 78 + STACK_LOOKUP[idx];
//         // out.push(UNDER_OFFSET + outer_idx as u16 * 78 + STACK_LOOKUP[idx]);
//         // let len = std::cmp::min(stack.len(), 17);
//         // for pos in 1..len {
//         //     let p = stack.from_top(pos).unwrap();
//         //     let c = (p.owner() == Color::White) as usize;
//         //     debug_assert!(p.is_flat());
//         //     let idx = c + 2 * pos - 1 + outer_idx * 32;
//         //     out.push(UNDER_OFFSET + idx as u16);
//         // }
//     }
//     out
// }

pub(crate) trait Scalar: FromStr + Copy {
    const ZERO: Self;
}

impl Scalar for i16 {
    const ZERO: Self = 0;
}

impl Scalar for i32 {
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

pub(crate) fn array_1d_boxed<const A: usize, T: Scalar>(line: &str) -> Box<[T]> {
    // dbg!(&line);
    let mut out = vec![T::ZERO; A];
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
    out.into_boxed_slice()
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

#[inline]
fn square_clipped_relu(x: f32) -> f32 {
    let val = x.clamp(0.0, 1.0);
    val * val
}

#[inline]
fn float_relu(x: f32) -> f32 {
    0.0_f32.max(x)
}

// #[inline]
// fn square_clipped_relu(x: i16) -> i32 {
//     let val = x.clamp(0, 255) as i32;
//     (val * val) / 255 // Still one factor of 255
// }

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
    storage: [f32; DIM2],
}

impl Default for Incremental {
    fn default() -> Self {
        Self::new()
    }
}

impl Incremental {
    pub fn new() -> Self {
        Incremental {
            storage: [0.0; DIM2],
        }
    }
    pub fn update_diff(&mut self, old: &NN_REPR, new: &NN_REPR, weights: &Weights) {
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

pub(crate) fn flip_stack_repr(val: i16) -> i16 {
    let table: [i16; 64] = [
        0, 1, 3, 2, 7, 6, 5, 4, 15, 14, 13, 12, 11, 10, 9, 8, 31, 30, 29, 28, 27, 26, 25, 24, 23,
        22, 21, 20, 19, 18, 17, 16, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
    ];
    table[val as usize]
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
}
