use crate::board::{BitIndexIterator, Bitboard6};
use crate::{Bitboard, Board6, Color};
use std::str::FromStr;

// const DIM1: usize = 2976;
// pub(crate) const DIM1: usize = 5696;
// pub const NN_REPR_SIZE: usize = 31 * 2;
pub const NN_REPR_SIZE: usize = 51;
pub(crate) const DIM1: usize = 440;
const NULL_IDX: u16 = DIM2 as u16 - 1;
// pub(crate) const DIM1: usize = 2808;
const DIM2: usize = 512;
const DIM3: usize = 8;

const SCALE_INT: i16 = 500;
const SCALE_FLOAT: f32 = SCALE_INT as f32;

// const OFFSET: u16 = FLAT_OFFSET + 36 * 4;
// const UNDER_OFFSET: u16 = 2648;
const UNDER_OFFSET: u16 = 0;
pub(crate) const STACK_LOOKUP: [u16; 343] = include!("stack_lookup.table");
// const CONV_COMPRESS: [u16; 3usize.pow(9)] = include!("conv.table");

pub struct Weights {
    white: Box<[[f32; DIM2]]>,
    black: Box<[[f32; DIM2]]>,
    linear: [f32; DIM2],
    bias: f32,
}

impl Weights {
    pub fn from_file<R: std::io::Read>(mut r: R) -> Option<Self> {
        let mut s = String::new();
        r.read_to_string(&mut s).unwrap();
        let mut lines: Vec<_> = s.lines().collect();
        Some(Self {
            white: take_n_lines::<DIM1, DIM2, f32>(&lines[0..DIM1]),
            black: take_n_lines::<DIM1, DIM2, f32>(&lines[DIM1..2 * DIM1]),
            linear: array_1d(lines[lines.len() - 2]),
            bias: lines.last()?.parse().ok()?,
        })
    }
    #[cfg(test)]
    fn ones() -> Self {
        let init = [1.0; DIM2];
        Self {
            white: vec![init; DIM1].into_boxed_slice(),
            black: vec![init; DIM1].into_boxed_slice(),
            linear: init,
            bias: 1.0,
        }
    }
    pub fn empty_incremental(&self) -> Incremental {
        Incremental::new()
    }
    pub fn build_incremental(&self, nonzero_input: &[usize], color: Color) -> Incremental {
        let mut out = Incremental {
            storage: [0.0; DIM2],
        };
        let outer = match color {
            Color::White => &self.white,
            Color::Black => &self.black,
        };
        for idx in nonzero_input.iter().copied() {
            out.move_in(idx, outer);
        }
        out
    }
    pub fn incremental_forward(&self, incremental: &Incremental) -> f32 {
        let mut out = self.bias;
        for i in 0..DIM2 {
            let val = incremental.storage[i];
            out += self.linear[i] * relu6(val);
        }
        out
    }
    fn get_embedding(&self, vals: &Incremental) -> [f32; DIM3] {
        todo!()
    }
}

pub fn nn_repr(board: &Board6) -> NNRepr {
    make_array(board)
}

// fn make_array(board: &Board6) -> NNRepr {
//     const SQ_OFFSET: u16 = 24;
//     // let mut out = Vec::with_capacity(31 * 2);
//     let mut out = NNRepr::empty();
//     for sq in BitIndexIterator::new(board.bits.white | board.bits.black) {
//         let mut offset = sq as u16 * SQ_OFFSET;
//         let stack = &board.board[sq];
//         let top = stack.top().unwrap() as u16 - 1; // 0 to 5
//         out.push(offset + top);
//         offset += 6;
//         for stack_idx in 1..=8 {
//             let piece_color = stack.from_top(stack_idx).map(|x| x.owner() as u16);
//             if let Some(piece_color) = piece_color {
//                 out.push(offset + (stack_idx as u16) * 2 + piece_color)
//             } else {
//                 break;
//             }
//         }
//     }
//     debug_assert!(out.len() <= 62);
//     out
// }

fn make_array(board: &Board6) -> NNRepr {
    use crate::{Position, TakBoard};
    use bitintr::Pext;
    const MISC_OFFSET: u16 = 6 * 4096;
    const STACK_OFFSET: u16 = MISC_OFFSET + 1152;
    let side = board.side_to_move();
    let mut out = NNRepr::empty();
    for sq_idx in [0, 7, 14, 21, 28, 35] {
        let mask = Bitboard6::orthogonal(sq_idx);
        for (color_idx, color) in [side, !side].into_iter().enumerate() {
            let bits = board.bits().road_pieces(color);
            let unique_idx = bits.raw_bits().pext(mask.raw_bits()); // 0 to 2^11
            assert!(unique_idx < 2048);
            out.push((sq_idx as u16 / 7) * 4096 + 2048 * color_idx as u16 + unique_idx as u16)
        }
    }
    let double_fcd = (board.flat_diff(side).clamp(-12, 12) + 12) as u16;
    let side_reserves = board.pieces_reserve(side) as u16;
    let enemy_reserves = board.pieces_reserve(!side) as u16;
    out.push(MISC_OFFSET + double_fcd);
    out.push(MISC_OFFSET + 32 + side_reserves * 32 + enemy_reserves);
    if let Color::White = side {
        out.push(MISC_OFFSET + 1150);
    } else {
        out.push(MISC_OFFSET + 1151);
    }
    // out.push(MISC_OFFSET + 24 + side_reserves);
    // out.push(MISC_OFFSET + 24 + 32 + enemy_reserves);
    let stacks = make_under_array(board);
    for val in stacks {
        out.push(val + STACK_OFFSET);
    }
    // for sq in BitIndexIterator::new(board.bits.white | board.bits.black) {
    //     let mut offset = sq as u16 * SQ_OFFSET;
    //     let stack = &board.board[sq];
    //     let top = if let Color::White = side {
    //         stack.top().unwrap()
    //     } else {
    //         stack.top().unwrap().swap_color()
    //     };
    //     let top_idx = top as u16 - 1; // 0 to 5
    //     offset += 6;
    //     let base_idx = STACK_OFFSET + SQ_OFFSET * sq as u16 + top_idx * 64;
    //     let mut friendly_idx = 0; // 0 - 64
    //     let mut enemy_idx = 0; // 0 - 64
    //     for stack_idx in 1..=6 {
    //         let piece_color = stack
    //             .from_top(stack_idx)
    //             .map(|x| (x.owner() == side) as u16);
    //         if let Some(piece_color) = piece_color {
    //             if piece_color == 1 {
    //                 friendly_idx |= 1 << (stack_idx - 1)
    //             } else {
    //                 enemy_idx |= 1 << (stack_idx - 1);
    //             }
    //         } else {
    //             break;
    //         }
    //     }
    // }
    out
}

// fn make_array(board: &Board6) -> NNRepr {
//     const SQ_OFFSET: u16 = 42;
//     // let mut out = Vec::with_capacity(31 * 2);
//     let mut out = NNRepr::empty();
//     for sq in BitIndexIterator::new(board.bits.white | board.bits.black) {
//         let mut offset = sq as u16 * SQ_OFFSET;
//         let stack = &board.board[sq];
//         let top = stack.top().unwrap() as u16 - 1; // 0 to 5
//         out.push(offset + top);
//         offset += 6;
//         if let Some(under) = stack.from_top(1).map(|x| x.owner()) {
//             out.push(offset + 2 * top + (under as u16)); // sq_offset + []
//             offset += 12;
//             let deep = stack.white_black_deep(); // [White, Black, White_Very_Deep, Black_Very_Deep]
//             for _ in 0..deep[0] {
//                 out.push(offset + top);
//             }
//             offset += 6;
//             for _ in 0..deep[1] {
//                 out.push(offset + top);
//             }
//             offset += 6;
//             for _ in 0..deep[2] {
//                 out.push(offset + top);
//             }
//             offset += 6;
//             for _ in 0..deep[3] {
//                 out.push(offset + top);
//             }
//             // let mut manual = [0, 0, 0, 0];
//             // while let Some(further) = stack.from_top(idx).map(|x| x.owner()) {
//             //     if idx > 7 {
//             //         // Todo how to treat the last piece that cannot be moved but can be revealed
//             //         out.push(sq_offset + further as u16 + 10);
//             //         manual[2 + further as usize] += 1;
//             //     } else {
//             //         out.push(sq_offset + further as u16 + 8);
//             //         manual[further as usize] += 1;
//             //     }
//             //     idx += 1;
//             // }
//         }
//     }
//     debug_assert!(out.len() <= 62);
//     out
// }

// fn make_array(board: &Board6) -> NNRepr {
//     const SQ_OFFSET: u16 = 12;
//     // let mut out = Vec::with_capacity(31 * 2);
//     let mut out = NNRepr::empty();
//     for sq in BitIndexIterator::new(board.bits.white | board.bits.black) {
//         let sq_offset = sq as u16 * SQ_OFFSET;
//         let stack = &board.board[sq];
//         let top = stack.top().unwrap() as u16 - 1; // 0 to 5
//         out.push(sq_offset + top);
//         if let Some(under) = stack.from_top(1).map(|x| x.owner()) {
//             out.push(sq_offset + under as u16 + 6); // 6 or 7
//             let deep = stack.white_black_deep(); // [White, Black, White_Very_Deep, Black_Very_Deep]
//             for _ in 0..deep[0] {
//                 out.push(sq_offset + 8);
//             }
//             for _ in 0..deep[1] {
//                 out.push(sq_offset + 9);
//             }
//             for _ in 0..deep[2] {
//                 out.push(sq_offset + 10);
//             }
//             for _ in 0..deep[3] {
//                 out.push(sq_offset + 11);
//             }
//             // let mut manual = [0, 0, 0, 0];
//             // while let Some(further) = stack.from_top(idx).map(|x| x.owner()) {
//             //     if idx > 7 {
//             //         // Todo how to treat the last piece that cannot be moved but can be revealed
//             //         out.push(sq_offset + further as u16 + 10);
//             //         manual[2 + further as usize] += 1;
//             //     } else {
//             //         out.push(sq_offset + further as u16 + 8);
//             //         manual[further as usize] += 1;
//             //     }
//             //     idx += 1;
//             // }
//         }
//     }
//     debug_assert!(out.len() <= 62);
//     out
// }

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

fn make_under_array(board: &Board6) -> [u16; 36] {
    use crate::board::TakBoard;
    use crate::Position;
    let mut out = [0; 36];
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
        out[outer_idx] = UNDER_OFFSET + outer_idx as u16 * 78 + STACK_LOOKUP[idx];
        // out.push(UNDER_OFFSET + outer_idx as u16 * 78 + STACK_LOOKUP[idx]);
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

fn take_n_lines<const N: usize, const I: usize, T: Scalar>(lines: &[&str]) -> Box<[[f32; I]]> {
    assert_eq!(N, lines.len());
    let mut out = Vec::new();
    for line in lines {
        out.push(array_1d(line));
    }
    out.into_boxed_slice()
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

#[derive(Debug)]
pub struct NNRepr {
    data: [u16; NN_REPR_SIZE],
    len: usize,
}

impl IntoIterator for NNRepr {
    type Item = u16;

    type IntoIter = std::iter::Take<std::array::IntoIter<u16, NN_REPR_SIZE>>;

    fn into_iter(self) -> Self::IntoIter {
        // self.data.into_iter().take_while(|&x| x != NULL_IDX)
        self.data.into_iter().take(self.len)
    }
}

impl NNRepr {
    pub const fn size() -> usize {
        NN_REPR_SIZE
    }
    pub fn empty() -> Self {
        Self {
            data: [NULL_IDX; NN_REPR_SIZE],
            len: 0,
        }
    }
    pub fn get(&self, idx: usize) -> Option<u16> {
        if idx >= self.len {
            None
        } else {
            Some(self.data[idx])
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_sorted(&self) -> bool {
        for (a, b) in self.iter().zip(self.iter().skip(1)) {
            if a > b {
                return false;
            }
        }
        true
    }
    fn push(&mut self, val: u16) {
        self.data[self.len] = val;
        self.len += 1;
    }
    fn iter(&self) -> impl Iterator<Item = u16> + '_ {
        // self.data.iter().copied().take_while(|&x| x != NULL_IDX)
        self.data.iter().copied().take(self.len)
    }
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
    pub fn update_diff(&mut self, old: &NNRepr, new: &NNRepr, weights: &Weights, color: Color) {
        let mut iter_old = old.iter();
        let mut iter_new = new.iter();
        let mut old_val = iter_old.next();
        let mut new_val = iter_new.next();
        let outer = match color {
            Color::White => &weights.white,
            Color::Black => &weights.black,
        };
        loop {
            match (old_val, new_val) {
                (Some(old), Some(new)) => {
                    if old == new {
                        // Duplicate, advance both pointers
                        old_val = iter_old.next();
                        new_val = iter_new.next();
                    } else if old > new {
                        // hash in new
                        self.move_in(new as usize, outer);
                        // advance new
                        new_val = iter_new.next();
                    } else {
                        // hash out old
                        self.move_out(old as usize, outer);
                        // advance old
                        old_val = iter_old.next();
                    }
                }
                (Some(old), None) => {
                    // hash out old
                    self.move_out(old as usize, outer);
                    // advance old
                    old_val = iter_old.next();
                }
                (None, Some(new)) => {
                    // hash in new
                    self.move_in(new as usize, outer);
                    // advance new
                    new_val = iter_new.next();
                }
                (None, None) => {
                    break;
                }
            }
        }
    }
    pub fn move_in(&mut self, idx: usize, weights: &[[f32; DIM2]]) {
        for i in 0..DIM2 {
            self.storage[i] += weights[idx][i];
        }
    }
    pub fn move_out(&mut self, idx: usize, weights: &[[f32; DIM2]]) {
        for i in 0..DIM2 {
            self.storage[i] -= weights[idx][i];
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
    use crate::TakBoard;

    #[test]
    fn sorted_array_repr() {
        let tps =
            "2,2,2,x2,1/x2,2,1,2,1/x,2,212S,x,1,1/x,2,21C,12C,x,1/1,1,1,x,2112,1/x3,1,2,2 1 20";
        let board = Board6::try_from_tps(tps).unwrap();
        let vals = make_array(&board);
        assert!(vals.is_sorted());
    }

    // #[test]
    // fn compute_diff() {
    //     let weights = &Weights::ones();
    //     let mut increment = Incremental::default();
    //     let old = vec![12, 56, 112, 1096];
    //     let new = vec![12, 40, 51, 52, 112, 1200];
    //     let mut iter_old = old.iter().copied();
    //     let mut iter_new = new.iter().copied();
    //     let mut old_val = iter_old.next();
    //     let mut new_val = iter_new.next();
    //     // Debug purposes only
    //     let mut hash_in = Vec::new();
    //     let mut hash_out = Vec::new();
    //     loop {
    //         match (old_val, new_val) {
    //             (Some(old), Some(new)) => {
    //                 if old == new {
    //                     // Duplicate, advance both pointers
    //                     old_val = iter_old.next();
    //                     new_val = iter_new.next();
    //                 } else if old > new {
    //                     // hash in new
    //                     increment.move_in(new, weights);
    //                     hash_in.push(new);
    //                     // advance new
    //                     new_val = iter_new.next();
    //                 } else {
    //                     // hash out old
    //                     increment.move_out(old, weights);
    //                     hash_out.push(old);
    //                     // advance old
    //                     old_val = iter_old.next();
    //                 }
    //             }
    //             (Some(old), None) => {
    //                 // hash out old
    //                 increment.move_out(old, weights);
    //                 hash_out.push(old);
    //                 // advance old
    //                 old_val = iter_old.next();
    //             }
    //             (None, Some(new)) => {
    //                 // hash in new
    //                 increment.move_in(new, weights);
    //                 hash_in.push(new);
    //                 // advance new
    //                 new_val = iter_new.next();
    //             }
    //             (None, None) => {
    //                 break;
    //             }
    //         }
    //     }
    //     assert_eq!(vec![56, 1096], hash_out);
    //     assert_eq!(vec![40, 51, 52, 1200], hash_in);
    // }
}
