use crate::{board::TakBoard, Board6};
use std::str::FromStr;

const DIM1: usize = 14016;
const DIM2: usize = 128;
const DIM3: usize = 64;

const SCALE_INT: i16 = 500;
const SCALE_FLOAT: f32 = SCALE_INT as f32;

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
    fn zeros() -> Self {
        let init = [0; DIM2];
        Self {
            outer: vec![init; DIM1].into_boxed_slice(),
            bias1: [0; DIM2],
            inner1: [[0.0; DIM2]; DIM3], // Todo see if more precision is needed
            bias2: [0.0; DIM3],
            white: [0.0; DIM3],
            black: [0.0; DIM3],
            white_bias: 0.0,
            black_bias: 0.0,
        }
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
    pub fn incremental_forward(&self, inc: &Incremental, is_white: bool) -> i32 {
        let mut input = [0.0; DIM2];
        for i in 0..input.len() {
            input[i] = (relu6_int(inc.storage[i]) as f32) / SCALE_FLOAT;
        }
        let mut middle = self.bias2.clone();
        for i in 0..DIM2 {
            for j in 0..DIM3 {
                middle[j] += input[i] * self.inner1[j][i];
            }
        }
        // dbg!(middle);
        // for w in self.inner1 {
        //     dbg!(w.len());
        //     for i in 0..w.len() {
        //         middle[i] += input[i] * w[i];
        //     }
        // }
        let (mut out, inner2) = if is_white {
            (self.white_bias, &self.white)
        } else {
            (self.black_bias, &self.black)
        };

        for i in 0..middle.len() {
            out += relu6(middle[i]) * inner2[i];
        }
        // dbg!(out);
        ((out as f64).tanh() * SCALE_FLOAT as f64).trunc() as i32
    }
}

pub fn nn_repr(board: &Board6) -> Vec<u16> {
    const OFFSET: usize = 12288;
    let mut arr1 = make_array(board);
    debug_assert!(arr1.iter().all(|&x| x < OFFSET as u16));
    arr1.extend(make_under_array(board).into_iter());
    arr1
}

fn make_array(board: &Board6) -> Vec<u16> {
    use crate::Bitboard;
    use bitintr::Pext;
    let mut bits = Vec::new();
    let mut out = Vec::with_capacity(24);
    for color in [board.bits().white, board.bits().black] {
        for piece in [board.bits().flat, board.bits().wall, board.bits().cap] {
            bits.push(color & piece);
        }
    }
    let mut counter = 0;
    for mask in [0xe0e0e00, 0x70707000, 0x70707000000000, 0xe0e0e00000000] {
        for b in bits.iter().copied() {
            let val = b.raw_bits().pext(mask);
            out.push(counter * 512 + val as u16);
            counter += 1;
        }
    }
    out
}

fn make_under_array(board: &Board6) -> Vec<u16> {
    const OFFSET: usize = 12288;
    let mut out = Vec::with_capacity(32);
    for (outer_idx, stack) in board.board().iter().enumerate() {
        let len = std::cmp::min(stack.len(), 17);
        for pos in 1..len {
            let p = stack.from_top(pos).unwrap();
            assert!(p.is_flat());
            let idx = pos - 1 + outer_idx * 16;
            out.push((OFFSET + idx) as u16);
        }
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
    let mut max_i = 0;
    let mut max_j = 0;
    for (i, sp) in line.split(";").enumerate() {
        for (j, val) in sp
            .split(",")
            .filter_map(|x| x.parse::<T>().ok())
            .enumerate()
        {
            max_i = i;
            max_j = j;
            out[i][j] = val;
            count += 1;
        }
    }
    dbg!(max_i);
    dbg!(max_j);
    assert_eq!(count, A * B);
    out.into_boxed_slice()
}

fn array_2d<const A: usize, const B: usize, T: Scalar>(line: &str) -> [[T; A]; B] {
    let mut out = [[T::ZERO; A]; B];
    let mut count = 0;
    let mut max_i = 0;
    let mut max_j = 0;
    for (i, sp) in line.split(";").enumerate() {
        for (j, val) in sp
            .split(",")
            .filter_map(|x| x.parse::<T>().ok())
            .enumerate()
        {
            max_i = i;
            max_j = j;
            out[i][j] = val;
            count += 1;
        }
    }
    dbg!(max_i);
    dbg!(max_j);
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

#[cfg(test)]
mod test {
    use super::{Incremental, Weights};

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
