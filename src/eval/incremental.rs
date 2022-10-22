use crate::{board::Bitboard6, Bitboard, BitboardStorage, Piece};
use crate::{board::TakBoard, Board6, Color};
use bitintr::Pdep;
use std::str::FromStr;

// const DIM1: usize = 14016 + 64;
const DIM1: usize = 5464;
const DIM2: usize = 128;
const DIM3: usize = 64;
const DIM_MOVE: usize = 256;

const SCALE_INT: i16 = 500;
const SCALE_FLOAT: f32 = SCALE_INT as f32;

const FLAT_OFFSET: u16 = 512 * 8;
const OFFSET: u16 = FLAT_OFFSET + 36 * 4;
const UNDER_OFFSET: u16 = OFFSET + 64;

pub struct Weights {
    outer: Box<[[i16; DIM2]]>,
    bias1: [i16; DIM2],
    white1: [[f32; DIM3]; DIM2],
    black1: [[f32; DIM3]; DIM2],
    white_bias1: [f32; DIM3],
    black_bias1: [f32; DIM3],
    white2: [f32; DIM3],
    black2: [f32; DIM3],
    white_bias2: f32,
    black_bias2: f32,
}

impl Weights {
    pub fn from_file<R: std::io::Read>(mut r: R) -> Option<Self> {
        let mut s = String::new();
        r.read_to_string(&mut s).unwrap();
        let lines: Vec<_> = s.lines().collect();

        Some(Self {
            outer: array_2d_boxed::<DIM2, DIM1, i16>(lines.get(0)?),
            bias1: array_1d(lines.get(1)?),
            white1: array_2d(lines.get(2)?),
            white_bias1: array_1d(lines.get(3)?),
            white2: array_1d(lines.get(4)?),
            white_bias2: array_1d::<1, _>(lines.get(5)?)[0],
            black1: array_2d(lines.get(6)?),
            black_bias1: array_1d(lines.get(7)?),
            black2: array_1d(lines.get(8)?),
            black_bias2: array_1d::<1, _>(lines.get(9)?)[0],
        })
    }
    #[cfg(test)]
    fn ones() -> Self {
        // let init = [1; DIM2];
        // Self {
        //     outer: vec![init; DIM1].into_boxed_slice(),
        //     bias1: [1; DIM2],
        //     inner1: [[1.0; DIM2]; DIM3],
        //     bias2: [1.0; DIM3],
        //     white: [1.0; DIM3],
        //     black: [1.0; DIM3],
        //     white_bias: 1.0,
        //     black_bias: 1.0,
        // }
        unimplemented!()
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
    #[cfg(feature = "rand")]
    pub fn noisy_incremental_forward<R: rand::RngCore>(
        &self,
        inc: &Incremental,
        is_white: bool,
        rng: &mut R,
    ) -> i32 {
        use rand::Rng;
        let mut input = [0.0; DIM2];
        for i in 0..input.len() {
            let sample = rng.sample::<f32, _>(rand_distr::StandardNormal) * SCALE_FLOAT / 10.0;
            input[i] = (relu6_int(inc.storage[i] + sample as i16) as f32) / SCALE_FLOAT;
        }
        let (mut middle, inner1) = if is_white {
            (self.white_bias1.clone(), &self.white1)
        } else {
            (self.black_bias1.clone(), &self.black1)
        };
        for i in 0..DIM2 {
            for j in 0..DIM3 {
                middle[j] += input[i] * inner1[i][j];
            }
        }
        let (mut out, inner2) = if is_white {
            (self.white_bias2, &self.white2)
        } else {
            (self.black_bias2, &self.black2)
        };

        for i in 0..middle.len() {
            out += relu6(middle[i]) * inner2[i];
        }
        ((out as f64).tanh() * SCALE_FLOAT as f64).trunc() as i32
    }
    pub fn incremental_forward(&self, inc: &Incremental, is_white: bool) -> i32 {
        let mut input = [0.0; DIM2];
        for i in 0..input.len() {
            input[i] = (relu6_int(inc.storage[i]) as f32) / SCALE_FLOAT;
        }
        let (mut middle, inner1) = if is_white {
            (self.white_bias1.clone(), &self.white1)
        } else {
            (self.black_bias1.clone(), &self.black1)
        };
        for i in 0..DIM2 {
            for j in 0..DIM3 {
                middle[j] += input[i] * inner1[i][j];
            }
        }
        let (mut out, inner2) = if is_white {
            (self.white_bias2, &self.white2)
        } else {
            (self.black_bias2, &self.black2)
        };

        for i in 0..middle.len() {
            out += relu6(middle[i]) * inner2[i];
        }
        // dbg!(out);
        ((out as f64).tanh() * SCALE_FLOAT as f64).trunc() as i32
    }
}

pub fn nn_repr<T: TakBoard>(board: &T) -> Vec<u16> {
    const OFFSET: usize = 4240;
    let mut arr1 = make_array(board);
    debug_assert!(arr1.iter().all(|&x| x < (OFFSET + 64) as u16));
    arr1.extend(make_under_array(board).into_iter());
    arr1
}

fn make_array<T: TakBoard>(board: &T) -> Vec<u16> {
    use bitintr::Pext;
    let mut bits = Vec::new();
    let mut out = Vec::with_capacity(26);
    for color in [board.bits().white, board.bits().black] {
        for piece in [board.bits().flat, board.bits().wall, board.bits().cap] {
            bits.push(color & piece);
        }
    }
    let mut counter = 0;
    for mask in [0xe0e0e00, 0x70707000, 0x70707000000000, 0xe0e0e00000000] {
        for b in [
            board.bits().white & board.bits().flat,
            board.bits().black & board.bits().flat,
        ] {
            let val = b.raw_bits().pext(mask);
            out.push(counter * 512 + val as u16);
            counter += 1;
        }
    }
    for x in (board.bits().white & board.bits().cap).index_iter() {
        let val = FLAT_OFFSET + x as u16;
        out.push(val);
    }
    for x in (board.bits().black & board.bits().cap).index_iter() {
        let val = FLAT_OFFSET + 36 + x as u16;
        out.push(val);
    }
    for x in (board.bits().white & board.bits().wall).index_iter() {
        let val = FLAT_OFFSET + 72 + x as u16;
        out.push(val);
    }
    for x in (board.bits().black & board.bits().wall).index_iter() {
        let val = FLAT_OFFSET + 108 + x as u16;
        out.push(val);
    }

    out.push(
        OFFSET
            + board.caps_reserve(Color::White) as u16
            + board.pieces_reserve(Color::White) as u16,
    );
    out.push(
        OFFSET
            + 32
            + board.caps_reserve(Color::Black) as u16
            + board.pieces_reserve(Color::Black) as u16,
    );
    out
}

fn make_under_array<T: TakBoard>(board: &T) -> Vec<u16> {
    let mut out = Vec::with_capacity(32);
    for (outer_idx, stack) in board.board().iter().enumerate() {
        let len = std::cmp::min(stack.len(), 17);
        for pos in 1..len {
            let p = stack.from_top(pos).unwrap();
            let c = (p.owner() == Color::White) as usize;
            debug_assert!(p.is_flat());
            let idx = c + 2 * pos - 1 + outer_idx * 32;
            out.push(UNDER_OFFSET + idx as u16);
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

pub fn undo_nn_repr(nn: Vec<u16>) -> Board6 {
    let mut flats = [30, 30];
    let mut caps = [1, 1];
    let mut stacks = vec![Vec::new(); 36];
    let mut bits = vec![Bitboard6::ZERO; 2];
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
        let index = ((x - FLAT_OFFSET) % 36) as usize;
        if stacks[index].len() != 0 {
            continue;
        }
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
        stacks[index].push(piece);
    }

    for under in nn.iter().copied().filter(|&x| x >= UNDER_OFFSET) {
        let x = under - UNDER_OFFSET - 1;
        let p = if x % 2 == 1 {
            flats[0] -= 1;
            Piece::WhiteFlat
        } else {
            flats[1] -= 1;
            Piece::BlackFlat
        };
        let outer = (x / 32) as usize;
        // assert!(stacks[outer].len() != 0);
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
    out
}

#[cfg(test)]
mod test {
    use crate::Position;

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
        let tps = "x3,2,x,1/x,1,1,2,2,1/x,2,1,12C,1,1/x2,21C,1,2,x/x4,2,x/2,x5 2 11";
        let board = Board6::try_from_tps(tps).unwrap();
        let repr: Vec<_> = nn_repr(&board).into_iter().map(|x| x as usize).collect();
        dbg!(&repr);
        let x = vec![
            180, 512, 1024, 1558, 2064, 2985, 3083, 4036, 4105, 4146, 4184, 4185, 4194, 4201, 4252,
            4286, 4593, 4753, 4756, 4758, 4760, 4762, 4764, 4849, 5330, 5331,
        ];
        dbg!(undo_nn_repr(x));
        // let inc = crate::eval::NN6.build_incremental(&repr);
        // let v = crate::eval::NN6.move_weights(&inc, board.side_to_move() == Color::White);
        // dbg!(&v[64]);
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
