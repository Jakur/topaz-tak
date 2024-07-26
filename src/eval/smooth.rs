use board::Bitboard6;

use super::*;

const MAX_FEATURES: usize = 88 + 24;
const PAD_IDX: usize = 823; // This should probably be something else

const SMOOTH_TABLE: [f32; 4344] = include!("smooth_weights.table");

pub struct SmoothWeights6 {
    weights: &'static [f32; 4344],
    phase_weights: [[f64; 4]; 20],
    output: [f64; 4],
}

impl SmoothWeights6 {
    pub fn empty() -> Self {
        Self {
            weights: &SMOOTH_TABLE,
            phase_weights: [
                [0.0, 0.0, 0.0, 0.0],
                [2.4902e-01, 2.6745e+00, 5.1621e-01, 1.8403e+00],
                [6.3719e-01, 2.1112e+00, 5.6241e-01, 1.9116e+00],
                [9.1352e-01, 1.5673e+00, 5.8154e-01, 1.8168e+00],
                [1.1294e+00, 1.2693e+00, 6.4078e-01, 1.7840e+00],
                [1.2986e+00, 1.0644e+00, 7.1512e-01, 1.7694e+00],
                [1.4005e+00, 9.5911e-01, 8.0030e-01, 1.7681e+00],
                [1.4794e+00, 8.7697e-01, 8.8408e-01, 1.7457e+00],
                [1.5572e+00, 7.6922e-01, 9.5319e-01, 1.6974e+00],
                [1.6055e+00, 6.9348e-01, 1.0519e+00, 1.6677e+00],
                [1.7161e+00, 5.2612e-01, 1.1290e+00, 1.5953e+00],
                [1.7874e+00, 4.0783e-01, 1.2169e+00, 1.5331e+00],
                [1.9041e+00, 2.4720e-01, 1.2772e+00, 1.4817e+00],
                [1.9967e+00, 1.2139e-01, 1.3926e+00, 1.4218e+00],
                [2.1361e+00, -3.6963e-02, 1.5513e+00, 1.3177e+00],
                [2.1006e+00, -1.6866e-01, 1.6023e+00, 1.1012e+00],
                [1.8961e+00, -1.8383e-01, 1.6598e+00, 8.6600e-01],
                [1.7976e+00, -2.9785e-01, 1.7063e+00, 5.2814e-01],
                [1.6359e+00, -4.6413e-01, 1.7454e+00, 1.1454e-01],
                [6.6663e-01, 2.2190e-01, 1.2335e+00, -1.6566e-01],
            ],
            output: [-2.2941, -1.8882, 1.7433, 2.8224],
        } // 8 extra
    }
    pub fn parse(data: &str) -> Self {
        todo!()
    }
    fn do_stack<'a, T: EvalComponents<'a>>(
        game: &<SmoothWeights6 as Evaluator>::Game,
        idx: usize,
        eval: &mut T,
    ) {
        const UNDER_OFFSET: usize = 128 + 270; // 395 necessary
        const PSQT_OFFSET: usize = UNDER_OFFSET + 400;
        const TARGETS_OFFSET: usize = PSQT_OFFSET + 252; // 36 extra onto this
        let stack = &game.board[idx];
        let mut flag = 0;
        let top = stack.top();
        if let None = top {
            // eval.mark(UNDER_OFFSET + 78 * 4);
            return;
        }
        let top = top.unwrap();
        let stack_piece = if game.side_to_move() == Color::White {
            top as usize
        } else {
            top.swap_color() as usize
        };
        eval.mark(PSQT_OFFSET + 7 * idx + stack_piece);
        if stack.len() == 1 {
            let stack_idx = stack_piece * 49;
            eval.mark(UNDER_OFFSET + incremental::STACK_LOOKUP[stack_idx] as usize);
            return;
        }
        let (captive, friendly) = stack.captive_friendly();

        let mut mobility = 0;
        let mut safety = 0;
        match top {
            Piece::WhiteFlat | Piece::BlackFlat => {}
            Piece::WhiteWall | Piece::BlackWall => {
                safety += 16;
            }
            Piece::WhiteCap | Piece::BlackCap => {
                safety += 64;
                mobility += 1;
            }
        }
        let cap_friendly_offset = std::cmp::min(captive, 6) * 7 + std::cmp::min(friendly, 6);
        let stack_idx = stack_piece * 49 + cap_friendly_offset as usize;

        let neighbors =
            <<SmoothWeights6 as Evaluator>::Game as TakBoard>::Bits::index_to_bit(idx).adjacent();
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
                            if stack.len() < <SmoothWeights6 as Evaluator>::Game::SIZE {
                                safety += 4;
                            }
                        }
                        Piece::WhiteCap | Piece::BlackCap => {
                            if stack.len() < <SmoothWeights6 as Evaluator>::Game::SIZE {
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
                            if stack.len() < <SmoothWeights6 as Evaluator>::Game::SIZE {
                                safety -= 4;
                            }
                        }
                        Piece::WhiteCap | Piece::BlackCap => {
                            if stack.len() < <SmoothWeights6 as Evaluator>::Game::SIZE {
                                safety -= 32;
                            }
                        }
                    }
                }
            } else {
                mobility += 2;
            }
        }
        // Positive
        if mobility < 2 {
            flag += 2;
        }
        if safety < 0 {
            flag += 1;
        }
        let out = UNDER_OFFSET + 78 * flag + incremental::STACK_LOOKUP[stack_idx] as usize;
        eval.mark(out);

        let all_friendly = friendly + 1;
        if all_friendly >= 3 {
            let enemy_flats = game.bits().flat & game.bits().all_pieces(!top.owner());
            let orth = <<SmoothWeights6 as Evaluator>::Game as TakBoard>::Bits::orthogonal(idx);
            let count = (orth & enemy_flats).pop_count() as i32;
            let all_friendly = friendly + 1;
            if count >= all_friendly {
                eval.mark(TARGETS_OFFSET + idx);
            }
        }
    }
    fn do_side<'a, T: EvalComponents<'a>>(
        game: &<SmoothWeights6 as Evaluator>::Game,
        color: Color,
        friendly_offset: usize,
        eval: &mut T,
    ) {
        let road_bits = game.bits.road_pieces(color);
        // let enemy_road_bits = game.bits.road_pieces(!color);
        let enemy_bits = game.bits.all_pieces(!color);
        let cap = game.bits.cap & road_bits;
        if cap != <<SmoothWeights6 as Evaluator>::Game as TakBoard>::Bits::ZERO {
            let lonely = cap.adjacent() & (game.bits.flat | game.bits.wall);
            if lonely == <<SmoothWeights6 as Evaluator>::Game as TakBoard>::Bits::ZERO {
                eval.mark(16 + friendly_offset);
            }
        }
        // Strongly positive
        eval.mark(18 + friendly_offset * 5 + attackable_cs(color, game).clamp(0, 4) as usize);
        let (road_score, comp) = flat_placement_road_short(road_bits, game.bits.empty());
        eval.mark(28 + 5 * friendly_offset + road_score.clamp(0, 4) as usize);
        eval.mark(38 + 7 * friendly_offset + comp.clamp(0, 6) as usize);
        // 52
        // Bridge
        // let cap_adj = cap.adjacent();
        // let line = ((cap.north() & road_bits).nonzero() && (cap.south() & road_bits).nonzero())
        //     || ((cap.east() & road_bits).nonzero() && (cap.west() & road_bits).nonzero());
        // let cutting_line = ((cap.north() & enemy_road_bits).nonzero()
        //     && (cap.south() & enemy_road_bits).nonzero())
        //     || ((cap.east() & enemy_road_bits).nonzero()
        //         && (cap.west() & enemy_road_bits).nonzero());
        // if line && cutting_line {
        //     eval.mark(52 + friendly_offset);
        // }
        Self::road_estimation(road_bits, enemy_bits, friendly_offset, eval); // Very slightly positive
                                                                             // Self::column_counts(game, color, friendly_offset, eval); // Negative
    }
    fn road_estimation<'a, B: Bitboard + std::fmt::Debug, T: EvalComponents<'a>>(
        road_bits: B,
        enemy_bits: B,
        side_offset: usize,
        eval: &mut T,
    ) {
        let mut iter = ComponentIterator::new(road_bits);
        while let Some(comp) = iter.next() {
            if comp.pop_count() >= 3 {
                let (score_a, score_b) = comp.simple_road_est();
                let enemy_idx = (comp.adjacent() & enemy_bits).nonzero() as usize;
                let idx = 52
                    + side_offset * 72
                    + enemy_idx * 36
                    + score_a as usize * 6
                    + score_b as usize;
                eval.mark(idx);
            }
        }
        // let empty = !road_bits & !enemy_bits;
        // let optimistic_road_bits = road_bits | (road_bits.adjacent() & empty);
        // let mut iter = ComponentIterator::new(optimistic_road_bits);
        // while let Some(comp) = iter.next() {
        //     if comp.pop_count() >= 4 {
        //         let (score_a, score_b) = comp.simple_road_est();
        //         let enemy_idx = (comp.adjacent() & enemy_bits).nonzero() as usize;
        //         let idx = 200
        //             + side_offset * 98
        //             + enemy_idx * 49
        //             + score_a as usize * 7
        //             + score_b as usize;
        //         eval.mark(idx);
        //     }
        // }
    }
    // fn column_counts<'a, T: EvalComponents<'a>>(
    //     game: &<SmoothWeights6 as Evaluator>::Game,
    //     color: Color,
    //     friendly_offset: usize,
    //     eval: &mut T,
    // ) {
    //     const COLUMN_OFFSET: usize = 128;
    //     let friendly_road_bits = game.bits.road_pieces(color);
    //     let enemy = game.bits.all_pieces(!color);
    //     let enemy_blocker = game.bits.blocker_pieces(!color);
    //     for array in [Bitboard6::EW, Bitboard6::NS] {
    //         for (col_mask, col_idx) in array.into_iter().zip([0, 1, 2, 2, 1, 0].into_iter()) {
    //             let col_adj = col_mask.adjacent(); // Slightly inefficient, only need 2 directions
    //             let road = friendly_road_bits & col_mask;
    //             let road_count = road.pop_count();
    //             if road_count == 0 {
    //                 continue;
    //             }
    //             let count_idx = road_count as usize - 1; // Min 0, Max 4 because if it were 6 you'd have a road.
    //             assert!(count_idx <= 4);
    //             let blocker_idx = if (enemy_blocker & col_mask).nonzero() {
    //                 2
    //             } else if (enemy & col_mask).nonzero() {
    //                 1
    //             } else {
    //                 0
    //             };
    //             let nearby_idx = if (enemy_blocker & col_adj).nonzero() {
    //                 2
    //             } else if (enemy & col_adj).nonzero() {
    //                 1
    //             } else {
    //                 0
    //             };
    //             let combined_idx = friendly_offset
    //                 + 2 * col_idx
    //                 + 6 * count_idx
    //                 + 30 * blocker_idx
    //                 + 90 * nearby_idx;
    //             assert!(combined_idx < 270);
    //             eval.mark(COLUMN_OFFSET + combined_idx);
    //         }
    //     }
    // }
    pub fn feature_length(&self, game: &Board6) -> usize {
        let fw = self.eval_components::<FeatureWriter>(game);
        fw.nonzero.len()
    }
    pub fn output<W: std::io::Write>(&self, game: &Board6, w: &mut W) -> std::io::Result<()> {
        let fw = self.eval_components::<FeatureWriter>(game);
        fw.output(w)
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
        let comp = self.eval_components::<IncrementalFeatures>(game);
        if depth % 2 == 0 {
            comp.score()
        } else {
            comp.score() - 150
        }
    }

    fn eval_stack(&self, game: &Self::Game, idx: usize, stack: &Stack) -> i32 {
        unimplemented!()
    }

    fn eval_components<'a, T: 'a>(&'a self, game: &Self::Game) -> T
    where
        T: EvalComponents<'a>,
    {
        let mut components = T::create(&self);
        components.mark(game.side_to_move() as usize);
        let (white_flat_diff, divisor) = endgame_calc(game);
        let fcd_index = if let Color::White = game.side_to_move() {
            white_flat_diff.clamp(-6, 6) + 6
        } else {
            -1 * white_flat_diff.clamp(-6, 6) + 6
        };
        components.set_phase(divisor as usize);
        components.mark(2 + fcd_index as usize);
        // 16
        Self::do_side(game, game.side_to_move(), 0, &mut components);
        Self::do_side(game, !game.side_to_move(), 1, &mut components);
        const SIZE: usize = <<SmoothWeights6 as Evaluator>::Game as TakBoard>::SIZE;
        for idx in 0..SIZE * SIZE {
            Self::do_stack(game, idx, &mut components);
        }
        components
    }
    fn is_quiet(&self, game: &Self::Game) -> bool {
        true
    }
}

struct IncrementalFeatures<'a> {
    middle: [f32; 4],
    phase: usize,
    weights: &'a SmoothWeights6,
    num_marks: usize,
}

impl<'a> IncrementalFeatures<'a> {
    fn new(weights: &'a SmoothWeights6) -> Self {
        Self {
            middle: [0.0, 0.0, 0.0, 0.0],
            phase: 1,
            weights,
            num_marks: 0,
        }
    }
}

impl<'a> EvalComponents<'a> for IncrementalFeatures<'a> {
    fn mark(&mut self, index: usize) {
        for i in 0..4 {
            self.middle[i] += self.weights.weights[index * 4 + i];
        }
        self.num_marks += 1;
    }

    fn set_phase(&mut self, phase: usize) {
        self.phase = phase;
    }

    fn create(weights: &'a SmoothWeights6) -> Self
    where
        Self: 'a,
    {
        Self::new(weights)
    }

    fn score(&self) -> i32 {
        let mut temp: [f64; 4] = [0.0; 4];
        // todo fix padding
        // let num_padding =
        let phase = &self.weights.phase_weights[self.phase];
        for i in 0..4 {
            temp[i] = phase[i] * sigmoid(self.middle[i]);
        }
        let out: f64 = temp
            .into_iter()
            .zip(self.weights.output.into_iter())
            .map(|(a, b)| a * b)
            .sum();
        // Omit final sigmoid because we don't care about win %
        (out * 250.0) as i32
    }
}

fn sigmoid(x: f32) -> f64 {
    1.0 / (1.0 + libm::exp(-1.0 * x as f64))
}

struct FeatureWriter<'a> {
    nonzero: Vec<usize>,
    phase: usize,
    weights: &'a SmoothWeights6,
}

impl<'a> FeatureWriter<'a> {
    fn new(weights: &'a SmoothWeights6) -> Self {
        Self {
            nonzero: Vec::new(),
            phase: 1,
            weights,
        }
    }
    fn output<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        assert!(self.nonzero.len() < MAX_FEATURES);
        write!(w, "{};", self.phase)?;
        for i in 0..MAX_FEATURES - 1 {
            if let Some(x) = self.nonzero.get(i) {
                write!(w, "{};", x)?;
            } else {
                write!(w, ";")?;
            }
        }
        if let Some(x) = self.nonzero.get(MAX_FEATURES - 1) {
            write!(w, "{}", x)?;
        }
        writeln!(w, "")
    }
}

impl<'a> EvalComponents<'a> for FeatureWriter<'a> {
    fn score(&self) -> i32 {
        todo!()
    }

    fn mark(&mut self, index: usize) {
        self.nonzero.push(index);
    }

    fn set_phase(&mut self, phase: usize) {
        self.phase = phase;
    }

    fn create(weights: &'a SmoothWeights6) -> Self
    where
        Self: 'a,
    {
        Self::new(weights)
    }
}
