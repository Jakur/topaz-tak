use super::Weights6;

pub const STACK_DEPTH: usize = 10;
pub const HIDDEN_SIZE: usize = 512;
pub const SCALE: i32 = 400;
pub const QA: i16 = 255;
pub const QB: i16 = 64;
pub const WHITE_FLAT: ValidPiece = ValidPiece(0);
pub const BLACK_FLAT: ValidPiece = ValidPiece(1);
pub const WHITE_WALL: ValidPiece = ValidPiece(2);
pub const BLACK_WALL: ValidPiece = ValidPiece(3);
pub const WHITE_CAP: ValidPiece = ValidPiece(4);
pub const BLACK_CAP: ValidPiece = ValidPiece(5);
const _ASS: () = assert!(
    WHITE_FLAT.flip_color().0 == BLACK_FLAT.0
        && BLACK_WALL.flip_color().0 == WHITE_WALL.0
        && BLACK_CAP.flip_color().0 == WHITE_CAP.0
);

pub static NNUE: Network = unsafe {
    let bytes = include_bytes!("quantised_6.bin");
    assert!(bytes.len() == std::mem::size_of::<Network>());
    std::mem::transmute(*bytes)
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidPiece(pub u8);

impl ValidPiece {
    pub const fn without_color(self) -> u8 {
        self.0 >> 1
    }
    const fn flip_color(self) -> Self {
        Self(self.0 ^ 1) // Toggle bit 0
    }
    pub const fn promote_cap(self) -> Self {
        Self(self.0 | 4) // Set bit 2
    }
    pub const fn is_white(self) -> bool {
        (self.0 & 1) == 0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PieceSquare(pub u8);

impl PieceSquare {
    pub fn new(square: usize, piece: u8) -> Self {
        Self((square as u8) | piece << 6)
    }
    pub fn square(self) -> u8 {
        self.0 & 63
    }
    pub fn piece(self) -> ValidPiece {
        let masked = 0b1100_0000 & self.0;
        ValidPiece(masked >> 6)
    }
    pub fn promote_wall(&mut self) {
        self.0 |= 128;
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BoardData {
    pub(crate) caps: [u8; 2],
    pub(crate) data: [PieceSquare; 62], // Each stack must be presented from top to bottom sequentially
    pub(crate) white_to_move: bool,
    pub(crate) score: i16,
    pub(crate) result: u8,
}

impl BoardData {
    pub fn minimal(caps: [u8; 2], data: [PieceSquare; 62], white_to_move: bool) -> Self {
        Self {
            caps,
            data,
            white_to_move,
            score: 0,
            result: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct TakSimple6 {}

impl TakSimple6 {
    pub const SQUARE_INPUTS: usize = 36 * (6 + 2 * STACK_DEPTH);
    // Squares + Side + Reserves
    pub const NUM_INPUTS: usize = TakSimple6::SQUARE_INPUTS + 8 + 80; // Pad to 1024

    pub fn handle_features<F: FnMut(usize, usize)>(&self, pos: &BoardData, mut f: F) {
        let mut reserves: [usize; 2] = [31, 31];
        for (piece, square, depth_idx) in pos.into_iter() {
            let c = (piece.is_white() ^ pos.white_to_move) as usize; // 0 if matches, else 1
            reserves[c] -= 1;
            let location = usize::from(piece.without_color() + depth_idx);
            let sq = usize::from(square);

            let stm = [0, 468][c] + 36 * location + sq;
            let ntm = [468, 0][c] + 36 * location + sq;
            f(stm, ntm);
        }
        let white_res_adv = (31 + reserves[0] - reserves[1]).clamp(23, 39);
        let black_res_adv = (31 + reserves[1] - reserves[0]).clamp(23, 39);
        if pos.white_to_move {
            // White to move
            f(
                Self::SQUARE_INPUTS + 8 + reserves[0],
                Self::SQUARE_INPUTS + 8 + reserves[1],
            );
            f(975 + white_res_adv, 975 + black_res_adv);
            f(Self::SQUARE_INPUTS, Self::SQUARE_INPUTS + 1);
        } else {
            // Black to move
            f(
                Self::SQUARE_INPUTS + 8 + reserves[1],
                Self::SQUARE_INPUTS + 8 + reserves[0],
            );
            f(975 + black_res_adv, 960 + white_res_adv);
            f(Self::SQUARE_INPUTS + 1, Self::SQUARE_INPUTS);
        }
    }
}

impl IntoIterator for BoardData {
    type Item = (ValidPiece, u8, u8);
    type IntoIter = TakBoardIter;
    fn into_iter(self) -> Self::IntoIter {
        TakBoardIter {
            board: self,
            idx: 0,
            last: u8::MAX,
            depth: 0,
        }
    }
}

pub struct TakBoardIter {
    board: BoardData,
    idx: usize,
    last: u8,
    depth: u8,
}

impl Iterator for TakBoardIter {
    type Item = (ValidPiece, u8, u8); // PieceType, Square, Depth
    fn next(&mut self) -> Option<Self::Item> {
        const DEPTH_TABLE: [u8; 10] = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        if self.idx > self.board.data.len() {
            return None;
        }
        let val = self.board.data[self.idx];
        let square = val.square();
        if square >= 36 {
            return None;
        }
        let mut piece = val.piece();
        if square == self.last {
            self.depth += 1;
        } else {
            self.depth = 0;
            if self.board.caps[0] == square || self.board.caps[1] == square {
                piece = piece.promote_cap();
            }
        }
        self.idx += 1;
        self.last = square;
        Some((piece, square, DEPTH_TABLE[self.depth as usize]))
    }
}

/// A column of the feature-weights matrix.
/// Note the `align(64)`.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    vals: [i16; HIDDEN_SIZE],
}

impl Accumulator {
    /// Initialised with bias so we can just efficiently
    /// operate on it afterwards.
    pub fn new(net: &Network) -> Self {
        net.feature_bias
    }

    pub fn from_old(old: &Self) -> Self {
        old.clone()
    }

    pub fn add_all(&mut self, features: &[u16], net: &Network) {
        for f in features {
            self.add_feature(*f as usize, net);
        }
    }

    pub fn remove_all(&mut self, features: &[u16], net: &Network) {
        for f in features {
            self.remove_feature(*f as usize, net);
        }
    }

    /// Add a feature to an accumulator.
    pub fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self
            .vals
            .iter_mut()
            .zip(&net.feature_weights[feature_idx].vals)
        {
            *i += *d
        }
    }

    /// Remove a feature from an accumulator.
    pub fn remove_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self
            .vals
            .iter_mut()
            .zip(&net.feature_weights[feature_idx].vals)
        {
            *i -= *d
        }
    }
}

pub struct NNUE6 {
    white: (Incremental, Incremental),
    black: (Incremental, Incremental),
    pub classical: Weights6,
}

impl NNUE6 {
    pub fn incremental_eval(&mut self, takboard: BoardData) -> i32 {
        let (ours, theirs) = build_features(takboard);
        let (old_ours, old_theirs) = if takboard.white_to_move {
            (&self.white.0, &self.white.1)
        } else {
            (&self.black.0, &self.black.1)
        };
        // Ours
        let mut ours_acc = Accumulator::from_old(&old_ours.vec);
        let (sub, add) = ours.diff(&old_ours.state);
        ours_acc.remove_all(&sub, &NNUE);
        ours_acc.add_all(&add, &NNUE);
        let ours = Incremental {
            state: ours,
            vec: ours_acc,
        };
        // Theirs
        let mut theirs_acc = Accumulator::from_old(&old_theirs.vec);
        let (sub, add) = theirs.diff(&old_theirs.state);
        theirs_acc.remove_all(&sub, &NNUE);
        theirs_acc.add_all(&add, &NNUE);
        let theirs = Incremental {
            state: theirs,
            vec: theirs_acc,
        };
        // Output
        let eval = NNUE.evaluate(
            &ours.vec,
            &theirs.vec,
            &ours.state.piece_data,
            &ours.state.meta,
        );
        if takboard.white_to_move {
            self.white = (ours, theirs);
        } else {
            self.black = (ours, theirs);
        }
        eval
    }
    pub(crate) fn manual_eval(takboard: BoardData) -> i32 {
        let (ours, theirs) = build_features(takboard);
        let ours = Incremental::fresh_new(&NNUE, ours);
        let theirs = Incremental::fresh_new(&NNUE, theirs);
        let eval = NNUE.evaluate(
            &ours.vec,
            &theirs.vec,
            &ours.state.piece_data,
            &ours.state.meta,
        );
        eval
    }
}

impl Default for NNUE6 {
    fn default() -> Self {
        Self {
            white: (
                Incremental::fresh_empty(&NNUE),
                Incremental::fresh_empty(&NNUE),
            ),
            black: (
                Incremental::fresh_empty(&NNUE),
                Incremental::fresh_empty(&NNUE),
            ),
            classical: Weights6::default(),
        }
    }
}

fn build_features(takboard: BoardData) -> (IncrementalState, IncrementalState) {
    let mut ours = Vec::new();
    let mut theirs = Vec::new();
    let simple = TakSimple6 {};
    simple.handle_features(&takboard, |x, y| {
        ours.push(x as u16);
        theirs.push(y as u16);
    });
    (
        IncrementalState::from_vec(ours),
        IncrementalState::from_vec(theirs),
    )
}

#[inline]
pub fn screlu(x: i16) -> i32 {
    i32::from(x.clamp(0, QA as i16)).pow(2)
}

/// This is the quantised format that bullet outputs.
#[repr(C)]
pub struct Network {
    /// Column-Major `HIDDEN_SIZE x 768` matrix.
    feature_weights: [Accumulator; TakSimple6::NUM_INPUTS],
    /// Vector with dimension `HIDDEN_SIZE`.
    feature_bias: Accumulator,
    /// Column-Major `1 x (2 * HIDDEN_SIZE)`
    /// matrix, we use it like this to make the
    /// code nicer in `Network::evaluate`.
    output_weights: [i16; 2 * HIDDEN_SIZE],
    /// Piece-Square Table for Input
    pqst: [i16; TakSimple6::NUM_INPUTS],
    /// Scalar output bias.
    output_bias: i16,
}

impl Network {
    /// Calculates the output of the network, starting from the already
    /// calculated hidden layer (done efficiently during makemoves).
    pub fn evaluate(
        &self,
        us: &Accumulator,
        them: &Accumulator,
        original_input: &[u16],
        original_meta: &[u16],
    ) -> i32 {
        // Initialise output with bias.
        let mut sum = 0;
        let mut psqt_out = 0;

        // Side-To-Move Accumulator -> Output.
        for (&input, &weight) in us.vals.iter().zip(&self.output_weights[..HIDDEN_SIZE]) {
            let val = screlu(input) * i32::from(weight);
            sum += val;
        }

        // Not-Side-To-Move Accumulator -> Output.
        for (&input, &weight) in them.vals.iter().zip(&self.output_weights[HIDDEN_SIZE..]) {
            sum += screlu(input) * i32::from(weight);
        }

        // Update Piece Square Table
        for idx in original_input {
            if *idx == u16::MAX {
                break;
            }
            psqt_out += i32::from(self.pqst[*idx as usize]);
        }
        // This is dumb but I'll fix it later
        for idx in original_meta {
            psqt_out += i32::from(self.pqst[*idx as usize]);
        }
        // Apply eval scale.
        psqt_out *= SCALE;
        // Remove quantisation.
        let output =
            (sum / (QA as i32) + i32::from(self.output_bias)) * SCALE / (QA as i32 * QB as i32);
        psqt_out /= i32::from(QA);
        output + psqt_out
    }
}

// Sorry this naming convention is so bad
struct Incremental {
    state: IncrementalState,
    vec: Accumulator,
}

impl Incremental {
    fn fresh_empty(net: &Network) -> Self {
        let mut acc = Accumulator::new(net);
        let inc = IncrementalState::from_vec(vec![0, 1, 2]); // Todo make this cleaner
        for d in inc.meta {
            acc.add_feature(d as usize, net);
        }
        Self {
            state: inc,
            vec: acc,
        }
    }
    fn fresh_new(net: &Network, data: IncrementalState) -> Self {
        let mut acc = Accumulator::new(net);
        for d in data.meta {
            acc.add_feature(d as usize, net);
        }
        for f in data.piece_data {
            let f = f as usize;
            if f > TakSimple6::SQUARE_INPUTS {
                break;
            }
            acc.add_feature(f, net);
        }
        Self {
            vec: acc,
            state: data,
        }
    }
}

struct IncrementalState {
    pub(crate) meta: [u16; 3],
    pub(crate) piece_data: [u16; 62],
}

impl IncrementalState {
    pub fn from_vec(mut vec: Vec<u16>) -> Self {
        let mut meta = [0; 3];
        for i in 0..3 {
            meta[i] = vec.pop().unwrap();
        }
        let mut piece_data = [u16::MAX; 62];
        piece_data[0..vec.len()].copy_from_slice(&vec);
        Self { meta, piece_data }
    }
    pub fn diff(&self, old: &Self) -> (Vec<u16>, Vec<u16>) {
        // Todo in the real algorithm, do not allocate vecs. This is just to demonstrate the idea
        let mut subtract = Vec::new();
        let mut add = Vec::new();
        Self::operate(&self.meta, &old.meta, &mut add);
        Self::operate(&old.meta, &self.meta, &mut subtract);
        // Piece data is not sorted, but it is grouped by square
        let mut new_st = 0;
        let mut old_st = 0;
        loop {
            let ol = Self::get_sq(old.piece_data[old_st]);
            let nw = Self::get_sq(self.piece_data[new_st]);
            if ol >= 36 && nw >= 36 {
                break;
            }
            if nw < ol {
                let new_end = Self::get_end(&self.piece_data, new_st);
                add.extend(self.piece_data[new_st..new_end].iter().copied());
                new_st = new_end;
            } else if ol < nw {
                let old_end = Self::get_end(&old.piece_data, old_st);
                subtract.extend(old.piece_data[old_st..old_end].iter().copied());
                old_st = old_end;
            } else {
                // They are equal
                let new_end = Self::get_end(&self.piece_data, new_st);
                let old_end = Self::get_end(&old.piece_data, old_st);
                Self::operate(
                    &self.piece_data[new_st..new_end],
                    &old.piece_data[old_st..old_end],
                    &mut add,
                );
                Self::operate(
                    &old.piece_data[old_st..old_end],
                    &self.piece_data[new_st..new_end],
                    &mut subtract,
                );
                new_st = new_end;
                old_st = old_end;
            }
        }
        // End
        (subtract, add)
    }
    fn get_end(slice: &[u16], st: usize) -> usize {
        let st_val = Self::get_sq(slice[st]);
        st + slice[st..]
            .iter()
            .position(|&x| Self::get_sq(x) != st_val)
            .unwrap()
    }
    /// Extend out with values in left which are not present in right
    fn operate(left: &[u16], right: &[u16], out: &mut Vec<u16>) {
        out.extend(left.iter().copied().filter(|x| !right.contains(x)));
    }
    fn get_sq(val: u16) -> u16 {
        if val == u16::MAX {
            return 64;
        }
        val % 36
    }
}
