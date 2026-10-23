use crate::{
    board::{Board6, TakBoardUpdate},
    pop_lowest, Color, Piece, Position, TakBoard,
};

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
    let bytes = include_bytes!("../quantised.bin");
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
    pub const fn color_index(self) -> usize {
        (self.0 & 1) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
    pub fn topaz_piece(self) -> Piece {
        match self.piece() {
            WHITE_FLAT => Piece::WhiteFlat,
            BLACK_FLAT => Piece::BlackFlat,
            WHITE_WALL => Piece::WhiteWall,
            BLACK_WALL => Piece::BlackWall,
            WHITE_CAP => Piece::WhiteCap,
            BLACK_CAP => Piece::BlackCap,
            _ => unimplemented!(),
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BoardData {
    pub caps: [u8; 2],
    pub data: [PieceSquare; 62], // Each stack must be presented from top to bottom sequentially
    pub data_len: u8,
    pub white_to_move: bool,
}

impl BoardData {
    const SIZE: u8 = 6;
    const SYM_TABLE: [[u8; 36]; 8] = Self::build_symmetry_table();
    pub fn new(caps: [u8; 2], data: [PieceSquare; 62], data_len: u8, white_to_move: bool) -> Self {
        Self {
            caps,
            data,
            data_len,
            white_to_move,
        }
    }
    pub fn symmetry(mut self, idx: usize) -> Self {
        if idx == 0 {
            return self;
        }
        assert!(idx < 8);
        let table = &Self::SYM_TABLE[idx];
        if self.caps[0] < 36 {
            self.caps[0] = table[self.caps[0] as usize];
        }
        if self.caps[1] < 36 {
            self.caps[1] = table[self.caps[1] as usize];
        }
        for i in 0..(self.data_len as usize) {
            let old = self.data[i];
            self.data[i] = PieceSquare::new(table[old.square() as usize] as usize, old.piece().0);
        }
        self
    }
    const fn build_symmetry_table() -> [[u8; 36]; 8] {
        [
            Self::transform(0),
            Self::transform(1),
            Self::transform(2),
            Self::transform(3),
            Self::transform(4),
            Self::transform(5),
            Self::transform(6),
            Self::transform(7),
        ]
    }
    const fn transform(rotation: usize) -> [u8; 36] {
        let mut data = [(0, 0); 36];
        let mut i = 0;
        while i < 36 {
            let (row, col) = Self::row_col(i as u8);
            data[i] = (row, col);
            i += 1;
        }
        match rotation {
            1 => Self::flip_ns(&mut data),
            2 => Self::flip_ew(&mut data),
            3 => Self::rotate(&mut data),
            4 => {
                Self::rotate(&mut data);
                Self::rotate(&mut data);
            }
            5 => {
                Self::rotate(&mut data);
                Self::rotate(&mut data);
                Self::rotate(&mut data);
            }
            6 => {
                Self::rotate(&mut data);
                Self::flip_ns(&mut data);
            }
            7 => {
                Self::rotate(&mut data);
                Self::flip_ew(&mut data);
            }
            _ => {}
        };
        let mut out = [0; 36];
        let mut i = 0;
        while i < 36 {
            let (row, col) = data[i];
            out[i] = Self::index(row, col);
            i += 1;
        }
        out
    }
    const fn flip_ns(arr: &mut [(u8, u8); 36]) {
        let mut i = 0;
        while i < 36 {
            let (row, _col) = &mut arr[i];
            *row = Self::SIZE - 1 - *row;
            i += 1;
        }
    }
    const fn flip_ew(arr: &mut [(u8, u8); 36]) {
        let mut i = 0;
        while i < 36 {
            let (_row, col) = &mut arr[i];
            *col = Self::SIZE - 1 - *col;
            i += 1;
        }
    }
    const fn rotate(arr: &mut [(u8, u8); 36]) {
        let mut i = 0;
        while i < 36 {
            let (row, col) = &mut arr[i];
            let new_row = Self::SIZE - 1 - *col;
            *col = *row;
            *row = new_row;
            i += 1;
        }
    }
    const fn row_col(index: u8) -> (u8, u8) {
        (index / Self::SIZE, index % Self::SIZE)
    }
    const fn index(row: u8, col: u8) -> u8 {
        row * Self::SIZE + col
    }
}

#[derive(Clone, Copy)]
pub struct TakSimple6 {}

impl TakSimple6 {
    pub const SQUARE_INPUTS: usize = 36 * (6 + 2 * 2 * STACK_DEPTH);
    const HALF: usize = Self::SQUARE_INPUTS / 2;
    // Squares + Side + Reserves
    pub const NUM_INPUTS: usize = TakSimple6::SQUARE_INPUTS + 8 + 80;

    pub fn handle_features<F: FnMut(usize, usize)>(&self, pos: &BoardData, mut f: F) {
        let mut reserves: [usize; 2] = [31, 31];
        let mut top = 0usize;
        for (piece, square, depth_idx) in pos.into_iter() {
            let c = (piece.is_white() ^ pos.white_to_move) as usize; // 0 if matches, else 1
            reserves[c] -= 1;
            let sq = usize::from(square);
            if depth_idx == 0 {
                // top = piece.0;
                top = c; // Friendly or enemy top
                let location = usize::from(piece.without_color()); // + depth, but we know it is 0
                let stm = [0, Self::HALF][c] + 36 * location + sq;
                let ntm = [Self::HALF, 0][c] + 36 * location + sq;
                f(stm, ntm);
            } else {
                let location = usize::from(depth_idx); // + top
                let stm = [0, Self::HALF][c] + 36 * (location + top) + sq;
                let ntm = [Self::HALF, 0][c] + 36 * (location + [1, 0][top]) + sq;
                f(stm, ntm);
            };
        }
        // Does not handle reserves, must be manually calced
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

impl TakBoardIter {
    const DEPTH_TABLE: [u8; 10] = {
        const fn init(x: usize) -> u8 {
            const TOP_KINDS: u8 = 2;
            if x == 0 {
                0 as u8
            } else {
                3 + TOP_KINDS * ((x - 1) as u8)
            }
        }
        [
            init(0),
            init(1),
            init(2),
            init(3),
            init(4),
            init(5),
            init(6),
            init(7),
            init(8),
            init(9),
        ]
    };
}

impl Iterator for TakBoardIter {
    type Item = (ValidPiece, u8, u8); // PieceType, Square, Depth
    fn next(&mut self) -> Option<Self::Item> {
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
        Some((piece, square, Self::DEPTH_TABLE[self.depth as usize]))
    }
}

/// A column of the feature-weights matrix.
/// Note the `align(64)`.
#[derive(Clone, Copy, PartialEq, Debug)]
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

struct AccPair {
    stm: Accumulator,
    ntm: Accumulator,
    // stm_vec: Vec<usize>,
    // ntm_vec: Vec<usize>,
}

impl AccPair {
    fn new(stm: Accumulator, ntm: Accumulator) -> Self {
        Self {
            stm,
            ntm,
            // stm_vec: Vec::new(),
            // ntm_vec: Vec::new(),
        }
    }
    fn flip(&mut self) {
        std::mem::swap(&mut self.stm, &mut self.ntm);
    }
    fn flip_from_parent(parent: &AccPair) -> Self {
        AccPair {
            stm: parent.ntm.clone(),
            ntm: parent.stm.clone(),
            // stm_vec: parent.ntm_vec.clone(),
            // ntm_vec: parent.stm_vec.clone(),
        }
    }
    fn f(&mut self, stm: usize, ntm: usize) {
        self.stm.add_feature(stm, &NNUE);
        self.ntm.add_feature(ntm, &NNUE);
        // self.stm_vec.push(stm);
        // self.ntm_vec.push(ntm);
    }
    fn f_inverse(&mut self, stm: usize, ntm: usize) {
        self.stm.remove_feature(stm, &NNUE);
        self.ntm.remove_feature(ntm, &NNUE);
        // let rem = self.ntm_vec.iter().position(|&x| x == ntm);
        // if let Some(rem) = rem {
        //     self.ntm_vec.remove(rem);
        // }
        // let rem = self.stm_vec.iter().position(|&x| x == stm);
        // if let Some(rem) = rem {
        //     self.stm_vec.remove(rem);
        // }
    }
}

struct AccUpdate {
    data: AccPair,
    pending: TakBoardUpdate,
}

impl AccUpdate {
    fn new(data: AccPair, pending: TakBoardUpdate) -> Self {
        Self { data, pending }
    }
    fn eat_pending(&mut self) -> TakBoardUpdate {
        let mut new = TakBoardUpdate::Empty;
        std::mem::swap(&mut self.pending, &mut new);
        new
    }
    fn apply_pending(&mut self, color: Color) {
        match self.eat_pending() {
            TakBoardUpdate::Empty => {
                // Null move
            }
            TakBoardUpdate::Placement(game_move, swap) => {
                let square = game_move.src_index();
                let mut place_piece = game_move.place_piece();
                if swap == 1 {
                    place_piece = place_piece.swap_color();
                }
                let psquare = super::build_piece_square(square, place_piece);
                self.top_update(square, psquare, color, place_piece.is_cap(), AccPair::f);
            }
            TakBoardUpdate::Stack(full_throw) => {
                // dbg!("Stack Move");
                for update in full_throw.data {
                    let square = update.sq as usize;
                    let old_top = update.old.top_piece();
                    let new_top = update.new.top_piece();
                    let top_out = self.lookup_top(old_top, color);
                    let top_in;
                    if old_top == new_top {
                        top_in = top_out;
                    } else {
                        top_in = self.lookup_top(new_top, color);
                        if let Some(old) = old_top {
                            let old_pq = super::build_piece_square(square, old);
                            self.top_update(
                                square,
                                old_pq,
                                color,
                                old.is_cap(),
                                AccPair::f_inverse,
                            );
                        }
                        if let Some(new) = new_top {
                            let new_pq = super::build_piece_square(square, new);
                            self.top_update(square, new_pq, color, new.is_cap(), AccPair::f);
                        }
                    }
                    let old_friendly = update.old.under_bits(color);
                    let new_friendly = update.new.under_bits(color);
                    let old_enemy = update.old.under_bits(!color);
                    let new_enemy = update.new.under_bits(!color);
                    if top_in == top_out {
                        // Figure out the diff
                        let friendly_diff = old_friendly ^ new_friendly;
                        let enemy_diff = old_enemy ^ new_enemy;
                        let friendly_in = friendly_diff & new_friendly;
                        let friendly_out = friendly_diff & old_friendly;
                        let enemy_in = enemy_diff & new_enemy;
                        let enemy_out = enemy_diff & old_enemy;
                        self.body_update(square, 0, friendly_out, top_out, AccPair::f_inverse);
                        self.body_update(square, 0, friendly_in, top_in, AccPair::f);
                        self.body_update(square, 1, enemy_out, top_out, AccPair::f_inverse);
                        self.body_update(square, 1, enemy_in, top_in, AccPair::f);
                    } else {
                        // Needs full rehash
                        self.body_update(square, 0, old_friendly, top_out, AccPair::f_inverse);
                        self.body_update(square, 0, new_friendly, top_in, AccPair::f);
                        self.body_update(square, 1, old_enemy, top_out, AccPair::f_inverse);
                        self.body_update(square, 1, new_enemy, top_in, AccPair::f);
                    }
                }
            }
        }
    }
    fn lookup_top(&self, p: Option<Piece>, side: Color) -> usize {
        if let Some(p) = p {
            let agrees = p.is_white() ^ (side == Color::White);
            agrees as usize
        } else {
            0
        }
    }
    fn top_update<U: FnMut(&mut AccPair, usize, usize)>(
        &mut self,
        square: usize,
        psquare: PieceSquare,
        color: Color,
        is_cap: bool,
        mut update: U,
    ) {
        let mut piece = psquare.piece();
        if is_cap {
            piece = piece.promote_cap();
        }
        let ours = (piece.is_white() ^ color.is_white()) as usize; // 0 if matches, else 1
        let location = usize::from(piece.without_color());

        let stm = [0, TakSimple6::HALF][ours] + 36 * location + square;
        let ntm = [TakSimple6::HALF, 0][ours] + 36 * location + square;
        update(&mut self.data, stm, ntm);
    }
    #[inline]
    fn body_update<U: FnMut(&mut AccPair, usize, usize)>(
        &mut self,
        square: usize,
        ours: usize,
        mut bits: u16,
        top: usize,
        mut update: U,
    ) {
        while bits != 0 {
            let lowest = bits.lowest_one().unwrap();
            if lowest < 9 {
                let location = TakBoardIter::DEPTH_TABLE[1 + lowest as usize] as usize;
                let stm = [0, TakSimple6::HALF][ours] + 36 * (location + top) + square;
                let ntm = [TakSimple6::HALF, 0][ours] + 36 * (location + [1, 0][top]) + square;
                update(&mut self.data, stm, ntm);
            }
            bits = bits & !(1 << lowest); // Should lop off lowest one
        }
    }
    fn update_acc_from_parent(&mut self, parent: &AccPair) {
        self.data = AccPair::flip_from_parent(parent);
    }
}

pub struct NNUE6 {
    dirty: [bool; 64],
    data: [AccUpdate; 64],
    pub(crate) tempo_offset: i32,
    root_player: Color,
}

impl NNUE6 {
    pub fn clear(&mut self) {
        for e in self.dirty.iter_mut() {
            *e = true;
        }
    }
    fn get_parity(&self, idx: usize) -> Color {
        if idx % 2 == 0 {
            self.root_player
        } else {
            !self.root_player
        }
    }
    pub fn _begin_search(&mut self, board: &mut crate::Board6) {
        let root_even = Self::rebuild_from_board(board);
        self.data[0] = AccUpdate::new(root_even, TakBoardUpdate::Empty);
        self.dirty[0] = false;
        self.root_player = board.side_to_move();
    }
    pub fn _nudge(&mut self, pending: TakBoardUpdate, ply_depth: usize) {
        assert!(ply_depth > 0);
        self.data[ply_depth].pending = pending;
        self.dirty[ply_depth] = true;
    }
    pub fn incremental_eval(&mut self, board: &Board6, ply_depth: usize) -> i32 {
        let mut idx = ply_depth;
        while self.dirty[idx] {
            idx -= 1; // Safe, because 0 is never dirty
            assert!(idx < usize::MAX - 10);
        }
        while idx < ply_depth {
            idx += 1; // Safe because we want the first dirty index
            let color = self.get_parity(idx);
            let (head, tail) = self.data.split_at_mut(idx);
            tail[0].update_acc_from_parent(&head.last().unwrap().data);
            tail[0].apply_pending(color);
            self.dirty[idx] = false;
        }
        let pair = &self.data[ply_depth].data;

        let mut cached_stm = pair.stm.clone();
        let mut cached_ntm = pair.ntm.clone();
        let w_reserves = board.pieces_reserve(Color::White) + board.caps_reserve(Color::White);
        let b_reserves = board.pieces_reserve(Color::Black) + board.caps_reserve(Color::Black);
        if let Color::White = board.side_to_move() {
            // Am white, with reserves vs. Am black with reserves
            cached_stm.add_feature(TakSimple6::SQUARE_INPUTS + w_reserves / 2, &NNUE);
            cached_ntm.add_feature(TakSimple6::SQUARE_INPUTS + 40 + b_reserves / 2, &NNUE);
        } else {
            cached_stm.add_feature(TakSimple6::SQUARE_INPUTS + 40 + b_reserves / 2, &NNUE);
            cached_ntm.add_feature(TakSimple6::SQUARE_INPUTS + w_reserves / 2, &NNUE);
        }
        let eval = NNUE.evaluate(&cached_stm, &cached_ntm);
        eval
    }
    fn rebuild_from_board(board: &Board6) -> AccPair {
        let takboard = super::build_nn_repr(board);
        let (ours, theirs) = build_features(takboard);
        let ours = Incremental::fresh_new(&NNUE, ours);
        let theirs = Incremental::fresh_new(&NNUE, theirs);
        AccPair::new(ours.vec, theirs.vec)
    }
    pub(crate) fn manual_eval(takboard: BoardData, board: &Board6) -> i32 {
        let (ours, theirs) = build_features(takboard);
        let mut ours = Incremental::fresh_new(&NNUE, ours);
        let mut theirs = Incremental::fresh_new(&NNUE, theirs);
        let w_reserves = board.pieces_reserve(Color::White) + board.caps_reserve(Color::White);
        let b_reserves = board.pieces_reserve(Color::Black) + board.caps_reserve(Color::Black);
        // let ours: Vec<_> = ours.state.clone().into_iter().collect();
        // dbg!(ours);
        // let theirs: Vec<_> = theirs.state.clone().into_iter().collect();
        // dbg!(theirs);
        if let Color::White = board.side_to_move() {
            // Am white, with reserves vs. Am black with reserves
            ours.vec
                .add_feature(TakSimple6::SQUARE_INPUTS + w_reserves / 2, &NNUE);
            theirs
                .vec
                .add_feature(TakSimple6::SQUARE_INPUTS + 40 + b_reserves / 2, &NNUE);
        } else {
            ours.vec
                .add_feature(TakSimple6::SQUARE_INPUTS + 40 + b_reserves / 2, &NNUE);
            theirs
                .vec
                .add_feature(TakSimple6::SQUARE_INPUTS + w_reserves / 2, &NNUE);
        }
        let eval = NNUE.evaluate(&ours.vec, &theirs.vec);
        eval
    }
}

impl Default for NNUE6 {
    fn default() -> Self {
        let acc = Incremental::fresh_empty(&NNUE).vec;
        Self {
            dirty: [true; 64],
            data: std::array::from_fn(|_| {
                AccUpdate::new(
                    AccPair::new(acc.clone(), acc.clone()),
                    TakBoardUpdate::Empty,
                )
            }),
            root_player: Color::White,
            tempo_offset: 100,
        }
    }
}

fn build_features(takboard: BoardData) -> (IncrementalState, IncrementalState) {
    let mut ours = IncrementalState::empty();
    let mut theirs = IncrementalState::empty();
    let simple = TakSimple6 {};
    simple.handle_features(&takboard, |x, y| {
        ours.add_feature(x as u16);
        theirs.add_feature(y as u16);
    });
    (ours, theirs)
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
    // pqst: [i16; TakSimple6::NUM_INPUTS],
    /// Scalar output bias.
    output_bias: i16,
}

impl Network {
    /// Calculates the output of the network, starting from the already
    /// calculated hidden layer (done efficiently during makemoves).
    fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        // Initialise output with bias.
        let mut sum = 0;
        // let mut psqt_out = 0;

        // Side-To-Move Accumulator -> Output.
        for (&input, &weight) in us.vals.iter().zip(&self.output_weights[..HIDDEN_SIZE]) {
            sum += screlu(input) * i32::from(weight);
        }

        // Not-Side-To-Move Accumulator -> Output.
        for (&input, &weight) in them.vals.iter().zip(&self.output_weights[HIDDEN_SIZE..]) {
            sum += screlu(input) * i32::from(weight);
        }

        // Remove quantisation.
        let output =
            (sum / (QA as i32) + i32::from(self.output_bias)) * SCALE / (QA as i32 * QB as i32);
        output
    }
}

// Sorry this naming convention is so bad
struct Incremental {
    state: IncrementalState,
    vec: Accumulator,
}

impl Incremental {
    fn fresh_empty(net: &Network) -> Self {
        let acc = Accumulator::new(net);
        let inc = IncrementalState::empty();
        Self {
            state: inc,
            vec: acc,
        }
    }
    fn fresh_new(net: &Network, data: IncrementalState) -> Self {
        const THRESHOLD: u16 = TakSimple6::SQUARE_INPUTS as u16;
        let mut acc = Accumulator::new(net);
        for d in data.clone().into_iter() {
            if d < THRESHOLD {
                acc.add_feature(d as usize, net);
            }
        }
        Self {
            vec: acc,
            state: data,
        }
    }
}

struct BitSetIterator {
    bitset: [u64; Self::SIZE],
    idx: usize,
}

impl BitSetIterator {
    const SIZE: usize = 73;
    const END: usize = Self::SIZE - 1;
}

impl Iterator for BitSetIterator {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        while self.bitset[self.idx] == 0 {
            if self.idx >= Self::END {
                return None;
            }
            self.idx += 1
        }
        let b_idx = pop_lowest(&mut self.bitset[self.idx]) as u16;
        let out = self.idx as u16 * 64 + b_idx;
        Some(out)
    }
}

#[derive(Clone, PartialEq)]
struct IncrementalState {
    pub(crate) bitset: [u64; 73],
}

impl IntoIterator for IncrementalState {
    type Item = u16;

    type IntoIter = BitSetIterator;

    fn into_iter(self) -> Self::IntoIter {
        BitSetIterator {
            bitset: self.bitset,
            idx: 0,
        }
    }
}

impl IncrementalState {
    pub fn empty() -> Self {
        let bitset = [0; 73];
        Self { bitset }
    }
    pub fn from_vec(vec: Vec<u16>) -> Self {
        let mut bitset = [0; 73];
        for val in vec {
            let b_idx = (val / 64) as usize;
            let b_val = 1 << (val % 64);
            bitset[b_idx] |= b_val
        }
        Self { bitset }
    }
    pub fn add_feature(&mut self, val: u16) {
        let b_idx = (val / 64) as usize;
        let b_val = 1 << (val % 64);
        self.bitset[b_idx] |= b_val
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board6;
    use crate::board::TakBoard;
    use crate::eval::build_nn_repr;
    #[test]
    fn check_rotation() {
        let tps = "2,x3,1,x/2,1,1,x,2112C,2/2,2,2,1S,1S,2/2,2,1,21,1,2/1,212S,1C,x,12S,1/2,x,1,1S,12,1 1 4";
        let board = Board6::try_from_tps(tps).unwrap();
        let data = build_nn_repr(&board);
        let syms: Vec<_> = (0..8)
            .into_iter()
            .map(|idx| {
                let d = data.symmetry(idx);
                let board = Board6::build_from_pieces(d.caps, d.data, d.white_to_move);
                format!("{:?}", board)
            })
            .collect();
        assert_eq!("2,x,1,1S,12,1/1,212S,1C,x,12S,1/2,2,1,21,1,2/2,2,2,1S,1S,2/2,1,1,x,2112C,2/2,x,x,x,1,x 1 4", &syms[1]);
        assert_eq!("x,1,x,x,x,2/2,2112C,x,1,1,2/2,1S,1S,2,2,2/2,1,21,1,2,2/1,12S,x,1C,212S,1/1,12,1S,1,x,2 1 4", &syms[2]);
        assert_eq!("x,2,2,2,1,1/1,2112C,1S,1,12S,12/x,x,1S,21,x,1S/x,1,2,1,1C,1/x,1,2,2,212S,x/2,2,2,2,1,2 1 4", &syms[3]);
        assert_eq!("1,12,1S,1,x,2/1,12S,x,1C,212S,1/2,1,21,1,2,2/2,1S,1S,2,2,2/2,2112C,x,1,1,2/x,1,x,x,x,2 1 4", &syms[4]);
        assert_eq!("2,1,2,2,2,2/x,212S,2,2,1,x/1,1C,1,2,1,x/1S,x,21,1S,x,x/12,12S,1,1S,2112C,1/1,1,2,2,2,x 1 4", &syms[5]);
        assert_eq!("2,2,2,2,1,2/x,1,2,2,212S,x/x,1,2,1,1C,1/x,x,1S,21,x,1S/1,2112C,1S,1,12S,12/x,2,2,2,1,1 1 4", &syms[6]);
        assert_eq!("1,1,2,2,2,x/12,12S,1,1S,2112C,1/1S,x,21,1S,x,x/1,1C,1,2,1,x/x,212S,2,2,1,x/2,1,2,2,2,2 1 4", &syms[7]);
    }
    // #[test]
    // fn check_repr() {
    //     let tps = "2,x3,1,x/2,1,1,x,2112C,2/2,2,2,1S,1S,2/2,2,1,21,1,2/1,212S,1C,x,12S,1/2,x,1,1S,12,1 1 24";
    //     let board = Board6::try_from_tps(tps).unwrap();
    //     let (caps, pieces, white_to_move) = super::super::build_nn_repr(&board);
    //     let mut rebuilt = Board6::build_from_pieces(caps, pieces, white_to_move);
    //     rebuilt.set_move_number(24);
    //     let tps_rebuilt = format!("{:?}", &rebuilt);
    //     let tps_original = format!("{:?}", &board);
    //     assert_eq!(tps_original, tps_rebuilt);
    //     assert!(board.bits() == rebuilt.bits());
    //     assert_eq!(board.hash(), rebuilt.hash());
    //     assert_eq!(board, rebuilt);
    // }
}
