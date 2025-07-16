use super::bitboard::{Bitboard, BitboardStorage};
use super::Piece;

#[derive(PartialEq, Clone, Debug)]
pub struct Stack {
    data: u64,
    length: u8,
    index: u8,
    top_piece: Piece,
}

impl Stack {
    pub const fn new() -> Self {
        Stack {
            data: 0,
            length: 0,
            index: 0,
            top_piece: Piece::WhiteFlat,
        }
    }
    pub(crate) fn set_index(&mut self, index: usize) {
        self.index = index as u8;
    }
    pub fn init(&mut self, index: usize) {
        self.index = index as u8;
    }
    pub fn interpret_coarse(&self, side_to_move: crate::Color) -> u16 {
        if self.len() == 0 {
            return 0;
        }
        let stack_piece = self
            .top()
            .map(|x| {
                if side_to_move == crate::Color::White {
                    x as usize
                } else {
                    x.swap_color() as usize
                }
            })
            .unwrap();
        let (cap, friendly) = self.captive_friendly();
        let cap_friendly_offset = std::cmp::min(cap, 6) * 7 + std::cmp::min(friendly, 6);
        let idx = (stack_piece * 49) as u16 + cap_friendly_offset as u16;
        idx
    }
    pub fn interpret(&self, side_to_move: crate::Color) -> u16 {
        if self.length == 0 {
            0
        } else {
            let len = std::cmp::min(6, self.length);
            let mask = (1 << len) - 1;
            let mut under = match side_to_move {
                crate::Color::Black => ((!self.data) & mask) >> 1,
                crate::Color::White => (self.data & mask) >> 1,
            };
            // let mut under = match self.top_piece.owner() {
            //     crate::Color::White => ((!self.data) & mask) >> 1,
            //     crate::Color::Black => (self.data & mask) >> 1,
            // };
            under |= 1 << (len - 1); // Stack length bit
            let top = (self.top_piece.kind_index() as u16) * 64;
            if self.top_piece.owner() == side_to_move {
                top + (under as u16)
            } else {
                192 + top + (under as u16)
            }
        }
    }
    pub fn from_bits(high_bits: u64, length: u8, top_piece: Piece) -> Self {
        let mut dummy: BitboardStorage<crate::board::Bitboard6> = BitboardStorage::default();
        match length {
            0 => Self::new(),
            1 => {
                let mut stack = Stack::new();
                stack.push(top_piece, &mut dummy);
                stack
            }
            _ => {
                let mut stack = Stack::new();
                stack.push(top_piece, &mut dummy);
                stack.data |= high_bits << 1;
                stack
            }
        }
    }
    fn get_inner_piece(&self, index: usize) -> Piece {
        let mask = 1 << index;
        if self.data & mask == 0 {
            Piece::WhiteFlat
        } else {
            Piece::BlackFlat
        }
    }
    pub fn bottom(&self) -> Option<Piece> {
        self.from_top(self.length.wrapping_sub(1) as usize)
    }
    /// Indexes the stack from the top, assuming the highest piece in the stack has index 0
    pub fn from_top(&self, index: usize) -> Option<Piece> {
        if index >= self.len() {
            None
        } else if index == 0 {
            Some(self.top_piece)
        } else {
            Some(self.get_inner_piece(index))
        }
    }
    pub fn has_wall(&self) -> bool {
        self.top_piece.is_wall()
    }
    pub fn has_cap(&self) -> bool {
        self.top_piece.is_cap()
    }
    pub fn push<T: Bitboard>(&mut self, item: Piece, bits: &mut BitboardStorage<T>) {
        self.hash_out_top(bits);
        bits.zobrist_middle(item, self.index as usize, self.len());
        self.length += 1;
        let color_bit = (item.owner() == crate::Color::Black) as u64;
        self.data = (self.data << 1) | color_bit;
        self.top_piece = item;
        self.hash_in_top(bits);
    }
    pub fn top(&self) -> Option<Piece> {
        if self.length > 0 {
            Some(self.top_piece)
        } else {
            None
        }
    }
    pub fn under_top(&self) -> Option<Piece> {
        if self.length > 1 {
            if self.data & 0b0010 == 0 {
                Some(Piece::WhiteFlat)
            } else {
                Some(Piece::BlackFlat)
            }
        } else {
            None
        }
    }
    pub fn limited_captive_friendly(&self) -> (i32, i32) {
        if self.length <= 1 {
            return (0, 0);
        }
        let vals = self.white_black_deep();
        if self
            .top()
            .map(|x| x.owner() == super::Color::White)
            .unwrap_or(true)
        {
            (vals[1], vals[0])
        } else {
            (vals[0], vals[1])
        }
    }
    pub fn white_black_deep(&self) -> [i32; 4] {
        // Todo variable cutoff, checks?
        let black = (self.data >> 1).count_ones() as i32;
        let white = self.length as i32 - 1 - black;
        let black_deep = (self.data >> 7).count_ones() as i32;
        let white_deep = std::cmp::max(0, self.length as i32 - 7 - black_deep);
        [
            white - white_deep,
            black - black_deep,
            white_deep,
            black_deep,
        ]
    }
    pub fn captive_friendly(&self) -> (i32, i32) {
        if self.length <= 1 {
            return (0, 0);
        }
        let black = (self.data >> 1).count_ones() as i32;
        let white = self.length as i32 - 1 - black;
        if self.top_piece.owner() == crate::Color::White {
            (black, white)
        } else {
            (white, black)
        }
    }
    pub fn pop<T: Bitboard>(&mut self, bits: &mut BitboardStorage<T>) -> Option<Piece> {
        self.hash_out_top(bits);
        let ret = self.top();
        if let Some(piece) = ret {
            self.data = self.data >> 1;
            self.length -= 1;
            if self.data & 1 == 0 {
                self.top_piece = Piece::WhiteFlat;
            } else {
                self.top_piece = Piece::BlackFlat;
            }
            bits.zobrist_middle(piece, self.index as usize, self.len());
        } else {
            self.top_piece = Piece::BlackFlat;
        }
        self.hash_in_top(bits);
        ret
    }
    pub fn iter(&self) -> StackIterator {
        StackIterator::new(&self, 0)
    }
    pub fn len(&self) -> usize {
        self.length as usize
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn add_stack<T: Bitboard>(
        &mut self,
        num: usize,
        other: &mut Pickup,
        bits: &mut BitboardStorage<T>,
    ) {
        self.hash_out_top(bits);
        let add = other.take(num as u8);
        // dbg!(format!("{:X}", add));
        self.data = self.data << num;
        self.data |= add;
        if other.length == 0 {
            self.top_piece = other.top;
        } else {
            if self.data & 1 == 0 {
                self.top_piece = Piece::WhiteFlat;
            } else {
                self.top_piece = Piece::BlackFlat;
            }
        }
        self.length += num as u8;
        self.hash_in_many(num, bits);
    }
    // Convert the top piece of a stack into a Capstone
    pub fn promote_top<T: Bitboard>(&mut self, bits: &mut BitboardStorage<T>) -> crate::Color {
        self.hash_out_top(bits);
        let top = match self.top_piece.owner() {
            crate::Color::Black => Piece::BlackCap,
            crate::Color::White => Piece::WhiteCap,
        };
        self.top_piece = top;
        self.hash_in_top(bits);
        top.owner()
    }
    pub fn uncrush_top<T: Bitboard>(&mut self, bits: &mut BitboardStorage<T>) {
        self.hash_out_top(bits);
        self.top_piece = self.top_piece.uncrush().unwrap();
        self.hash_in_top(bits);
    }
    pub fn split_off<T: Bitboard>(
        &mut self,
        top_n: usize,
        bits: &mut BitboardStorage<T>,
    ) -> Pickup {
        self.hash_out_many(top_n, bits);
        let mask = (1 << top_n) - 1;
        let take_bits = self.data & mask;
        let pickup = Pickup::new(take_bits as u8, self.top_piece, top_n as u8);
        self.data = self.data >> top_n;
        if self.data & 1 == 0 {
            self.top_piece = Piece::WhiteFlat;
        } else {
            self.top_piece = Piece::BlackFlat;
        }
        self.length = self.length - top_n as u8;
        self.hash_in_top(bits);
        pickup
    }
    fn hash_out_many<T: Bitboard>(&self, top_n: usize, bits: &mut BitboardStorage<T>) {
        if self.length == 0 {
            return;
        }
        let tile_index = self.index as usize;
        self.hash_out_top(bits);
        let mut stack_bits = self.data;
        for i in (self.len() - top_n..self.len()).rev() {
            let piece = if stack_bits & 1 == 0 {
                Piece::WhiteFlat
            } else {
                Piece::BlackFlat
            };
            bits.zobrist_middle(piece, tile_index, i);
            stack_bits = stack_bits >> 1;
        }
        // Todo zobrist operations
    }
    fn hash_in_many<T: Bitboard>(&self, top_n: usize, bits: &mut BitboardStorage<T>) {
        if self.length == 0 {
            return;
        }
        let tile_index = self.index as usize;
        self.hash_in_top(bits);
        // Todo off by one ?
        let mut stack_bits = self.data;
        for i in (self.len() - top_n..self.len()).rev() {
            let piece = if stack_bits & 1 == 0 {
                Piece::WhiteFlat
            } else {
                Piece::BlackFlat
            };
            bits.zobrist_middle(piece, tile_index, i);
            stack_bits = stack_bits >> 1;
        }
        // Todo zobrist operations
    }
    fn hash_in_top<T: Bitboard>(&self, bits: &mut BitboardStorage<T>) {
        let index = self.index as usize;
        let bitboard = T::index_to_bit(index);
        match self.top() {
            Some(Piece::WhiteFlat) => {
                bits.white |= bitboard;
                bits.flat |= bitboard;
                bits.zobrist_top(Piece::WhiteFlat, index);
            }
            Some(Piece::BlackFlat) => {
                bits.black |= bitboard;
                bits.flat |= bitboard;
                bits.zobrist_top(Piece::BlackFlat, index);
            }
            Some(Piece::WhiteWall) => {
                bits.white |= bitboard;
                bits.wall |= bitboard;
                bits.zobrist_top(Piece::WhiteWall, index);
            }
            Some(Piece::BlackWall) => {
                bits.black |= bitboard;
                bits.wall |= bitboard;
                bits.zobrist_top(Piece::BlackWall, index);
            }
            Some(Piece::WhiteCap) => {
                bits.white |= bitboard;
                bits.cap |= bitboard;
                bits.zobrist_top(Piece::WhiteCap, index);
            }
            Some(Piece::BlackCap) => {
                bits.black |= bitboard;
                bits.cap |= bitboard;
                bits.zobrist_top(Piece::BlackCap, index);
            }
            _ => {}
        }
    }
    fn hash_out_top<T: Bitboard>(&self, bits: &mut BitboardStorage<T>) {
        let index = self.index as usize;
        let bitboard = T::index_to_bit(index);
        match self.top() {
            Some(Piece::WhiteFlat) => {
                bits.white -= bitboard;
                bits.flat -= bitboard;
                bits.zobrist_top(Piece::WhiteFlat, index);
            }
            Some(Piece::BlackFlat) => {
                bits.black -= bitboard;
                bits.flat -= bitboard;
                bits.zobrist_top(Piece::BlackFlat, index);
            }
            Some(Piece::WhiteWall) => {
                bits.white -= bitboard;
                bits.wall -= bitboard;
                bits.zobrist_top(Piece::WhiteWall, index);
            }
            Some(Piece::BlackWall) => {
                bits.black -= bitboard;
                bits.wall -= bitboard;
                bits.zobrist_top(Piece::BlackWall, index);
            }
            Some(Piece::WhiteCap) => {
                bits.white -= bitboard;
                bits.cap -= bitboard;
                bits.zobrist_top(Piece::WhiteCap, index);
            }
            Some(Piece::BlackCap) => {
                bits.black -= bitboard;
                bits.cap -= bitboard;
                bits.zobrist_top(Piece::BlackCap, index);
            }
            _ => {}
        }
    }
}

pub struct StackIterator<'a> {
    stack: &'a Stack,
    count: usize,
}

impl<'a> StackIterator<'a> {
    fn new(stack: &'a Stack, count: usize) -> Self {
        Self { stack, count }
    }
}

impl<'a> Iterator for StackIterator<'a> {
    type Item = Piece;

    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        if self.count > self.stack.len() {
            return None;
        } else if self.count == self.stack.len() {
            return Some(self.stack.top_piece);
        }
        let mask = 1 << (self.stack.len() - self.count);
        if self.stack.data & mask == 0 {
            Some(Piece::WhiteFlat)
        } else {
            Some(Piece::BlackFlat)
        }
    }
}

pub struct Pickup {
    pieces: u8,
    top: Piece,
    length: u8,
}

impl Pickup {
    pub fn new(pieces: u8, top: Piece, length: u8) -> Self {
        Self {
            pieces,
            top,
            length,
        }
    }
    fn take(&mut self, count: u8) -> u64 {
        let keep = self.length - count;
        let keep_mask: u64 = (1 << keep) - 1;
        let out = self.pieces >> keep;
        self.length -= count;
        self.pieces = self.pieces & (keep_mask as u8);
        out as u64
    }
    pub fn append(&mut self, other: Self) {
        self.top = other.top;
        self.pieces = self.pieces << other.length;
        self.pieces |= other.pieces;
        self.length += other.length;
    }
}

impl Default for Pickup {
    fn default() -> Self {
        Self {
            pieces: 0,
            top: Piece::WhiteFlat,
            length: 0,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{board::Board6, Position, TakBoard};
    #[test]
    fn deep_stack_counts() {
        let tps = "1,1S,2,112S,2,x/x2,1,1,1,x/x,1,x,11212211112C,21,1/x2,21,21C,1,1/x,221,1,x,1,2/2,2,1,2,2,2 1 34";
        let board = Board6::try_from_tps(tps).unwrap();
        let array = board.board[15].white_black_deep();
        assert_eq!(array, [3, 2, 3, 1]);
    }
    // #[test]
    // fn stack_interpret() {
    //     let board = crate::board::Board6::try_from_tps("212,x,1,x3/1S,1,x,1,x2/x,122212C,11,x3/12,2,1,121,x2/2,2,21,2S,x2/221C,21,1222111,x3 2 33").unwrap();
    //     for idx in 0..36 {
    //         let stack = board.index(idx);
    //         assert_eq!(
    //             stack.interpret(board.side_to_move()) % 64,
    //             stack.interpret(!board.side_to_move()) % 64
    //         );
    //     }
    //     let board = crate::board::Board6::try_from_tps(
    //         "1,1,1,1,x2/x,1S,x,1221C,2C,x/x2,2,121,1,x/x,22,1,1111112,x,12/x2,22S,2,1,1/2,2,x3,1 1 28",
    //     )
    //     .unwrap();
    //     let repr = nn_repr(&board);
    //     let expected = [
    //         0b1,
    //         0b1,
    //         0b1,
    //         0b1,
    //         0,
    //         0,
    //         0,
    //         0b1 + 64,
    //         0,
    //         0b1100 + 128,
    //         64 * 5 + 0b1,
    //         0,
    //         0,
    //         0,
    //         64 * 3 + 0b1,
    //         0b110,
    //         0b1,
    //         0,
    //         0,
    //         64 * 3 + 0b11,
    //         1,
    //         64 * 3 + 0b100000,
    //     ];
    //     for i in 0..36 {
    //         let val = repr[i];
    //         if let Some(x) = expected.get(i) {
    //             eprintln!("{val:#b}");
    //             assert_eq!(val, *x);
    //             if val == 64 * 3 + 0b100000 {
    //                 eprintln!("Extra test{:#b}", val - 192);
    //             }
    //         }
    //     }
    // }
}
