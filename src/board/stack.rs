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
    fn get_inner_piece(&self, index: usize) -> Piece {
        let mask = 1 << index;
        if self.data & mask == 0 {
            Piece::WhiteFlat
        } else {
            Piece::BlackFlat
        }
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
