use super::bitboard::{Bitboard, BitboardStorage};
use super::Piece;

#[derive(PartialEq, Clone, Debug)]
pub struct Stack {
    data: Vec<Piece>,
    index: usize,
}

impl Stack {
    pub const fn new() -> Self {
        const VEC: Vec<Piece> = Vec::new();
        const IDX: usize = 0;
        Stack {
            data: VEC,
            index: IDX,
        }
    }
    pub fn init(&mut self, index: usize) {
        self.data.reserve(8);
        self.index = index;
    }
    pub fn push<T: Bitboard>(&mut self, item: Piece, bits: &mut BitboardStorage<T>) {
        self.hash_out_top(bits);
        self.data.push(item);
        self.hash_in_top(bits);
    }
    pub fn last(&self) -> Option<&Piece> {
        self.data.last()
    }
    pub fn pop<T: Bitboard>(&mut self, bits: &mut BitboardStorage<T>) -> Option<Piece> {
        self.hash_out_top(bits);
        let ret = self.data.pop();
        self.hash_in_top(bits);
        ret
    }
    pub fn iter(&self) -> std::slice::Iter<Piece> {
        self.data.iter()
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Mimicking the Extend trait from std, but we need an extra parameter
    pub fn extend<T, U>(&mut self, iter: T, bits: &mut BitboardStorage<U>)
    where
        T: IntoIterator<Item = Piece>,
        U: Bitboard,
    {
        for item in iter {
            self.push(item, bits);
        }
    }
    pub fn try_crush_wall<T: Bitboard>(&mut self, bits: &mut BitboardStorage<T>) -> bool {
        if self.len() >= 2 {
            let wall_idx = self.len() - 2;
            if let Some(crushed) = self.data[wall_idx].crush() {
                self.data[wall_idx] = crushed;
                return true;
            }
        }
        false
    }
    pub fn uncrush_wall<T: Bitboard>(&mut self, bits: &mut BitboardStorage<T>) {
        if self.len() >= 2 {
            let wall_idx = self.len() - 2;
            if let Some(uncrushed) = self.data[wall_idx].uncrush() {
                self.data[wall_idx] = uncrushed;
                return;
            }
        }
        panic!("Could not find piece to uncrush!");
    }
    pub fn reverse_top<T: Bitboard>(&mut self, top_n: usize, bits: &mut BitboardStorage<T>) {
        // If top_n is only 1 then we just reverse the first piece, which does nothing
        if top_n > 1 {
            self.hash_out_many(top_n, bits);
            let range_st = self.len() - top_n;
            let slice = &mut self.data[range_st..];
            slice.reverse();
            self.hash_in_many(top_n, bits);
        }
    }
    pub fn split_off<T: Bitboard>(
        &mut self,
        top_n: usize,
        bits: &mut BitboardStorage<T>,
    ) -> Vec<Piece> {
        self.hash_out_many(top_n, bits);
        let split_idx = self.len() - top_n;
        let vec = self.data.split_off(split_idx);
        self.hash_in_top(bits);
        vec
    }
    fn hash_out_many<T: Bitboard>(&self, top_n: usize, bits: &mut BitboardStorage<T>) {
        self.hash_out_top(bits);
        // Todo zobrist operations
    }
    fn hash_in_many<T: Bitboard>(&self, top_n: usize, bits: &mut BitboardStorage<T>) {
        self.hash_in_top(bits);
        // Todo zobrist operations
    }
    fn hash_in_top<T: Bitboard>(&self, bits: &mut BitboardStorage<T>) {
        let bitboard = T::index_to_bit(self.index);
        match self.data.last() {
            Some(Piece::WhiteFlat) => {
                bits.white |= bitboard;
                bits.flat |= bitboard;
            }
            Some(Piece::BlackFlat) => {
                bits.black |= bitboard;
                bits.flat |= bitboard;
            }
            Some(Piece::WhiteWall) => {
                bits.white |= bitboard;
                bits.wall |= bitboard;
            }
            Some(Piece::BlackWall) => {
                bits.black |= bitboard;
                bits.wall |= bitboard;
            }
            Some(Piece::WhiteCap) => {
                bits.white |= bitboard;
                bits.cap |= bitboard;
            }
            Some(Piece::BlackCap) => {
                bits.black |= bitboard;
                bits.cap |= bitboard;
            }
            _ => {}
        }
    }
    fn hash_out_top<T: Bitboard>(&self, bits: &mut BitboardStorage<T>) {
        let bitboard = T::index_to_bit(self.index);
        match self.data.last() {
            Some(Piece::WhiteFlat) => {
                bits.white -= bitboard;
                bits.flat -= bitboard;
            }
            Some(Piece::BlackFlat) => {
                bits.black -= bitboard;
                bits.flat -= bitboard;
            }
            Some(Piece::WhiteWall) => {
                bits.white -= bitboard;
                bits.wall -= bitboard;
            }
            Some(Piece::BlackWall) => {
                bits.black -= bitboard;
                bits.wall -= bitboard;
            }
            Some(Piece::WhiteCap) => {
                bits.white -= bitboard;
                bits.cap -= bitboard;
            }
            Some(Piece::BlackCap) => {
                bits.black -= bitboard;
                bits.cap -= bitboard;
            }
            _ => {}
        }
    }
}
