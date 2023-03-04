use crate::{Color, GameMove, GameResult, TakBoard};
use anyhow::Result;
use miniserde::{json, Deserialize, Serialize};
use std::{collections::HashMap, io::Write};

#[derive(Serialize, Deserialize, Clone)]
pub struct Book {
    mode: BookMode,
    map: HashMap<u64, BookEval>,
}

impl Book {
    pub fn new(mode: BookMode, map: HashMap<u64, BookEval>) -> Self {
        Self { mode, map }
    }
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        write!(writer, "{}", &json::to_string(self))?;
        Ok(())
    }
    pub fn get<T: TakBoard + Clone>(&self, board: &T) -> Option<i32> {
        if board.ply() <= 3 {
            return None;
        }
        for hash in board.symmetries().into_iter().map(|x| x.hash()) {
            if let Some(val) = self.map.get(&hash) {
                let record = -1 * val.record(self.mode); // This seems to be backwards
                let offset = (board.ply() - 3) * 15;
                return Some(offset as i32 * record);
            }
        }
        None
    }
    pub fn update<T: TakBoard>(&mut self, mut board: T, winner: GameResult, moves: Vec<GameMove>) {
        for (idx, mv) in (0..=10).zip(moves.into_iter()) {
            let rotations = board.symmetries();
            for r_board in rotations.into_iter() {
                if idx > 0 {
                    let entry = self
                        .map
                        .entry(r_board.hash())
                        .or_insert(BookEval::new(0, 0, 0));
                    let winner = match winner {
                        GameResult::WhiteWin => Some(Color::White),
                        GameResult::BlackWin => Some(Color::Black),
                        GameResult::Draw => None,
                    };
                    if let Some(winner) = winner {
                        // Normal, not reversed ?
                        if winner == r_board.side_to_move() {
                            entry.add_win();
                        } else {
                            entry.add_loss();
                        }
                    } else {
                        entry.add_draw();
                    }
                }
            }
            board.do_move(mv);
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct BookEval {
    win: i32,
    loss: i32,
    draw: i32,
}

impl BookEval {
    pub fn new(win: i32, loss: i32, draw: i32) -> Self {
        Self { win, loss, draw }
    }
    pub fn add_win(&mut self) {
        self.win += 1;
    }
    pub fn add_loss(&mut self) {
        self.loss += 1;
    }
    pub fn add_draw(&mut self) {
        self.draw += 1;
    }
    pub fn empty() -> Self {
        Self::new(0, 0, 0)
    }
    pub fn record(&self, mode: BookMode) -> i32 {
        match mode {
            BookMode::Learn => self.win + self.draw - self.loss,
            BookMode::Best => self.win - self.loss,
            BookMode::Explore => self.win + self.draw + self.loss,
        }
    }
}

// #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
// pub struct BookMove {
//     mv: GameMove,
//     score: i32,
//     played: i32,
// }

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BookMode {
    Learn,
    Explore,
    Best,
}

#[cfg(test)]
mod test {

    use super::super::*;
    use super::*;
    use crate::eval::Weights5;
    use crate::Position;
    use crate::{board::Board5, board::Board6, TakBoard};

    #[test]
    fn move_is_lava() {
        let mut board = Board5::new();
        assert_eq!(board.ply(), 2);
        board.do_move(GameMove::try_from_ptn("a5", &board).unwrap());
        let e1 = GameMove::try_from_ptn("e1", &board).unwrap();
        let zo_0 = board.zobrist();
        let rev = board.do_move(e1);
        // dbg!(&board);
        let mut map = HashMap::new();
        let zo_1 = board.zobrist();
        map.insert(board.zobrist(), BookEval::new(10, 0, 0));
        // dbg!(board.zobrist());
        let book = Book::new(BookMode::Learn, map);
        board.reverse_move(rev);
        let rev = board.do_move(e1);
        assert_eq!(zo_1, board.zobrist());
        board.reverse_move(rev);
        dbg!(&board);
        dbg!(board.ply());
        assert_eq!(zo_0, board.zobrist());
        let mut eval = Weights5::default();
        // let mut moves = Vec::new();
        let mut info = SearchInfo::new(4, 2 << 14)
            .time_bank(TimeBank::flat(1_000))
            .book(book.clone());
        let outcome = search(&mut board, &mut eval, &mut info).unwrap();
        let mv = outcome.next().unwrap();
        let mv_ptn = mv.to_ptn::<Board5>();
        dbg!(&mv_ptn);
        assert!(mv_ptn != "e1");
    }
    #[test]
    fn ser_deser() {
        let mut board = Board6::new();
        assert_eq!(board.ply(), 2);
        board.do_move(GameMove::try_from_ptn("a6", &board).unwrap());
        let mut map = HashMap::new();
        map.insert(board.zobrist(), BookEval::new(1, 0, 0));
        let book = Book::new(BookMode::Learn, map);
        let copy: Book = json::from_str(&json::to_string(&book)).unwrap();
        assert_eq!(copy.map, book.map);
        assert_eq!(copy.mode, book.mode);
        // book.save(&mut std::fs::File::create("book.json").unwrap())
        //     .unwrap();
    }
}
