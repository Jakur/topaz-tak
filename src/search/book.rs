use once_cell::sync::OnceCell;
use rand_core::{RngCore, SeedableRng};
use weighted_trie::WeightedTrie;

static BOOK: OnceCell<Book> = OnceCell::new();

pub fn get_book_move(move_history: &str) -> Option<String> {
    BOOK.get()?.lookup(move_history)
}

pub fn load_book_data(book: Vec<String>) -> &'static Book {
    BOOK.get_or_init(|| build_book(book))
}

fn build_book(book: Vec<String>) -> Book {
    let mut trie = WeightedTrie::new();
    for line in book {
        trie.insert(line, 1);
    }
    Book { trie }
}

pub struct Book {
    trie: WeightedTrie,
}

impl Book {
    pub fn lookup(&self, s: &str) -> Option<String> {
        let vec = self.trie.search(s);
        let full = match vec.len() {
            0 => return None,
            1 => vec[0].clone(),
            _ => {
                let mut seed: [u8; 16] = [0; 16];
                getrandom::fill(&mut seed).unwrap();
                let mut rng = rand_xoshiro::Xoshiro128PlusPlus::from_seed(seed);
                let idx = rng.next_u32() as usize % vec.len();
                vec[idx].clone()
            }
        };
        let reduced = full.strip_prefix(s)?.split_whitespace().nth(0)?;
        Some(reduced.to_string())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn get_book_move() {
        let mvs = vec![
            "a1 f1 d3 c3 d4 Cc4 c2 b2".to_string(),
            "a1 f6 d4 c3 c4 d3 b4 b3".to_string(),
            "a6 f6 d4 c4 d3 c3 d5 c5".to_string(),
        ];
        let book = build_book(mvs);
        assert_eq!(book.lookup("a1 f1").unwrap(), "d3");
        assert_eq!(book.lookup("a6 f6 d4 c4").unwrap(), "d3");
        assert_eq!(book.lookup("a6 f1"), None);
    }
}
