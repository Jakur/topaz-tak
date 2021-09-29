use board_game_traits::Position;
use std::env;
use std::io::{self, BufRead};
use topaz_tak::eval::Evaluate;
use topaz_tak::search::{search, SearchInfo};
use topaz_tak::*;

pub fn main() {
    let args: Vec<String> = env::args().collect();
    let computer_first = if let Some(color_str) = args.get(1) {
        if color_str == "black" {
            false
        } else if color_str == "white" {
            true
        } else {
            println!("Assuming computer plays black.");
            false
        }
    } else {
        println!("Assuming computer plays black.");
        false
    };
    play_game(computer_first);
}

fn play_game(mut computer_turn: bool) {
    let mut board = Board6::new();
    while let None = board.game_result() {
        println!("{:?}", &board);
        if computer_turn {
            let mut info = SearchInfo::new(6);
            search(&mut board, &mut info);
            let pv_move = info.pv_move(&board).unwrap();
            println!("Computer Choose: {}", pv_move.to_ptn());
            board.do_move(pv_move);
        } else {
            let stdin = io::stdin();
            let line = stdin.lock().lines().next().unwrap().unwrap();
            if line == "q" {
                return;
            } else if line == "tps" {
                println!("{:?}", &board);
                continue;
            }
            let ptn_move = GameMove::try_from_ptn(&line, &board).unwrap();
            if !board.legal_move(ptn_move) {
                println!("Illegal Move Attempted!");
                continue;
            }
            board.do_move(ptn_move);
        }
        computer_turn = !computer_turn;
    }
}
