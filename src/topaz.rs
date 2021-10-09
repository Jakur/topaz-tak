use anyhow::Result;
use board_game_traits::Position;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::env;
use std::io::{self, BufRead};
use std::thread;
use topaz_tak::eval::Evaluate;
use topaz_tak::search::{search, SearchInfo};
use topaz_tak::*;

pub fn main() {
    let args: Vec<String> = env::args().collect();

    if let Some(color_str) = args.get(1) {
        if color_str == "black" {
            play_game(false);
        } else if color_str == "white" {
            play_game(true);
        } else {
            println!("Unknown argument: {}", color_str);
        }
    }
    let mut buffer = String::new();
    std::io::stdin()
        .read_line(&mut buffer)
        .expect("Could not read line");
    if buffer.trim() == "tei" {
        let (s, r) = unbounded();
        tei_loop(s);
        identify();
        play_game_tei(r);
    } else if buffer == "play white" {
        play_game(true)
    } else if buffer == "play black" {
        play_game(false)
    } else {
        println!("Unknown command: {}", buffer);
    }
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

fn play_game_tei(receiver: Receiver<TeiCommand>) -> Result<()> {
    let mut board = Board6::new();
    loop {
        let message = receiver.recv()?;
        match message {
            TeiCommand::Go(s) => {
                let mut info = SearchInfo::new(6);
                search(&mut board, &mut info);
                let pv_move = info.pv_move(&board).unwrap();
                println!("bestmove {}", pv_move.to_ptn());
            }
            TeiCommand::Position(s) => {
                board = Board6::new();
                for m in s.split_whitespace() {
                    if let Some(m) = GameMove::try_from_ptn(m, &board) {
                        board.do_move(m);
                    }
                }
            }
            TeiCommand::Quit => {
                break;
            }
            _ => println!("Unknown command: {:?}", message),
        }
    }
    Ok(())
}

fn identify() {
    println!("id name Topaz");
    println!("id author Justin Kur");
    println!("teiok");
}

fn tei_loop(sender: Sender<TeiCommand>) {
    thread::spawn(move || {
        let mut buffer = String::new();
        loop {
            std::io::stdin()
                .read_line(&mut buffer)
                .expect("Could not read line");
            let line = buffer.trim();
            if line == "tei" {
                identify();
            } else if line == "isready" {
                println!("readyok");
            } else if line == "quit" {
                sender.send(TeiCommand::Quit).unwrap();
                break;
            } else if line == "stop" {
                sender.send(TeiCommand::Stop).unwrap();
            } else if line.starts_with("position") {
                sender.send(TeiCommand::Position(line.to_string())).unwrap();
            } else if line.starts_with("go") {
                sender.send(TeiCommand::Go(line.to_string())).unwrap();
            } else {
                println!("Unknown Tei Command: {}", buffer);
            }
            buffer.clear();
        }
    });
}
