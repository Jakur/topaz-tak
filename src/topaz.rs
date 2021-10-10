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
            play_game_cmd(false);
        } else if color_str == "white" {
            play_game_cmd(true);
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
        play_game_cmd(true)
    } else if buffer == "play black" {
        play_game_cmd(false)
    } else {
        println!("Unknown command: {}", buffer);
    }
}

fn play_game_cmd(mut computer_turn: bool) {
    let mut board = Board6::new();
    while let None = board.game_result() {
        println!("{:?}", &board);
        if computer_turn {
            let mut info = SearchInfo::new(6, 5000);
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

struct TimeLeft {
    wtime: u64,
    btime: u64,
    winc: u64,
    binc: u64,
}

impl TimeLeft {
    pub fn new(tei_str: &str) -> Self {
        let mut ret = Self {
            wtime: 1000,
            btime: 1000,
            winc: 0,
            binc: 0,
        };
        for (field, val) in tei_str
            .split_whitespace()
            .zip(tei_str.split_whitespace().skip(1))
        {
            match (field, val.parse()) {
                ("wtime", Ok(val)) => ret.wtime = val,
                ("btime", Ok(val)) => ret.btime = val,
                ("winc", Ok(val)) => ret.winc = val,
                ("binc", Ok(val)) => ret.binc = val,
                _ => {}
            }
        }
        ret
    }
    fn use_time(&self, est_plies: usize, side_to_move: Color) -> u64 {
        let (time_bank, inc) = match side_to_move {
            Color::White => (self.wtime, self.winc),
            Color::Black => (self.btime, self.binc),
        };
        let use_bank = time_bank / (est_plies + 2) as u64 / 1000;
        use_bank + inc / 1000
    }
}

fn play_game_tei(receiver: Receiver<TeiCommand>) -> Result<()> {
    let mut board = Board6::new();
    let mut info = SearchInfo::new(6, 1000000);
    loop {
        let message = receiver.recv()?;
        match message {
            TeiCommand::Go(s) => {
                let low_flats = std::cmp::min(
                    board.pieces_reserve(Color::White),
                    board.pieces_reserve(Color::Black),
                );
                let est_plies = low_flats * 2;
                let time_left = TimeLeft::new(&s);
                let use_time = time_left.use_time(est_plies, board.side_to_move());
                info = SearchInfo::new(6, 0)
                    .take_table(&mut info)
                    .max_time(use_time);
                let res = search(&mut board, &mut info);
                if let Some(outcome) = res {
                    println!("info {}", outcome);
                    println!(
                        "bestmove {}",
                        outcome
                            .best_move()
                            .expect("Could not find best move!")
                            .trim_end_matches('*')
                    );
                } else {
                    println!("Something went wrong, search failed!");
                }
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
