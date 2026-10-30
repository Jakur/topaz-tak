const MAX_DEPTH: usize = 64; // 64

pub const INT_CONVERSION: i32 = 1024;
pub const APPROX_LN_FLOAT: [f32; MAX_DEPTH] = generate_lookup_table();
pub const APPROX_LN_INT: [i32; MAX_DEPTH] = generate_int_table();
pub const APPROX_LOG2_INT: [i32; MAX_DEPTH] = generate_int_table_log2();
const LOG_CONVERSION: f32 = 1.4427;

const fn generate_lookup_table() -> [f32; MAX_DEPTH] {
    let mut table = [0.0; MAX_DEPTH];
    let mut i = 1;
    while i < MAX_DEPTH {
        table[i] = approx_ln(i as f32);
        i += 1;
    }
    table
}

const fn generate_int_table() -> [i32; MAX_DEPTH] {
    let mut table = [0; MAX_DEPTH];
    let mut i = 0;
    while i < MAX_DEPTH {
        table[i] = (APPROX_LN_FLOAT[i] * INT_CONVERSION as f32) as i32;
        i += 1;
    }
    table
}

const fn generate_int_table_log2() -> [i32; MAX_DEPTH] {
    let mut table = [0; MAX_DEPTH];
    let mut i = 0;
    while i < MAX_DEPTH {
        table[i] = (APPROX_LN_FLOAT[i] * INT_CONVERSION as f32 * LOG_CONVERSION) as i32;
        i += 1;
    }
    table
}

const fn approx_ln(mut x: f32) -> f32 {
    // https://quadst.rip/ln-approx
    let mut bx = x.to_bits();

    let ex = bx >> 23;
    let t = ex as i32 - 127;

    bx = 1065353216 | (bx & 8388607);
    x = f32::from_bits(bx);

    -1.49278 + (2.11263 + (-0.729104 + 0.10969 * x) * x) * x + 0.6931471806 * t as f32
}
