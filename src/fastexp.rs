// Copyright 2014-2016 Johannes Köster.
// Licensed under the MIT license (http://opensource.org/licenses/MIT)
// This file may not be copied, modified, or distributed
// except according to those terms.
//
// From src/utils/fastexp.rs in the `bio` crate

//! This module provides a trait adding a fast approximation of the exponential function to f32.
//! This can be very useful if the exact value is not too important.

const COEFF_0: f32 = 1.0;
const COEFF_1: f32 = 4.831794110;
const COEFF_2: f32 = 0.143440676;
const COEFF_3: f32 = 0.019890581;
const COEFF_4: f32 = 0.006935931;
const ONEBYLOG2: f32 = 1.442695041;
const OFFSET_F64: i64 = 1023;
const FRACTION_F64: u32 = 52;
const MIN_VAL: f32 = -500.0;

/// This trait adds a fast approximation of exp to float types.
pub trait FastExp<V> {
    fn fastexp(&self) -> V;
}

impl FastExp<f32> for f32 {
    /// Fast approximation of exp() as shown by Kopcynski 2017:
    /// https://eldorado.tu-dortmund.de/bitstream/2003/36203/1/Dissertation_Kopczynski.pdf
    fn fastexp(&self) -> f32 {
        if *self > MIN_VAL {
            let mut x = ONEBYLOG2 * self;

            #[repr(C)]
            union F1 {
                i: i64,
                f: f32,
            }
            let mut f1 = F1 { i: x as i64 };

            x -= unsafe { f1.i } as f32;
            let mut f2 = x;
            let mut x_tmp = x;

            unsafe {
                f1.i += OFFSET_F64;
                f1.i <<= FRACTION_F64;
            }

            f2 *= COEFF_4;
            x_tmp += COEFF_1;
            f2 += COEFF_3;
            x_tmp *= x;
            f2 *= x;
            f2 += COEFF_2;
            f2 *= x_tmp;
            f2 += COEFF_0;

            unsafe { f1.f * f2 }
        } else {
            0.0
        }
    }
}
