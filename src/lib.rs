#![feature(static_nobundle)]

#[macro_use(s)]

extern crate ndarray;

use numpy::{PyArray2};
use ndarray::{Ix2, ArrayBase, Data};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn beam_search(result: &PyArray2<f32>, alphabet: String, beam_size: usize, beam_cut_threshold: f32) -> String {
    beam_search_(&result.as_array(), alphabet, beam_size, beam_cut_threshold)
}

#[pymodule]
fn fast_ctc_decode(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(beam_search))?;
    Ok(())
}

fn beam_search_<D: Data<Elem=f32>>(result: &ArrayBase<D, Ix2>, alphabet: String, beam_size: usize, beam_cut_threshold: f32) -> String {

    let alphabet: Vec<char> = alphabet.chars().collect();
    // alphabet_size minus the blank label
    let alphabet_size = alphabet.len() - 1;

    // (base, what)
    let mut beam_prevs = vec![(0, 0)];
    let mut beam_forward: Vec<Vec<i32>> = vec![vec![-1; alphabet_size]];

    let mut cur_probs = vec![(0i32, 0.0, 1.0)];
    let mut new_probs = Vec::new();

    for pr in result.slice(s![..;-1, ..]).outer_iter() {
        new_probs.clear();

        for &(beam, base_prob, n_prob) in &cur_probs {
            // add N to beam
            if pr[0] > beam_cut_threshold {
                new_probs.push((beam, 0.0, (n_prob + base_prob) * pr[0]));
            }

            for b in 1..alphabet.len() {
                if pr[b] < beam_cut_threshold {
                    continue
                }
                if b == beam_prevs[beam as usize].0 {
                    new_probs.push((beam, base_prob * pr[b], 0.0));
                    let mut new_beam = beam_forward[beam as usize][b-1];
                    if new_beam == -1 {
                        new_beam = beam_prevs.len() as i32;
                        beam_prevs.push((b, beam));
                        beam_forward[beam as usize][b-1] = new_beam;
                        beam_forward.push(vec![-1; alphabet_size]);
                    }

                    new_probs.push((new_beam, n_prob * pr[b], 0.0));

                } else {
                    let mut new_beam = beam_forward[beam as usize][b-1];
                    if new_beam == -1 {
                        new_beam = beam_prevs.len() as i32;
                        beam_prevs.push((b, beam));
                        beam_forward[beam as usize][b-1] = new_beam;
                        beam_forward.push(vec![-1; alphabet_size]);
                    }

                    new_probs.push((new_beam, (base_prob + n_prob) * pr[b], 0.0));
                }
            }
        }
        std::mem::swap(&mut cur_probs, &mut new_probs);

        cur_probs.sort_by_key(|x| x.0);
        let mut last_key: i32 = -1;
        let mut last_key_pos = 0;
        for i in 0..cur_probs.len() {
            if cur_probs[i].0 == last_key {
                cur_probs[last_key_pos].1 = cur_probs[last_key_pos].1 + cur_probs[i].1;
                cur_probs[last_key_pos].2 = cur_probs[last_key_pos].2 + cur_probs[i].2;
                cur_probs[i].0 = -1;
            } else {
                last_key_pos = i;
                last_key = cur_probs[i].0;
            }
        }

        cur_probs.retain(|x| x.0 != -1);
        cur_probs.sort_by(|a, b| (b.1 + b.2).partial_cmp(&(a.1 + a.2)).unwrap());
        cur_probs.truncate(beam_size);
        let top = cur_probs[0].1 + cur_probs[0].2;
        for mut x in &mut cur_probs {
            x.1 /= top;
            x.2 /= top;
        }
    }

    let mut out = String::new();
    let mut beam = cur_probs[0].0;
    while beam != 0 {
        out.push(alphabet[beam_prevs[beam as usize].0]);
        beam = beam_prevs[beam as usize].1;
    }
    out
}
