#![feature(static_nobundle)]

#[macro_use(s)]
extern crate ndarray;

use numpy::PyArray2;

use pyo3::exceptions::{RuntimeError, ValueError};
use pyo3::prelude::*;
use pyo3::types::PySequence;
use pyo3::wrap_pyfunction;
use std::fmt;

mod search;
mod vec2d;

#[derive(Clone, Copy, Debug)]
pub enum SearchError {
    RanOutOfBeam,
    IncomparableValues,
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::RanOutOfBeam => {
                write!(f, "Ran out of search space (beam_cut_threshold too high)")
            }
            SearchError::IncomparableValues => {
                write!(f, "Failed to compare values (NaNs in input?)")
            }
        }
    }
}

impl std::error::Error for SearchError {}

/// Perform a CTC beam search decode on an RNN output.
///
/// The ultimate aim here is to find a way of labelling some input data. In the case of nanopore
/// sequencing data, we're taking the electrical current samples, and labelling them with what we
/// think the DNA/RNA base is at any given time.
///
/// CTC decoding (and hence this function) takes as input the result of a neural network that has
/// figured out, for each sample and each label, the probability that the sample corresponds to
/// that label. It also outputs a probability for the data point corresponding to an extra "blank"
/// label (a sort of "none of the above" option). This is represented as a 2D matrix of size ``N x
/// (L+1)``, where ``N`` is the number of samples and ``L`` is the number of labels we're
/// interested in (the ``+1`` is to account for the blank label).
///
/// What this function does is a beam search variant of the prefix search decoding mentioned (and
/// described in fairly vague terms) in the original CTC paper (Graves et al, 2006, section 3.2).
///
/// A quick overview: a _path_ through the matrix is an assignment of a label or blank to each
/// sample. The probability that the path is correct is the product of the selected entries in the
/// matrix. Each path produces a labelling: first collapse all duplicate labels or blanks, then
/// remove the remaining blanks - AAAGGbGGbbbC would become AGbGbC, and then AGGC. The probability
/// that the labelling is correct is the sum of the probabilities of the paths that produce it. We
/// want the most likely labelling.
///
/// The paper mentioned above provides recursive equations that give an efficient way to find the
/// probability for a specific labelling. The possible suffix_tree, together with their
/// probabilities, can be built up by starting at one end and trying every possible label at each
/// stage. The "beam" part of the search is how we keep the search space managable - at each step,
/// we discard all but the most-probable suffix_tree (like searching with a torch beam). This means
/// we may not actually find the most likely labelling, but it often works very well.
///
/// Args:
///     network_output (numpy.ndarray): The 2D array output of the neural network. Must be the
///         output of a softmax layer, with values between 0.0 and 1.0 representing probabilities.
///         The first (outer) axis is time, and the second (inner) axis is label. The first entry
///         on the label axis is the blank label.
///     alphabet (sequence): The labels (including the blank label) in the order given on the label
///          axis of `network_output`. Length must match the size of the inner axis of `network_output`.
///     beam_size (int): How many suffix_tree should be kept at each step. Higher numbers are less
///         likely to discard the true labelling, but also make it slower and more memory
///         intensive. Must be at least 1.
///     beam_cut_threshold (float): Ignore any entries in `network_output` below this value. Must
///         be at least 0.0, and less than ``1/len(alphabet)``.
///
/// Returns:
///     tuple of (str, numpy.ndarray): The decoded sequence and an array of the final
///         timepoints of each label (as indices into the outer axis of `network_output`).
///
/// Raises:
///     ValueError: The constraints on the arguments have not been met.
#[pyfunction(beam_size = "5", beam_cut_threshold = "0.0")]
#[text_signature = "(network_output, alphabet, beam_size=5, beam_cut_threshold=0.0)"]
fn beam_search(
    network_output: &PyArray2<f32>,
    alphabet: &PySequence,
    beam_size: usize,
    beam_cut_threshold: f32,
) -> PyResult<(String, Vec<usize>)> {
    let alphabet: Vec<String> = alphabet.tuple()?.iter().map(|x| x.to_string()).collect();
    let max_beam_cut = 1.0 / (alphabet.len() as f32);
    if alphabet.len() != network_output.shape()[1] {
        Err(ValueError::py_err(format!(
            "alphabet size {} does not match probability matrix dimensions {}",
            alphabet.len(),
            network_output.shape()[1]
        )))
    } else if beam_size == 0 {
        Err(ValueError::py_err("beam_size cannot be 0"))
    } else if beam_cut_threshold < -0.0 {
        Err(ValueError::py_err(
            "beam_cut_threshold must be at least 0.0",
        ))
    } else if beam_cut_threshold >= max_beam_cut {
        Err(ValueError::py_err(format!(
            "beam_cut_threshold cannot be more than {}",
            max_beam_cut
        )))
    } else {
        search::beam_search(
            &network_output.as_array(),
            &alphabet,
            beam_size,
            beam_cut_threshold,
        )
        .map_err(|e| RuntimeError::py_err(format!("{}", e)))
    }
}

/// Methods for labelling RNN results using CTC decoding.
#[pymodule]
fn fast_ctc_decode(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(beam_search))?;
    Ok(())
}
