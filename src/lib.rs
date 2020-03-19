#![feature(static_nobundle)]

#[macro_use(s)]
extern crate ndarray;

use ndarray::{ArrayBase, Data, Ix2};
use numpy::PyArray2;

use pyo3::exceptions::ValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod vec2d;
use crate::vec2d::Vec2D;

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
/// # Arguments
///
/// * `network_output` - The 2D array output of the neural network. Must be the output of a softmax
///                      layer, with values between 0.0 and 1.0 representing probabilities. The
///                      first (outer) axis is time, and the second (inner) axis is label. The
///                      first entry on the label axis is the blank label.
/// * `alphabet` - The labels (excluding the blank label) in the order given on the label axis of
///                `network_output`.
/// * `beam_size` - How many suffix_tree should be kept at each step. Higher numbers are less likely
///                 to discard the true labelling, but also make it slower and more memory
///                 intensive.
/// * `beam_cut_threshold` - Ignore any entries in `network_output` below this value.
#[pyfunction]
fn beam_search(
    network_output: &PyArray2<f32>,
    alphabet: String,
    beam_size: usize,
    beam_cut_threshold: f32,
) -> PyResult<(String, Vec<usize>)> {
    if alphabet.len() != network_output.shape()[1] {
        Err(ValueError::py_err(
            "alphabet size does not match probability matrix dimensions",
        ))
    } else if beam_size == 0 {
        Err(ValueError::py_err("beam_size cannot be 0"))
    } else {
        Ok(beam_search_(
            &network_output.as_array(),
            alphabet,
            beam_size,
            beam_cut_threshold,
        ))
    }
}

#[pymodule]
fn fast_ctc_decode(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(beam_search))?;
    Ok(())
}

/// An element in a possible labelling.
#[derive(Clone, Copy, Debug)]
struct LabelNode {
    /// The index into the alphabet of this label.
    ///
    /// Note that blanks are not represented by a LabelNode - this is an actual label.
    label: usize,
    /// The index of the next LabelNode.
    ///
    /// Can also be considered the parent edge in the tree of labelling suffixes.
    next: i32,
    /// The last(?) sample to which the label applies.
    time: usize,
}

/// A node in the labelling tree to build from.
#[derive(Clone, Copy, Debug)]
struct SearchPoint {
    /// The node search should progress from.
    node: i32,
    /// The cumulative probability of the labelling so far for paths without any leading blank
    /// labels.
    label_prob: f32,
    /// The cumulative probability of the labelling so far for paths with one or more leading
    /// blank labels.
    gap_prob: f32,
}

impl SearchPoint {
    /// The total probability of the labelling so far.
    ///
    /// This sums the probabilities of the paths with and without leading blank labels.
    fn probability(&self) -> f32 {
        self.label_prob + self.gap_prob
    }
}

fn beam_search_<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: String,
    beam_size: usize,
    beam_cut_threshold: f32,
) -> (String, Vec<usize>) {
    let alphabet: Vec<char> = alphabet.chars().collect();

    // alphabet_size minus the blank label
    let alphabet_size = alphabet.len() - 1;
    let duration = network_output.nrows();

    // suffix_tree and suffix_children, between them, describe a tree of labelling suffixes
    // (partial labellings pinned to the end of the network output). Zipping the two vectors
    // together gives all the nodes in the tree, with the first entry being the root node and the
    // next field of LabelNode giving the index of each node's parent.
    let mut suffix_tree = vec![LabelNode {
        label: 0,
        next: 0,
        time: 0,
    }];
    // suffix_children is a 2D array.
    //
    // Invariants:
    //
    // suffix_tree.len() == suffix_children.len()
    //
    // For all n, l where suffix_children[n][l] != -1:
    //     suffix_tree[suffix_children[n][l]].label == l (child edge label matches child label)
    //     suffix_tree[suffix_children[n][l]].next == n  (child's parent pointer is correct)
    //
    // For all n > 0:
    //     suffix_children[suffix_tree[n].next][suffix_tree[n].label] == n
    //     (the parent node has a child edge back to this node labelled correctly)
    let mut suffix_children: Vec2D<i32> = Vec2D::new(alphabet_size);
    suffix_children.add_row_with_value(-1);
    let mut beam = vec![SearchPoint {
        node: 0,
        label_prob: 0.0,
        gap_prob: 1.0,
    }];
    let mut next_beam = Vec::new();

    for (idx, pr) in network_output.slice(s![..;-1, ..]).outer_iter().enumerate() {
        next_beam.clear();

        // forward index in time
        let fidx = duration - idx - 1;

        for &SearchPoint {
            node,
            label_prob,
            gap_prob,
        } in &beam
        {
            // add N to beam
            if pr[0] > beam_cut_threshold {
                next_beam.push(SearchPoint {
                    node,
                    label_prob: 0.0,
                    gap_prob: (label_prob + gap_prob) * pr[0],
                });
            }

            for (label, &pr_b) in (1..=alphabet_size).zip(pr.iter().skip(1)) {
                if pr_b < beam_cut_threshold {
                    continue;
                }
                if label == suffix_tree[node as usize].label {
                    next_beam.push(SearchPoint {
                        node,
                        label_prob: label_prob * pr_b,
                        gap_prob: 0.0,
                    });
                    let mut new_node_idx = suffix_children[[node as usize, label - 1]];
                    if new_node_idx == -1 && gap_prob > 0.0 {
                        new_node_idx = suffix_tree.len() as i32;
                        suffix_tree.push(LabelNode {
                            label: label,
                            next: node,
                            time: fidx,
                        });
                        suffix_children[[node as usize, label - 1]] = new_node_idx;
                        suffix_children.add_row_with_value(-1);
                    }

                    next_beam.push(SearchPoint {
                        node: new_node_idx,
                        label_prob: gap_prob * pr_b,
                        gap_prob: 0.0,
                    });
                } else {
                    let mut new_node_idx = suffix_children[[node as usize, label - 1]];
                    if new_node_idx == -1 {
                        new_node_idx = suffix_tree.len() as i32;
                        suffix_tree.push(LabelNode {
                            label: label,
                            next: node,
                            time: fidx,
                        });
                        suffix_children[[node as usize, label - 1]] = new_node_idx;
                        suffix_children.add_row_with_value(-1);
                    }

                    next_beam.push(SearchPoint {
                        node: new_node_idx,
                        label_prob: (label_prob + gap_prob) * pr_b,
                        gap_prob: 0.0,
                    });
                }
            }
        }
        std::mem::swap(&mut beam, &mut next_beam);

        beam.sort_by_key(|x| x.node);
        let mut last_key: i32 = -1;
        let mut last_key_pos = 0;
        for i in 0..beam.len() {
            let beam_item = beam[i];
            if beam_item.node == last_key {
                beam[last_key_pos].label_prob += beam_item.label_prob;
                beam[last_key_pos].gap_prob += beam_item.gap_prob;
                beam[i].node = -1;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }

        beam.retain(|x| x.node != -1);
        beam.sort_by(|a, b| (b.probability()).partial_cmp(&(a.probability())).unwrap());
        beam.truncate(beam_size);
        if beam.is_empty() {
            // we've run out of beam (probably the threshold is too high)
            return (String::new(), Vec::new());
        }
        let top = beam[0].probability();
        for mut x in &mut beam {
            x.label_prob /= top;
            x.gap_prob /= top;
        }
    }

    let mut node_idx = beam[0].node;
    let mut path = Vec::new();
    let mut sequence = String::new();

    while node_idx != 0 {
        let node = &suffix_tree[node_idx as usize];
        path.push(node.time);
        sequence.push(alphabet[node.label]);
        node_idx = node.next;
    }

    (sequence, path)
}
