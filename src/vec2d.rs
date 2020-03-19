use std::iter;
use std::ops::{Index, IndexMut};

/// A 2D vector that can grow along one dimension.
pub struct Vec2D<T> {
    vec: Vec<T>,
    inner_size: usize,
}

impl<T> Vec2D<T> {
    pub fn new(inner_size: usize) -> Self {
        Self {
            vec: Vec::new(),
            inner_size,
        }
    }
}

impl<T> Vec2D<T>
where
    T: Clone,
{
    pub fn add_row_with_value(&mut self, value: T) {
        self.vec.reserve(self.inner_size);
        self.vec.extend(iter::repeat(value).take(self.inner_size))
    }
}

impl<T> Index<[usize; 2]> for Vec2D<T> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.vec[index[0] * self.inner_size + index[1]]
    }
}

impl<T> IndexMut<[usize; 2]> for Vec2D<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.vec[index[0] * self.inner_size + index[1]]
    }
}
