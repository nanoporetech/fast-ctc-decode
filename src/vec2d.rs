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

impl<T> Index<(usize, usize)> for Vec2D<T> {
    type Output = T;

    fn index(&self, (outer, inner): (usize, usize)) -> &Self::Output {
        &self.vec[outer * self.inner_size + inner]
    }
}

impl<T> IndexMut<(usize, usize)> for Vec2D<T> {
    fn index_mut(&mut self, (outer, inner): (usize, usize)) -> &mut Self::Output {
        &mut self.vec[outer * self.inner_size + inner]
    }
}
