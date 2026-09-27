pub trait Argmax {
    type Output;
    fn argmax(self) -> Option<(usize, Self::Output)>;
}

impl<I, T> Argmax for I
where
    I: Iterator<Item = T>,
    T: std::cmp::PartialOrd,
{
    type Output = T;

    fn argmax(self) -> Option<(usize, T)> {
        let mut enumerated = self.enumerate();
        enumerated.next().map(move |first| {
            enumerated.fold(first, |(i_max, val_max), (i, val)| {
                if val >= val_max {
                    (i, val)
                } else {
                    (i_max, val_max)
                }
            })
        })
    }
}
