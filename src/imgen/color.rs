use cv::core::Scalar;
use libc::c_int;

#[derive(Debug)]
pub enum ChannelCombination {
    R,
    G,
    B,
    RG,
    RB,
    GB,
    RGB,
}

impl ChannelCombination {
    pub fn from_c_int(value: c_int) -> ChannelCombination {
        match value {
             0 => ChannelCombination::R,
             1 => ChannelCombination::G,
             2 => ChannelCombination::B,
             3 => ChannelCombination::RG,
             4 => ChannelCombination::RB,
             5 => ChannelCombination::GB,
             6 => ChannelCombination::RGB,
             _ => panic!("Value out of range"),
        }
    }

    pub fn to_scalar_with_intensity(&self, intensity: c_int) -> Scalar {
        match *self {
            ChannelCombination::R   => {
                Scalar::new(0, 0, intensity, 255)
            },
            ChannelCombination::G   => { 
                Scalar::new(0, intensity, 0, 255)
            },
            ChannelCombination::B   => {
                Scalar::new(intensity, 0, 0, 255)
            },
            ChannelCombination::RG  => {
                 Scalar::new(0, intensity, intensity, 255)
            },
            ChannelCombination::RB  => {
                 Scalar::new(intensity, 0, intensity, 255)
            },
            ChannelCombination::GB  => {
                 Scalar::new(intensity, intensity, 0, 255)
            },
            ChannelCombination::RGB => {
                 Scalar::new(intensity, intensity, intensity, 255)
            },
        }
    }
}

#[allow(dead_code)]
pub struct SampleSpace {
    pub colors: Vec<Scalar>,
}

impl SampleSpace {
    pub fn new(samples: c_int) -> SampleSpace {
        if samples < 7 {
            panic!("Too few color samples requested");
        }

        let size = samples + (7 - (samples % 7)) + 1; /* Round up to multiple of 7 
                                                         + 1 for black */

        let mut space = SampleSpace{ colors: Vec::with_capacity(size as usize) };
        space.colors.push(Scalar::new(0,0,0,255));
        let per_color = size / 7;
        let intensity_step = 255 / per_color; 
        let mut intensity = intensity_step;
        let mut channels = -1;

        for i in 0..size-1 {
            if i % per_color == 0 { /* Next channel(s) */
                channels += 1;
                intensity = intensity_step;
            }
            else {
                intensity += intensity_step;
            }
            space.colors.push(ChannelCombination::from_c_int(channels)
                              .to_scalar_with_intensity(intensity));
        }

        space
    }
}

impl IntoIterator for SampleSpace {
    type Item = Scalar;
    type IntoIter = ::std::vec::IntoIter<Scalar>;

    fn into_iter(self) -> Self::IntoIter {
        self.colors.into_iter()
    }
}

