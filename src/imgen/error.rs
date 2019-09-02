use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Error {
    what: String,
}

impl Error {
    pub fn new(msg: &str) -> Error {
        Error{what: msg.to_string()}
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.what)
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        &self.what
    }

    fn cause(&self) -> Option<&dyn error::Error> {
        None /* Generic error */
    }
}
