extern crate regex;

use crate::imgen::cmp::Weights;
use crate::imgen::core::{Filter, ExecutionPolicy, PostProc};
use crate::imgen::error::Error;
use crate::imgen::math::Size2u;

use regex::Regex;

use std::fs::File;
use std::error::Error as SError;
use std::io::prelude::*;
use std::path::Path;

#[derive(Debug)]
enum Tag {
    NoOpt,
    IO,
    Params,
    Visuals,
}

impl Tag {
    fn from_str(string: &str) -> Result<Tag, Error> {
        match string.as_ref() {
            "io" => Ok(Tag::IO),
            "params" => Ok(Tag::Params),
            "visuals" => Ok(Tag::Visuals),
            _ => Err(Error::new("Unknown tag")),
        }
    }
}

#[derive(Debug)]
pub enum Show {
    None,
    Res,
    Orig,
    Both,
}

impl Show {
    fn from_str(string: &str) -> Result<Show, Error> {
        match string.as_ref() {
            "none" => Ok(Show::None),
            "result" => Ok(Show::Res),
            "original" => Ok(Show::Orig),
            "both" => Ok(Show::Both),
            _ => Err(Error::new(&format!("{} is not a valid Show enumerator", string))),
        }
    }
}


#[derive(Debug)]
pub struct ParseResult {
    pub input: String,
    pub output: String,
    pub subsize: Size2u,
    pub ncolors: u32,
    pub circsize: Size2u,
    pub weights: Weights,
    pub filter: Filter,
    pub exec: ExecutionPolicy,
    pub postproc: PostProc,
    pub show: Show,
}

impl ParseResult {
    fn default() -> ParseResult {
        ParseResult {
            input: "".to_string(),
            output: "".to_string(),
            subsize: Size2u::new(0,0),
            ncolors: 0,
            circsize: Size2u::new(0,0),
            weights: Weights::new(1.0, 0.0).unwrap(),
            filter: Filter::None,
            exec: ExecutionPolicy::Sequential,
            postproc: PostProc::None,
            show: Show::None,
        }
    }

}

#[allow(dead_code)]
pub fn parse(config: &str) -> Result<ParseResult, Error> {
    let contents = file_contents(config)?
                    .to_lowercase();
    let mut tag = Tag::NoOpt;

    let tag_re = Regex::new(r"^\[(?P<tag>\w+)\]").unwrap();
    let cont_re = Regex::new(r"(?P<content>[\w\d., ]+)").unwrap();
    let path_re = Regex::new(r"(?P<path>[/\w\d.]+)").unwrap();
    let comment_re = Regex::new(r"(?P<content>.*)#").unwrap();

    let mut result = ParseResult::default();

    for line in contents.lines() {
        /* Strip comments */
        //let caps = comment_re.captures(line).unwrap();
        //let line = &caps["content"];
        let caps = comment_re.captures(line);

        let sline = match caps {
            Some(cap) => cap["content"].to_string(),
            None => line.to_string(),
        };

        let line = &sline;

        if tag_re.is_match(line) {
            let caps = tag_re.captures(line).unwrap();

            tag = Tag::from_str(&caps["tag"])?;
        }
        else if !line.trim().is_empty() {
            let comps = line.split("=").collect::<Vec<&str>>();

            if comps.len() == 1 {
                return Err(Error::new(&format!("Malformed line: {}", line)));
            }

            let val = comps[1].trim();

            let val = match tag {
                Tag::IO => {
                    let caps = path_re.captures(val).unwrap();
                    caps["path"].to_string()
                },
                Tag::Params => {
                    let caps = cont_re.captures(val).unwrap();
                    caps["content"].to_string()
                },
                Tag::Visuals => {
                    let caps = cont_re.captures(val).unwrap();
                    caps["content"].to_string()
                },
                _ => return Err(Error::new("Unknown tag")),
            };

            match comps[0].trim() {
                "input" => result.input = val,
                "output" => result.output = val,
                "subsize" => result.subsize = parse_size2u(&val)?,
                "ncolors" => {
                    result.ncolors = match val.parse::<u32>() {
                        Ok(num) => num,
                        Err(_) => return Err(Error::new(&format!("{} is not a u32", val))),
                    };
                },
                "circsize" => result.circsize = parse_size2u(&val)?,
                "weights" => result.weights = parse_weights(&val)?,
                "filter" => result.filter = parse_filter(&val)?,
                "exec" => result.exec = parse_exec_policy(&val)?,
                "postproc" => result.postproc = parse_postproc(&val)?,
                "show" => result.show = Show::from_str(&val)?,
                _ => return Err(Error::new(&format!("Unknown key: {}", comps[0]))),
            };
        }

    }

    Ok(result)
}

fn parse_size2u(s2u: &str) -> Result<Size2u, Error> {
    let comps = s2u.split("x").collect::<Vec<&str>>();

    if comps.len() != 2 {
        return Err(Error::new(&format!("Unknown size specification: {}", s2u)));
    }
    let first = comps[0].trim();
    let second = comps[1].trim();

    let first = match first.parse::<u32>() {
        Ok(num) => num,
        Err(_) => return Err(Error::new(&format!("{} is not an unsigned integral type", first))),
    };

    let second = match second.parse::<u32>() {
        Ok(num) => num,
        Err(_) => return Err(Error::new(&format!("{} is not an unsigned integral type", second))),
    };

    Ok(Size2u::new(first, second))
}

fn parse_weights(weights: &str) -> Result<Weights, Error> {
    let comps = weights.split(",").collect::<Vec<&str>>();

    if comps.len() != 2 {
        return Err(Error::new(&format!("Unknown size specification: {}", weights)));
    }

    let first = comps[0].trim();
    let second = comps[1].trim();

    let first = match first.parse::<f32>() {
        Ok(num) => num,
        Err(_) => return Err(Error::new(&format!("{} is not an unsigned integral type", first))),
    };

    let second = match second.parse::<f32>() {
        Ok(num) => num,
        Err(_) => return Err(Error::new(&format!("{} is not an unsigned integral type", second))),
    };

    Ok(Weights::new(first, second)?)
}

fn parse_filter(filter: &str) -> Result<Filter, Error> {
    match filter {
        "none" => Ok(Filter::None),
        "sdev" => Ok(Filter::Sdev),
        _ => Err(Error::new(&format!("{} is not a filter", filter))),
    }
}

fn parse_exec_policy(policy: &str) -> Result<ExecutionPolicy, Error> {
    match policy {
        "sequential" => Ok(ExecutionPolicy::Sequential),
        "parallelx4" => Ok(ExecutionPolicy::Parallelx4),
        "parallelx8" => Ok(ExecutionPolicy::Parallelx8),
        _ => Err(Error::new(&format!("{} is not an exection policy", policy))),
    }
}

fn parse_postproc(proc: &str) -> Result<PostProc, Error> {
    match proc {
        "none" => Ok(PostProc::None),
        "origsize" => Ok(PostProc::OrigSize),
        _ => Err(Error::new(&format!("{} is not an enumerator of type PostProc", proc))),
    }
}

fn file_contents(config: &str) -> Result<String, Error> {
    let path = Path::new(config);
    let display = path.display();

    let mut file = match File::open(config) {
        Err(why) => {
            return Err(Error::new(&format!("Could not open {} due to: {}", display, 
                                                                           why.description())));
        },
        Ok(file) => file,
    };

    let mut contents = String::new();

    match file.read_to_string(&mut contents) {
        Err(why) => {
            return Err(Error::new(&format!("Could not read {} due to: {}", display,
                                                                           why.description())));
        },
        Ok(_) => (),
    };

    Ok(contents)
}
