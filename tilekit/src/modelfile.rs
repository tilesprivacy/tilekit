// A parser for ollama modelfile
// https://github.com/ollama/ollama/blob/main/docs/modelfile.md

// Modelfile grammar
// command -> Instruction arguments*
// Instruction -> "FROM" | "PARAMETER" | "TEMPLATE"...
// arguments -> WORD | quoted_string | multiline_string
// quoted_string -> "<str>"
// multiline_string -> """<str>"""

use std::{fmt::Display, fs, str::FromStr};

use nom::{
    AsChar, IResult, Parser,
    branch::alt,
    bytes::complete::{tag_no_case, take_until1, take_while1},
    character::complete::multispace0,
    combinator::map,
    multi::separated_list1,
    sequence::{delimited, pair},
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ParamValue {
    Int(i32),
    Float(f32),
    Str(String),
}

impl Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::Int(value) => write!(f, "{}", value),
            ParamValue::Str(value) => write!(f, "{}", value),
            ParamValue::Float(value) => write!(f, "{}", value),
        }
    }
}

#[derive(Debug, Clone)]
enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug)]
enum Output<'a> {
    Single(&'a str),
    Pair((&'a str, &'a str)),
}
impl<'a> Display for Output<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(word) => write!(f, "{}", word),
            Self::Pair((word, word_1)) => write!(f, "{} {}", word, word_1),
        }
    }
}

impl FromStr for Role {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            _ => Err("Invalid Role".to_owned()),
        }
    }
}
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Parameter {
    pub param_type: String,
    pub value: ParamValue,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Message {
    role: Role,
    message: String,
}

impl Parameter {
    fn new(param_type: String, value: ParamValue) -> Self {
        Self { param_type, value }
    }
}

#[derive(Debug, Clone)]
pub struct Modelfile {
    pub from: Option<String>,
    pub parameters: Vec<Parameter>,
    pub template: Option<String>,
    pub adapter: Option<String>,
    pub system: Option<String>,
    pub license: Option<String>,
    pub messages: Vec<Message>,
    pub data: Vec<String>,
    pub errors: Vec<String>,
}

impl Modelfile {
    pub fn new() -> Self {
        Self {
            from: None,
            data: vec![],
            parameters: vec![],
            template: None,
            messages: vec![],
            license: None,
            adapter: None,
            system: None,
            errors: vec![],
        }
    }

    pub fn add_from(&mut self, value: &str) -> Result<(), String> {
        if self.from.is_some() {
            let error = "Modelfile can only have one FROM instruction".to_owned();
            self.errors.push(error.clone());
            Err(error)
        } else {
            self.from = Some(value.to_owned());
            self.data.push(format!("FROM {}", value));
            Ok(())
        }
    }

    pub fn add_template(&mut self, value: &str) -> Result<(), String> {
        if self.template.is_some() {
            let error = "Modelfile can only have one TEMPLATE instruction".to_owned();
            self.errors.push(error.clone());
            Err(error)
        } else {
            self.template = Some(value.to_owned());
            self.data.push(format!("TEMPLATE {}", value));
            Ok(())
        }
    }

    pub fn add_license(&mut self, value: &str) -> Result<(), String> {
        if self.license.is_some() {
            let error = "Modelfile can only have one LICENSE instruction".to_owned();
            self.errors.push(error.clone());
            Err(error)
        } else {
            self.license = Some(value.to_owned());
            self.data.push(format!("LICENSE {}", value));
            Ok(())
        }
    }

    pub fn add_adapter(&mut self, value: &str) -> Result<(), String> {
        if self.adapter.is_some() {
            let error = "Modelfile can only have one ADAPTER instruction".to_owned();
            self.errors.push(error.clone());
            Err(error)
        } else {
            self.adapter = Some(value.to_owned());
            self.data.push(format!("ADAPTER {}", value));
            Ok(())
        }
    }

    pub fn add_system(&mut self, value: &str) -> Result<(), String> {
        self.data.push(format!("SYSTEM \"{}\"", value));
        if self.system.is_some() {
            let error = "Modelfile can only have one SYSTEM instruction".to_owned();
            self.errors.push(error.clone());

            Err(error)
        } else {
            self.system = Some(value.to_owned());
            Ok(())
        }
    }

    pub fn update_system(&mut self, value: &str) {
        self.system = Some(value.to_owned());
        // Find and replace or add to data
        let mut found = false;
        for line in self.data.iter_mut() {
            if line.to_uppercase().starts_with("SYSTEM") {
                *line = format!("SYSTEM \"{}\"", value);
                found = true;
                break;
            }
        }
        if !found {
            self.data.push(format!("SYSTEM \"{}\"", value));
        }
    }

    pub fn add_comment(&mut self, value: &str) -> Result<(), String> {
        self.data.push(format!("# {}", value));
        Ok(())
    }

    pub fn add_parameter(&mut self, param_type: &str, param_value: &str) -> Result<(), String> {
        match parse_parameter(param_type, param_value) {
            Ok(parameter) => {
                self.parameters.push(parameter);
                self.data
                    .push(format!("PARAMETER {} {}", param_type, param_value));
                Ok(())
            }
            Err(err) => {
                self.errors.push(err.clone());
                Err(err)
            }
        }
    }

    pub fn add_message(&mut self, role: &str, message: &str) -> Result<(), String> {
        match parse_message(role, message) {
            Ok(msg) => {
                self.messages.push(msg);
                self.data.push(format!("MESSAGE {} {}", role, message));
                Ok(())
            }
            Err(err) => {
                self.errors.push(err.clone());
                Err(err)
            }
        }
    }

    pub fn build(&mut self) -> Result<(), String> {
        if self.from.is_none() {
            let error = String::from("Modelfile should need a FROM instruction");
            self.errors.push(error.clone());
            Err(error)
        } else {
            Ok(())
        }
    }
}

impl Default for Modelfile {
    fn default() -> Self {
        Modelfile::new()
    }
}

impl FromStr for Modelfile {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse(s)
    }
}

impl Display for Modelfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data.join("\n"))
    }
}

pub fn parse_from_file(path: &str) -> Result<Modelfile, String> {
    match fs::read_to_string(path) {
        Ok(content) => parse(content.as_str()),
        Err(err) => Err(format!("Parsing Modelfile failed due to {}", err)),
    }
}

pub fn parse(input: &str) -> Result<Modelfile, String> {
    match parse_file(input) {
        Ok((_rest, parsed_data)) => create_modelfile(parsed_data.clone()),
        Err(err) => Err(format!("Modelfile failed to parse due to {:?}", err)),
    }
}

fn parse_file(input: &str) -> IResult<&str, Vec<(&str, Output<'_>)>> {
    separated_list1(multispace0, parse_command).parse(input)
}

fn parse_command(input: &str) -> IResult<&str, (&str, Output<'_>)> {
    pair(
        delimited(multispace0, parse_instruction, multispace0),
        alt((
            map(parse_multiquote, Output::Single),
            map(parse_singlequote, Output::Single),
            map(parse_multi_arguments, Output::Pair),
            map(parse_singleline, Output::Single),
        )),
    )
    .parse(input)
}

fn parse_instruction(input: &str) -> IResult<&str, &str> {
    alt((
        tag_no_case("FROM"),
        tag_no_case("PARAMETER"),
        tag_no_case("TEMPLATE"),
        tag_no_case("SYSTEM"),
        tag_no_case("ADAPTER"),
        tag_no_case("LICENSE"),
        tag_no_case("MESSAGE"),
        tag_no_case("#"),
    ))
    .parse(input)
}

fn parse_multi_arguments(input: &str) -> IResult<&str, (&str, &str)> {
    pair(
        delimited(
            multispace0,
            alt((
                tag_no_case("stop"),
                tag_no_case("num_ctx"),
                tag_no_case("repeat_last_n"),
                tag_no_case("temperature"),
                tag_no_case("seed"),
                tag_no_case("top_k"),
                tag_no_case("top_p"),
                tag_no_case("min_p"),
                tag_no_case("num_predict"),
                tag_no_case("repeat_penalty"),
                tag_no_case("user"),
                tag_no_case("assistant"),
                tag_no_case("system"),
            )),
            multispace0,
        ),
        alt((parse_multiquote, parse_singlequote, parse_singleline)),
    )
    .parse(input)
}

fn parse_multiquote(input: &str) -> IResult<&str, &str> {
    delimited(
        tag_no_case("\"\"\""),
        take_until1("\"\"\""),
        tag_no_case("\"\"\""),
    )
    .parse(input)
}

fn parse_singlequote(input: &str) -> IResult<&str, &str> {
    delimited(tag_no_case("\""), take_until1("\""), tag_no_case("\"")).parse(input)
}
fn parse_singleline(input: &str) -> IResult<&str, &str> {
    delimited(
        multispace0,
        take_while1(|c: char| !c.is_newline()),
        multispace0,
    )
    .parse(input)
}
fn create_modelfile(commands: Vec<(&str, Output)>) -> Result<Modelfile, String> {
    // TODO: There might be a better way
    let mut modelfile: Modelfile = Modelfile::new();
    for command in commands {
        let _ = match (command.0.to_lowercase().as_str(), command.1) {
            //TODO: Can add validations for path if its a gguf file later
            ("from", Output::Single(from)) => modelfile.add_from(from.trim()),
            ("parameter", Output::Pair((param, argument))) => {
                modelfile.add_parameter(param, argument.trim())
            }
            ("template", Output::Single(template)) => modelfile.add_template(template.trim()),
            ("system", Output::Single(system)) => modelfile.add_system(system.trim()),
            ("adapter", Output::Single(adapter)) => modelfile.add_adapter(adapter.trim()),
            ("message", Output::Pair((role, message))) => {
                modelfile.add_message(role, message.trim())
            }
            ("license", Output::Single(license)) => modelfile.add_license(license.trim()),
            ("#", comment) => {
                let comment_str = comment.to_string();
                modelfile.add_comment(&comment_str)
            }
            (instruction, command) => {
                modelfile.errors.push(format!(
                    "Invalid instruction Instruction: `{}` command: `{}`",
                    instruction, command
                ));
                Ok(())
            }
        };
    }

    modelfile.build()?;
    if modelfile.errors.is_empty() {
        Ok(modelfile.clone())
    } else {
        let error = modelfile.errors.join(" , ");
        Err(error)
    }
}

fn parse_parameter(param: &str, argument: &str) -> Result<Parameter, String> {
    let param_type: String = param.to_lowercase();
    match (param_type.as_str(), argument) {
        ("num_ctx", value) => parse_int(param_type, value),
        ("repeat_last_n", value) => parse_int(param_type, value),
        ("repeat_penalty", value) => parse_float(param_type, value),
        ("temperature", value) => parse_float(param_type, value),
        ("seed", value) => parse_int(param_type, value),

        ("stop", value) => Ok(Parameter::new(
            param_type,
            ParamValue::Str(value.to_owned()),
        )),

        ("num_predict", value) => parse_int(param_type, value),

        ("top_k", value) => parse_int(param_type, value),

        ("top_p", value) => parse_float(param_type, value),

        ("min_p", value) => parse_float(param_type, value),
        _ => Err("Invalid Parameter type".to_owned()),
    }
}

fn parse_int(param_type: String, value: &str) -> Result<Parameter, String> {
    if let Ok(parsed_val) = value.parse::<i32>() {
        Ok(Parameter::new(param_type, ParamValue::Int(parsed_val)))
    } else {
        Err(format!("{} not an Integer", param_type))
    }
}

fn parse_float(param_type: String, value: &str) -> Result<Parameter, String> {
    if let Ok(parsed_val) = value.parse::<f32>() {
        Ok(Parameter::new(param_type, ParamValue::Float(parsed_val)))
    } else {
        Err(format!("{} not a Float", param_type))
    }
}

fn parse_message(role: &str, message: &str) -> Result<Message, String> {
    let binding = role.to_lowercase();
    let param_type = binding.as_str();
    if let Ok(role) = param_type.parse::<Role>() {
        Ok(Message {
            role,
            message: message.to_owned(),
        })
    } else {
        Err(format!("{} not a valid role", param_type))
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn test_empty_modelfile() {
        let res = parse("");
        assert!(res.is_err());
    }
    #[test]
    fn test_wrong_instruction() {
        assert!(parse("FRO llama").is_err());
    }

    #[test]
    fn test_valid_modelfile() {
        let modelfile = "
            FROM llama3.2
            PARAMETER num_ctx 4096
        ";

        assert!(parse(modelfile).is_ok());
    }
    #[test]
    fn test_invalid_instruction() {
        let modelfile = "
            FROM llama3.2
            adapter num_ctx 4096
        ";

        assert!(parse(modelfile).is_err());
    }

    #[test]
    fn test_valid_comment() {
        let modelfile = "
            FROM llama3.2
            # system num_ctx 4096
        ";

        assert!(parse(modelfile).is_ok());
    }

    #[test]
    fn test_parse_modelfile_without_from() {
        let modelfile = "
            PARAMETER num_ctx 4096
        ";
        assert!(parse(modelfile).is_err())
    }

    #[test]
    fn test_parse_multiline_single_arguments() {
        let modelfile = "
        FROM llama3.2
        SYSTEM \"\"\"
            You are a bot
            You also not a bot
            \"\"\"";
        let res = parse(modelfile);
        println!("{:?}", res);
        assert!(parse(modelfile).is_ok())
    }

    #[test]
    fn test_modelfile_builder() -> Result<(), Box<dyn Error>> {
        let mut modelfile = Modelfile::new();
        modelfile.add_from("llama3.2")?;
        modelfile.add_parameter("num_ctx", "4096")?;
        assert!(modelfile.build().is_ok());
        Ok(())
    }

    #[test]
    fn test_building_over_parsed_modelfile() -> Result<(), Box<dyn Error>> {
        let modelfile_content = "
            FROM llama3.2
            PARAMETER num_ctx 4096
        ";
        let mut modelfile = parse(modelfile_content)?;
        modelfile.add_parameter("temperature", "3.2")?;
        assert!(modelfile.build().is_ok());
        Ok(())
    }

    #[test]
    fn test_parse_modelfile_from_file() {
        let modelfile = parse_from_file("fixtures/a.modelfile").unwrap();
        assert_eq!(modelfile.from, Some("llama3.2:latest".to_owned()))
    }

    #[test]
    fn test_e2e_pipeline() -> Result<(), Box<dyn Error>> {
        let modelfile_content = "
            FROM llama3.2
            PARAMETER num_ctx 4096
        ";
        let mut modelfile = parse(modelfile_content)?;
        modelfile.add_parameter("temperature", "3.2")?;
        assert!(modelfile.build().is_ok());
        let modelfile_str = modelfile.to_string();
        assert!(modelfile_str.contains("temperature"));
        Ok(())
    }

    #[test]
    fn test_parse_modelfile_from_file_mistral() -> Result<(), Box<dyn Error>> {
        let modelfile = parse_from_file("fixtures/mistral.modelfile")?;
        // There should be 8 tokens in the modelfile including comments
        assert_eq!(modelfile.data.len(), 8);
        Ok(())
    }

    #[test]
    fn test_parse_bad_modelfile() {
        // modelfile has more than 2 FROM
        assert!(parse_from_file("fixtures/llama_bad.Modelfile").is_err())
    }

    #[test]
    fn test_values_should_be_trimmed() -> Result<(), String> {
        let modelfile_content = "
            FROM llama3.2 
            PARAMETER num_ctx 4096
        ";
        let modelfile = parse(modelfile_content)?;
        assert_eq!(modelfile.from.unwrap(), String::from("llama3.2"));
        Ok(())
    }
}
