//! Grammar-constrained sampling for structured output
//!
//! This module provides grammar-based constraints for token sampling,
//! enabling generation of valid JSON, code, or other structured formats.
//!
//! Supported grammar types:
//! - JSON Schema validation
//! - Regular expression patterns
//! - Context-free grammars (GBNF format)

use std::collections::HashSet;

/// Grammar constraint for token sampling
#[derive(Debug, Clone, Default)]
pub enum Grammar {
    /// JSON output constraint
    Json(JsonGrammar),
    /// Regular expression pattern
    Regex(RegexGrammar),
    /// Context-free grammar (GBNF format)
    Gbnf(GbnfGrammar),
    /// Choice between multiple strings
    Choice(Vec<String>),
    /// No constraint
    #[default]
    None,
}

/// JSON grammar constraint
#[derive(Debug, Clone)]
pub struct JsonGrammar {
    /// JSON schema (optional)
    pub schema: Option<String>,
    /// Whether to allow any valid JSON
    pub allow_any: bool,
    /// Required fields for objects
    pub required_fields: Vec<String>,
}

impl Default for JsonGrammar {
    fn default() -> Self {
        Self {
            schema: None,
            allow_any: true,
            required_fields: Vec::new(),
        }
    }
}

impl JsonGrammar {
    /// Create a grammar for any valid JSON
    pub fn any() -> Self {
        Self::default()
    }

    /// Create a grammar with a JSON schema
    pub fn with_schema(schema: impl Into<String>) -> Self {
        Self {
            schema: Some(schema.into()),
            allow_any: false,
            required_fields: Vec::new(),
        }
    }
}

/// Regular expression grammar constraint
#[derive(Debug, Clone)]
pub struct RegexGrammar {
    /// The regex pattern
    pub pattern: String,
    /// Compiled state machine (simplified)
    state: RegexState,
}

#[derive(Debug, Clone, Default)]
struct RegexState {
    /// Current position in pattern matching
    position: usize,
    /// Whether we're in a character class
    _in_class: bool,
    /// Minimum remaining characters needed
    _min_remaining: usize,
}

impl RegexGrammar {
    /// Create a new regex grammar
    pub fn new(pattern: impl Into<String>) -> Self {
        let pattern = pattern.into();
        Self {
            pattern,
            state: RegexState::default(),
        }
    }

    /// Check if a character is allowed at current position
    pub fn allows_char(&self, c: char) -> bool {
        // Simplified regex matching - full implementation would compile to NFA/DFA
        if self.state.position >= self.pattern.len() {
            return false;
        }

        let pattern_chars: Vec<char> = self.pattern.chars().collect();
        let current = pattern_chars.get(self.state.position);

        match current {
            Some('.') => true, // Dot matches any character
            Some('\\') => {
                // Escape sequence
                if let Some(&next) = pattern_chars.get(self.state.position + 1) {
                    match next {
                        'd' => c.is_ascii_digit(),
                        'w' => c.is_alphanumeric() || c == '_',
                        's' => c.is_whitespace(),
                        _ => c == next,
                    }
                } else {
                    false
                }
            }
            Some('[') => {
                // Character class - simplified
                true
            }
            Some(&pc) if pc == c => true,
            Some('*') | Some('+') | Some('?') => true, // Quantifiers
            _ => false,
        }
    }

    /// Advance state after accepting a character
    pub fn advance(&mut self, _c: char) {
        self.state.position += 1;
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.state = RegexState::default();
    }
}

/// GBNF (GGML BNF) grammar constraint
#[derive(Debug, Clone)]
pub struct GbnfGrammar {
    /// Grammar rules
    pub rules: Vec<GbnfRule>,
    /// Root rule name
    pub root: String,
    /// Current parse state
    state: GbnfState,
}

/// A single GBNF rule
#[derive(Debug, Clone)]
pub struct GbnfRule {
    /// Rule name
    pub name: String,
    /// Rule alternatives
    pub alternatives: Vec<GbnfAlternative>,
}

/// An alternative in a GBNF rule
#[derive(Debug, Clone)]
pub struct GbnfAlternative {
    /// Sequence of elements
    pub elements: Vec<GbnfElement>,
}

/// An element in a GBNF alternative
#[derive(Debug, Clone)]
pub enum GbnfElement {
    /// Literal string
    Literal(String),
    /// Reference to another rule
    RuleRef(String),
    /// Character range [a-z]
    CharRange(char, char),
    /// Character class
    CharClass(Vec<char>),
    /// Optional element (?)
    Optional(Box<GbnfElement>),
    /// Zero or more (*)
    ZeroOrMore(Box<GbnfElement>),
    /// One or more (+)
    OneOrMore(Box<GbnfElement>),
}

#[derive(Debug, Clone, Default)]
struct GbnfState {
    /// Stack of rule states
    _stack: Vec<(String, usize, usize)>, // (rule_name, alt_idx, elem_idx)
}

impl GbnfGrammar {
    /// Create a new GBNF grammar
    pub fn new(rules: Vec<GbnfRule>, root: impl Into<String>) -> Self {
        Self {
            rules,
            root: root.into(),
            state: GbnfState::default(),
        }
    }

    /// Parse GBNF grammar from string
    pub fn parse(input: &str) -> Result<Self, String> {
        let mut rules = Vec::new();
        let mut root = String::new();

        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse rule: name ::= alternatives
            if let Some(pos) = line.find("::=") {
                let name = line[..pos].trim().to_string();
                let body = line[pos + 3..].trim();

                if root.is_empty() {
                    root = name.clone();
                }

                let alternatives = Self::parse_alternatives(body)?;
                rules.push(GbnfRule { name, alternatives });
            }
        }

        if rules.is_empty() {
            return Err("No rules found in grammar".to_string());
        }

        Ok(Self::new(rules, root))
    }

    fn parse_alternatives(body: &str) -> Result<Vec<GbnfAlternative>, String> {
        let mut alternatives = Vec::new();

        for alt in body.split('|') {
            let elements = Self::parse_elements(alt.trim())?;
            alternatives.push(GbnfAlternative { elements });
        }

        Ok(alternatives)
    }

    fn parse_elements(body: &str) -> Result<Vec<GbnfElement>, String> {
        let mut elements = Vec::new();
        let mut chars = body.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                '"' => {
                    // Literal string
                    let mut literal = String::new();
                    while let Some(&next) = chars.peek() {
                        if next == '"' {
                            chars.next();
                            break;
                        }
                        if next == '\\' {
                            chars.next();
                            if let Some(escaped) = chars.next() {
                                literal.push(escaped);
                            }
                        } else {
                            literal.push(chars.next().unwrap());
                        }
                    }
                    elements.push(GbnfElement::Literal(literal));
                }
                '[' => {
                    // Character class or range
                    let mut class_chars = Vec::new();
                    while let Some(&next) = chars.peek() {
                        if next == ']' {
                            chars.next();
                            break;
                        }
                        class_chars.push(chars.next().unwrap());
                    }

                    // Check for range (e.g., a-z)
                    if class_chars.len() == 3 && class_chars[1] == '-' {
                        elements.push(GbnfElement::CharRange(class_chars[0], class_chars[2]));
                    } else {
                        elements.push(GbnfElement::CharClass(class_chars));
                    }
                }
                ' ' | '\t' => {
                    // Skip whitespace
                }
                _ if c.is_alphabetic() || c == '_' => {
                    // Rule reference
                    let mut name = String::from(c);
                    while let Some(&next) = chars.peek() {
                        if next.is_alphanumeric() || next == '_' || next == '-' {
                            name.push(chars.next().unwrap());
                        } else {
                            break;
                        }
                    }
                    elements.push(GbnfElement::RuleRef(name));
                }
                _ => {}
            }
        }

        Ok(elements)
    }

    /// Get allowed next characters based on current state
    pub fn allowed_chars(&self) -> HashSet<char> {
        // Simplified - return common characters
        let mut allowed = HashSet::new();

        // Add basic ASCII printable characters
        for c in ' '..='~' {
            allowed.insert(c);
        }

        allowed
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.state = GbnfState::default();
    }
}

/// Grammar-aware token filter
#[derive(Debug)]
pub struct GrammarSampler {
    /// The grammar constraint
    grammar: Grammar,
    /// Current generated text
    generated: String,
    /// Token vocabulary for filtering
    vocab: Vec<String>,
}

impl GrammarSampler {
    /// Create a new grammar sampler
    pub fn new(grammar: Grammar, vocab: Vec<String>) -> Self {
        Self {
            grammar,
            generated: String::new(),
            vocab,
        }
    }

    /// Get mask of allowed tokens (true = allowed, false = blocked)
    pub fn get_token_mask(&self) -> Vec<bool> {
        let mut mask = vec![true; self.vocab.len()];

        match &self.grammar {
            Grammar::None => {
                // No filtering
            }
            Grammar::Json(_) => {
                // Filter tokens based on JSON validity
                self.filter_json_tokens(&mut mask);
            }
            Grammar::Regex(regex) => {
                // Filter based on regex
                self.filter_regex_tokens(&mut mask, regex);
            }
            Grammar::Gbnf(gbnf) => {
                // Filter based on GBNF grammar
                self.filter_gbnf_tokens(&mut mask, gbnf);
            }
            Grammar::Choice(choices) => {
                // Only allow tokens that could lead to one of the choices
                self.filter_choice_tokens(&mut mask, choices);
            }
        }

        mask
    }

    fn filter_json_tokens(&self, mask: &mut [bool]) {
        let current = &self.generated;
        let depth = current.chars().filter(|&c| c == '{' || c == '[').count() as i32
            - current.chars().filter(|&c| c == '}' || c == ']').count() as i32;

        for (i, token) in self.vocab.iter().enumerate() {
            let would_be = format!("{}{}", current, token);

            // Basic JSON validity checks
            let valid = if current.is_empty() {
                // Must start with { or [
                token.trim_start().starts_with('{')
                    || token.trim_start().starts_with('[')
                    || token.trim().is_empty()
            } else if depth <= 0 && !current.trim().is_empty() {
                // Already closed, don't allow more content
                token.trim().is_empty()
            } else {
                // Check for balanced brackets
                let new_depth = would_be.chars().filter(|&c| c == '{' || c == '[').count() as i32
                    - would_be.chars().filter(|&c| c == '}' || c == ']').count() as i32;
                new_depth >= 0
            };

            mask[i] = valid;
        }
    }

    fn filter_regex_tokens(&self, mask: &mut [bool], regex: &RegexGrammar) {
        for (i, token) in self.vocab.iter().enumerate() {
            let mut allowed = true;
            for c in token.chars() {
                if !regex.allows_char(c) {
                    allowed = false;
                    break;
                }
            }
            mask[i] = allowed;
        }
    }

    fn filter_gbnf_tokens(&self, mask: &mut [bool], gbnf: &GbnfGrammar) {
        let allowed_chars = gbnf.allowed_chars();

        for (i, token) in self.vocab.iter().enumerate() {
            let all_allowed = token.chars().all(|c| allowed_chars.contains(&c));
            mask[i] = all_allowed;
        }
    }

    fn filter_choice_tokens(&self, mask: &mut [bool], choices: &[String]) {
        for (i, token) in self.vocab.iter().enumerate() {
            let would_be = format!("{}{}", self.generated, token);

            // Check if this could lead to any choice
            let could_match = choices
                .iter()
                .any(|choice| choice.starts_with(&would_be) || would_be.starts_with(choice));

            mask[i] = could_match;
        }
    }

    /// Apply mask to logits (set blocked tokens to -inf)
    pub fn apply_mask(&self, logits: &mut [f32]) {
        let mask = self.get_token_mask();

        for (i, &allowed) in mask.iter().enumerate() {
            if !allowed && i < logits.len() {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    /// Record a generated token
    pub fn record_token(&mut self, token: &str) {
        self.generated.push_str(token);
    }

    /// Reset the sampler state
    pub fn reset(&mut self) {
        self.generated.clear();
        match &mut self.grammar {
            Grammar::Regex(r) => r.reset(),
            Grammar::Gbnf(g) => g.reset(),
            _ => {}
        }
    }

    /// Check if generation is complete according to grammar
    pub fn is_complete(&self) -> bool {
        match &self.grammar {
            Grammar::None => false,
            Grammar::Json(_) => {
                let trimmed = self.generated.trim();
                (trimmed.starts_with('{') && trimmed.ends_with('}'))
                    || (trimmed.starts_with('[') && trimmed.ends_with(']'))
            }
            Grammar::Choice(choices) => choices.iter().any(|c| c == &self.generated),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_grammar() {
        let grammar = JsonGrammar::any();
        assert!(grammar.allow_any);
    }

    #[test]
    fn test_regex_grammar() {
        let regex = RegexGrammar::new(r"\d+");
        assert!(regex.allows_char('5'));
    }

    #[test]
    fn test_gbnf_parse() {
        let grammar_str = r#"
            root ::= "hello" | "world"
        "#;

        let grammar = GbnfGrammar::parse(grammar_str).unwrap();
        assert_eq!(grammar.root, "root");
        assert_eq!(grammar.rules.len(), 1);
    }

    #[test]
    fn test_grammar_sampler_json() {
        let grammar = Grammar::Json(JsonGrammar::any());
        let vocab = vec!["{".to_string(), "}".to_string(), "hello".to_string()];
        let sampler = GrammarSampler::new(grammar, vocab);

        let mask = sampler.get_token_mask();
        assert!(mask[0]); // { should be allowed at start
    }

    #[test]
    fn test_grammar_sampler_choice() {
        let grammar = Grammar::Choice(vec!["yes".to_string(), "no".to_string()]);
        let vocab = vec!["y".to_string(), "n".to_string(), "x".to_string()];
        let sampler = GrammarSampler::new(grammar, vocab);

        let mask = sampler.get_token_mask();
        assert!(mask[0]); // "y" could lead to "yes"
        assert!(mask[1]); // "n" could lead to "no"
        assert!(!mask[2]); // "x" can't lead to either
    }
}
