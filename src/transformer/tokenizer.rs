use std::{collections::HashMap, iter::Peekable};

use crate::Result;

// TODO:build Byte Pair Encoding Tokenizer
// Optional: encoding rules???
#[derive(Debug)]
pub struct BytePairEncoder {
    vocabulary: HashMap<usize, String>,
    merges: HashMap<(usize, usize), usize>,
    max_token_id: usize,
}

impl Default for BytePairEncoder {
    fn default() -> Self {
        Self {
            vocabulary: HashMap::new(),
            merges: HashMap::new(),
            max_token_id: 255, //TODO: potentially set this dynamically, currently coupled to utf-8 encoding
        }
    }
}

pub enum Encoding {
    Utf8,
}

impl BytePairEncoder {
    fn from_utf8(&mut self, input: &str) -> Result<Vec<usize>> {
        let byte_rep = input.as_bytes().to_vec();
        let encoded_byte_rep = byte_rep.into_iter().map(u8::into).collect();

        let vocab = input
            .chars()
            .into_iter()
            .zip(&encoded_byte_rep)
            .map(|x: (char, &usize)| (x.1.to_owned(), x.0.to_string()))
            .collect::<HashMap<usize, String>>();
        self.vocabulary.extend(vocab);
        Ok(encoded_byte_rep)
    }

    fn get_token_id(&mut self) -> usize {
        self.max_token_id += 1;
        let id = self.max_token_id.clone();
        id
    }

    pub fn train(
        &mut self,
        text: String,
        encoding: Encoding,
        max_vocab_size: Option<usize>,
    ) -> Result<()> {
        match encoding {
            Encoding::Utf8 => {
                let byte_rep = self.from_utf8(&text)?;
                self.train_loop(byte_rep, max_vocab_size)?; // _ => Err("Not implemented!".into()),
            }
        }

        // finish creating vocabulary
        let mut total_vocab = self.vocabulary.to_owned();
        for rule in self.get_merge_rules() {
            total_vocab.insert(
                *rule.1,
                format!(
                    "{}{}",
                    total_vocab
                        .get(&rule.0 .0)
                        .expect(&format!(
                            "{:?} exists, current vocab: {:?}",
                            &rule, total_vocab
                        ))
                        .as_str(),
                    total_vocab
                        .get(&rule.0 .1)
                        .expect(&format!(
                            "{:?} exists, current vocab:  {:?}",
                            &rule, total_vocab
                        ))
                        .as_str()
                ),
            );
        }
        self.vocabulary = total_vocab;
        Ok(())
    }

    pub fn encode(&mut self, input: &str, encoding: Encoding) -> Result<Vec<usize>> {
        match encoding {
            Encoding::Utf8 => {
                let byte_rep = self.from_utf8(input)?;
                self.encode_input(&byte_rep) // TODO: Separate training and encoding
            } // _ => Err("Not implemented!".into()),
        }
    }

    fn encode_input(&self, input: &Vec<usize>) -> Result<Vec<usize>> {
        let mut output: Vec<usize> = Vec::new();

        let mut it = input.into_iter().peekable();
        while let Some(token) = it.next() {
            if !it.peek().is_none() {
                let mut result = self.check_merge_rules_recursively(*token, &mut it)?;
                output.append(&mut result);
            } else {
                output.push(*token)
            }
        }

        Ok(output)
    }

    fn check_merge_rules_recursively<'a, I: Iterator<Item = &'a usize>>(
        &self,
        token: usize,
        it: &mut Peekable<I>,
    ) -> Result<Vec<usize>> {
        let mut output: Vec<usize> = Vec::new();
        if !it.peek().is_none() {
            let test = (token, **it.peek().expect("value can be found"));
            match self.merges.get(&test) {
                Some(value) => {
                    // Consume tokens if found
                    it.next();
                    let mut deeper = self.check_merge_rules_recursively(*value, it)?;
                    output.append(&mut deeper);
                    Ok(output)
                }
                None => {
                    output.push(token);

                    Ok(output)
                }
            }
        } else {
            output.push(token);
            Ok(output)
        }
    }

    fn train_loop(
        &mut self,
        input: Vec<usize>,
        max_vocab_size: Option<usize>,
    ) -> Result<Vec<usize>> {
        // TODO: this is not just encode, but encode+train, refactor so that train and encode separate steps
        let mut input_vec = input;
        let loop_iter = match max_vocab_size {
            Some(v) => v,
            None => 20 + self.vocabulary.len(), // default max vocab size
        };
        loop {
            if input_vec.len() <= 1 || self.vocabulary.len() + self.merges.len() >= loop_iter {
                // TODO: control loop exit condition with parameter instead, e.g. vocab size
                break;
            } else {
                let input_vec_once = self.train_once(&input_vec)?;
                if input_vec_once == input_vec {
                    break; // no more merges were done
                } else {
                    input_vec = input_vec_once;
                }
            }
        }
        if input_vec.len() > 0 {
            Ok(input_vec)
        } else {
            unreachable!("we checked in the loop");
        }
    }
    fn train_once(&mut self, input: &Vec<usize>) -> Result<Vec<usize>> {
        let mut counts: HashMap<(usize, usize), usize> = HashMap::new();

        let _ = input
            .windows(2)
            .map(|x| match counts.get(&(x[0], x[1])) {
                Some(i) => {
                    counts.insert((x[0], x[1]), i + 1);
                }
                None => {
                    counts.insert((x[0], x[1]), 1);
                }
            })
            .collect::<Vec<_>>();
        //TODO: validate which pair to mint, if only one pair is minted.
        //TODO: also validate if more than one pair is minted
        let encoding_candidate = counts
            // let (top_count, count) = counts
            .iter()
            .max_by(|a, b| a.1.cmp(&b.1))
            .map(|(k, &v)| match v > 1 {
                true => Some((k, v)),
                false => None,
            })
            .expect("can find a top count");
        match encoding_candidate {
            Some((top_count, _)) => {
                let id = self.get_token_id();
                self.merges.insert(top_count.to_owned(), id);
                let mut output: Vec<usize> = Vec::new();
                let mut it = input.into_iter().peekable();
                while let Some(token) = it.next() {
                    if !it.peek().is_none() {
                        let test = (token.to_owned(), **it.peek().expect("value can be found"));
                        match self.merges.get(&test) {
                            Some(value) => {
                                // consume both tokens and replace with merge value
                                it.next();
                                output.push(value.to_owned())
                            }
                            None => output.push(*token), // add the not found token to the list
                        }
                    } else {
                        output.push(*token)
                    }
                }
                Ok(output)
            }
            None => Ok(input.clone()),
        }
    }
    pub fn decode(&self, input: Vec<usize>) -> Result<String> {
        let mut output: String = String::new();
        for token in input {
            match self.vocabulary.get(&token) {
                Some(v) => output.push_str(v),
                None => unreachable!("the token should exist in vocab!"),
            }
        }
        return Ok(output);
    }

    fn get_merge_rules(&self) -> Vec<(&(usize, usize), &usize)> {
        let mut merge_rules = self
            .merges
            .iter()
            .collect::<Vec<(&(usize, usize), &usize)>>();
        merge_rules.sort_by(|a, b| a.1.cmp(b.1));
        merge_rules
    }
    pub fn to_utf8(&self, input: Vec<usize>) -> Result<String> {
        input
            .into_iter()
            .map(|ch| match self.vocabulary.get(&ch) {
                Some(letter) => Ok(letter.to_owned()),
                None => Err(format!("invalid token in input {ch}").into()),
            })
            .collect::<Result<String>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_string() -> Result<()> {
        let input = "hello world, we are programming!";
        let mut encoder = BytePairEncoder::default();
        let output = encoder.from_utf8(input)?;
        let expected = vec![
            104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 44, 32, 119, 101, 32, 97, 114,
            101, 32, 112, 114, 111, 103, 114, 97, 109, 109, 105, 110, 103, 33,
        ];

        assert_eq!(output, expected);

        Ok(())
    }

    #[test]
    fn test_bpe_encode() -> Result<()> {
        // let input = "hello world, we are programming!";
        let input = "ababbcbc";
        let mut encoder = BytePairEncoder::default();
        encoder.train(input.to_string(), Encoding::Utf8, None)?;
        let output = encoder.encode(input, Encoding::Utf8)?;
        let expected = vec![257, 257, 256, 256];
        println!("encoder: {:?}", encoder);
        println!("encoded string: {:?}", output);

        assert_eq!(output, expected);

        Ok(())
    }
    #[test]
    fn test_bpe_decode() -> Result<()> {
        let input = "abbbcbcd";
        let mut encoder = BytePairEncoder::default();
        encoder.train(input.to_string(), Encoding::Utf8, None)?;
        let encode_once = encoder.from_utf8(input)?;
        println!("initial encoding: {:?}", encode_once);
        let encoded = encoder.encode(input, Encoding::Utf8)?;
        println!("final encoding: {:?}", &encoded);
        let decoded = encoder.decode(encoded)?;
        assert_eq!(decoded, input);
        Ok(())
    }

    #[test]
    fn test_bpe_to_utf8() -> Result<()> {
        let input = vec![
            104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 44, 32, 119, 101, 32, 97, 114,
            101, 32, 112, 114, 111, 103, 114, 97, 109, 109, 105, 110, 103, 33,
        ];
        let mut encoder = BytePairEncoder::default();
        let expected = "hello world, we are programming!".to_string();
        encoder.train(expected.to_string(), Encoding::Utf8, None)?;

        let _ = encoder.encode(&expected, Encoding::Utf8); // building vocab
        let output = encoder.to_utf8(input)?;

        assert_eq!(output, expected);

        Ok(())
    }

    #[test]
    fn test_transformer_flow() -> Result<()> {
        let input = "aabbabax";
        println!("input string: {input}");
        let mut encoder = BytePairEncoder::default();
        encoder.train(input.to_string(), Encoding::Utf8, None)?;

        let encoded = encoder.encode(input, Encoding::Utf8)?;

        println!("{:?}", encoder);
        println!("last encoded string: {:?}", encoded);

        let decoded = encoder.decode(encoded)?;
        println!("decoded string: {:?}", decoded);
        assert_eq!(input, decoded);

        Ok(())
    }
}
