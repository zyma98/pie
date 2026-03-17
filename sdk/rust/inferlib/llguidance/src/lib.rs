wit_bindgen::generate!({
    path: "wit",
    world: "llguidance-provider",
    generate_all,
});

use exports::inferlib::llguidance::constrained_sampling::{
    Guest, GuestConstrainedSampler, GuestGrammarMatcher, GuestTokenMask, TokenMask,
};

use fancy_regex::Regex;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

struct Component;

export!(Component);

impl Guest for Component {
    type GrammarMatcher = GrammarMatcherImpl;
    type TokenMask = TokenMaskImpl;
    type ConstrainedSampler = ConstrainedSamplerImpl;
}

pub struct GrammarMatcherImpl {
    inner: RefCell<Matcher>,
}

pub struct TokenMaskImpl {
    vob: SimpleVob,
}

impl GuestGrammarMatcher for GrammarMatcherImpl {
    fn new(
        vocab_ids: Vec<u32>,
        vocab_bytes: Vec<Vec<u8>>,
        special_token_ids: Vec<u32>,
        special_token_bytes: Vec<Vec<u8>>,
        split_regex: String,
        grammar: String,
        eos_token_id: u32,
        escape_non_printable: bool,
    ) -> Self {
        let rank_map: HashMap<u32, Vec<u8>> = vocab_ids.into_iter().zip(vocab_bytes).collect();

        let special_tokens: HashMap<String, u32> = special_token_bytes
            .into_iter()
            .map(|w| String::from_utf8(w).unwrap())
            .zip(special_token_ids)
            .collect();

        let tokenizer = BytePairEncoder::new(
            rank_map,
            special_tokens,
            &split_regex,
            eos_token_id,
            escape_non_printable,
        )
        .unwrap();
        let tokenizer_env = tokenizer.to_env();

        let grammar = TopLevelGrammar::from_lark(grammar);
        let factory = ParserFactory::new_simple(&tokenizer_env).unwrap();
        let parser = factory.create_parser(grammar);
        let constraint = Matcher::new(parser);

        GrammarMatcherImpl {
            inner: RefCell::new(constraint),
        }
    }

    fn compute_mask(&self) -> Option<TokenMask> {
        let mut inner = self.inner.borrow_mut();
        match inner.compute_mask() {
            Ok(vob) => Some(TokenMask::new(TokenMaskImpl { vob })),
            Err(_) => None,
        }
    }

    fn consume_token(&self, token_id: u32) {
        let mut inner = self.inner.borrow_mut();
        let _ = inner.consume_token(token_id);
    }
}

impl GuestTokenMask for TokenMaskImpl {
    fn is_empty(&self) -> bool {
        self.vob.is_empty()
    }

    fn is_allowed(&self, token_id: u32) -> bool {
        self.vob.is_allowed(token_id)
    }

    fn first_bit_set(&self) -> Option<u32> {
        self.vob.first_bit_set().map(|v| v as u32)
    }
}

pub struct ConstrainedSamplerImpl {
    matcher: RefCell<Matcher>,
    eos_token_id: u32,
}

impl GuestConstrainedSampler for ConstrainedSamplerImpl {
    fn new(
        vocab_ids: Vec<u32>,
        vocab_bytes: Vec<Vec<u8>>,
        special_token_ids: Vec<u32>,
        special_token_bytes: Vec<Vec<u8>>,
        split_regex: String,
        grammar: String,
        eos_token_id: u32,
        escape_non_printable: bool,
    ) -> Self {
        let rank_map: HashMap<u32, Vec<u8>> = vocab_ids.into_iter().zip(vocab_bytes).collect();

        let special_tokens: HashMap<String, u32> = special_token_bytes
            .into_iter()
            .map(|w| String::from_utf8(w).unwrap())
            .zip(special_token_ids)
            .collect();

        let tokenizer = BytePairEncoder::new(
            rank_map,
            special_tokens,
            &split_regex,
            eos_token_id,
            escape_non_printable,
        )
        .unwrap();
        let tokenizer_env = tokenizer.to_env();

        let grammar = TopLevelGrammar::from_lark(grammar);
        let factory = ParserFactory::new_simple(&tokenizer_env).unwrap();
        let parser = factory.create_parser(grammar);
        let constraint = Matcher::new(parser);

        ConstrainedSamplerImpl {
            matcher: RefCell::new(constraint),
            eos_token_id,
        }
    }

    fn sample(&self, token_ids: Vec<u32>, probs: Vec<f32>) -> u32 {
        let mut inner = self.matcher.borrow_mut();

        let vob = match inner.compute_mask() {
            Ok(vob) => vob,
            Err(_) => return self.eos_token_id,
        };

        if vob.is_empty() {
            return self.eos_token_id;
        }

        let mut max_prob = f32::NEG_INFINITY;
        let mut best_token = None;

        for (i, &token_id) in token_ids.iter().enumerate() {
            if vob.is_allowed(token_id) && probs[i] > max_prob {
                max_prob = probs[i];
                best_token = Some(token_id);
            }
        }

        let sampled_token_id = match best_token {
            Some(token) => token,
            None => {
                return vob.first_bit_set().map(|v| v as u32).unwrap_or(0);
            }
        };

        let _ = inner.consume_token(sampled_token_id);

        sampled_token_id
    }
}

// --- BPE implementation (from tiktoken-rs) ---

type Rank = u32;

fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut parts = Vec::with_capacity(piece.len() + 1);

    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = |parts: &Vec<(usize, Rank)>, i: usize| {
        if (i + 3) < parts.len() {
            *ranks
                .get(&piece[parts[i].0..parts[i + 3].0])
                .unwrap_or(&Rank::MAX)
        } else {
            Rank::MAX
        }
    };

    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;

        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

#[derive(Clone)]
struct BytePairEncoder {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens: HashSet<String>,
    special_tokens_encoder: HashMap<String, Rank>,
    regex: Regex,
    special_regex: Regex,
    tok_trie: TokTrie,
    escape_non_printable: bool,
}

impl BytePairEncoder {
    fn new(
        decoder: HashMap<Rank, Vec<u8>>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
        eos_token: u32,
        escape_non_printable: bool,
    ) -> Result<Self, String> {
        let regex = Regex::new(pattern).map_err(|e| e.to_string())?;
        let special_regex = {
            let parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&parts.join("|")).map_err(|e| e.to_string())?
        };

        let encoder: HashMap<Vec<u8>, Rank> =
            decoder.iter().map(|(k, v)| (v.clone(), *k)).collect();

        if encoder.len() != decoder.len() {
            return Err(
                "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?".to_string()
            );
        }

        let mut max_rank = 0;
        for rank in decoder.keys() {
            max_rank = max_rank.max(*rank);
        }
        for rank in special_tokens_encoder.values() {
            max_rank = max_rank.max(*rank);
        }
        let n_vocab = (max_rank + 1) as usize;

        let mut tokens = vec![vec![]; n_vocab];

        for (rank, bytes) in decoder.iter() {
            let raw_bytes = if escape_non_printable {
                unescape_non_printable(bytes)
            } else {
                bytes.clone()
            };
            tokens[*rank as usize] = raw_bytes;
        }

        for (name, rank) in special_tokens_encoder.iter() {
            let mut spec_bytes = Vec::with_capacity(name.len() + 1);
            spec_bytes.push(TokTrie::SPECIAL_TOKEN_MARKER);
            spec_bytes.extend_from_slice(name.as_bytes());
            tokens[*rank as usize] = spec_bytes;
        }

        let special_tokens = special_tokens_encoder.keys().cloned().collect();

        for (i, token) in tokens.iter_mut().enumerate() {
            if token.is_empty() {
                let mut name = format!(".<[{i}]>").into_bytes();
                name[0] = TokTrie::SPECIAL_TOKEN_MARKER;
                *token = name;
            }
        }

        let tok_trie = TokTrie::from(
            &TokRxInfo {
                vocab_size: n_vocab as u32,
                tok_eos: eos_token,
                tok_end_of_turn: None,
                tok_unk: None,
                tok_pad: None,
                tok_bos: None,
            },
            &tokens,
        );

        Ok(Self {
            encoder,
            special_tokens,
            special_tokens_encoder,
            regex,
            special_regex,
            tok_trie,
            escape_non_printable,
        })
    }

    fn encode(&self, text: &str) -> Vec<Rank> {
        let mut ret = vec![];

        let mut start = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                next_special = self.special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if self.special_tokens.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            for mat in self.regex.find_iter(&text[start..end]) {
                let mut piece = mat.unwrap().as_str().as_bytes();

                let escaped_piece = escape_non_printable(piece);
                if self.escape_non_printable {
                    piece = escaped_piece.as_bytes();
                }

                if let Some(token) = self.encoder.get(piece) {
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                ret.extend(&tokens);
            }

            match next_special {
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                }
                None => break,
            }
        }

        ret
    }

    fn to_env(self) -> TokEnv {
        Arc::new(self)
    }
}

impl TokenizerEnv for BytePairEncoder {
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie
            .tokenize_with_greedy_fallback(s, |s| self.encode(s))
    }

    fn tokenize_bytes_special(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(s, |s| {
            self.tok_trie.tokenize_with_special(s, |s| self.encode(s))
        })
    }
}

/// Generate the 256-entry byte-level maps for Qwen/GPT-style token encoding.
fn build_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();

    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let cs: Vec<char> = cs.into_iter().map(|u| char::from_u32(u).unwrap()).collect();

    let mut enc = ['\0'; 256];
    let mut dec = HashMap::with_capacity(256);
    for (b, ch) in bs.into_iter().zip(cs.into_iter()) {
        enc[b as usize] = ch;
        dec.insert(ch, b);
    }
    (enc, dec)
}

fn escape_non_printable(bytes: &[u8]) -> String {
    static TABLES: once_cell::sync::Lazy<([char; 256], HashMap<char, u8>)> =
        once_cell::sync::Lazy::new(build_tables);

    bytes.iter().map(|&b| TABLES.0[b as usize]).collect()
}

fn unescape_non_printable(bytes: &[u8]) -> Vec<u8> {
    static TABLES: once_cell::sync::Lazy<([char; 256], HashMap<char, u8>)> =
        once_cell::sync::Lazy::new(build_tables);

    match std::str::from_utf8(bytes) {
        Ok(s) => s
            .chars()
            .filter_map(|c| TABLES.1.get(&c).copied())
            .collect(),
        Err(_) => bytes.to_vec(),
    }
}
