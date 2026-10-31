//! Concrete context store with refcounted, page-aligned KV segments.
//! This is the memory model the planner reasons about: a fork shares every
//! segment, each append/gen allocates a new segment rounded up to whole pages,
//! and a splice allocates the replacement plus copies of the tail *before*
//! releasing the old chain.

use std::collections::{BTreeSet, HashMap};

pub type CtxId = u64;
type SegId = u64;

#[derive(Clone, Debug)]
struct Seg {
    text: String,
    tokens: u64,
    alloc: u64,
    labels: BTreeSet<String>,
    refs: u32,
    bytes_per_token: f64,
}

#[derive(Clone, Debug)]
struct Ctx {
    model: String,
    segs: Vec<SegId>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StoreStats {
    pub kv_bytes_now: f64,
    pub peak_kv_bytes: f64,
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub reused_tokens: u64,
    pub live_contexts: usize,
}

#[derive(Default)]
pub struct Store {
    segs: HashMap<SegId, Seg>,
    ctxs: HashMap<CtxId, Ctx>,
    prefixes: HashMap<String, (String, SegId)>,
    next: u64,
    pub stats: StoreStats,
}

pub struct ModelGeom {
    pub name: String,
    pub bytes_per_token: f64,
    pub block: u64,
}

impl Store {
    fn id(&mut self) -> u64 {
        self.next += 1;
        self.next
    }

    fn alloc_seg(&mut self, geom_bpt: f64, block: u64, text: String, tokens: u64, labels: BTreeSet<String>) -> SegId {
        let id = self.id();
        let alloc = tokens.div_ceil(block) * block;
        self.segs.insert(id, Seg { text, tokens, alloc, labels, refs: 0, bytes_per_token: geom_bpt });
        id
    }

    fn incref(&mut self, s: SegId) {
        let seg = self.segs.get_mut(&s).unwrap();
        if seg.refs == 0 {
            self.stats.kv_bytes_now += seg.alloc as f64 * seg.bytes_per_token;
        }
        seg.refs += 1;
        if self.stats.kv_bytes_now > self.stats.peak_kv_bytes {
            self.stats.peak_kv_bytes = self.stats.kv_bytes_now;
        }
    }

    fn decref(&mut self, s: SegId) {
        let seg = self.segs.get_mut(&s).unwrap();
        seg.refs -= 1;
        if seg.refs == 0 {
            self.stats.kv_bytes_now -= seg.alloc as f64 * seg.bytes_per_token;
            self.segs.remove(&s);
        }
    }

    fn new_ctx(&mut self, model: &str, segs: Vec<SegId>) -> CtxId {
        for &s in &segs {
            self.incref(s);
        }
        let id = self.id();
        self.ctxs.insert(id, Ctx { model: model.to_string(), segs });
        self.stats.live_contexts = self.ctxs.len();
        id
    }

    fn ctx(&self, c: CtxId) -> Result<&Ctx, String> {
        self.ctxs.get(&c).ok_or_else(|| format!("internal: unknown context handle c{c}"))
    }

    pub fn register_prefix(&mut self, name: &str, g: &ModelGeom, text: &str, tokens: u64) {
        let s = self.alloc_seg(g.bytes_per_token, g.block, text.to_string(), 1 + tokens, BTreeSet::new());
        self.incref(s); // pinned
        self.stats.prefill_tokens += 1 + tokens;
        self.prefixes.insert(name.to_string(), (g.name.clone(), s));
    }

    pub fn open(&mut self, g: &ModelGeom, prefix: Option<&str>) -> CtxId {
        if let Some(p) = prefix {
            let (model, seg) = self.prefixes[p].clone();
            self.stats.reused_tokens += self.segs[&seg].tokens;
            return self.new_ctx(&model, vec![seg]);
        }
        let s = self.alloc_seg(g.bytes_per_token, g.block, String::new(), 1, BTreeSet::new());
        self.stats.prefill_tokens += 1;
        self.new_ctx(&g.name, vec![s])
    }

    pub fn model_of(&self, c: CtxId) -> Result<String, String> {
        Ok(self.ctx(c)?.model.clone())
    }

    pub fn text(&self, c: CtxId) -> Result<String, String> {
        Ok(self.ctx(c)?.segs.iter().map(|s| self.segs[s].text.as_str()).collect())
    }

    pub fn tokens(&self, c: CtxId) -> Result<u64, String> {
        Ok(self.ctx(c)?.segs.iter().map(|s| self.segs[s].tokens).sum())
    }

    /// Append a segment; `decode` marks generated (not prefilled) tokens.
    pub fn append(&mut self, g: &ModelGeom, c: CtxId, text: &str, tokens: u64, label: Option<&str>, decode: bool) -> Result<CtxId, String> {
        let mut segs = self.ctx(c)?.segs.clone();
        let model = self.ctx(c)?.model.clone();
        let labels = label.map(|l| BTreeSet::from([l.to_string()])).unwrap_or_default();
        let s = self.alloc_seg(g.bytes_per_token, g.block, text.to_string(), tokens, labels);
        segs.push(s);
        if decode {
            self.stats.decode_tokens += tokens;
        } else {
            self.stats.prefill_tokens += tokens;
        }
        let n = self.new_ctx(&model, segs);
        self.release(c);
        Ok(n)
    }

    pub fn fork(&mut self, c: CtxId) -> Result<(CtxId, CtxId), String> {
        let ctx = self.ctx(c)?.clone();
        self.stats.reused_tokens += ctx.segs.iter().map(|s| self.segs[s].tokens).sum::<u64>();
        let a = self.new_ctx(&ctx.model, ctx.segs.clone());
        let b = self.new_ctx(&ctx.model, ctx.segs.clone());
        self.release(c);
        Ok((a, b))
    }

    /// Replace the segment labelled `label`; every later segment is re-prefilled.
    pub fn splice(&mut self, g: &ModelGeom, c: CtxId, label: &str, new_label: Option<&str>, text: &str, tokens: u64) -> Result<(CtxId, u64), String> {
        let ctx = self.ctx(c)?.clone();
        let idx = ctx.segs.iter().position(|s| self.segs[s].labels.contains(label)).ok_or_else(|| format!("span `{label}` not found"))?;
        let mut segs = ctx.segs[..idx].to_vec();
        segs.push(self.alloc_seg(g.bytes_per_token, g.block, text.to_string(), tokens, new_label.map(|l| BTreeSet::from([l.to_string()])).unwrap_or_default()));
        let mut tail = 0;
        for s in &ctx.segs[idx + 1..] {
            let old = self.segs[s].clone();
            tail += old.tokens;
            segs.push(self.alloc_seg(g.bytes_per_token, g.block, old.text, old.tokens, old.labels));
        }
        self.stats.prefill_tokens += tokens + tail;
        let n = self.new_ctx(&ctx.model, segs);
        self.release(c);
        Ok((n, tail))
    }

    pub fn release(&mut self, c: CtxId) {
        if let Some(ctx) = self.ctxs.remove(&c) {
            for s in ctx.segs {
                self.decref(s);
            }
        }
        self.stats.live_contexts = self.ctxs.len();
    }
}
