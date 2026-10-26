# assets

`tokenizer.json` is not committed: it is 6.07 MB, MIT licensed, and belongs to
the model rather than to this crate. Fetch the one matching your serving model.

| directory | model | upstream |
| --- | --- | --- |
| `v4/` | DeepSeek-V4-Pro | <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro> |
| `v41/` | DeepSeek-V4-Flash-Vision-Exp | <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp> |

Both are redistributed in <https://github.com/deepseek-ai/deepseek-recipe> under
`static/tokenizers/`, with the local modifications that align the image special
token with that repository's prompt encodings. If you only count text, either
copy works; if you also render prompts with `deepseek-recipe-encoding`, use the
matching one.

```powershell
# from the workspace root
.\assets\fetch-tokenizer.ps1 -Variant v41
```

Then point the estimator at it:

```rust
let tokenizer = HfTokenizer::from_file("assets/tokenizer.json")?;
```

Any `tokenizer.json` in this format works, not just DeepSeek's — see the
`billion_context::tokenizer` module docs for the other model families. Tests look for
`../assets/tokenizer.json` relative to `billion-context/`, or at `$env:ACP_TOKENIZER_JSON`.

