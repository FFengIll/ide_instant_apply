# Note

## How it works
Below is a usage of `Speculative Decoding`.  
Let us explain the code step by step.  

```python
draft: list[int] = []
target_idx = target_edit_dist.index(min(target_edit_dist))
if target_idx > 0 and token == target_tokens[target_idx - 1]:
    draft = target_tokens[target_idx:]
else:
    edit_idx = edit_edit_dist.index(min(edit_edit_dist))
    if edit_idx > 0 and token == edit_tokens[edit_idx - 1]:
        draft = edit_tokens[edit_idx:]
    else:
        # to recover quickly from the LLM deleting a large chunk of text
        # (otherwise keeps drafting from pre-deletion position due to edit dist)
        target_idx = min(
            (i for i, t in enumerate(target_tokens) if t == token),
            default=0,
            key=lambda i: target_edit_dist[i + 1],
        )
        draft = target_tokens[target_idx + 1 :]
```

`draft`: used to store the draft tokens  
`target_edit_dist`: used to store the index (I keep the original name) of the target tokens  

check next token from `target` (the original code), then `edit` (the edited code),  
use one of them if matched.  

> target - the original code  
> edit - insert / modify

if not matched, move idx (smallest) to position until mismatch `target`.

> mismatch - delete

## More details
Pls learn `Speculative Decoding`.

A common explanation:
- For decoder model, it predicts next-one token. if we can guess next-k token and verify them in parallele, it becomes quick.
- When we talk about code edit, many code is kept as original, so we can guess next-k and confirm it's right in most case which make token generation more fast.
