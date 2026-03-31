---
name: summarize
description: Summarize a URL or text content when the user asks for a summary
---

# Summarize

Fetch and summarize the given URL or text. Structure the output as:

1. **TL;DR** — one sentence
2. **Key Points** — 3-5 bullet points
3. **Takeaway** — one actionable conclusion

Be concise. Avoid filler. If the content is in a foreign language, summarize in the same language as the user's request.

## Script

For bulk summarization from a file list:

```bash
python3 {baseDir}/scripts/summarize.py --input urls.txt
```
