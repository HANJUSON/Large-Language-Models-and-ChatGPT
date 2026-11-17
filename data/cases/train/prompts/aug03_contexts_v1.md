# AUG-03 context-passage generation prompt v1 (training augmentation, 2026-11-17)

Purpose: benign context passages for TRAINING augmentation of a prompt-injection classifier (README 5.3, condition C). Each passage is placed before and/or after a training sentence (injection or benign) so that surrounding ordinary content carries no information about the label. These passages are used for training only. They must not overlap with the development or final evaluation passages (README DATA-07), so the kinds below differ from the development list.

Write 240 English passages, numbered 1 to 240: items 1 to 160 are SHORT (15 to 35 words), items 161 to 240 are LONG (50 to 90 words).

1. Each passage is ordinary everyday content that could appear around a message in a chat or a pasted document. Cycle through these kinds in order (item k uses kind number ((k - 1) mod 10) + 1): 1 hobby forum post, 2 internal newsletter item, 3 lab or experiment log entry, 4 event announcement, 5 job posting excerpt, 6 podcast or video show notes, 7 appliance manual excerpt, 8 sports match report, 9 book or film blurb, 10 neighbourhood or club notice.
2. The passage must be neutral: it must not mention AI, assistants, chatbots, prompts, instructions to a model, rules, policies, security, hacking, passwords, jailbreaks or anything about following or ignoring directions; it must not contain questions or requests addressed to the reader; it must not contain imperative sentences addressed to the reader (for kind 7 describe what the appliance does instead of telling the reader what to do).
3. Plain English prose in a single paragraph, no line breaks, no bullet lists, no markdown, no URLs or e-mail addresses, no names of real people. Use ASCII characters only (straight quotes and apostrophes, a plain hyphen instead of a dash, no accented letters, no symbols such as the degree sign).
4. Vary topics and wording; no two passages may share the same first five words.
5. Do not write an e-mail, meeting notes, a product review, a recipe step, a travel plan, study notes, a customer-support ticket, a news-style paragraph, a code comment or changelog line, or a diary entry (those kinds belong to the development set).

Output only JSON in this form:
{"items": [{"n": 1, "length": "short", "kind": "hobby forum post", "text": "..."}, ...]}
