# AUG-04 benign-input generation prompt v1 (training augmentation, 2026-11-17)

Purpose: BENIGN (label 0) training inputs for a prompt-injection classifier (README 5.3, condition C). Stage 3 showed that the classifier flags benign inputs that merely talk about AI security or use attack-related words. Every input here is benign but uses such vocabulary. These inputs are used for training only; the families below are defined by situation and task format and are different from the development topic list (README DATA-07).

Write 300 English user inputs, numbered 1 to 300, in six families of 50 (family b1 = items 1 to 50, b2 = 51 to 100, b3 = 101 to 150, b4 = 151 to 200, b5 = 201 to 250, b6 = 251 to 300).

Families:
- b1 intent_criteria: a question about how to tell a harmless question from a manipulation attempt: criteria, signals, checklists or borderline cases, asked by a moderator, a support agent, a classifier designer or a curious user.
- b2 handling_meta: a question about how an assistant, a filter or a classifier should treat inputs that contain hidden instructions, embedded commands or requests about system prompts, or about how such attacks work in general terms.
- b3 it_operations: a statement or question about software or IT operations that has nothing to do with AI assistants but uses words such as system, access, permission, override, admin, bypass, ignore, command, root, privilege, policy or restricted (databases, deployment, networking, version control, operating systems, access-control audits).
- b4 illustrative_document: a request for a document about prompt injection or AI security (training handout, briefing, blog post, slide outline, FAQ, glossary) that says the examples must be invented, fictional, generic or redacted. It never asks for the assistant's own or any real system prompt.
- b5 labelled_examples: a task that treats an attack-like string as data: label it, annotate it, explain why a detector should flag it, turn it into a detection rule or a test case, or compare two quoted strings. The attack phrase is quoted and is not addressed to the assistant.
- b6 workplace_process: organisational work around AI security: an incident postmortem, an audit checklist, a vendor questionnaire, a staff training plan, bug-bounty triage notes or a compliance review. It asks for structure, wording or a summary, not for an attack.

Rules for every input:
1. It must stay benign: it must not ask the assistant to ignore, override or forget its own instructions, to reveal or summarise its own hidden or system prompt, configuration or rules, to drop its safety rules, to play an unrestricted persona, or to produce a working attack payload against a specific real system.
2. Mentioning, quoting or analysing attack phrases (for example "ignore previous instructions", "system prompt", "jailbreak", "developer mode") is allowed when the purpose is clearly analysis, teaching, labelling or defence.
3. Length 40 to 200 characters, one or two sentences, no line breaks, plain English, ordinary casing and spacing (no deliberate obfuscation). Use ASCII characters only (straight quotes and apostrophes, a plain hyphen instead of a dash).
4. Vary wording and structure: no two inputs may start with the same first four words, and do not reuse a sentence template with only a swapped noun. Mix questions, requests and statements.
5. Do not copy sentences from any public prompt-injection dataset.

Output only JSON in this form:
{"items": [{"n": 1, "family": "b1", "text": "..."}, ...]}
