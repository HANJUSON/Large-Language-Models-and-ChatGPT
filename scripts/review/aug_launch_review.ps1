param([string]$Round = 'a1')
$host.UI.RawUI.WindowTitle = "aug-review $Round"
$prompt = "Round: $Round. Read data/cases/train/prompts/aug_review_v1.md and follow it exactly. Process every batch data/cases/train/view/llm_review/in/$Round-*.md in order."
claude $prompt --model claude-opus-5-5 --name "aug-review-$Round" --allowedTools 'Read,Glob,Write(./data/cases/train/view/llm_review/out/**),Edit(./data/cases/train/view/llm_review/out/**)' --disallowedTools 'Bash,PowerShell,WebFetch,WebSearch,Agent,Task,Read(./datasets/**),Read(./results/**),Read(./runs/**),Read(./.omc/**),Read(./data/processed/**),Read(./data/cases/dev/**),Read(./data/cases/final/**),Read(./data/external/**),Read(./data/error_review/**)'
