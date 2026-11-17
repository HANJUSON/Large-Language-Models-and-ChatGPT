param([string]$Round = 'r1')
$host.UI.RawUI.WindowTitle = "dev-case-review $Round"
$prompt = "Round: $Round. Read data/cases/dev/prompts/review_v1.md and follow it exactly. Process every batch data/cases/dev/view/llm_review/in/$Round-*.md in order."
claude $prompt --model claude-opus-5-5 --name "dev-case-review-$Round" --allowedTools 'Read,Glob,Write(./data/cases/dev/view/llm_review/out/**),Edit(./data/cases/dev/view/llm_review/out/**)' --disallowedTools 'Bash,PowerShell,WebFetch,WebSearch,Agent,Task,Read(./datasets/**),Read(./results/**),Read(./runs/**),Read(./.omc/**),Read(./data/processed/**)'
