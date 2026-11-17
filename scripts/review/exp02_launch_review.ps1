param([string]$Round = 'e1')
$host.UI.RawUI.WindowTitle = "exp02-review $Round"
$prompt = "Round: $Round. Read data/error_review/prompts/exp02_review_v1.md and follow it exactly. Process every batch data/error_review/view/in/$Round-*.md in order."
claude $prompt --model claude-opus-5-5 --name "exp02-review-$Round" --allowedTools 'Read,Glob,Write(./data/error_review/view/out/**),Edit(./data/error_review/view/out/**)' --disallowedTools 'Bash,PowerShell,WebFetch,WebSearch,Agent,Task,Read(./datasets/**),Read(./results/**),Read(./runs/**),Read(./.omc/**),Read(./data/processed/**),Read(./data/cases/**)'
