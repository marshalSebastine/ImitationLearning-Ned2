# Links this repository's shared Claude memory and skills into Claude Code's own folders (Windows).
# Run once per laptop after cloning, and again whenever the repository folder moves:
#     powershell -ExecutionPolicy Bypass -File setup_claude.ps1
# Uses directory junctions, which need no administrator rights.
$ErrorActionPreference = "Stop"

$ProjectDirectory = $PSScriptRoot
$ClaudeDirectory = if ($env:CLAUDE_CONFIG_DIR) { $env:CLAUDE_CONFIG_DIR } else { Join-Path $HOME ".claude" }
# Claude Code names a project's folder after its path, with every character that is not a letter or digit replaced by "-".
$ProjectSlug = $ProjectDirectory -replace "[^A-Za-z0-9]", "-"
$MemoryLink = Join-Path $ClaudeDirectory "projects\$ProjectSlug\memory"
$SharedMemory = Join-Path $ProjectDirectory "claude_memory"

New-Item -ItemType Directory -Force -Path (Split-Path $MemoryLink) | Out-Null
$Existing = Get-Item $MemoryLink -ErrorAction SilentlyContinue
if ($Existing -and -not $Existing.LinkType) {
    # Keep any memory Claude already wrote here: copy files the repository does not have yet, then swap in the link.
    Get-ChildItem $MemoryLink -Filter *.md | Where-Object { -not (Test-Path (Join-Path $SharedMemory $_.Name)) } |
        Copy-Item -Destination $SharedMemory
    Rename-Item $MemoryLink "memory.before_link"
    Write-Host "Old memory folder kept as memory.before_link; new files were copied into claude_memory\."
} elseif ($Existing) {
    $Existing.Delete()
}
New-Item -ItemType Junction -Path $MemoryLink -Target $SharedMemory | Out-Null
Write-Host "Memory:  $MemoryLink -> $SharedMemory"

$SkillsDirectory = Join-Path $ClaudeDirectory "skills"
New-Item -ItemType Directory -Force -Path $SkillsDirectory | Out-Null
foreach ($Skill in Get-ChildItem (Join-Path $ProjectDirectory "claude_skills") -Directory) {
    $SkillLink = Join-Path $SkillsDirectory $Skill.Name
    $OldSkill = Get-Item $SkillLink -ErrorAction SilentlyContinue
    if ($OldSkill) { Rename-Item $SkillLink "$($Skill.Name).before_link" }
    New-Item -ItemType Junction -Path $SkillLink -Target $Skill.FullName | Out-Null
    Write-Host "Skill:   $SkillLink -> $($Skill.FullName)"
}
