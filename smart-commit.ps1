# Smart Commit Generator
# Analyzes your changes and creates professional commit messages
# Makes your GitHub activity look impressive!

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Smart Commit Generator" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check for changes
$hasChanges = git status --porcelain

if (-not $hasChanges) {
    Write-Host "No changes detected." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "1. Make some code changes first" -ForegroundColor White
    Write-Host "2. Run .\daily-commit.ps1 for activity commit" -ForegroundColor White
    Write-Host ""
    exit
}

# Stage changes
git add .

# Analyze changes
Write-Host "Analyzing changes..." -ForegroundColor Yellow
Write-Host ""

$changedFiles = git diff --cached --name-status
$stats = git diff --cached --stat

# Count file types
$pythonCount = ($changedFiles | Where-Object { $_ -match "\.py" }).Count
$docCount = ($changedFiles | Where-Object { $_ -match "\.md" }).Count
$configCount = ($changedFiles | Where-Object { $_ -match "config|json|yml" }).Count
$scriptCount = ($changedFiles | Where-Object { $_ -match "\.ps1|\.sh" }).Count

# Smart commit message generation
$commitTypes = @()
$commitDetails = @()

if ($pythonCount -gt 0) {
    $commitTypes += "🐍 Python"
    $commitDetails += "- Enhanced $pythonCount Python module(s)"
}

if ($docCount -gt 0) {
    $commitTypes += "📚 Docs"
    $commitDetails += "- Updated $docCount documentation file(s)"
}

if ($configCount -gt 0) {
    $commitTypes += "⚙️ Config"
    $commitDetails += "- Modified $configCount configuration file(s)"
}

if ($scriptCount -gt 0) {
    $commitTypes += "🔧 Scripts"
    $commitDetails += "- Improved $scriptCount automation script(s)"
}

# Generate commit title
$commitTitle = if ($pythonCount -gt 2) {
    "feat: Major code improvements and refactoring"
} elseif ($docCount -gt 0) {
    "docs: Update documentation and examples"
} elseif ($pythonCount -gt 0) {
    "feat: Enhance core functionality"
} else {
    "chore: Project maintenance and updates"
}

# Build full message
$fullMessage = "$commitTitle`n`n"
$fullMessage += "Changes: " + ($commitTypes -join " | ") + "`n`n"
$fullMessage += $commitDetails -join "`n"
$fullMessage += "`n`n📊 Stats:`n$stats"
$fullMessage += "`n`n✨ Continuous improvement of CryptoVisionAI"

# Show commit
Write-Host "Commit Preview:" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Gray
Write-Host $fullMessage -ForegroundColor White
Write-Host "----------------------------------------" -ForegroundColor Gray
Write-Host ""

# Confirm
Write-Host "Commit and push? (Y/n): " -ForegroundColor Yellow -NoNewline
$confirm = Read-Host

if ($confirm -eq "" -or $confirm -eq "Y" -or $confirm -eq "y") {
    git commit -m $fullMessage
    
    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    git push origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "=========================================" -ForegroundColor Green
        Write-Host "  ✅ Successfully committed!" -ForegroundColor Green
        Write-Host "=========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your GitHub profile just got more impressive! 🌟" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "View: https://github.com/saber-barhoumi/CryptoVisionAI" -ForegroundColor Blue
        Write-Host ""
    }
} else {
    Write-Host ""
    Write-Host "Commit cancelled." -ForegroundColor Yellow
    git reset HEAD
    Write-Host ""
}
