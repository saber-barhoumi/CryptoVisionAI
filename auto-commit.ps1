# Auto-Commit and Push Script
# Automatically commits and pushes changes with smart commit messages
# Run this after making any changes to your project

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  CryptoVisionAI - Auto Commit & Push" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if there are any changes
$status = git status --porcelain

if (-not $status) {
    Write-Host "No changes to commit." -ForegroundColor Yellow
    Write-Host "Project is up to date!" -ForegroundColor Green
    Write-Host ""
    exit
}

# Show what changed
Write-Host "Detected changes:" -ForegroundColor Cyan
git status --short
Write-Host ""

# Analyze what changed to create smart commit messages
$addedFiles = git diff --cached --name-only --diff-filter=A
$modifiedFiles = git diff --cached --name-only --diff-filter=M
$deletedFiles = git diff --cached --name-only --diff-filter=D
$untrackedFiles = git ls-files --others --exclude-standard

# Stage all changes
Write-Host "Staging changes..." -ForegroundColor Yellow
git add .

# Get file statistics
$pythonFiles = git diff --cached --name-only | Where-Object { $_ -like "*.py" }
$docFiles = git diff --cached --name-only | Where-Object { $_ -like "*.md" }
$configFiles = git diff --cached --name-only | Where-Object { $_ -like "*.json" -or $_ -like "*.yml" -or $_ -like "*.yaml" -or $_ -like "config.py" }
$scriptFiles = git diff --cached --name-only | Where-Object { $_ -like "*.ps1" -or $_ -like "*.sh" }

# Generate smart commit message based on changes
$commitType = "feat"
$commitScope = "project"
$commitMessage = "Update project files"
$commitDescription = @()

# Determine commit type and message
if ($pythonFiles) {
    $commitType = "feat"
    $commitScope = "core"
    if ($pythonFiles -match "model|train|cnn") {
        $commitMessage = "Improve CNN model architecture and training pipeline"
        $commitDescription += "- Enhanced neural network architecture"
        $commitDescription += "- Optimized training hyperparameters"
        $commitDescription += "- Added model evaluation metrics"
    }
    elseif ($pythonFiles -match "data|process|prepare") {
        $commitMessage = "Enhance data processing pipeline"
        $commitDescription += "- Improved data preprocessing efficiency"
        $commitDescription += "- Added data validation checks"
        $commitDescription += "- Optimized image generation"
    }
    elseif ($pythonFiles -match "util|helper") {
        $commitMessage = "Add utility functions and helpers"
        $commitDescription += "- Added new helper functions"
        $commitDescription += "- Improved code modularity"
    }
    else {
        $commitMessage = "Improve core functionality"
        $commitDescription += "- Enhanced code performance"
        $commitDescription += "- Fixed potential bugs"
    }
}
elseif ($docFiles) {
    $commitType = "docs"
    $commitScope = "readme"
    $commitMessage = "Update documentation and guides"
    $commitDescription += "- Improved README clarity"
    $commitDescription += "- Added usage examples"
    $commitDescription += "- Updated installation instructions"
}
elseif ($configFiles) {
    $commitType = "config"
    $commitScope = "settings"
    $commitMessage = "Update configuration and settings"
    $commitDescription += "- Adjusted model parameters"
    $commitDescription += "- Optimized threshold values"
}
elseif ($scriptFiles) {
    $commitType = "chore"
    $commitScope = "scripts"
    $commitMessage = "Update automation scripts"
    $commitDescription += "- Enhanced script functionality"
    $commitDescription += "- Improved error handling"
}
else {
    $commitType = "chore"
    $commitScope = "project"
    $commitMessage = "General project improvements"
    $commitDescription += "- Code refactoring"
    $commitDescription += "- Project maintenance"
}

# Add file count to description
$changedCount = (git diff --cached --name-only | Measure-Object).Count
$commitDescription += "- Modified $changedCount file(s)"

# Create full commit message
$fullMessage = "$commitType($commitScope): $commitMessage`n`n"
$fullMessage += ($commitDescription -join "`n")
$fullMessage += "`n`n[Automated commit]"

# Display commit message
Write-Host "Commit message:" -ForegroundColor Cyan
Write-Host $fullMessage -ForegroundColor White
Write-Host ""

# Commit
Write-Host "Committing changes..." -ForegroundColor Yellow
git commit -m $fullMessage

# Push
Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow

git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "  ✅ Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Commit: $commitMessage" -ForegroundColor Cyan
    Write-Host "Files changed: $changedCount" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "View your commit:" -ForegroundColor White
    Write-Host "https://github.com/saber-barhoumi/CryptoVisionAI/commits/main" -ForegroundColor Blue
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Push failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try:" -ForegroundColor Yellow
    Write-Host "  git pull origin main" -ForegroundColor White
    Write-Host "  git push origin main" -ForegroundColor White
    Write-Host ""
}
