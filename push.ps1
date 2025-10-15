# Push to GitHub - Simple Script

Write-Host ""
Write-Host "========================================="
Write-Host "  Pushing to GitHub"
Write-Host "========================================="
Write-Host ""

# Initialize git if needed
if (-not (Test-Path .git)) {
    git init
    Write-Host "Git initialized"
    Write-Host ""
}

# Add all files (gitignore will exclude large files)
Write-Host "Adding files..."
git add .

# Show status
Write-Host ""
Write-Host "Files to commit:"
git status --short

# Commit
Write-Host ""
Write-Host "Committing..."
git commit -m "Initial commit: CryptoVisionAI

- Complete Binance data to CNN images pipeline
- 260,000+ balanced images (26% Buy, 27% Sell, 47% Hold)  
- Optimized 0.15% threshold for better trading signals
- Comprehensive documentation
- Ready for CNN training and backtesting

Dataset source: https://www.kaggle.com/datasets/jorijnsmit/binance-full-history"

# Set main branch
git branch -M main

# Add remote
$remote = "https://github.com/saber-barhoumi/CryptoVisionAI.git"

try {
    git remote add origin $remote
} catch {
    git remote set-url origin $remote
}

Write-Host ""
Write-Host "Remote set: $remote"

# Push
Write-Host ""
Write-Host "Pushing to GitHub..."
Write-Host "(Enter your GitHub credentials if prompted)"
Write-Host ""

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================="
    Write-Host "  Successfully pushed to GitHub!"
    Write-Host "========================================="
    Write-Host ""
    Write-Host "Repository: $remote"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Visit: https://github.com/saber-barhoumi/CryptoVisionAI"
    Write-Host "2. Add repository description"
    Write-Host "3. Add topics: machine-learning, cnn, cryptocurrency, trading"
    Write-Host "4. Star your own repo!"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Push failed!"
    Write-Host "Try: git push -f origin main"
    Write-Host ""
}
