# CryptoVisionAI - Push to GitHub Script
# This script pushes the essential files to GitHub (without large datasets)

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "  Pushing CryptoVisionAI to GitHub" -ForegroundColor Yellow
Write-Host "=========================================`n" -ForegroundColor Cyan

# Check if git is initialized
if (-not (Test-Path .git)) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✓ Git initialized" -ForegroundColor Green
}

# Replace old README with new one
if (Test-Path "README_NEW.md") {
    Copy-Item "README_NEW.md" "README.md" -Force
    Remove-Item "README_NEW.md"
    Write-Host "✓ README updated" -ForegroundColor Green
}

# Add essential files
Write-Host "`nAdding files to Git..." -ForegroundColor Yellow

git add README.md
git add LICENSE
git add .gitignore
git add .gitattributes
git add requirements.txt
git add config.py

# Add Python scripts
git add explore_data.py
git add calculate_stats.py
git add data_to_images.py
git add reprocess_balanced.py
git add view_images.py

# Add scripts
git add check_balance.ps1
git add check_progress.ps1

# Add documentation
git add docs/

Write-Host "✓ Files added" -ForegroundColor Green

# Show what will be committed
Write-Host "`nFiles to be committed:" -ForegroundColor Cyan
git status --short

# Commit
Write-Host "`nCommitting changes..." -ForegroundColor Yellow
git commit -m "Initial commit: CryptoVisionAI - CNN-based crypto trading with balanced dataset

- Complete pipeline for Binance data to CNN images
- 260,000+ balanced images (26% Buy, 27% Sell, 47% Hold)
- Optimized 0.15% threshold for better signals
- Comprehensive documentation and guides
- Ready for CNN training and backtesting"

Write-Host "✓ Changes committed" -ForegroundColor Green

# Set branch to main
git branch -M main
Write-Host "✓ Branch set to main" -ForegroundColor Green

# Add remote
$remote = "https://github.com/saber-barhoumi/CryptoVisionAI.git"
Write-Host "`nAdding remote origin..." -ForegroundColor Yellow

try {
    git remote add origin $remote 2>$null
} catch {
    Write-Host "Remote already exists, updating..." -ForegroundColor Gray
    git remote set-url origin $remote
}

Write-Host "✓ Remote added: $remote" -ForegroundColor Green

# Push to GitHub
Write-Host "`nPushing to GitHub..." -ForegroundColor Yellow
Write-Host "(You may need to enter your credentials)" -ForegroundColor Gray

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================="-ForegroundColor Cyan
    Write-Host "  ✅ Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "=========================================`n" -ForegroundColor Cyan
    Write-Host "Repository: $remote" -ForegroundColor Cyan
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "1. Visit your repository on GitHub" -ForegroundColor White
    Write-Host "2. Add a description and topics" -ForegroundColor White
    Write-Host "3. Enable GitHub Pages (optional)" -ForegroundColor White
    Write-Host "4. Share with the community!`n" -ForegroundColor White
} else {
    Write-Host "`n❌ Push failed!" -ForegroundColor Red
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "- Check your GitHub credentials" -ForegroundColor White
    Write-Host "- Make sure you have push access" -ForegroundColor White
    Write-Host "- Try: git push -f origin main (if needed)`n" -ForegroundColor White
}
