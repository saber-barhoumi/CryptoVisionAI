# 🧹 Project Cleanup Script
# Organizes files and removes duplicates before pushing to GitHub

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "  Cleaning CryptoVisionAI Project" -ForegroundColor Yellow
Write-Host "=========================================`n" -ForegroundColor Cyan

# Create organized directory structure
Write-Host "Creating organized structure..." -ForegroundColor Yellow

$directories = @(
    "data_preparation",
    "scripts",
    "models",
    "utils",
    "archive"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✓ Created $dir/" -ForegroundColor Green
    }
}

# Move Python files to proper directories
Write-Host "`nOrganizing Python files..." -ForegroundColor Yellow

# Data preparation scripts
$dataPrepFiles = @(
    "explore_data.py",
    "calculate_stats.py",
    "reprocess_balanced.py"
)

foreach ($file in $dataPrepFiles) {
    if (Test-Path $file) {
        Move-Item $file "data_preparation\" -Force
        Write-Host "  ✓ Moved $file -> data_preparation/" -ForegroundColor Green
    }
}

# Scripts directory
$scriptFiles = @(
    "view_images.py",
    "check_balance.ps1",
    "check_progress.ps1",
    "quick_check.ps1"
)

foreach ($file in $scriptFiles) {
    if (Test-Path $file) {
        Move-Item $file "scripts\" -Force
        Write-Host "  ✓ Moved $file -> scripts/" -ForegroundColor Green
    }
}

# Archive old/duplicate files
Write-Host "`nArchiving old files..." -ForegroundColor Yellow

$archiveFiles = @(
    "data_to_images.py",
    "config_full_processing.py",
    "process_full_dataset.py",
    "DATA_ANALYSIS_RESULTS.md",
    "FINAL_SUMMARY.md",
    "MODERATE_STRATEGY_GUIDE.md",
    "PROCESSING_STATUS.md",
    "PROJECT_SUMMARY.md",
    "QUICKSTART.md",
    "REBALANCING_STATUS.md",
    "START_PROCESSING.bat",
    "comparison_buy_sell_hold.png"
)

foreach ($file in $archiveFiles) {
    if (Test-Path $file) {
        Move-Item $file "archive\" -Force
        Write-Host "  ✓ Archived $file" -ForegroundColor Gray
    }
}

# Use the new README
if (Test-Path "README_NEW.md") {
    Copy-Item "README_NEW.md" "README.md" -Force
    Remove-Item "README_NEW.md"
    Write-Host "`n✓ Updated README.md" -ForegroundColor Green
}

# Create __init__.py files for Python packages
Write-Host "`nCreating Python package files..." -ForegroundColor Yellow

$packages = @("data_preparation", "models", "utils")
foreach ($pkg in $packages) {
    $initFile = Join-Path $pkg "__init__.py"
    if (-not (Test-Path $initFile)) {
        New-Item -ItemType File -Path $initFile -Force | Out-Null
        Add-Content $initFile "# $pkg package"
        Write-Host "  ✓ Created $pkg/__init__.py" -ForegroundColor Green
    }
}

# Remove Python cache
if (Test-Path "__pycache__") {
    Remove-Item "__pycache__" -Recurse -Force
    Write-Host "`n✓ Removed __pycache__" -ForegroundColor Green
}

# Show final structure
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Project Cleaned Successfully!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Final Structure:" -ForegroundColor Yellow
Write-Host ""
Write-Host "data_preparation/  - Data processing scripts" -ForegroundColor White
Write-Host "scripts/           - Helper scripts" -ForegroundColor White
Write-Host "models/            - CNN models (ready)" -ForegroundColor White
Write-Host "utils/             - Utilities (ready)" -ForegroundColor White
Write-Host "docs/              - Documentation" -ForegroundColor White
Write-Host "archive/           - Old files (not pushed)" -ForegroundColor Gray
Write-Host ""
Write-Host "Files ready for GitHub push!" -ForegroundColor Green
Write-Host "Run: .\push_to_github.ps1" -ForegroundColor Cyan
Write-Host ""
