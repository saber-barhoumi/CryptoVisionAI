# Project Cleanup Script
# Organizes files before pushing to GitHub

Write-Host ""
Write-Host "========================================="
Write-Host "  Cleaning CryptoVisionAI Project"
Write-Host "========================================="
Write-Host ""

# Create directories
$directories = @("data_preparation", "scripts", "models", "utils", "archive")

Write-Host "Creating organized structure..."
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created $dir/"
    }
}

# Move data preparation files
Write-Host ""
Write-Host "Organizing Python files..."

if (Test-Path "explore_data.py") {
    Move-Item "explore_data.py" "data_preparation\" -Force
    Write-Host "  Moved explore_data.py"
}

if (Test-Path "calculate_stats.py") {
    Move-Item "calculate_stats.py" "data_preparation\" -Force
    Write-Host "  Moved calculate_stats.py"
}

if (Test-Path "reprocess_balanced.py") {
    Move-Item "reprocess_balanced.py" "data_preparation\" -Force
    Write-Host "  Moved reprocess_balanced.py"
}

# Move scripts
if (Test-Path "view_images.py") {
    Move-Item "view_images.py" "scripts\" -Force
    Write-Host "  Moved view_images.py"
}

if (Test-Path "check_balance.ps1") {
    Move-Item "check_balance.ps1" "scripts\" -Force
    Write-Host "  Moved check_balance.ps1"
}

if (Test-Path "check_progress.ps1") {
    Move-Item "check_progress.ps1" "scripts\" -Force
    Write-Host "  Moved check_progress.ps1"
}

if (Test-Path "quick_check.ps1") {
    Move-Item "quick_check.ps1" "scripts\" -Force
    Write-Host "  Moved quick_check.ps1"
}

# Archive old files
Write-Host ""
Write-Host "Archiving old files..."

$oldFiles = @(
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

foreach ($file in $oldFiles) {
    if (Test-Path $file) {
        Move-Item $file "archive\" -Force
        Write-Host "  Archived $file"
    }
}

# Use new README
if (Test-Path "README_NEW.md") {
    Copy-Item "README_NEW.md" "README.md" -Force
    Remove-Item "README_NEW.md"
    Write-Host ""
    Write-Host "Updated README.md"
}

# Create package files
Write-Host ""
Write-Host "Creating Python packages..."

$packages = @("data_preparation", "models", "utils")
foreach ($pkg in $packages) {
    $initFile = "$pkg\__init__.py"
    if (-not (Test-Path $initFile)) {
        New-Item -ItemType File -Path $initFile -Force | Out-Null
        Set-Content $initFile "# $pkg package"
        Write-Host "  Created $pkg/__init__.py"
    }
}

# Remove cache
if (Test-Path "__pycache__") {
    Remove-Item "__pycache__" -Recurse -Force
    Write-Host ""
    Write-Host "Removed __pycache__"
}

Write-Host ""
Write-Host "========================================="
Write-Host "  Project Cleaned Successfully!"
Write-Host "========================================="
Write-Host ""
Write-Host "Final Structure:"
Write-Host "  data_preparation/  - Data processing"
Write-Host "  scripts/           - Helper scripts"
Write-Host "  models/            - CNN models"
Write-Host "  utils/             - Utilities"
Write-Host "  docs/              - Documentation"
Write-Host "  archive/           - Old files"
Write-Host ""
Write-Host "Ready for GitHub!"
Write-Host "Run: .\push_to_github.ps1"
Write-Host ""
