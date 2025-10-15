# 🚀 CryptoVisionAI - Training Launcher
# Quick start script for CNN training

Clear-Host
Write-Host "`n"
Write-Host "╔════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                            ║" -ForegroundColor Cyan
Write-Host "║                 🚀 CryptoVisionAI - CNN Training Pipeline                 ║" -ForegroundColor Cyan
Write-Host "║                                                                            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "📊 Dataset Status:" -ForegroundColor Yellow
Write-Host "   ✅ 254,424 candlestick images ready" -ForegroundColor Green
Write-Host "   ✅ 70/15/15 train/val/test split" -ForegroundColor Green
Write-Host "   ✅ Balanced distribution (Buy/Sell/Hold)" -ForegroundColor Green
Write-Host ""

Write-Host "🎯 What would you like to do?" -ForegroundColor Cyan
Write-Host ""
Write-Host "  [1] 🔥 Start Training (PyTorch - Recommended)" -ForegroundColor Yellow
Write-Host "  [2] 🔧 Install PyTorch" -ForegroundColor White
Write-Host "  [3] 🔧 Fix TensorFlow Installation" -ForegroundColor White
Write-Host "  [4] 📚 View Quick Start Guide" -ForegroundColor White
Write-Host "  [5] 📋 View Full Training Guide" -ForegroundColor White
Write-Host "  [6] 📊 View Dataset Statistics" -ForegroundColor White
Write-Host "  [7] 💾 Commit & Push to GitHub" -ForegroundColor White
Write-Host "  [0] ❌ Exit" -ForegroundColor Red
Write-Host ""

$choice = Read-Host "Enter your choice"

switch ($choice) {
    "1" {
        Write-Host "`n🔥 Checking PyTorch installation..." -ForegroundColor Yellow
        $torchInstalled = python -c "import torch; print('OK')" 2>$null
        
        if ($torchInstalled -eq "OK") {
            Write-Host "✅ PyTorch is installed!" -ForegroundColor Green
            Write-Host "`n🚀 Starting training..." -ForegroundColor Cyan
            Write-Host "   This will take 2-6 hours depending on your hardware" -ForegroundColor Gray
            Write-Host ""
            
            cd models
            python train_pytorch.py
        } else {
            Write-Host "❌ PyTorch not installed!" -ForegroundColor Red
            Write-Host ""
            Write-Host "📦 Would you like to install it now? [Y/n]: " -ForegroundColor Yellow -NoNewline
            $install = Read-Host
            
            if ($install -eq "" -or $install -eq "Y" -or $install -eq "y") {
                cd ..
                .\install-pytorch.ps1
            }
        }
    }
    
    "2" {
        Write-Host "`n📦 Installing PyTorch..." -ForegroundColor Cyan
        .\install-pytorch.ps1
    }
    
    "3" {
        Write-Host "`n🔧 Fixing TensorFlow installation..." -ForegroundColor Cyan
        .\fix-tensorflow.ps1
    }
    
    "4" {
        Write-Host "`n📚 Opening Quick Start Guide..." -ForegroundColor Cyan
        if (Get-Command code -ErrorAction SilentlyContinue) {
            code QUICKSTART.md
        } else {
            notepad QUICKSTART.md
        }
    }
    
    "5" {
        Write-Host "`n📋 Opening Training Guide..." -ForegroundColor Cyan
        if (Get-Command code -ErrorAction SilentlyContinue) {
            code TRAINING_GUIDE.md
        } else {
            notepad TRAINING_GUIDE.md
        }
    }
    
    "6" {
        Write-Host "`n📊 Dataset Statistics:" -ForegroundColor Cyan
        Write-Host ""
        Get-Content dataset\dataset_stats.json
        Write-Host ""
    }
    
    "7" {
        Write-Host "`n💾 Committing changes to GitHub..." -ForegroundColor Cyan
        git add .
        $msg = Read-Host "Commit message (or press Enter for auto-message)"
        
        if ($msg -eq "") {
            git commit -m "✨ Update CryptoVisionAI project"
        } else {
            git commit -m $msg
        }
        
        git push
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
            Write-Host "🔗 https://github.com/saber-barhoumi/CryptoVisionAI" -ForegroundColor Blue
        }
    }
    
    "0" {
        Write-Host "`n👋 Goodbye!" -ForegroundColor Cyan
    }
    
    default {
        Write-Host "`n❌ Invalid choice!" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
