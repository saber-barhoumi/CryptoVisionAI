# Install PyTorch (Better Windows compatibility than TensorFlow)
# PyTorch often installs more smoothly on Windows

Write-Host "`n" -NoNewline
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "PYTORCH INSTALLATION (TENSORFLOW ALTERNATIVE)" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

Write-Host "📦 Installing PyTorch + torchvision..." -ForegroundColor Yellow
Write-Host "  This is a reliable alternative to TensorFlow" -ForegroundColor Gray
Write-Host "  Works great for CNN training!" -ForegroundColor Gray
Write-Host ""

# Install PyTorch (CPU version for compatibility)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "="*80 -ForegroundColor Green
    Write-Host "✅ PYTORCH INSTALLED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "="*80 -ForegroundColor Green
    Write-Host ""
    
    # Verify installation
    Write-Host "🧪 Verifying installation..." -ForegroundColor Cyan
    python -c "import torch; print('✅ PyTorch version:', torch.__version__); print('✅ CUDA available:', torch.cuda.is_available())"
    
    Write-Host ""
    Write-Host "🎯 Next step: Use PyTorch training scripts" -ForegroundColor Cyan
    Write-Host "   We'll create PyTorch versions of the CNN models" -ForegroundColor Gray
    Write-Host ""
    
} else {
    Write-Host ""
    Write-Host "❌ Installation failed" -ForegroundColor Red
    Write-Host ""
}
