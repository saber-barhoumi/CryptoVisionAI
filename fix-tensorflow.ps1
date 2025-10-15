# Fix TensorFlow Installation Issues on Windows
# This script closes Python processes and cleans up locked files

Write-Host "`n" -NoNewline
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "TENSORFLOW INSTALLATION FIX" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

# Step 1: Close all Python processes
Write-Host "🔧 Step 1: Closing Python processes..." -ForegroundColor Yellow
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "  Found $($pythonProcesses.Count) Python process(es)" -ForegroundColor Gray
    $pythonProcesses | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-Host "  ✅ Closed Python processes" -ForegroundColor Green
} else {
    Write-Host "  ✅ No Python processes running" -ForegroundColor Green
}

# Step 2: Clean up pip cache
Write-Host "`n🧹 Step 2: Cleaning pip cache..." -ForegroundColor Yellow
python -m pip cache purge 2>$null
Write-Host "  ✅ Cache cleared" -ForegroundColor Green

# Step 3: Install TensorFlow with --no-cache-dir
Write-Host "`n📦 Step 3: Installing TensorFlow (fresh install)..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes..." -ForegroundColor Gray
Write-Host ""

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir tensorflow tensorboard scikit-learn

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "="*80 -ForegroundColor Green
    Write-Host "✅ TENSORFLOW INSTALLED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "="*80 -ForegroundColor Green
    Write-Host ""
    
    # Verify installation
    Write-Host "🧪 Verifying installation..." -ForegroundColor Cyan
    python -c "import tensorflow as tf; print('✅ TensorFlow version:', tf.__version__); print('✅ GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
    
} else {
    Write-Host ""
    Write-Host "="*80 -ForegroundColor Red
    Write-Host "⚠️  TENSORFLOW INSTALLATION FAILED" -ForegroundColor Red
    Write-Host "="*80 -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Alternative solution: Use PyTorch instead" -ForegroundColor Yellow
    Write-Host "   Run: .\install-pytorch.ps1" -ForegroundColor Cyan
    Write-Host ""
}
