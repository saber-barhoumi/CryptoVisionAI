# Choose your ML framework
Write-Host "`n"
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "CHOOSE YOUR ML FRAMEWORK" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

Write-Host "We have TWO options for CNN training:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1️⃣  TensorFlow/Keras" -ForegroundColor Green
Write-Host "    ✅ Industry standard" -ForegroundColor Gray
Write-Host "    ✅ Excellent documentation" -ForegroundColor Gray
Write-Host "    ⚠️  Can have Windows installation issues" -ForegroundColor Gray
Write-Host ""
Write-Host "2️⃣  PyTorch" -ForegroundColor Green
Write-Host "    ✅ Better Windows compatibility" -ForegroundColor Gray
Write-Host "    ✅ More Pythonic and flexible" -ForegroundColor Gray
Write-Host "    ✅ Faster installation" -ForegroundColor Gray
Write-Host ""

Write-Host "Which do you want to try?" -ForegroundColor Cyan
Write-Host "[1] TensorFlow (run fix-tensorflow.ps1)" -ForegroundColor White
Write-Host "[2] PyTorch (run install-pytorch.ps1) - RECOMMENDED" -ForegroundColor Yellow
Write-Host ""

$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host "`n🚀 Running TensorFlow installation fix..." -ForegroundColor Green
    .\fix-tensorflow.ps1
} elseif ($choice -eq "2") {
    Write-Host "`n🚀 Running PyTorch installation..." -ForegroundColor Green
    .\install-pytorch.ps1
    
    Write-Host "`n📝 Creating PyTorch training scripts..." -ForegroundColor Cyan
    Write-Host "   Models will be compatible with your dataset!" -ForegroundColor Gray
} else {
    Write-Host "`n❌ Invalid choice" -ForegroundColor Red
}

Write-Host ""
