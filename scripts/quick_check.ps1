# QUICK CHECK - Run this anytime!
$buy = @(Get-ChildItem "C:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Balanced\Buy" -File -Recurse -EA Silent).Count
$sell = @(Get-ChildItem "C:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Balanced\Sell" -File -Recurse -EA Silent).Count
$hold = @(Get-ChildItem "C:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Balanced\Hold" -File -Recurse -EA Silent).Count
$total = $buy + $sell + $hold

Write-Host "`n===========================================" -ForegroundColor Cyan
Write-Host "         REBALANCING STATUS" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Cyan

if($total -gt 0) {
    $buyPct = [math]::Round($buy/$total*100,1)
    $sellPct = [math]::Round($sell/$total*100,1)
    $holdPct = [math]::Round($hold/$total*100,1)
    $progress = [math]::Round(($total/258500)*100,1)
    
    Write-Host "`nProgress: $progress% ($total / 258,500 images)" -ForegroundColor White
    Write-Host "`nCurrent Distribution:" -ForegroundColor White
    Write-Host "  🟢 Buy:  $buy ($buyPct%)" -ForegroundColor Green
    Write-Host "  🔴 Sell: $sell ($sellPct%)" -ForegroundColor Red
    Write-Host "  🟡 Hold: $hold ($holdPct%)" -ForegroundColor Yellow
    Write-Host "`nTarget: 30% Buy | 30% Sell | 40% Hold" -ForegroundColor Gray
} else {
    Write-Host "`n⏳ Processing... No images yet." -ForegroundColor Yellow
    Write-Host "   (First images take 2-3 minutes)" -ForegroundColor Gray
}

Write-Host "===========================================`n" -ForegroundColor Cyan
