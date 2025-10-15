# LIVE PROGRESS MONITOR
# Run this to check rebalancing progress anytime!

$buy = @(Get-ChildItem "C:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Balanced\Buy" -File -Recurse -ErrorAction SilentlyContinue).Count
$sell = @(Get-ChildItem "C:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Balanced\Sell" -File -Recurse -ErrorAction SilentlyContinue).Count
$hold = @(Get-ChildItem "C:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Balanced\Hold" -File -Recurse -ErrorAction SilentlyContinue).Count

$total = $buy + $sell + $hold

if($total -gt 0) {
    $buyPct = [math]::Round($buy/$total*100,1)
    $sellPct = [math]::Round($sell/$total*100,1)
    $holdPct = [math]::Round($hold/$total*100,1)
    
    # Estimate files processed (avg 500 images per file)
    $filesProcessed = [math]::Round(($total/500))
    $progress = [math]::Round(($filesProcessed/517)*100,1)
    
    # Estimate time remaining
    $estimatedTotal = 258500
    $percentComplete = [math]::Round(($total/$estimatedTotal)*100,1)
    
    Write-Host ""
    Write-Host "===========================================================" -ForegroundColor Cyan
    Write-Host "          REBALANCING PROGRESS - LIVE UPDATE" -ForegroundColor Yellow
    Write-Host "===========================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Files Progress: $filesProcessed / 517 ($progress%)" -ForegroundColor White
    Write-Host "Image Progress: $total / $estimatedTotal ($percentComplete%)" -ForegroundColor White
    Write-Host ""
    Write-Host "Current Distribution:" -ForegroundColor White
    Write-Host "  🟢 Buy:  $buy images ($buyPct%)" -ForegroundColor Green
    Write-Host "  🔴 Sell: $sell images ($sellPct%)" -ForegroundColor Red
    Write-Host "  🟡 Hold: $hold images ($holdPct%)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Target Distribution:" -ForegroundColor Gray
    Write-Host "  🎯 Buy:  30%" -ForegroundColor Gray
    Write-Host "  🎯 Sell: 30%" -ForegroundColor Gray  
    Write-Host "  🎯 Hold: 40%" -ForegroundColor Gray
    Write-Host ""
    
    # Compare to old dataset
    $oldBuy = 16.4
    $oldSell = 16.9
    $oldHold = 66.7
    
    $buyImprovement = $buyPct - $oldBuy
    $sellImprovement = $sellPct - $oldSell
    $holdImprovement = $holdPct - $oldHold
    
    Write-Host "Improvement vs OLD Dataset (0.3% threshold):" -ForegroundColor Cyan
    $buySign = if($buyImprovement -gt 0){"+"}else{""}
    $sellSign = if($sellImprovement -gt 0){"+"}else{""}
    $holdSign = if($holdImprovement -gt 0){"+"}else{""}
    Write-Host "  Buy:  $oldBuy% → $buyPct% ($buySign$([math]::Round($buyImprovement,1))%)" -ForegroundColor Green
    Write-Host "  Sell: $oldSell% → $sellPct% ($sellSign$([math]::Round($sellImprovement,1))%)" -ForegroundColor Green
    Write-Host "  Hold: $oldHold% → $holdPct% ($holdSign$([math]::Round($holdImprovement,1))%)" -ForegroundColor Yellow
    
    Write-Host ""
    Write-Host "===========================================================" -ForegroundColor Cyan
    Write-Host ""
    
} else {
    Write-Host ""
    Write-Host "⏳ Processing just started... No images generated yet." -ForegroundColor Yellow
    Write-Host "   Check again in 5-10 minutes!" -ForegroundColor Gray
    Write-Host ""
}
