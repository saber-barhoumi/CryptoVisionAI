# QUICK CHECK - Dataset Balance Verification
$buy = @(Get-ChildItem "C:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Balanced\Buy" -File -Recurse -EA Silent).Count
$sell = @(Get-ChildItem "C:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Balanced\Sell" -File -Recurse -EA Silent).Count
$hold = @(Get-ChildItem "C:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Balanced\Hold" -File -Recurse -EA Silent).Count
$total = $buy + $sell + $hold

Write-Host ""
Write-Host "==========================================="
Write-Host "         REBALANCING STATUS"
Write-Host "==========================================="

if($total -gt 0) {
    $buyPct = [math]::Round($buy/$total*100,1)
    $sellPct = [math]::Round($sell/$total*100,1)
    $holdPct = [math]::Round($hold/$total*100,1)
    $progress = [math]::Round(($total/258500)*100,1)
    
    Write-Host ""
    Write-Host "Progress: $progress percent ($total / 258,500 images)"
    Write-Host ""
    Write-Host "Current Distribution:"
    Write-Host "  Buy:  $buy ($buyPct percent)"
    Write-Host "  Sell: $sell ($sellPct percent)"
    Write-Host "  Hold: $hold ($holdPct percent)"
    Write-Host ""
    Write-Host "Target: 30 percent Buy | 30 percent Sell | 40 percent Hold"
} else {
    Write-Host ""
    Write-Host "Processing... No images yet."
    Write-Host "(First images take 2-3 minutes)"
}

Write-Host "==========================================="
Write-Host ""
