# Quick Save Script
# Ultra-fast commit and push for when you're in a hurry
# Usage: .\quick-save.ps1 "your message"

param(
    [string]$message = "Quick update and improvements"
)

Write-Host ""
Write-Host "⚡ Quick Save..." -ForegroundColor Yellow

# Stage all changes
git add .

# Generate detailed commit based on simple message
$timestamp = Get-Date -Format "HH:mm"
$fullMessage = "chore: $message`n`n- Updated at $timestamp`n- Code improvements and optimizations`n`n[Quick save]"

# Commit and push
git commit -m $fullMessage
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Saved and pushed!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "❌ Failed!" -ForegroundColor Red
    Write-Host ""
}
