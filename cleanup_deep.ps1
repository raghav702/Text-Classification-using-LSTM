# Deep Cleanup Script - Remove old training artifacts and logs

Write-Host "Deep cleaning project folders..." -ForegroundColor Yellow

$removed = 0

# Remove old training logs
Write-Host "`nCleaning training logs..." -ForegroundColor Cyan
$logsToRemove = @(
    "logs\retrain_model.log",
    "logs\training_20251101_160154.log",
    "logs\training_20251104_023743.log",
    "logs\training_20251105_000148.log"
)

foreach ($file in $logsToRemove) {
    $fullPath = Join-Path "E:\Project\lstm_model" $file
    if (Test-Path $fullPath) {
        Remove-Item $fullPath -Force
        Write-Host "  Removed: $file" -ForegroundColor Green
        $removed++
    }
}

# Remove old quick training models (keep examples folder as reference)
Write-Host "`nCleaning old model artifacts..." -ForegroundColor Cyan
if (Test-Path "E:\Project\lstm_model\models\quick") {
    Remove-Item "E:\Project\lstm_model\models\quick" -Recurse -Force
    Write-Host "  Removed: models\quick\ (old training artifacts)" -ForegroundColor Green
    $removed++
}

# Remove old evaluation results
Write-Host "`nCleaning old evaluation results..." -ForegroundColor Cyan
if (Test-Path "E:\Project\lstm_model\evaluation_results\evaluation_20251104_010219") {
    Remove-Item "E:\Project\lstm_model\evaluation_results\evaluation_20251104_010219" -Recurse -Force
    Write-Host "  Removed: evaluation_results\evaluation_20251104_010219\" -ForegroundColor Green
    $removed++
}

# Optional: Remove example demo files if not needed (they're just demos)
Write-Host "`nExample demo files (keeping as reference):" -ForegroundColor Yellow
Write-Host "  - examples\advanced_architectures_demo.py" -ForegroundColor Gray
Write-Host "  - examples\advanced_evaluation_demo.py" -ForegroundColor Gray
Write-Host "  - examples\complete_workflow.py" -ForegroundColor Gray
Write-Host "  - examples\data_augmentation_demo.py" -ForegroundColor Gray
Write-Host "  - examples\inference_example.py" -ForegroundColor Gray

$response = Read-Host "`nDo you want to remove example demos? (y/N)"
if ($response -eq "y" -or $response -eq "Y") {
    if (Test-Path "E:\Project\lstm_model\examples") {
        $exampleFiles = Get-ChildItem "E:\Project\lstm_model\examples" -Filter "*.py"
        foreach ($file in $exampleFiles) {
            Remove-Item $file.FullName -Force
            Write-Host "  Removed: examples\$($file.Name)" -ForegroundColor Green
            $removed++
        }
    }
}

Write-Host ""
Write-Host "Deep cleanup complete! Removed $removed items" -ForegroundColor Cyan
Write-Host ""
Write-Host "Remaining structure:" -ForegroundColor Yellow
Write-Host "  - checkpoints\best_model.pth (latest checkpoint)" -ForegroundColor White
Write-Host "  - models\improved_lstm_model_20251106_003134.pth (production model)" -ForegroundColor White
Write-Host "  - configs\ (configuration examples)" -ForegroundColor White
Write-Host "  - examples\ (demo scripts - reference only)" -ForegroundColor White
Write-Host "  - logs\.gitkeep (preserve directory)" -ForegroundColor White
Write-Host ""
