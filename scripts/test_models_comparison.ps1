#!/usr/bin/env pwsh
# PowerShell script to compare multiple Ollama models

param(
    [int]$NumTests = 2,
    [int]$MaxTokens = 200
)

$models = @(
    "mistral:7b-instruct",
    "gemma2:2b",
    "qwen2.5:7b-instruct"
)

$testPrompt = "Come posso lavare la mia auto senza graffiare la vernice? Rispondi in italiano in modo professionale e cordiale."

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Model Performance Comparison" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Testing $($models.Count) models with $NumTests runs each" -ForegroundColor White
Write-Host ""

$OllamaUrl = "http://localhost:11434/api/generate"
$headers = @{
    "Content-Type" = "application/json"
}

$allResults = @()

foreach ($model in $models) {
    Write-Host "Testing $model..." -ForegroundColor Cyan
    Write-Host "---------------------------------------------" -ForegroundColor Gray
    
    # Test if model is available
    try {
        $listResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -ErrorAction Stop
        $availableModels = $listResponse.models | ForEach-Object { $_.name }
        
        if ($availableModels -notcontains $model) {
            Write-Host "  Model not available, skipping..." -ForegroundColor Yellow
            Write-Host ""
            continue
        }
    } catch {
        Write-Host "  Could not check model availability: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host ""
        continue
    }
    
    $modelResults = @()
    
    for ($i = 1; $i -le $NumTests; $i++) {
        Write-Host "  Run $i/$NumTests..." -ForegroundColor Gray -NoNewline
        
        $prompt = @"
Sei un assistente esperto di prodotti per la cura dell'auto.

Domanda: $testPrompt

Risposta:
"@
        
        $body = @{
            model = $model
            prompt = $prompt
            options = @{
                num_predict = $MaxTokens
                temperature = 0.5
                top_k = 40
                top_p = 0.9
            }
        } | ConvertTo-Json -Depth 10
        
        $startTime = Get-Date
        
        try {
            $response = Invoke-RestMethod -Uri $OllamaUrl -Method Post -Headers $headers -Body $body -ErrorAction Stop
            
            $endTime = Get-Date
            $elapsed = ($endTime - $startTime).TotalSeconds
            
            $responseText = ""
            if ($response.response) {
                $responseText = $response.response
            }
            
            Write-Host " $([math]::Round($elapsed, 2))s ($($responseText.Length) chars)" -ForegroundColor Green
            
            $modelResults += @{
                Time = [math]::Round($elapsed, 2)
                Tokens = $responseText.Length
                Success = $true
            }
            
        } catch {
            Write-Host " FAILED" -ForegroundColor Red
            $modelResults += @{
                Time = 0
                Tokens = 0
                Success = $false
            }
        }
        
        Start-Sleep -Milliseconds 300
    }
    
    $successful = $modelResults | Where-Object { $_.Success -eq $true }
    if ($successful.Count -gt 0) {
        $avgTime = ($successful | Measure-Object -Property Time -Average).Average
        $minTime = ($successful | Measure-Object -Property Time -Minimum).Minimum
        $maxTime = ($successful | Measure-Object -Property Time -Maximum).Maximum
        
        $allResults += @{
            Model = $model
            AvgTime = [math]::Round($avgTime, 2)
            MinTime = [math]::Round($minTime, 2)
            MaxTime = [math]::Round($maxTime, 2)
            SuccessRate = "$($successful.Count)/$NumTests"
        }
    }
    
    Write-Host ""
}

# Comparison Table
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Comparison Results" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

if ($allResults.Count -gt 0) {
    Write-Host ("{0,-25} {1,-10} {2,-10} {3,-10} {4,-10}" -f "Model", "Avg Time", "Min Time", "Max Time", "Success") -ForegroundColor Yellow
    Write-Host ("-" * 70) -ForegroundColor Gray
    
    foreach ($result in $allResults) {
        Write-Host ("{0,-25} {1,-10}s {2,-10}s {3,-10}s {4,-10}" -f `
            $result.Model, `
            $result.AvgTime, `
            $result.MinTime, `
            $result.MaxTime, `
            $result.SuccessRate)
    }
    
    Write-Host ""
    
    # Find fastest
    $fastest = $allResults | Sort-Object AvgTime | Select-Object -First 1
    Write-Host "Fastest Model: $($fastest.Model) ($($fastest.AvgTime)s average)" -ForegroundColor Green
    Write-Host ""
}

Write-Host "=============================================" -ForegroundColor Cyan


