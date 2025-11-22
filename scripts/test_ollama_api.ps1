#!/usr/bin/env pwsh
# PowerShell script to test Ollama API performance

param(
    [string]$Model = "mistral:7b-instruct",
    [int]$NumTests = 3,
    [int]$MaxTokens = 200,
    [string]$Prompt = "Come posso lavare la mia auto senza graffiare la vernice? Rispondi in italiano in modo professionale e cordiale."
)

# Try different endpoint formats
$OllamaUrl = "http://localhost:11434/api/generate"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Ollama API Performance Test" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Model: $Model" -ForegroundColor White
Write-Host "Tests: $NumTests" -ForegroundColor White
Write-Host "Max Tokens: $MaxTokens" -ForegroundColor White
Write-Host ""

# Test query
$testPrompt = @"
Sei un assistente esperto di prodotti per la cura dell'auto.

Contesto:
- Ticket storici e guide tecniche sono disponibili nel database
- Devi rispondere in italiano in modo professionale e cordiale

Domanda: $Prompt

Risposta:
"@

$headers = @{
    "Content-Type" = "application/json"
}

$body = @{
    model = $Model
    prompt = $testPrompt
    options = @{
        num_predict = $MaxTokens
        temperature = 0.5
        top_k = 40
        top_p = 0.9
    }
} | ConvertTo-Json -Depth 10

Write-Host "Running $NumTests tests..." -ForegroundColor Yellow
Write-Host ""

$results = @()

for ($i = 1; $i -le $NumTests; $i++) {
    Write-Host "Test $i/$NumTests..." -ForegroundColor Cyan -NoNewline
    
    $startTime = Get-Date
    
    try {
        $response = Invoke-RestMethod -Uri $OllamaUrl -Method Post -Headers $headers -Body $body -ErrorAction Stop
        
        $endTime = Get-Date
        $elapsed = ($endTime - $startTime).TotalSeconds
        
        # Handle different response formats
        $responseText = ""
        
        # Check if response has 'response' property (standard Ollama format)
        if ($response | Get-Member -Name 'response' -ErrorAction SilentlyContinue) {
            $responseText = $response.response
        } elseif ($response | Get-Member -Name 'Response' -ErrorAction SilentlyContinue) {
            $responseText = $response.Response
        } elseif ($response -is [string]) {
            $responseText = $response
        } elseif ($response -is [hashtable] -and $response.ContainsKey('response')) {
            $responseText = $response['response']
        } else {
            # Debug: show what we got
            Write-Host " (DEBUG: Response type: $($response.GetType().Name))" -ForegroundColor Yellow
            # Try to extract text from JSON response
            $responseText = ($response | ConvertTo-Json -Depth 10)
        }
        
        $result = [PSCustomObject]@{
            Test = $i
            Time = [math]::Round($elapsed, 2)
            Tokens = $responseText.Length
            Success = $true
            Response = $responseText
        }
        
        Write-Host " Done in $($result.Time)s ($($result.Tokens) chars)" -ForegroundColor Green
        
        $results += $result
        
    } catch {
        Write-Host " FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $results += [PSCustomObject]@{
            Test = $i
            Time = 0
            Tokens = 0
            Success = $false
            Error = $_.Exception.Message
        }
    }
    
    # Small delay between tests
    if ($i -lt $NumTests) {
        Start-Sleep -Milliseconds 500
    }
}

# Summary
Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Test Results Summary" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

$successfulTests = @($results | Where-Object { $_.Success -eq $true })

if ($successfulTests.Count -gt 0) {
    $avgTime = ($successfulTests | Measure-Object -Property Time -Average).Average
    $minTime = ($successfulTests | Measure-Object -Property Time -Minimum).Minimum
    $maxTime = ($successfulTests | Measure-Object -Property Time -Maximum).Maximum
    $avgTokens = [math]::Round(($successfulTests | Measure-Object -Property Tokens -Average).Average, 0)
    
    Write-Host "Successful Tests: $($successfulTests.Count)/$NumTests" -ForegroundColor Green
    Write-Host ""
    Write-Host "Timing:" -ForegroundColor Yellow
    Write-Host "  Average: $([math]::Round($avgTime, 2))s" -ForegroundColor White
    Write-Host "  Minimum: $([math]::Round($minTime, 2))s" -ForegroundColor White
    Write-Host "  Maximum: $([math]::Round($maxTime, 2))s" -ForegroundColor White
    Write-Host ""
    Write-Host "Response:" -ForegroundColor Yellow
    Write-Host "  Average Length: $avgTokens characters" -ForegroundColor White
    Write-Host ""
    
    # Show first response as example
    if ($successfulTests.Count -gt 0) {
        Write-Host "Sample Response (Test 1):" -ForegroundColor Yellow
        Write-Host $successfulTests[0].Response -ForegroundColor Gray
        Write-Host ""
    }
} else {
    Write-Host "All tests failed!" -ForegroundColor Red
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

