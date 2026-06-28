# deploy-local.ps1
# Usage: .\deploy\deploy-local.ps1

Write-Host "=== Starting Local Docker Deployment ===" -ForegroundColor Cyan

# Start containers
Write-Host "Building and starting local docker containers..." -ForegroundColor Green
docker compose -f docker/docker-compose.yml up -d --build --remove-orphans

# Poll Streamlit until it is healthy
$Url = "http://localhost:8501"
Write-Host "Waiting for Streamlit app to become healthy at $Url..." -ForegroundColor Yellow
while ($true) {
    try {
        $response = Invoke-WebRequest -Uri $Url -Method Head -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ App is online!" -ForegroundColor Green
            break
        }
    } catch {
        # Sleep for 3 seconds before trying again
        Start-Sleep -Seconds 3
    }
}

# Define browser incognito opener function
function Open-BrowserIncognito {
    param ([string]$Url)
    $chromePath = Get-Command chrome -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if (-not $chromePath) {
        $standardPaths = @(
            "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe",
            "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
            "${env:LocalAppData}\Google\Chrome\Application\chrome.exe"
        )
        foreach ($path in $standardPaths) {
            if (Test-Path $path) { $chromePath = $path; break }
        }
    }
    if ($chromePath) {
        Start-Process $chromePath -ArgumentList "--incognito", $Url
        return
    }
    $edgePath = Get-Command msedge -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if (-not $edgePath) {
        $standardPaths = @(
            "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
            "${env:ProgramFiles}\Microsoft\Edge\Application\msedge.exe"
        )
        foreach ($path in $standardPaths) {
            if (Test-Path $path) { $edgePath = $path; break }
        }
    }
    if ($edgePath) {
        Start-Process $edgePath -ArgumentList "-inprivate", $Url
        return
    }
    Start-Process $Url
}

# Open URL
Open-BrowserIncognito -Url $Url
