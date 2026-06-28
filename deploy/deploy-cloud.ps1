# deploy-cloud.ps1
# Usage:
#   To deploy latest updates:
#     .\deploy\deploy-cloud.ps1 -vmIp "74.248.184.155" -sshKeyPath "C:\path\to\xai-key.pem"
#   To clean, reset, and re-provision the VM:
#     .\deploy\deploy-cloud.ps1 -vmIp "74.248.184.155" -sshKeyPath "C:\path\to\xai-key.pem" -reset

param (
    [Parameter(Mandatory=$true)]
    [string]$vmIp,
    [Parameter(Mandatory=$true)]
    [string]$sshKeyPath,
    [switch]$reset
)

if (-not (Test-Path $sshKeyPath)) {
    Write-Error "SSH key path does not exist: $sshKeyPath"
    exit 1
}

$sshUser = "azureuser"

if ($reset) {
    Write-Host "=== Resetting Azure VM (Stopping old containers and pruning data) ===" -ForegroundColor Cyan
    ssh -i $sshKeyPath "${sshUser}@${vmIp}" "docker stop \$(docker ps -aq) 2>/dev/null || true; docker system prune -a --volumes -f; rm -rf ~/xai-app"
    
    Write-Host "=== Provisioning Azure VM (Running setup script) ===" -ForegroundColor Cyan
    Get-Content -Raw deploy/setup-cloud.sh | ssh -i $sshKeyPath "${sshUser}@${vmIp}" 'bash -s'
} else {
    Write-Host "=== Deploying latest changes to Azure VM ===" -ForegroundColor Cyan
    # Remote pull
    ssh -i $sshKeyPath "${sshUser}@${vmIp}" "cd ~/xai-app && git fetch --all && git reset --hard origin/v0.3/ai-imagery-detector"
}

# Remote start containers
Write-Host "Building and starting remote containers..." -ForegroundColor Green
ssh -i $sshKeyPath "${sshUser}@${vmIp}" "cd ~/xai-app && docker compose -f docker/docker-compose.cloud.yml up -d --build --remove-orphans"

# Poll remote Streamlit until it is healthy
$Url = "http://$vmIp:8501"
Write-Host "Waiting for Azure app to become healthy at $Url..." -ForegroundColor Yellow
while ($true) {
    try {
        $response = Invoke-WebRequest -Uri $Url -Method Head -TimeoutSec 3 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ Azure app is online!" -ForegroundColor Green
            break
        }
    } catch {
        # Sleep for 10 seconds before trying again (cloud builds might take longer)
        Start-Sleep -Seconds 10
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
