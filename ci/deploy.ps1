param(
    [Parameter(Mandatory = $true)]
    [string]$RepoPath,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{7,40}$')]
    [string]$ExpectedRevision
)

$ErrorActionPreference = 'Stop'

function Get-ToolPath {
    param(
        [string]$Name,
        [string[]]$FallbackPaths = @()
    )

    $command = Get-Command "$Name.exe" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    foreach ($path in $FallbackPaths) {
        if (Test-Path $path) {
            return $path
        }
    }

    throw "Required executable '$Name.exe' was not found."
}

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

$RepoPath = (Resolve-Path $RepoPath).Path
Set-Location $RepoPath

$git = Get-ToolPath -Name 'git' -FallbackPaths @(
    'C:\Program Files\Git\cmd\git.exe'
)
$actualRevisionOutput = & $git rev-parse --short HEAD
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to read the Windows working tree revision.'
}
$actualRevision = ($actualRevisionOutput -join '').Trim()
if ($actualRevision -ne $ExpectedRevision) {
    throw "Windows working tree revision '$actualRevision' does not match Jenkins revision '$ExpectedRevision'."
}

$changes = & $git status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to check the Windows working tree status.'
}
if ($changes) {
    throw 'The Windows working tree has uncommitted changes; commit or clean it before deploying.'
}

$docker = Get-ToolPath -Name 'docker' -FallbackPaths @(
    'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
)
$kubectl = Get-ToolPath -Name 'kubectl' -FallbackPaths @(
    'C:\Program Files\Docker\Docker\resources\bin\kubectl.exe'
)
$kind = Get-ToolPath -Name 'kind' -FallbackPaths @(
    (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links\kind.exe')
)

$clusters = & $kind get clusters
if ($LASTEXITCODE -ne 0 -or $clusters -notcontains 'desktop') {
    throw "The Docker Desktop kind cluster 'desktop' was not found."
}

Invoke-Checked -Executable $docker -Arguments @(
    '--context', 'desktop-linux',
    'build',
    '--build-arg', "APP_VERSION=$ExpectedRevision",
    '-t', 'local-chat-app:initial',
    '.'
)
Invoke-Checked -Executable $kind -Arguments @(
    'load', 'docker-image', 'local-chat-app:initial', '--name', 'desktop'
)
Invoke-Checked -Executable $kubectl -Arguments @(
    'apply', '-f', 'k8s/service.yaml'
)
Invoke-Checked -Executable $kubectl -Arguments @(
    'apply', '-f', 'k8s/deployment.yaml'
)
Invoke-Checked -Executable $kubectl -Arguments @(
    'rollout', 'restart', 'deployment/chat-app'
)
Invoke-Checked -Executable $kubectl -Arguments @(
    'rollout', 'status', 'deployment/chat-app', '--timeout=600s'
)

$health = & $kubectl exec deployment/chat-app -- python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/health").read().decode())'
if ($LASTEXITCODE -ne 0) {
    throw 'The in-pod health check failed.'
}

$healthResponse = $health | ConvertFrom-Json
if ($healthResponse.status -ne 'ok' -or $healthResponse.version -ne $ExpectedRevision) {
    throw "Unexpected health response: $health"
}

Write-Output $health
