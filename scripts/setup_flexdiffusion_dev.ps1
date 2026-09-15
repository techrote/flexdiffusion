[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EasyDiffusionPath,

    [string]$RepoPath,

    [switch]$Restore
)

$ErrorActionPreference = "Stop"

# Windows PowerShell can leave $PSScriptRoot empty while evaluating parameter
# default expressions. Resolve the repository path only after parameter binding,
# using the script invocation path, so callers may safely omit -RepoPath.
if ([string]::IsNullOrWhiteSpace($RepoPath)) {
    $scriptPath = $MyInvocation.MyCommand.Path
    if ([string]::IsNullOrWhiteSpace($scriptPath)) {
        throw "Could not determine this setup script's path. Pass -RepoPath explicitly."
    }
    $scriptDir = Split-Path -Parent $scriptPath
    $RepoPath = Split-Path -Parent $scriptDir
}

function Resolve-ExistingPath([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Backup-FileOnce([string]$Path) {
    $backup = "$Path.flexdiffusion-upstream.bak"
    if (-not (Test-Path -LiteralPath $backup)) {
        Copy-Item -LiteralPath $Path -Destination $backup
        Write-Host "Backed up $Path"
    }
    return $backup
}

function Disable-Line([string]$Content, [string]$Regex, [string]$Marker) {
    if ($Content -match [regex]::Escape($Marker)) {
        return $Content
    }
    $match = [regex]::Match($Content, $Regex, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if (-not $match.Success) {
        throw "Could not find expected Easy Diffusion startup line for: $Marker"
    }
    return [regex]::Replace(
        $Content,
        $Regex,
        "@REM FLEXDIFFUSION DEV DISABLED: $Marker",
        [System.Text.RegularExpressions.RegexOptions]::Multiline
    )
}

function Remove-UiLinkOrDirectory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        & cmd.exe /d /c "rmdir `"$Path`""
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to remove existing UI junction: $Path"
        }
    }
    else {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

$EasyDiffusionPath = Resolve-ExistingPath $EasyDiffusionPath "Easy Diffusion install"
$RepoPath = Resolve-ExistingPath $RepoPath "FlexDiffusion repository"

$repoUi = Join-Path $RepoPath "ui"
$installUi = Join-Path $EasyDiffusionPath "ui"
$onEnv = Join-Path $EasyDiffusionPath "scripts\on_env_start.bat"
$onSd = Join-Path $EasyDiffusionPath "scripts\on_sd_start.bat"

$null = Resolve-ExistingPath $repoUi "FlexDiffusion ui directory"
$null = Resolve-ExistingPath $onEnv "Easy Diffusion on_env_start.bat"
$null = Resolve-ExistingPath $onSd "Easy Diffusion on_sd_start.bat"

$listener = Get-NetTCPConnection -LocalPort 9000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    throw "Something is listening on TCP port 9000. Close Easy Diffusion before changing the development junction."
}

$onEnvBackup = "$onEnv.flexdiffusion-upstream.bak"
$onSdBackup = "$onSd.flexdiffusion-upstream.bak"

if ($Restore) {
    if (-not (Test-Path -LiteralPath $onEnvBackup) -or -not (Test-Path -LiteralPath $onSdBackup)) {
        throw "FlexDiffusion startup-script backups were not found; refusing an incomplete restore."
    }

    Remove-UiLinkOrDirectory $installUi
    Copy-Item -LiteralPath $onEnvBackup -Destination $onEnv -Force
    Copy-Item -LiteralPath $onSdBackup -Destination $onSd -Force

    Write-Host "Restored Easy Diffusion startup scripts."
    Write-Host "The next normal Easy Diffusion start will recreate its upstream UI directory."
    exit 0
}

Backup-FileOnce $onEnv | Out-Null
Backup-FileOnce $onSd | Out-Null

$onEnvContent = Get-Content -LiteralPath $onEnv -Raw
$onEnvContent = Disable-Line `
    $onEnvContent `
    '(?m)^\s*@xcopy\s+sd-ui-files\\ui\s+ui\s+.*$' `
    'copy upstream ui into install ui'
$onEnvContent = Disable-Line `
    $onEnvContent `
    '(?m)^\s*@copy\s+sd-ui-files\\scripts\\on_sd_start\.bat\s+scripts\\\s+/Y\s*$' `
    'overwrite patched on_sd_start.bat'
Set-Content -LiteralPath $onEnv -Value $onEnvContent -Encoding ASCII

$onSdContent = Get-Content -LiteralPath $onSd -Raw
$onSdContent = Disable-Line `
    $onSdContent `
    '(?m)^\s*@copy\s+sd-ui-files\\scripts\\on_env_start\.bat\s+scripts\\\s+/Y\s*$' `
    'overwrite patched on_env_start.bat'
Set-Content -LiteralPath $onSd -Value $onSdContent -Encoding ASCII

Remove-UiLinkOrDirectory $installUi
New-Item -ItemType Junction -Path $installUi -Target $repoUi | Out-Null

$branch = $null
try {
    $branch = (& git -C $RepoPath branch --show-current 2>$null).Trim()
}
catch {
    $branch = $null
}

Write-Host ""
Write-Host "FlexDiffusion development link installed successfully."
Write-Host "  Easy Diffusion: $EasyDiffusionPath"
Write-Host "  Repository:     $RepoPath"
Write-Host "  UI junction:    $installUi -> $repoUi"
if ($branch) {
    Write-Host "  Git branch:     $branch"
}
Write-Host ""
Write-Host "You can now start Easy Diffusion normally."
Write-Host "For Trajectory Lab testing, select Settings -> Engine to use -> v2.0 (ed_classic), save, and restart."
Write-Host ""
Write-Host "To undo this development setup later:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -EasyDiffusionPath `"$EasyDiffusionPath`" -RepoPath `"$RepoPath`" -Restore"
