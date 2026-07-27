[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BundlePath,

    [string]$PrivateRoot = ".ai-local\profiles\T32"
)

$ErrorActionPreference = "Stop"
$privateDirectory = New-Item -ItemType Directory -Force -Path $PrivateRoot
$captureId = (
    (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ") +
    "-" +
    [guid]::NewGuid().ToString("N")
)
$transcriptPath = Join-Path `
    $privateDirectory.FullName `
    "geniex-session-$captureId.private.txt"
$fixedPrompt = (
    "Reply with five consecutive integers beginning at 41, separated by spaces."
)
$fixedPromptSha256 = (
    "e36ded0e32a5d70a5b1c3d36d4e625ef98377475295d568b05b69d4719cfa055"
)
$normalizedPrompt = $fixedPrompt.Normalize(
    [System.Text.NormalizationForm]::FormC
)
$promptBytes = [System.Text.Encoding]::UTF8.GetBytes($normalizedPrompt)
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $computedPromptSha256 = [System.BitConverter]::ToString(
        $sha256.ComputeHash($promptBytes)
    ).Replace("-", "").ToLowerInvariant()
}
finally {
    $sha256.Dispose()
}
if ($computedPromptSha256 -ne $fixedPromptSha256) {
    throw "Pinned prompt digest self-check failed."
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [string[]]$Arguments = @()
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Native command failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Keep the transcript private. It may contain local paths and runtime details."
Write-Host "Use the fixed prompt from README.md and record only observed timings."

Start-Transcript -Path $transcriptPath -NoClobber | Out-Null
try {
    Write-Host "=== exact tool versions ==="
    Invoke-NativeChecked -Command "geniex" -Arguments @("--version")
    Invoke-NativeChecked `
        -Command "python" `
        -Arguments @(
            "-c",
            "import importlib.metadata as m; print(m.version('qai-hub-models-cli'))"
        )

    Write-Host "=== public Qwen3-0.6B Q4_0 asset fetch command ==="
    Write-Host "qai-hub-models fetch Qwen3-0.6B --runtime geniex_llamacpp --precision q4_0"

    Write-Host "=== register the already-fetched local GGUF bundle ==="
    $resolvedBundlePath = (Resolve-Path -Path $BundlePath -ErrorAction Stop).Path
    Invoke-NativeChecked `
        -Command "geniex" `
        -Arguments @(
            "pull",
            "local/qwen3-0.6b",
            "--local-path",
            $resolvedBundlePath
        )

    Write-Host "=== persistent device-side generation loop ==="
    Write-Host "Paste this exact UTF-8 NFC prompt with no trailing newline:"
    Write-Host $fixedPrompt
    Write-Host "Pinned prompt SHA-256: $fixedPromptSha256"
    $env:GENIEX_LOG = "INFO"
    Invoke-NativeChecked `
        -Command "geniex" `
        -Arguments @(
            "infer",
            "local/qwen3-0.6b",
            "--compute",
            "npu",
            "--think=false",
            "--seed",
            "0",
            "--nctx",
            "4096",
            "--max-tokens",
            "32"
        )
}
finally {
    Stop-Transcript | Out-Null
}

Write-Host "Private transcript written. Do not commit it:"
Write-Host $transcriptPath
