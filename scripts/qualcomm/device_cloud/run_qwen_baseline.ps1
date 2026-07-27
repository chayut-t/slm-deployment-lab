[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BundlePath,

    [string]$PrivateRoot = ".ai-local\profiles\T32"
)

$ErrorActionPreference = "Stop"
$privateDirectory = New-Item -ItemType Directory -Force -Path $PrivateRoot
$transcriptPath = Join-Path $privateDirectory.FullName "geniex-session.private.txt"

Write-Host "Keep the transcript private. It may contain local paths and runtime details."
Write-Host "Use the fixed prompt from README.md and record only observed timings."

Start-Transcript -Path $transcriptPath -Force | Out-Null
try {
    Write-Host "=== exact tool versions ==="
    geniex --version
    python -c "import importlib.metadata as m; print(m.version('qai-hub-models-cli'))"

    Write-Host "=== public Qwen3-0.6B Q4_0 asset fetch command ==="
    Write-Host "qai-hub-models fetch Qwen3-0.6B --runtime geniex_llamacpp --precision q4_0"

    Write-Host "=== register the already-fetched local GGUF bundle ==="
    geniex pull local/qwen3-0.6b --local-path (Resolve-Path $BundlePath).Path

    Write-Host "=== persistent device-side generation loop ==="
    $env:GENIEX_LOG = "INFO"
    geniex infer local/qwen3-0.6b `
        --compute npu `
        --think=false `
        --seed 0 `
        --nctx 4096 `
        --max-tokens 32
}
finally {
    Stop-Transcript | Out-Null
}

Write-Host "Private transcript written. Do not commit it:"
Write-Host $transcriptPath
