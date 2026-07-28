[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [string]$PrivateRoot = ".ai-local\profiles\T32",

    [string]$GenieXRoot = "$env:LOCALAPPDATA\GenieX CLI",

    [switch]$TokenizerOnly,

    [string]$FormattedPromptPath,

    [string]$TokenizerOutputPath
)

$ErrorActionPreference = "Stop"

$fixedPrompt = (
    "Reply with five consecutive integers beginning at 41, separated by spaces."
)
$fixedPromptSha256 = (
    "e36ded0e32a5d70a5b1c3d36d4e625ef98377475295d568b05b69d4719cfa055"
)

$resolvedModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
$resolvedGenieXRoot = (Resolve-Path -LiteralPath $GenieXRoot).Path
$llamaCppRoot = Join-Path $resolvedGenieXRoot "llama_cpp"
$privateDirectory = New-Item -ItemType Directory -Force -Path $PrivateRoot
$captureId = (
    (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ") +
    "-" +
    [guid]::NewGuid().ToString("N")
)
$rawLogPath = Join-Path `
    $privateDirectory.FullName `
    "geniex-boundaries-$captureId.private.txt"
$rawJsonPath = Join-Path `
    $privateDirectory.FullName `
    "geniex-boundaries-$captureId.private.json"

$promptBytes = [System.Text.Encoding]::UTF8.GetBytes(
    $fixedPrompt.Normalize([System.Text.NormalizationForm]::FormC)
)
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

$nativeSource = @'
using System;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;

public static class T32Native {
    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaModelParams {
        public IntPtr devices;
        public IntPtr tensor_buft_overrides;
        public int n_gpu_layers;
        public int split_mode;
        public int main_gpu;
        public IntPtr tensor_split;
        public IntPtr progress_callback;
        public IntPtr progress_callback_user_data;
        public IntPtr kv_overrides;
        public byte vocab_only;
        public byte use_mmap;
        public byte use_direct_io;
        public byte use_mlock;
        public byte check_tensors;
        public byte use_extra_bufts;
        public byte no_host;
        public byte no_alloc;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ModelConfig {
        public int n_ctx;
        public int n_threads;
        public int n_threads_batch;
        public int n_batch;
        public int n_ubatch;
        public int n_seq_max;
        public int n_gpu_layers;
        public IntPtr chat_template_path;
        public IntPtr chat_template_content;
        public IntPtr system_prompt;
        public byte enable_sampling;
        public IntPtr grammar_str;
        public int max_tokens;
        public byte enable_thinking;
        public byte verbose;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LlmCreateInput {
        public IntPtr model_name;
        public IntPtr model_path;
        public IntPtr tokenizer_path;
        public ModelConfig config;
        public IntPtr plugin_id;
        public IntPtr device_id;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ChatMessage {
        public IntPtr role;
        public IntPtr content;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ApplyChatTemplateInput {
        public IntPtr messages;
        public int message_count;
        public IntPtr tools;
        public byte enable_thinking;
        public byte add_generation_prompt;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ApplyChatTemplateOutput {
        public IntPtr formatted_text;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SamplerConfig {
        public float temperature;
        public float top_p;
        public int top_k;
        public float min_p;
        public float repetition_penalty;
        public float presence_penalty;
        public float frequency_penalty;
        public int seed;
        public IntPtr grammar_path;
        public IntPtr grammar_string;
        public byte enable_json;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct GenerationConfig {
        public int max_tokens;
        public IntPtr stop;
        public int stop_count;
        public int n_past;
        public IntPtr sampler_config;
        public IntPtr image_paths;
        public int image_count;
        public int image_max_length;
        public IntPtr audio_paths;
        public int audio_count;
        public byte sliding_window;
        public int sliding_window_n_keep;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct GenerateInput {
        public IntPtr prompt_utf8;
        public IntPtr config;
        public IntPtr on_token;
        public IntPtr user_data;
        public IntPtr input_ids;
        public int input_ids_count;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ProfileData {
        public long ttft;
        public long prompt_time;
        public long decode_time;
        public long prompt_tokens;
        public long generated_tokens;
        public long audio_duration;
        public double prefill_speed;
        public double decoding_speed;
        public double real_time_factor;
        public IntPtr stop_reason;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct GenerateOutput {
        public IntPtr full_text;
        public ProfileData profile_data;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool SetDllDirectory(string path);

    [DllImport("llama.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void llama_backend_init();

    [DllImport("llama.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void llama_backend_free();

    [DllImport("llama.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern LlamaModelParams llama_model_default_params();

    [DllImport("llama.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr llama_model_load_from_file(
        [MarshalAs(UnmanagedType.LPStr)] string path,
        LlamaModelParams parameters
    );

    [DllImport("llama.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void llama_model_free(IntPtr model);

    [DllImport("llama.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr llama_model_get_vocab(IntPtr model);

    [DllImport("ggml.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr ggml_backend_dev_by_name(
        [MarshalAs(UnmanagedType.LPStr)] string name
    );

    [DllImport(
        "llama.dll",
        EntryPoint = "llama_tokenize",
        CallingConvention = CallingConvention.Cdecl
    )]
    public static extern int llama_tokenize_size(
        IntPtr vocab,
        byte[] text,
        int textLength,
        IntPtr tokens,
        int maxTokens,
        byte addSpecial,
        byte parseSpecial
    );

    [DllImport(
        "llama.dll",
        EntryPoint = "llama_tokenize",
        CallingConvention = CallingConvention.Cdecl
    )]
    public static extern int llama_tokenize(
        IntPtr vocab,
        byte[] text,
        int textLength,
        int[] tokens,
        int maxTokens,
        byte addSpecial,
        byte parseSpecial
    );

    [DllImport("geniex.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int geniex_init();

    [DllImport("geniex.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int geniex_deinit();

    [DllImport("geniex.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int geniex_llm_create(
        ref LlmCreateInput input,
        out IntPtr handle
    );

    [DllImport("geniex.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int geniex_llm_destroy(IntPtr handle);

    [DllImport("geniex.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int geniex_llm_apply_chat_template(
        IntPtr handle,
        ref ApplyChatTemplateInput input,
        out ApplyChatTemplateOutput output
    );

    [DllImport("geniex.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int geniex_llm_generate(
        IntPtr handle,
        ref GenerateInput input,
        out GenerateOutput output
    );

    [DllImport("geniex.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void geniex_free(IntPtr pointer);

    public static IntPtr Utf8(string value) {
        if (value == null) {
            return IntPtr.Zero;
        }
        byte[] bytes = Encoding.UTF8.GetBytes(value + "\0");
        IntPtr pointer = Marshal.AllocHGlobal(bytes.Length);
        Marshal.Copy(bytes, 0, pointer, bytes.Length);
        return pointer;
    }

    public static string Utf8String(IntPtr pointer) {
        if (pointer == IntPtr.Zero) {
            return null;
        }
        int length = 0;
        while (Marshal.ReadByte(pointer, length) != 0) {
            length++;
        }
        byte[] bytes = new byte[length];
        Marshal.Copy(pointer, bytes, 0, length);
        return Encoding.UTF8.GetString(bytes);
    }

    public static double MeasureArtifactMapMilliseconds(string path) {
        Stopwatch stopwatch = Stopwatch.StartNew();
        using (FileStream stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read
        ))
        using (MemoryMappedFile mapped = MemoryMappedFile.CreateFromFile(
            stream,
            null,
            0,
            MemoryMappedFileAccess.Read,
            HandleInheritability.None,
            true
        ))
        using (MemoryMappedViewAccessor view = mapped.CreateViewAccessor(
            0,
            1,
            MemoryMappedFileAccess.Read
        )) {
            view.ReadByte(0);
        }
        stopwatch.Stop();
        return stopwatch.Elapsed.TotalMilliseconds;
    }
}
'@

if (-not ("T32Native" -as [type])) {
    Add-Type -TypeDefinition $nativeSource
}

if (-not [T32Native]::SetDllDirectory($llamaCppRoot)) {
    throw "SetDllDirectory failed for the installed llama.cpp runtime."
}
$env:Path = (
    $resolvedGenieXRoot +
    [IO.Path]::PathSeparator +
    $llamaCppRoot +
    [IO.Path]::PathSeparator +
    $env:Path
)

if ($TokenizerOnly) {
    if (-not $FormattedPromptPath -or -not $TokenizerOutputPath) {
        throw "Tokenizer-only mode requires prompt and output paths."
    }
    $cpuRuntime = Join-Path $privateDirectory.FullName "llama-cpu-runtime"
    New-Item -ItemType Directory -Force -Path $cpuRuntime | Out-Null
    foreach ($runtimeFile in @(
        "llama.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll"
    )) {
        Copy-Item `
            -LiteralPath (Join-Path $llamaCppRoot $runtimeFile) `
            -Destination $cpuRuntime `
            -Force
    }
    if (-not [T32Native]::SetDllDirectory($cpuRuntime)) {
        throw "SetDllDirectory failed for the CPU-only tokenizer runtime."
    }
    $env:Path = (
        $cpuRuntime +
        [IO.Path]::PathSeparator +
        $env:Path
    )

    $tokenizerModel = [IntPtr]::Zero
    try {
        [T32Native]::llama_backend_init()
        $parameters = [T32Native]::llama_model_default_params()
        $parameters.vocab_only = 1
        $parameters.use_mmap = 1
        $parameters.n_gpu_layers = 0
        $tokenizerModel = [T32Native]::llama_model_load_from_file(
            $resolvedModelPath,
            $parameters
        )
        if ($tokenizerModel -eq [IntPtr]::Zero) {
            throw "Failed to load the exact llama.cpp vocabulary."
        }
        $tokenizerVocab = [T32Native]::llama_model_get_vocab($tokenizerModel)
        if ($tokenizerVocab -eq [IntPtr]::Zero) {
            throw "Failed to get the exact llama.cpp vocabulary."
        }

        $formattedPrompt = [IO.File]::ReadAllText(
            (Resolve-Path -LiteralPath $FormattedPromptPath).Path,
            [Text.Encoding]::UTF8
        )
        $formattedPromptBytes = [Text.Encoding]::UTF8.GetBytes($formattedPrompt)
        $tokenizationStopwatch = [Diagnostics.Stopwatch]::StartNew()
        $required = [T32Native]::llama_tokenize_size(
            $tokenizerVocab,
            $formattedPromptBytes,
            $formattedPromptBytes.Length,
            [IntPtr]::Zero,
            0,
            1,
            1
        )
        if ($required -ge 0) {
            throw "Tokenizer sizing call did not report a required capacity."
        }
        $tokens = New-Object int[] (-$required)
        $tokenCount = [T32Native]::llama_tokenize(
            $tokenizerVocab,
            $formattedPromptBytes,
            $formattedPromptBytes.Length,
            $tokens,
            $tokens.Length,
            1,
            1
        )
        $tokenizationStopwatch.Stop()
        if ($tokenCount -le 0) {
            throw "Exact llama.cpp tokenization failed with code $tokenCount."
        }
        if ($tokenCount -ne $tokens.Length) {
            $tokens = $tokens[0..($tokenCount - 1)]
        }
        [ordered]@{
            tokenization_ms = [math]::Round(
                $tokenizationStopwatch.Elapsed.TotalMilliseconds,
                6
            )
            token_count = $tokenCount
            token_ids = @($tokens | ForEach-Object { [int]$_ })
        } | ConvertTo-Json -Depth 4 | Set-Content `
            -LiteralPath $TokenizerOutputPath `
            -Encoding UTF8 `
            -NoNewline
    }
    finally {
        if ($tokenizerModel -ne [IntPtr]::Zero) {
            [T32Native]::llama_model_free($tokenizerModel)
        }
        [T32Native]::llama_backend_free()
    }
    exit 0
}

$modelNamePointer = [IntPtr]::Zero
$modelPathPointer = [IntPtr]::Zero
$pluginPointer = [IntPtr]::Zero
$devicePointer = [IntPtr]::Zero
$rolePointer = [IntPtr]::Zero
$contentPointer = [IntPtr]::Zero
$messagesPointer = [IntPtr]::Zero
$samplerPointer = [IntPtr]::Zero
$generationPointer = [IntPtr]::Zero
$geniexHandle = [IntPtr]::Zero
$formattedPromptPointer = [IntPtr]::Zero
$generatedTextPointer = [IntPtr]::Zero
$tokenHandle = $null

Start-Transcript -LiteralPath $rawLogPath -NoClobber | Out-Null
try {
    Write-Host "Private T32 boundary capture. Do not commit this transcript."
    Write-Host "Model SHA-256:"
    (Get-FileHash -LiteralPath $resolvedModelPath -Algorithm SHA256).Hash.ToLowerInvariant()

    $requestStopwatch = [Diagnostics.Stopwatch]::StartNew()
    $artifactLoadMs = [T32Native]::MeasureArtifactMapMilliseconds(
        $resolvedModelPath
    )

    $initCode = [T32Native]::geniex_init()
    if ($initCode -ne 0) {
        throw "geniex_init failed with code $initCode."
    }

    $modelNamePointer = [T32Native]::Utf8("T32-Qwen3-0.6B-Q4_0")
    $modelPathPointer = [T32Native]::Utf8($resolvedModelPath)
    $pluginPointer = [T32Native]::Utf8("llama_cpp")
    $devicePointer = [T32Native]::Utf8("HTP0")

    $modelConfig = New-Object T32Native+ModelConfig
    $modelConfig.n_ctx = 4096
    $modelConfig.n_gpu_layers = -1
    $modelConfig.max_tokens = 32
    $modelConfig.enable_thinking = 0
    $modelConfig.verbose = 1

    $createInput = New-Object T32Native+LlmCreateInput
    $createInput.model_name = $modelNamePointer
    $createInput.model_path = $modelPathPointer
    $createInput.config = $modelConfig
    $createInput.plugin_id = $pluginPointer
    $createInput.device_id = $devicePointer

    $modelLoadStopwatch = [Diagnostics.Stopwatch]::StartNew()
    $createCode = [T32Native]::geniex_llm_create(
        [ref]$createInput,
        [ref]$geniexHandle
    )
    $modelLoadStopwatch.Stop()
    if ($createCode -ne 0 -or $geniexHandle -eq [IntPtr]::Zero) {
        throw "geniex_llm_create failed with code $createCode."
    }

    $rolePointer = [T32Native]::Utf8("user")
    $contentPointer = [T32Native]::Utf8($fixedPrompt)
    $message = New-Object T32Native+ChatMessage
    $message.role = $rolePointer
    $message.content = $contentPointer
    $messagesPointer = [Runtime.InteropServices.Marshal]::AllocHGlobal(
        [Runtime.InteropServices.Marshal]::SizeOf($message)
    )
    [Runtime.InteropServices.Marshal]::StructureToPtr(
        $message,
        $messagesPointer,
        $false
    )

    $templateInput = New-Object T32Native+ApplyChatTemplateInput
    $templateInput.messages = $messagesPointer
    $templateInput.message_count = 1
    $templateInput.enable_thinking = 0
    $templateInput.add_generation_prompt = 1
    $templateOutput = New-Object T32Native+ApplyChatTemplateOutput
    $templateStopwatch = [Diagnostics.Stopwatch]::StartNew()
    $templateCode = [T32Native]::geniex_llm_apply_chat_template(
        $geniexHandle,
        [ref]$templateInput,
        [ref]$templateOutput
    )
    $templateStopwatch.Stop()
    if ($templateCode -ne 0 -or $templateOutput.formatted_text -eq [IntPtr]::Zero) {
        throw "geniex_llm_apply_chat_template failed with code $templateCode."
    }
    $formattedPromptPointer = $templateOutput.formatted_text
    $formattedPrompt = [T32Native]::Utf8String($formattedPromptPointer)
    $formattedPromptPath = Join-Path `
        $privateDirectory.FullName `
        "formatted-prompt-$captureId.private.txt"
    $tokenizerResultPath = Join-Path `
        $privateDirectory.FullName `
        "tokenizer-$captureId.private.json"
    [IO.File]::WriteAllText(
        $formattedPromptPath,
        $formattedPrompt,
        [Text.UTF8Encoding]::new($false)
    )
    & powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File $MyInvocation.MyCommand.Path `
        -ModelPath $resolvedModelPath `
        -PrivateRoot $privateDirectory.FullName `
        -GenieXRoot $resolvedGenieXRoot `
        -TokenizerOnly `
        -FormattedPromptPath $formattedPromptPath `
        -TokenizerOutputPath $tokenizerResultPath
    if ($LASTEXITCODE -ne 0) {
        throw "Exact llama.cpp tokenizer subprocess failed."
    }
    $tokenizerResult = Get-Content `
        -LiteralPath $tokenizerResultPath `
        -Raw | ConvertFrom-Json
    $tokenCount = [int]$tokenizerResult.token_count
    if ($null -ne $tokenizerResult.token_ids.value) {
        $tokens = [int[]]@($tokenizerResult.token_ids.value)
    }
    else {
        $tokens = [int[]]@($tokenizerResult.token_ids)
    }
    if ($tokenCount -le 0 -or $tokens.Length -ne $tokenCount) {
        throw "Tokenizer subprocess returned an invalid token vector."
    }

    $sampler = New-Object T32Native+SamplerConfig
    $sampler.temperature = 0.0
    $sampler.top_p = 1.0
    $sampler.top_k = 0
    $sampler.min_p = 0.0
    $sampler.repetition_penalty = 1.0
    $sampler.seed = 0
    $samplerPointer = [Runtime.InteropServices.Marshal]::AllocHGlobal(
        [Runtime.InteropServices.Marshal]::SizeOf($sampler)
    )
    [Runtime.InteropServices.Marshal]::StructureToPtr(
        $sampler,
        $samplerPointer,
        $false
    )

    $generationConfig = New-Object T32Native+GenerationConfig
    $generationConfig.max_tokens = 32
    $generationConfig.sampler_config = $samplerPointer
    $generationPointer = [Runtime.InteropServices.Marshal]::AllocHGlobal(
        [Runtime.InteropServices.Marshal]::SizeOf($generationConfig)
    )
    [Runtime.InteropServices.Marshal]::StructureToPtr(
        $generationConfig,
        $generationPointer,
        $false
    )

    $tokenHandle = [Runtime.InteropServices.GCHandle]::Alloc(
        $tokens,
        [Runtime.InteropServices.GCHandleType]::Pinned
    )
    $generateInput = New-Object T32Native+GenerateInput
    $generateInput.config = $generationPointer
    $generateInput.input_ids = $tokenHandle.AddrOfPinnedObject()
    $generateInput.input_ids_count = $tokenCount
    $generateOutput = New-Object T32Native+GenerateOutput

    $generationHostStopwatch = [Diagnostics.Stopwatch]::StartNew()
    $generateCode = [T32Native]::geniex_llm_generate(
        $geniexHandle,
        [ref]$generateInput,
        [ref]$generateOutput
    )
    $generationHostStopwatch.Stop()
    $requestStopwatch.Stop()
    if ($generateCode -ne 0) {
        throw "geniex_llm_generate failed with code $generateCode."
    }

    $generatedTextPointer = $generateOutput.full_text
    $generatedText = [T32Native]::Utf8String($generatedTextPointer)
    $profile = $generateOutput.profile_data
    $firstDecodeUs = $profile.ttft - $profile.prompt_time
    if ($firstDecodeUs -lt 0) {
        throw "Runtime counters produced a negative first-decode boundary."
    }
    $remainingDecodeUs = $profile.decode_time - $firstDecodeUs
    if ($remainingDecodeUs -lt 0) {
        throw "Runtime counters produced a negative remaining-decode boundary."
    }
    $requestStageSumMs = (
        $artifactLoadMs +
        $modelLoadStopwatch.Elapsed.TotalMilliseconds +
        $templateStopwatch.Elapsed.TotalMilliseconds +
        [double]$tokenizerResult.tokenization_ms +
        $generationHostStopwatch.Elapsed.TotalMilliseconds
    )
    $normalizedGeneratedText = [regex]::Replace(
        $generatedText.Trim(),
        "[,\s]+",
        " "
    )

    $outputBytes = [Text.Encoding]::UTF8.GetBytes($generatedText)
    $outputHasher = [Security.Cryptography.SHA256]::Create()
    try {
        $outputSha256 = [BitConverter]::ToString(
            $outputHasher.ComputeHash($outputBytes)
        ).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $outputHasher.Dispose()
    }

    $record = [ordered]@{
        schema_version = 1
        captured_at = (Get-Date).ToUniversalTime().ToString(
            "yyyy-MM-ddTHH:mm:ss.fffZ"
        )
        model = [ordered]@{
            sha256 = (
                Get-FileHash -LiteralPath $resolvedModelPath -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            size_bytes = (Get-Item -LiteralPath $resolvedModelPath).Length
        }
        prompt = [ordered]@{
            sha256 = $fixedPromptSha256
            formatted_prompt_token_count = $tokenCount
        }
        output = [ordered]@{
            sha256 = $outputSha256
            generated_token_count = $profile.generated_tokens
            stop_reason = [T32Native]::Utf8String($profile.stop_reason)
            valid_multi_token_output_confirmed = (
                $profile.generated_tokens -ge 2 -and
                $normalizedGeneratedText -eq "41 42 43 44 45"
            )
        }
        placement = [ordered]@{
            status = "observed"
            compute_unit = "NPU"
            backend = "HTP"
            device_id = "HTP0"
        }
        timings_ms = [ordered]@{
            artifact_load = [math]::Round($artifactLoadMs, 6)
            model_load = [math]::Round(
                $modelLoadStopwatch.Elapsed.TotalMilliseconds,
                6
            )
            chat_template = [math]::Round(
                $templateStopwatch.Elapsed.TotalMilliseconds,
                6
            )
            tokenization = [math]::Round(
                [double]$tokenizerResult.tokenization_ms,
                6
            )
            prefill = [math]::Round($profile.prompt_time / 1000.0, 6)
            first_decode = [math]::Round($firstDecodeUs / 1000.0, 6)
            decode = [math]::Round($remainingDecodeUs / 1000.0, 6)
            generation_total = [math]::Round(
                ($profile.prompt_time + $profile.decode_time) / 1000.0,
                6
            )
            generation_host_total = [math]::Round(
                $generationHostStopwatch.Elapsed.TotalMilliseconds,
                6
            )
            request_stage_sum = [math]::Round(
                $requestStageSumMs,
                6
            )
            request_total = [math]::Round(
                $requestStopwatch.Elapsed.TotalMilliseconds,
                6
            )
        }
        timing_sources = [ordered]@{
            artifact_load = "instrumented_host_clock"
            model_load = "instrumented_host_clock"
            tokenization = "instrumented_host_clock"
            prefill = "geniex_runtime_report"
            first_decode = "derived_from_runtime_counters"
            decode = "derived_from_runtime_counters"
            generation_total = "derived_from_runtime_counters"
            request_total = "instrumented_host_clock"
        }
    }
    $record | ConvertTo-Json -Depth 8 | Set-Content `
        -LiteralPath $rawJsonPath `
        -Encoding UTF8 `
        -NoNewline

    Write-Host "Observed generated text:"
    Write-Host $generatedText
    Write-Host "Private raw JSON:"
    Write-Host $rawJsonPath
}
finally {
    if ($tokenHandle -ne $null -and $tokenHandle.IsAllocated) {
        $tokenHandle.Free()
    }
    if ($generatedTextPointer -ne [IntPtr]::Zero) {
        [T32Native]::geniex_free($generatedTextPointer)
    }
    if ($formattedPromptPointer -ne [IntPtr]::Zero) {
        [T32Native]::geniex_free($formattedPromptPointer)
    }
    if ($geniexHandle -ne [IntPtr]::Zero) {
        [void][T32Native]::geniex_llm_destroy($geniexHandle)
    }
    [void][T32Native]::geniex_deinit()

    foreach ($pointer in @(
        $modelNamePointer,
        $modelPathPointer,
        $pluginPointer,
        $devicePointer,
        $rolePointer,
        $contentPointer,
        $messagesPointer,
        $samplerPointer,
        $generationPointer
    )) {
        if ($pointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::FreeHGlobal($pointer)
        }
    }
    Stop-Transcript | Out-Null
}
