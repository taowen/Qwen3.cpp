param(
  [Parameter(Mandatory = $true, Position = 0)]
  [ValidateSet("baseline", "env", "doctor", "export", "pack", "run", "bench", "bench-compare", "chatbot-build", "chatbot-run", "vendor-sync-runtime", "vendor-sync-vulkan")]
  [string]$Command,

  [string]$ConfigPath = "",
  [switch]$DryRun,
  [switch]$Recreate,
  [string]$PythonVersion = "",

  [string]$Model = "",
  [string]$OutputName = "",
  [int]$Runs = 0,
  [int]$MaxNewTokens = 0,
  [double]$Temperature = [double]::NaN,
  [switch]$IgnoreEos,
  [switch]$NoIgnoreEos,
  [string]$PromptFile = "",
  [string]$Prompt = "",
  [string]$OutDir = "",

  [string]$BuildDir = "",
  [string]$ExecuTorchRepoRoot = "",
  [string]$ExecuTorchRoot = "",
  [string]$ExecuTorchPrefix = "",
  [string]$TokenizerPath = "",

  [ValidateSet("pull", "push")]
  [string]$Mode = "pull",

  [switch]$SkipExternal,
  [switch]$SkipRuntimeArtifacts
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "lib\common.ps1")

$repoRoot = Get-RepoRoot -FromPath $PSScriptRoot
$cfgBundle = Import-Qwen3Config -RepoRoot $repoRoot -ConfigPath $ConfigPath
$cfg = $cfgBundle.Data

function Select-StringValue {
  param([string]$Override, [string]$Default)
  if (![string]::IsNullOrWhiteSpace($Override)) { return $Override }
  return $Default
}

function Select-IntValue {
  param([int]$Override, [int]$Default)
  if ($Override -gt 0) { return $Override }
  return $Default
}

function Select-DoubleValue {
  param([double]$Override, [double]$Default)
  if (![double]::IsNaN($Override)) { return $Override }
  return $Default
}

function Resolve-ConfigPathValue {
  param([string]$Override, [string]$Default)
  $effective = Select-StringValue -Override $Override -Default $Default
  return Resolve-FullPath -Path $effective -BaseDir $repoRoot
}

if ($IgnoreEos -and $NoIgnoreEos) {
  throw "Cannot use -IgnoreEos and -NoIgnoreEos at the same time."
}

$runtimeRoot = Resolve-FullPath -Path $cfg.Paths.RuntimeRoot -BaseDir $repoRoot
$defaultModel = $cfg.Paths.DefaultModelArtifact
$defaultBuildDir = $cfg.Paths.ExecuTorchBuildDir
$defaultUpstreamRepo = $cfg.Paths.ExecuTorchRepoRoot
$defaultExecuTorchRoot = $cfg.Paths.ExecuTorchRoot
$defaultExecuTorchPrefix = $cfg.Paths.ExecuTorchInstallPrefix
$defaultPromptFile = Resolve-FullPath -Path $cfg.Paths.RuntimePromptFile -BaseDir $repoRoot
$defaultTokenizerPath = $cfg.Paths.TokenizerPath

switch ($Command) {
  "baseline" {
    Write-Step "Freezing baseline metadata"
    $docsDir = Join-Path $repoRoot "docs"
    Ensure-Directory -Path $docsDir

    $commit = (git -C $repoRoot rev-parse HEAD).Trim()
    $branch = (git -C $repoRoot rev-parse --abbrev-ref HEAD).Trim()
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"

    $baselineMd = @"
# Baseline Snapshot

- Timestamp: $timestamp
- Branch: $branch
- Commit: $commit
- Local tag: baseline-pre-refactor
- Config: $($cfgBundle.Path)

## Current Workflow

1. .\scripts\qwen3.ps1 env
2. .\scripts\qwen3.ps1 export
3. .\scripts\qwen3.ps1 pack
4. .\scripts\qwen3.ps1 run
5. .\scripts\qwen3.ps1 bench

## Key Paths

- Repo root: $repoRoot
- Runtime root: $runtimeRoot
- External ExecuTorch repo: $defaultUpstreamRepo
- External ExecuTorch source root: $defaultExecuTorchRoot
- External ExecuTorch install prefix: $defaultExecuTorchPrefix
"@

    $baselinePath = Join-Path $docsDir "BASELINE.md"
    if ($DryRun) {
      Write-Step "DryRun enabled; skip writing $baselinePath"
      Write-Step "DryRun enabled; skip git tag update"
    } else {
      Set-Content -Path $baselinePath -Value $baselineMd -Encoding UTF8
      git -C $repoRoot tag -f baseline-pre-refactor | Out-Null
    }
    Write-Step "Baseline updated: $baselinePath"
    break
  }

  "env" {
    $targetPy = Select-StringValue -Override $PythonVersion -Default $cfg.Python.Version
    Write-Step "Preparing uv environment (.venv, python=$targetPy)"

    Push-Location $repoRoot
    try {
      if ($Recreate -and (Test-Path ".venv")) {
        if ($DryRun) {
          Write-Step "DryRun: Remove-Item .venv"
        } else {
          Remove-Item -Recurse -Force ".venv"
        }
      }

      if ($DryRun) {
        if (Test-Path ".venv") {
          Write-Step "DryRun: keep existing .venv"
        } else {
          Write-Step "DryRun: uv venv .venv --python $targetPy"
        }
        Write-Step "DryRun: uv sync --locked"
      } else {
        if (!(Test-Path ".venv")) {
          Invoke-Checked -Executable "uv" -Arguments @("venv", ".venv", "--python", $targetPy)
        } else {
          Write-Step "Using existing .venv"
        }
        Invoke-Checked -Executable "uv" -Arguments @("sync", "--locked")
      }
    } finally {
      Pop-Location
    }
    break
  }

  "doctor" {
    Write-Step "Running doctor checks"
    $errors = @()

    if ($null -eq (Get-Command uv -ErrorAction SilentlyContinue)) {
      $errors += "uv command not found in PATH."
    }

    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (!(Test-Path $venvPython)) {
      $errors += "Missing virtualenv python: $venvPython"
    }

    foreach ($required in @("scripts", "runtime", "vendor-runtime", "vendor-executorch")) {
      $p = Join-Path $repoRoot $required
      if (!(Test-Path $p)) {
        $errors += "Missing required directory: $p"
      }
    }

    if (!$SkipExternal) {
      $upstreamRepo = Resolve-ConfigPathValue -Override $ExecuTorchRepoRoot -Default $defaultUpstreamRepo
      $buildDir = Select-StringValue -Override $BuildDir -Default $defaultBuildDir
      $flatc = Join-Path $upstreamRepo ($buildDir + "\third-party\flatc_ep\bin\flatc.exe")
      if (!(Test-Path $upstreamRepo)) {
        $errors += "Missing external ExecuTorch repo: $upstreamRepo"
      }
      if (!(Test-Path $flatc)) {
        $errors += "Missing external flatc: $flatc"
      }
    }

    if (!$SkipRuntimeArtifacts) {
      $runner = Join-Path $runtimeRoot "bin\llama_main.exe"
      $model = Join-Path $runtimeRoot ("models\" + $defaultModel)
      $tokenizer = Join-Path $runtimeRoot "tokenizer\tokenizer.json"
      foreach ($p in @($runner, $model, $tokenizer)) {
        if (!(Test-Path $p)) {
          $errors += "Missing runtime artifact: $p"
        }
      }
    }

    if ($errors.Count -gt 0) {
      $errors | ForEach-Object { Write-Host "[doctor] $_" }
      throw "Doctor failed with $($errors.Count) issue(s)."
    }

    Write-Step "Doctor passed"
    break
  }

  "export" {
    $upstreamRepo = Resolve-ConfigPathValue -Override $ExecuTorchRepoRoot -Default $defaultUpstreamRepo
    $buildDir = Select-StringValue -Override $BuildDir -Default $defaultBuildDir
    $outName = Select-StringValue -Override $OutputName -Default $defaultModel

    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $flatc = Join-Path $upstreamRepo ($buildDir + "\third-party\flatc_ep\bin\flatc.exe")
    $dstModel = Join-Path $runtimeRoot ("models\" + $outName)

    if (!$DryRun) {
      Require-Path -Path $python -Description "local venv python"
      Require-Path -Path $upstreamRepo -Description "external ExecuTorch repo"
      Require-Path -Path $flatc -Description "flatc"
      Ensure-Directory -Path (Split-Path $dstModel -Parent)
    }

    $args = @(
      "-m", "executorch.examples.models.llama.export_llama",
      "--model", $cfg.Export.ModelId,
      "--params", $cfg.Export.ParamsRelativePath,
      "-qmode", $cfg.Export.QuantMode,
      "-G", [string]$cfg.Export.GroupSize
    )
    if ($cfg.Export.EnableVulkan) { $args += "-V" }
    if ($cfg.Export.EnableKVCache) { $args += "-kv" }
    if ($cfg.Export.EnableSDPAWithKVCache) { $args += "--use_sdpa_with_kv_cache" }
    $args += @("--output_name", $outName)

    $oldPyIo = $env:PYTHONIOENCODING
    $oldPyPath = $env:PYTHONPATH
    $oldFlatc = $env:FLATC_EXECUTABLE
    try {
      $env:PYTHONIOENCODING = "utf-8"
      $env:PYTHONPATH = Join-Path $upstreamRepo "third_party"
      $env:FLATC_EXECUTABLE = $flatc

      if ($DryRun) {
        Write-Step "DryRun: $python $($args -join ' ')"
      } else {
        Invoke-Checked -Executable $python -Arguments $args -WorkingDirectory $upstreamRepo
        Copy-Item -Force (Join-Path $upstreamRepo $outName) $dstModel
      }
    } finally {
      $env:PYTHONIOENCODING = $oldPyIo
      $env:PYTHONPATH = $oldPyPath
      $env:FLATC_EXECUTABLE = $oldFlatc
    }

    Write-Step "Export completed: $dstModel"
    break
  }

  "pack" {
    $upstreamRepo = Resolve-ConfigPathValue -Override $ExecuTorchRepoRoot -Default $defaultUpstreamRepo
    $buildDir = Select-StringValue -Override $BuildDir -Default $defaultBuildDir
    $modelName = Select-StringValue -Override $Model -Default $defaultModel
    $tokenizer = Resolve-ConfigPathValue -Override $TokenizerPath -Default $defaultTokenizerPath

    $runnerSrc = Join-Path $upstreamRepo ($buildDir + "\" + $cfg.Paths.RunnerRelativePath)
    $modelSrc = Join-Path $upstreamRepo $modelName
    $promptSrc = Join-Path $upstreamRepo $cfg.Paths.PromptSourceFile

    $runnerDst = Join-Path $runtimeRoot "bin\llama_main.exe"
    $modelDst = Join-Path $runtimeRoot ("models\" + [System.IO.Path]::GetFileName($modelName))
    $tokDst = Join-Path $runtimeRoot "tokenizer\tokenizer.json"
    $promptDst = Join-Path $runtimeRoot "prompts\perf_prompt.txt"

    foreach ($dir in @((Split-Path $runnerDst -Parent), (Split-Path $modelDst -Parent), (Split-Path $tokDst -Parent), (Split-Path $promptDst -Parent))) {
      Ensure-Directory -Path $dir
    }

    if ($DryRun) {
      Write-Step "DryRun: copy $runnerSrc -> $runnerDst"
      Write-Step "DryRun: copy $modelSrc -> $modelDst"
      Write-Step "DryRun: copy $tokenizer -> $tokDst"
      Write-Step "DryRun: copy $promptSrc -> $promptDst"
    } else {
      foreach ($p in @($runnerSrc, $modelSrc, $tokenizer, $promptSrc)) {
        Require-Path -Path $p -Description "pack source"
      }
      Copy-Item -Force $runnerSrc $runnerDst
      Copy-Item -Force $modelSrc $modelDst
      Copy-Item -Force $tokenizer $tokDst
      Copy-Item -Force $promptSrc $promptDst
    }
    Write-Step "Pack completed: $runtimeRoot"
    break
  }

  "run" {
    $modelName = Select-StringValue -Override $Model -Default $defaultModel
    $maxTok = Select-IntValue -Override $MaxNewTokens -Default $cfg.Runtime.DefaultMaxNewTokens
    $temp = Select-DoubleValue -Override $Temperature -Default $cfg.Runtime.DefaultTemperature

    $ignore = [bool]$cfg.Runtime.DefaultIgnoreEos
    if ($IgnoreEos) { $ignore = $true }
    if ($NoIgnoreEos) { $ignore = $false }

    $exe = Join-Path $runtimeRoot "bin\llama_main.exe"
    $modelPath = Join-Path $runtimeRoot ("models\" + $modelName)
    $tokenizerPath = Join-Path $runtimeRoot "tokenizer\tokenizer.json"
    $promptPath = $defaultPromptFile

    if (![string]::IsNullOrWhiteSpace($PromptFile)) {
      $promptPath = Resolve-FullPath -Path $PromptFile -BaseDir $repoRoot
    }

    $args = @(
      "--model_path", $modelPath,
      "--tokenizer_path", $tokenizerPath,
      "--temperature", [string]$temp,
      "--max_new_tokens", [string]$maxTok
    )
    if ($ignore) { $args += "--ignore_eos" }
    if (![string]::IsNullOrWhiteSpace($Prompt)) {
      $args += @("--prompt", $Prompt)
    } else {
      $args += @("--prompt_file", $promptPath)
    }

    if ($DryRun) {
      Write-Step "DryRun: $exe $($args -join ' ')"
    } else {
      foreach ($p in @($exe, $modelPath, $tokenizerPath)) {
        Require-Path -Path $p -Description "run input"
      }
      if ([string]::IsNullOrWhiteSpace($Prompt)) {
        Require-Path -Path $promptPath -Description "prompt file"
      }
      & $exe @args
      if ($LASTEXITCODE -ne 0) {
        throw "Run failed with exit code $LASTEXITCODE"
      }
    }
    break
  }

  "bench" {
    $modelName = Select-StringValue -Override $Model -Default $defaultModel
    $runsCount = Select-IntValue -Override $Runs -Default $cfg.Bench.DefaultRuns
    $maxTok = Select-IntValue -Override $MaxNewTokens -Default $cfg.Runtime.DefaultMaxNewTokens

    $exe = Join-Path $runtimeRoot "bin\llama_main.exe"
    $modelPath = Join-Path $runtimeRoot ("models\" + $modelName)
    $tokenizerPath = Join-Path $runtimeRoot "tokenizer\tokenizer.json"
    $promptPath = $defaultPromptFile
    if (![string]::IsNullOrWhiteSpace($PromptFile)) {
      $promptPath = Resolve-FullPath -Path $PromptFile -BaseDir $repoRoot
    }

    $outDir = Join-Path $runtimeRoot "bench"
    Ensure-Directory -Path $outDir

    if ($DryRun) {
      Write-Step "DryRun: bench runs=$runsCount max_new_tokens=$maxTok"
      break
    }

    foreach ($p in @($exe, $modelPath, $tokenizerPath, $promptPath)) {
      Require-Path -Path $p -Description "bench input"
    }

    $rows = @()
    for ($i = 1; $i -le $runsCount; $i++) {
      $stdout = Join-Path $outDir ("run{0}_stdout.log" -f $i)
      $stderr = Join-Path $outDir ("run{0}_stderr.log" -f $i)
      $args = @(
        "--model_path", $modelPath,
        "--tokenizer_path", $tokenizerPath,
        "--prompt_file", $promptPath,
        "--temperature", "0",
        "--max_new_tokens", [string]$maxTok,
        "--ignore_eos"
      )

      $p = Start-Process -FilePath $exe -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
      $obs = Select-String -Path $stdout -Pattern "PyTorchObserver\s+\{.*\}" | Select-Object -Last 1
      if ($obs) {
        $obj = (($obs.Line -replace "^.*PyTorchObserver\s+", "") | ConvertFrom-Json)
        $rows += [pscustomobject]@{
          run = $i
          exit = $p.ExitCode
          prefill = [double]$obj.prefill_token_per_sec
          decode = [double]$obj.decode_token_per_sec
          prompt_tokens = [int]$obj.prompt_tokens
          generated_tokens = [int]$obj.generated_tokens
        }
      } else {
        $rows += [pscustomobject]@{
          run = $i
          exit = $p.ExitCode
          prefill = $null
          decode = $null
          prompt_tokens = $null
          generated_tokens = $null
        }
      }
    }

    $runsCsv = Join-Path $outDir "runs.csv"
    $rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $runsCsv

    $ok = @($rows | Where-Object { $_.prefill -ne $null })
    $summary = [pscustomobject]@{
      runs = $runsCount
      ok_runs = $ok.Count
      avg_prefill = if ($ok.Count) { [math]::Round((($ok | Measure-Object prefill -Average).Average), 4) } else { $null }
      avg_decode = if ($ok.Count) { [math]::Round((($ok | Measure-Object decode -Average).Average), 4) } else { $null }
    }
    $summaryCsv = Join-Path $outDir "summary.csv"
    $summary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryCsv
    $summary | Format-List
    Write-Host "runs=$runsCsv"
    Write-Host "summary=$summaryCsv"
    break
  }

  "bench-compare" {
    $perfScript = Join-Path $repoRoot "scripts\perf_compare.ps1"
    Require-Path -Path $perfScript -Description "perf compare script"

    $forward = @{}
    if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
    if ($Runs -gt 0) { $forward.Runs = $Runs }
    if ($MaxNewTokens -gt 0) { $forward.MaxNewTokens = $MaxNewTokens }
    if (![double]::IsNaN($Temperature)) { $forward.Temperature = $Temperature }
    if ($PromptFile) { $forward.PromptFile = $PromptFile }
    if ($OutDir) { $forward.OutDir = $OutDir }

    if ($DryRun) {
      Write-Step "DryRun: $perfScript"
    } else {
      & $perfScript @forward
      if ($LASTEXITCODE -ne 0) {
        throw "bench-compare failed with exit code $LASTEXITCODE"
      }
    }
    break
  }

  "chatbot-build" {
    $buildOutDir = Select-StringValue -Override $BuildDir -Default $cfg.Build.ChatbotBuildDir
    $prefix = Resolve-ConfigPathValue -Override $ExecuTorchPrefix -Default $defaultExecuTorchPrefix
    $generator = $cfg.Build.Generator
    $toolset = $cfg.Build.Toolset
    $srcDir = Join-Path $repoRoot "vendor-runtime"
    $binDir = Resolve-FullPath -Path $buildOutDir -BaseDir $repoRoot

    $configureArgs = @(
      "-S", $srcDir,
      "-B", $binDir,
      "-G", $generator,
      "-T", $toolset,
      "-DEXECUTORCH_PREFIX=$prefix"
    )

    if ($DryRun) {
      Write-Step "DryRun: cmake $($configureArgs -join ' ')"
      Write-Step "DryRun: cmake --build $binDir --config Release --target qwen3_chatbot -- /m"
    } else {
      Require-Path -Path $prefix -Description "ExecuTorch install prefix"
      Invoke-Checked -Executable "cmake" -Arguments $configureArgs
      Invoke-Checked -Executable "cmake" -Arguments @("--build", $binDir, "--config", "Release", "--target", "qwen3_chatbot", "--", "/m")
    }
    break
  }

  "chatbot-run" {
    $buildOutDir = Select-StringValue -Override $BuildDir -Default $cfg.Build.ChatbotBuildDir
    $modelName = Select-StringValue -Override $Model -Default $defaultModel
    $maxTok = Select-IntValue -Override $MaxNewTokens -Default $cfg.Runtime.DefaultMaxNewTokens
    $temp = Select-DoubleValue -Override $Temperature -Default $cfg.Runtime.DefaultTemperature

    $ignore = [bool]$cfg.Runtime.DefaultIgnoreEos
    if ($IgnoreEos) { $ignore = $true }
    if ($NoIgnoreEos) { $ignore = $false }

    $exe = Resolve-FullPath -Path (Join-Path $buildOutDir "Release\qwen3_chatbot.exe") -BaseDir $repoRoot
    $modelPath = Join-Path $runtimeRoot ("models\" + $modelName)
    $tokPath = Resolve-ConfigPathValue -Override $TokenizerPath -Default (Join-Path $runtimeRoot "tokenizer\tokenizer.json")

    $args = @(
      "--model_path", $modelPath,
      "--tokenizer_path", $tokPath,
      "--max_new_tokens", [string]$maxTok,
      "--temperature", [string]$temp,
      "--ignore_eos", $(if ($ignore) { "1" } else { "0" })
    )

    if ($DryRun) {
      Write-Step "DryRun: $exe $($args -join ' ')"
    } else {
      foreach ($p in @($exe, $modelPath, $tokPath)) {
        Require-Path -Path $p -Description "chatbot input"
      }
      & $exe @args
      if ($LASTEXITCODE -ne 0) {
        throw "chatbot run failed with exit code $LASTEXITCODE"
      }
    }
    break
  }

  "vendor-sync-runtime" {
    $srcExecuTorch = Resolve-ConfigPathValue -Override $ExecuTorchRoot -Default $defaultExecuTorchRoot
    $srcRoot = Join-Path $srcExecuTorch "examples\models\llama"
    $dstRoot = Join-Path $repoRoot "vendor-runtime"

    Require-Path -Path $srcRoot -Description "ExecuTorch llama source root"
    Require-Path -Path $dstRoot -Description "vendor runtime root"

    $map = @(
      @{ src = "runner\runner.cpp"; dst = "src\runner.cpp" },
      @{ src = "runner\runner.h"; dst = "include\runner.h" },
      @{ src = "tokenizer\llama_tiktoken.cpp"; dst = "src\llama_tiktoken.cpp" },
      @{ src = "tokenizer\llama_tiktoken.h"; dst = "include\llama_tiktoken.h" }
    )

    foreach ($entry in $map) {
      $srcFile = Join-Path $srcRoot $entry.src
      $dstFile = Join-Path $dstRoot $entry.dst
      $upstreamSnapshot = Join-Path $dstRoot ("upstream\" + [IO.Path]::GetFileName($entry.src))
      if ($DryRun) {
        Write-Step "DryRun: copy $srcFile -> $dstFile"
      } else {
        Copy-Item -Force $srcFile $dstFile
        Copy-Item -Force $srcFile $upstreamSnapshot
      }
    }

    if (!$DryRun) {
      (Get-Content (Join-Path $dstRoot "src\runner.cpp")) `
        -replace "<executorch/examples/models/llama/runner/runner.h>", '"runner.h"' `
        -replace "<executorch/examples/models/llama/tokenizer/llama_tiktoken.h>", '"llama_tiktoken.h"' |
        Set-Content (Join-Path $dstRoot "src\runner.cpp") -Encoding UTF8

      (Get-Content (Join-Path $dstRoot "include\runner.h")) `
        -replace "<executorch/examples/models/llama/tokenizer/llama_tiktoken.h>", '"llama_tiktoken.h"' |
        Set-Content (Join-Path $dstRoot "include\runner.h") -Encoding UTF8

      (Get-Content (Join-Path $dstRoot "src\llama_tiktoken.cpp")) `
        -replace "<executorch/examples/models/llama/tokenizer/llama_tiktoken.h>", '"llama_tiktoken.h"' |
        Set-Content (Join-Path $dstRoot "src\llama_tiktoken.cpp") -Encoding UTF8
    }

    Write-Step "Vendor runtime sync complete: $dstRoot"
    break
  }

  "vendor-sync-vulkan" {
    $srcExecuTorch = Resolve-ConfigPathValue -Override $ExecuTorchRoot -Default $defaultExecuTorchRoot
    $upstreamRoot = Join-Path $srcExecuTorch "backends\vulkan"
    $vendorRoot = Join-Path $repoRoot "vendor-executorch\backends\vulkan"

    Require-Path -Path $upstreamRoot -Description "ExecuTorch Vulkan backend"
    Ensure-Directory -Path $vendorRoot

    $allowDirs = @("_passes", "cmake", "partitioner", "patterns", "quantizer", "runtime", "serialization", "third-party")
    $allowFiles = @("BUCK", "CMakeLists.txt", "README.md", "__init__.py", "custom_ops_lib.py", "op_registry.py", "targets.bzl", "utils.py", "vulkan_preprocess.py")

    if ($Mode -eq "pull") {
      foreach ($name in $allowDirs) {
        $src = Join-Path $upstreamRoot $name
        if (!(Test-Path $src)) { continue }
        $dst = Join-Path $vendorRoot $name
        $opts = @("/MIR", "/R:1", "/W:1", "/XF", "*.pyc", "/XD", "__pycache__")
        if ($name -eq "third-party") {
          $opts += @("/XD", "docs", "test", "tests", "tools", "media", ".github")
        }
        if ($DryRun) {
          Write-Step "DryRun: robocopy $src $dst $($opts -join ' ')"
        } else {
          Invoke-RobocopyChecked -Source $src -Target $dst -Options $opts
        }
      }

      foreach ($name in $allowFiles) {
        $srcFile = Join-Path $upstreamRoot $name
        if (!(Test-Path $srcFile)) { continue }
        $dstFile = Join-Path $vendorRoot $name
        if ($DryRun) {
          Write-Step "DryRun: copy $srcFile -> $dstFile"
        } else {
          Copy-Item -Force $srcFile $dstFile
        }
      }

      if (!$DryRun) {
        foreach ($item in Get-ChildItem -Force $vendorRoot) {
          if ($item.PSIsContainer) {
            if ($allowDirs -notcontains $item.Name) {
              Remove-Item -Recurse -Force $item.FullName
            }
          } else {
            if ($allowFiles -notcontains $item.Name) {
              Remove-Item -Force $item.FullName
            }
          }
        }
      }
    } else {
      foreach ($name in $allowDirs) {
        $src = Join-Path $vendorRoot $name
        if (!(Test-Path $src)) { continue }
        $dst = Join-Path $upstreamRoot $name
        $opts = @("/E", "/R:1", "/W:1", "/XF", "*.pyc", "/XD", "__pycache__")
        if ($name -eq "third-party") {
          $opts += @("/XD", "docs", "test", "tests", "tools", "media", ".github")
        }
        if ($DryRun) {
          Write-Step "DryRun: robocopy $src $dst $($opts -join ' ')"
        } else {
          Invoke-RobocopyChecked -Source $src -Target $dst -Options $opts
        }
      }

      foreach ($name in $allowFiles) {
        $srcFile = Join-Path $vendorRoot $name
        if (!(Test-Path $srcFile)) { continue }
        $dstFile = Join-Path $upstreamRoot $name
        if ($DryRun) {
          Write-Step "DryRun: copy $srcFile -> $dstFile"
        } else {
          Copy-Item -Force $srcFile $dstFile
        }
      }
    }

    Write-Step "Vendor Vulkan sync complete. mode=$Mode"
    break
  }
}
