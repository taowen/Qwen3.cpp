param(
  [string]$Model = "qwen3_0_6b_vulkan_8da4w_kv_sdpa_dynamic.pte",
  [int]$Runs = 3,
  [int]$MaxNewTokens = 80
)

$ErrorActionPreference = 'Stop'
$runtimeRoot = Resolve-Path (Join-Path $PSScriptRoot '..\runtime')
$exe = Join-Path $runtimeRoot 'bin\llama_main.exe'
$modelPath = Join-Path $runtimeRoot ("models\{0}" -f $Model)
$tokenizerPath = Join-Path $runtimeRoot 'tokenizer\tokenizer.json'
$promptFile = Join-Path $runtimeRoot 'prompts\perf_prompt.txt'
$outDir = Join-Path $runtimeRoot 'bench'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$rows = @()
for ($i = 1; $i -le $Runs; $i++) {
  $stdout = Join-Path $outDir ("run{0}_stdout.log" -f $i)
  $stderr = Join-Path $outDir ("run{0}_stderr.log" -f $i)
  $args = @(
    '--model_path', $modelPath,
    '--tokenizer_path', $tokenizerPath,
    '--prompt_file', $promptFile,
    '--temperature', '0',
    '--max_new_tokens', $MaxNewTokens,
    '--ignore_eos'
  )
  $p = Start-Process -FilePath $exe -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  $obs = Select-String -Path $stdout -Pattern 'PyTorchObserver\s+\{.*\}' | Select-Object -Last 1
  if ($obs) {
    $obj = (($obs.Line -replace '^.*PyTorchObserver\s+', '') | ConvertFrom-Json)
    $rows += [pscustomobject]@{
      run = $i
      exit = $p.ExitCode
      prefill = [double]$obj.prefill_token_per_sec
      decode = [double]$obj.decode_token_per_sec
      prompt_tokens = [int]$obj.prompt_tokens
      generated_tokens = [int]$obj.generated_tokens
    }
  } else {
    $rows += [pscustomobject]@{ run=$i; exit=$p.ExitCode; prefill=$null; decode=$null; prompt_tokens=$null; generated_tokens=$null }
  }
}

$runsCsv = Join-Path $outDir 'runs.csv'
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $runsCsv

$ok = @($rows | Where-Object { $_.prefill -ne $null })
$summary = [pscustomobject]@{
  runs = $Runs
  ok_runs = $ok.Count
  avg_prefill = if ($ok.Count) { [math]::Round((($ok | Measure-Object prefill -Average).Average), 4) } else { $null }
  avg_decode = if ($ok.Count) { [math]::Round((($ok | Measure-Object decode -Average).Average), 4) } else { $null }
}
$summaryCsv = Join-Path $outDir 'summary.csv'
$summary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryCsv
$summary | Format-List
Write-Host "runs=$runsCsv"
Write-Host "summary=$summaryCsv"
