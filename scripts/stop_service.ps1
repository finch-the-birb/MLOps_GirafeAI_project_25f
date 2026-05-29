param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [int]$Port,
    [string]$PidFile = ""
)

function Stop-ProcessTree {
    param([int]$ProcessId)
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ParentProcessId -eq $ProcessId } |
        ForEach-Object { Stop-ProcessTree -ProcessId $_.ProcessId }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

if ($PidFile -and (Test-Path $PidFile)) {
    $filePid = [int](Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($filePid -gt 0) {
        Stop-ProcessTree -ProcessId $filePid
        Write-Output "Stopped $Name (pid file: $filePid)"
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

$listenerPids = @(
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        Where-Object { $_ -gt 0 }
)

foreach ($procId in $listenerPids) {
    Stop-ProcessTree -ProcessId $procId
    Write-Output "Stopped $Name (port $Port listener pid $procId)"
}

# Fallback: orphaned uv/python workers (e.g. mlflow server started without pid file)
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -in @("python.exe", "python3.exe", "uv.exe") -and
        $_.CommandLine -and (
            ($Name -eq "mlflow" -and $_.CommandLine -match "mlflow") -or
            ($Name -eq "api" -and $_.CommandLine -match "uvicorn" -and $_.CommandLine -match ":$Port|--port\s+$Port") -or
            ($Name -eq "ui" -and $_.CommandLine -match "streamlit") -or
            ($Name -eq "docs" -and $_.CommandLine -match "mkdocs")
        )
    } |
    ForEach-Object {
        Stop-ProcessTree -ProcessId $_.ProcessId
        Write-Output "Stopped $Name (matched cmdline pid $($_.ProcessId))"
    }
