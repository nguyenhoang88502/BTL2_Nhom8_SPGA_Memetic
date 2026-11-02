# demo.ps1 - khởi động demo SPGA gốc, GA-TS, GA-SA (Nhóm 8)
# Cách dùng: bấm đúp CHAY_DEMO.bat  (hoặc: powershell -ExecutionPolicy Bypass -File demo.ps1)
#
# Lưu ý: tên lệnh trong PowerShell không phân biệt hoa thường và hàm được ưu tiên hơn file .exe,
# nên tuyệt đối không đặt tên hàm/alias là py, python (trước đây `function Py` gọi lại chính nó).
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { $Host.UI.RawUI.WindowTitle = "Demo SPGA - Nhom 8 (dong cua so nay de tat demo)" } catch {}
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = "1"

function Write-Line($msg, $color = "Gray") { Write-Host $msg -ForegroundColor $color }

Write-Line "=============================================================" Cyan
Write-Line "  DEMO: SPGA gốc, GA-TS, GA-SA  -  Nhóm 8" Cyan
Write-Line "=============================================================" Cyan

# ---------------------------------------------------------------- 1. tìm Python 3
# Trả về đường dẫn python.exe thật nếu lệnh chạy được và là Python 3, ngược lại trả về $null.
# Bản giả của Microsoft Store (WindowsApps) không in ra được phiên bản nên tự bị loại.
function Resolve-RealPython($exePath, [string[]]$preArgs = @()) {
    try {
        $lines = @(& $exePath @preArgs -c "import sys; print(sys.version_info[0]); print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -ne 0 -or $lines.Count -lt 2 -or "$($lines[0])".Trim() -ne "3") { return $null }
        $real = "$($lines[1])".Trim()
        if ($real -and (Test-Path -LiteralPath $real)) { return $real }
    } catch {}
    return $null
}

function Find-RealPython {
    # -CommandType Application: chỉ lấy file .exe thật, không bao giờ trúng hàm hoặc alias.
    foreach ($item in @(@("py", "-3"), @("python"), @("python3"))) {
        $preArgs = @($item | Select-Object -Skip 1)
        foreach ($cmd in @(Get-Command $item[0] -CommandType Application -All -ErrorAction SilentlyContinue)) {
            $real = Resolve-RealPython $cmd.Source $preArgs
            if ($real) { return $real }
        }
    }
    $installed = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.Directory.Name -replace '\D', '') } -Descending
    foreach ($file in @($installed)) {
        $real = Resolve-RealPython $file.FullName
        if ($real) { return $real }
    }
    return $null
}

$PythonExe = Find-RealPython
if (-not $PythonExe) {
    Write-Line "`n[!] Không tìm thấy Python 3 trên máy." Red
    Write-Line "    Cài Python tại https://www.python.org/downloads/ (nhớ tick 'Add python.exe to PATH'), rồi chạy lại." Yellow
    exit 1
}
$PythonVersion = & $PythonExe -c "import sys; print(sys.version.split()[0])"
Write-Line "[1/3] Python $PythonVersion  ($PythonExe)" Green

# ---------------------------------------------------------------- 2. thư viện
function Test-PythonModule($module) {
    & $PythonExe -c "import $module" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}
if (-not (Test-PythonModule "numpy")) {
    Write-Line "[2/3] Đang cài numpy (cần mạng)..." Yellow
    & $PythonExe -m pip install --user --disable-pip-version-check numpy
    if (-not (Test-PythonModule "numpy")) {
        Write-Line "[!] Không cài được numpy. Thử thủ công: `"$PythonExe`" -m pip install numpy" Red
        exit 1
    }
}
$missing = @("scipy", "pandas", "matplotlib" | Where-Object { -not (Test-PythonModule $_) })
if ($missing.Count -gt 0) {
    Write-Line "[2/3] Đang cài thêm $($missing -join ', ') (cho nút 'Chạy lại main.py')..." Yellow
    & $PythonExe -m pip install --user --disable-pip-version-check @missing
    $still = @($missing | Where-Object { -not (Test-PythonModule $_) })
    if ($still.Count -gt 0) {
        Write-Line "      Cảnh báo: thiếu $($still -join ', ') -> demo vẫn chạy, chỉ nút 'Chạy lại main.py' không dùng được." DarkYellow
    }
}
Write-Line "[2/3] Thư viện OK" Green

# ---------------------------------------------------------------- 3. chạy server + mở trình duyệt
function Test-PortFree($p) {
    try {
        $listener = New-Object System.Net.Sockets.TcpListener ([System.Net.IPAddress]::Loopback, $p)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch { return $false }
}
$port = 8765
while (-not (Test-PortFree $port)) {
    $port++
    if ($port -gt 8865) {
        Write-Line "[!] Không tìm được cổng trống từ 8765 đến 8865." Red
        exit 1
    }
}
$url = "http://localhost:$port"

# Cùng cửa sổ console (-NoNewWindow) để log có màu của server hiện ra ở đây.
$server = Start-Process -FilePath $PythonExe -ArgumentList @("-u", "demo_server.py", "--port", "$port") `
    -WorkingDirectory $PSScriptRoot -NoNewWindow -PassThru
try {
    # Server chỉ nghe ở 127.0.0.1, còn Invoke-WebRequest tới localhost thử ::1 trước và mất khoảng 2 giây,
    # nên -TimeoutSec phải lớn hơn hẳn 2 (để 2 thì luôn hết giờ dù server đã chạy).
    $ready = $false
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline -and -not $server.HasExited) {
        try {
            Invoke-WebRequest "$url/api/defaults" -UseBasicParsing -TimeoutSec 5 | Out-Null
            $ready = $true
            break
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) {
        Write-Line "`n[!] Server không khởi động được (xem lỗi phía trên)." Red
        exit 1
    }
    Write-Line "[3/3] Demo đang chạy tại $url" Green
    Start-Process $url

    Write-Line "`n-------------------------------------------------------------" Cyan
    Write-Line "  Trình duyệt đã mở. Nếu không thấy, tự mở: $url" White
    Write-Line "  Cửa sổ này hiện LOG THUẬT TOÁN (thế hệ, Z tốt nhất, từng bước tìm kiếm cục bộ)." White
    Write-Line "  -> Đặt cạnh trình duyệt để thầy xem. GIỮ cửa sổ này mở trong lúc demo." White
    Write-Line "  Nhấn Enter để tắt demo." White
    Write-Line "-------------------------------------------------------------" Cyan
    Read-Host | Out-Null
}
finally {
    # /T: tắt cả tiến trình con (main.py khi đang chạy lại từ trình duyệt).
    if (-not $server.HasExited) { & taskkill.exe /PID $server.Id /T /F 2>$null | Out-Null }
    Write-Line "Đã tắt demo." DarkGray
}
