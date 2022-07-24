& "$PSScriptRoot/compile_resources.ps1"

$arguments = @(
  '--noconfirm',
  '--windowed',
  '--onefile',
  '--additional-hooks-dir=Pyinstaller/hooks',
  '--icon=res/icon.ico')
if (-not $IsMacOS) {
  # Splash screen is not supported on macOS.
  # https://pyinstaller.org/en/stable/usage.html#splash-screen-experimental
  $arguments += @('--splash=res/splash.png')
}
if ($IsLinux) {
  $arguments += @(
    # Required on the CI for PyWinCtl
    '--hidden-import pynput.keyboard._xorg',
    '--hidden-import pynput.mouse._xorg')
}
Start-Process pyinstaller -Wait -ArgumentList "$arguments `"$PSScriptRoot/../src/AutoSplit.py`""

If ($IsLinux) {
  Move-Item -Force $PSScriptRoot/../dist/AutoSplit $PSScriptRoot/../dist/AutoSplit.elf
  If ($?) {
    Write-Host 'Added .elf extension'
  }
  chmod +x $PSScriptRoot/../dist/AutoSplit.elf
  Write-Host 'Added execute permission'
}
