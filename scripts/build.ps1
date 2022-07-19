& "$PSScriptRoot/compile_resources.ps1"

$arguments = @(
  '--windowed',
  '--onefile',
  '--additional-hooks-dir=Pyinstaller/hooks',
  '--icon=res/icon.ico',
  '--splash=res/splash.png')
if ($IsLinux) {
  $arguments += @(
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
