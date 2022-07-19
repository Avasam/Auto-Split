& "$PSScriptRoot/compile_resources.ps1"
pyinstaller `
  --windowed `
  --onefile `
  --additional-hooks-dir=Pyinstaller/hooks `
  --icon=res/icon.ico `
  --splash=res/splash.png `
  "$PSScriptRoot/../src/AutoSplit.py"

If ($IsLinux) {
  Move-Item -Force $PSScriptRoot/../dist/AutoSplit $PSScriptRoot/../dist/AutoSplit.elf
  If ($?) {
    Write-Host 'Added .elf extension'
  }
  chmod +x $PSScriptRoot/../dist/AutoSplit.elf
  Write-Host 'Added execute permission'
}
