; SpyDE Windows installer (NSIS) — per-user, no admin.
;
; Layout shipped to the install dir (staged by tools/build_installer_payload.py):
;   pyproject.toml, uv.lock, main.py, spyde\, installer\launch.py, uv.exe,
;   SpyDE.exe (launcher stub), Spyde.ico
;
; First launch runs `uv sync` to build the managed venv with the GPU-correct
; torch wheel; updates re-sync incrementally.
;
; Build:  makensis /DVERSION=0.1.0 installer\spyde.nsi
; Output: dist\SpyDE-Setup-<version>.exe

!ifndef VERSION
  !define VERSION "0.0.0"
!endif

!define APPNAME    "SpyDE"
!define COMPANY    "Direct Electron"
!define PAYLOAD    "..\dist\installer_payload"   ; staged by the build script
!define ICON       "..\spyde\Spyde.ico"

Name "${APPNAME} ${VERSION}"
OutFile "..\dist\SpyDE-Setup-${VERSION}.exe"
Unicode True
RequestExecutionLevel user                       ; per-user, no UAC
InstallDir "$LOCALAPPDATA\Programs\${APPNAME}"
InstallDirRegKey HKCU "Software\${APPNAME}" "InstallDir"
SetCompressor /SOLID lzma
!if /FileExists "${ICON}"
  Icon "${ICON}"
  UninstallIcon "${ICON}"
!endif

!include "MUI2.nsh"
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\SpyDE.exe"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"

Section "SpyDE" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  ; Whole staged payload (project + uv + launcher stub).
  File /r "${PAYLOAD}\*.*"

  ; Start-menu + desktop shortcuts → the launcher stub.
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\SpyDE.exe" "" "$INSTDIR\Spyde.ico"
  CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\SpyDE.exe" "" "$INSTDIR\Spyde.ico"

  ; File associations (best-effort; HKCU so no admin needed).
  WriteRegStr HKCU "Software\Classes\.hspy" "" "SpyDE.Dataset"
  WriteRegStr HKCU "Software\Classes\.zspy" "" "SpyDE.Dataset"
  WriteRegStr HKCU "Software\Classes\SpyDE.Dataset" "" "SpyDE Dataset"
  WriteRegStr HKCU "Software\Classes\SpyDE.Dataset\DefaultIcon" "" "$INSTDIR\Spyde.ico"
  WriteRegStr HKCU "Software\Classes\SpyDE.Dataset\shell\open\command" "" '"$INSTDIR\SpyDE.exe" "%1"'

  ; Add/Remove Programs entry.
  WriteRegStr HKCU "Software\${APPNAME}" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayName" "${APPNAME}"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTKEY}" "Publisher" "${COMPANY}"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\Spyde.ico"
  WriteRegStr HKCU "${UNINSTKEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoRepair" 1

  WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Uninstall"
  ; Remove the managed venv + everything we installed.
  RMDir /r "$INSTDIR\.venv"
  RMDir /r "$INSTDIR\spyde"
  RMDir /r "$INSTDIR\installer"
  Delete "$INSTDIR\*.*"
  RMDir /r "$INSTDIR"

  Delete "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk"
  RMDir  "$SMPROGRAMS\${APPNAME}"
  Delete "$DESKTOP\${APPNAME}.lnk"

  DeleteRegKey HKCU "${UNINSTKEY}"
  DeleteRegKey HKCU "Software\${APPNAME}"
  DeleteRegKey HKCU "Software\Classes\SpyDE.Dataset"
SectionEnd
