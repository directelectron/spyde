; installer.nsh -- SpyDE's replacement for electron-builder's "is the app still
; running?" check. electron-builder picks this file up automatically from
; buildResources (the `nsis.include` default), and defining
; `customCheckAppRunning` overrides the stock macro in both the installer and
; the uninstaller.
;
; WHY THIS EXISTS
;
; The stock check (app-builder-lib/templates/nsis/include/
; allowOnlyOneInstallerInstance.nsh) asks PowerShell for every process whose
; executable path sits under $INSTDIR, stops them one at a time with
; Stop-Process, re-checks with no pause, and after two failed rounds ends in a
; "SpyDE cannot be closed. Please close it manually and click Retry" box whose
; Retry re-runs the identical check. That is a dead end for the user: nothing
; they can do in the dialog changes the answer, and the only way out is to
; uninstall by hand.
;
; Three things make it trip during a real SpyDE update:
;
;   * Stop-Process ends ONE process. SpyDE runs a Python sidecar which itself
;     runs Dask workers, so killing a parent leaves children that are found
;     again on the next round.
;   * The install directory holds more than the app binary -- resources\python
;     carries uv and a vendored git -- so the path-prefix match can catch a
;     helper the app is mid-way through running.
;   * Re-checking with no delay counts processes that are already exiting.
;   * MATCHING ON THE EXECUTABLE PATH MISSES THE PYTHON SIDECAR ENTIRELY. Its
;     interpreter lives in the managed env under $APPDATA, not in $INSTDIR --
;     only its working directory was in the install dir. A working directory is
;     an open handle, so an orphaned sidecar (and everything it spawned, which
;     inherits that directory) blocks the install dir from being removed while
;     staying invisible to any scan keyed on image path. That is why this pass
;     also sweeps $APPDATA\*\python-env\*, and why the sidecar no longer runs
;     from the install dir at all (de-shell's pythonEnv.ts).
;
; So: kill whole process TREES by pid, look outside $INSTDIR for the processes
; that hold it, wait between rounds, give the app several seconds to go before
; asking the user for anything, and make Retry actually retry the kill rather
; than just the question.
;
; THIS MACRO IS NOT THE ONLY SOURCE OF THAT DIALOG. electron-builder's
; uninstallOldVersion (app-builder-lib templates/nsis/include/installUtil.nsh)
; runs the PREVIOUS version's uninstaller and, if it returns non-zero five
; times, shows the very same "$(appCannotBeClosed)" -- with a Retry that only
; re-runs the uninstaller and never kills anything. There is no hook to override
; that loop, so the only way to stay out of it is to leave nothing holding the
; install directory by the time it runs. Hence the sweep above.
;
; The app side of the same race is in packages/shell-main/src/updater.ts -- it
; tears the sidecar down and force-exits rather than trusting Electron's
; graceful quit to win. Both halves are needed: this macro cannot reach a
; sidecar that lives outside $INSTDIR, and the app cannot clean up after a
; crash it did not survive.

Var spydeSetupPid

!macro customCheckAppRunning
  Push $0
  Push $R0
  Push $R1

  ; Our own pid, so a tree-kill can never take down the installer itself. (The
  ; stock macro's $pid is not declared when this override is in play.)
  System::Call 'kernel32::GetCurrentProcessId() i .r0'
  StrCpy $spydeSetupPid $0

  StrCpy $R1 0   ; rounds spent asking the user

  spydeCloseRetry:
    DetailPrint "Closing ${PRODUCT_NAME}..."

    ; One PowerShell pass: up to 12 rounds of "find everything holding the
    ; install directory, kill each one's whole tree, wait". Exits 0 the moment
    ; nothing is left, 1 if the directory is still busy after ~6 s. Two things
    ; count as holding it: a process running out of $INSTDIR, and one running
    ; out of the managed Python env ($APPDATA\...\python-env\...), which is the
    ; sidecar -- see the header for why it needs naming separately.
    ; The trailing backslash on the directory keeps a sibling install (...\spyde
    ; vs ...\spyde-old) from matching. `$$` is NSIS's escape for a literal `$`,
    ; so `$$_` reaches PowerShell as `$_`.
    nsExec::Exec `"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -Command "$$dir='$INSTDIR\'; $$roam='$APPDATA\'; $$self=$spydeSetupPid; for($$round=0; $$round -lt 12; $$round++){ $$busy=@(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $$p=$$_.Path; $$p -and $$_.ProcessId -ne $$self -and ($$p.StartsWith($$dir,'CurrentCultureIgnoreCase') -or ($$p.StartsWith($$roam,'CurrentCultureIgnoreCase') -and $$p -like '*\python-env\*')) }); if($$busy.Count -eq 0){ exit 0 }; foreach($$victim in $$busy){ $$null = & taskkill.exe /PID $$victim.ProcessId /T /F 2>&1 }; Start-Sleep -Milliseconds 500 }; exit 1"`
    Pop $R0

    ; nsExec pushes "error" when the program could not be started at all -- a
    ; machine with no PowerShell, or one where it is blocked by policy. Fall
    ; back to killing the app's own image tree, which needs nothing but
    ; taskkill. It exits 128 ("no tasks") when there was nothing to kill, so
    ; both 0 and 128 mean the directory is clear afterwards.
    ${if} $R0 == "error"
      nsExec::Exec `"$SYSDIR\cmd.exe" /C taskkill /IM "${APP_EXECUTABLE_FILENAME}" /T /F /FI "USERNAME eq %USERNAME%"`
      Pop $R0
      Sleep 1000
      ${if} $R0 == 128
        StrCpy $R0 0
      ${endif}
    ${endif}

    ${if} $R0 == 0
      Goto spydeClosed
    ${endif}

    ; Still busy. An update handoff is allowed one silent second round: the app
    ; that spawned this installer may simply be slow to die.
    IntOp $R1 $R1 + 1
    ${if} $R1 < 2
      Sleep 1000
      Goto spydeCloseRetry
    ${endif}

    ; Genuinely stuck -- something is holding the install directory that we are
    ; not allowed to end. Now the user's help is worth asking for, and Retry
    ; runs the whole kill pass again rather than re-asking the same question.
    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "$(appCannotBeClosed)" /SD IDCANCEL IDRETRY spydeCloseRetry
    Quit

  spydeClosed:
  Pop $R1
  Pop $R0
  Pop $0
!macroend
