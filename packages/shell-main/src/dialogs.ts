/**
 * dialogs.ts — native open/save dialogs, on the shell's channel.
 *
 * A sandboxed renderer cannot open a file dialog and has no filesystem paths,
 * so this is the only way an app can ask the user for a file. Registered once
 * per app by `registerShellDialogs`, on `<appId>:open-file` / `<appId>:save-file`.
 *
 * Both resolve to a PATH or null. Null means cancelled, which is a normal
 * outcome and not an error — the caller should do nothing rather than report a
 * failure.
 */
import { BrowserWindow, dialog, ipcMain } from 'electron'

import { channel } from './config'

export interface FileFilter { name: string; extensions: string[] }

/** Wire the open/save dialog handlers. Call once, after `configureShell`. */
export function registerShellDialogs(): void {
  ipcMain.handle(channel('open-file'), async (event, filters?: FileFilter[]) => {
    // Parent the dialog to the window that asked, so it is modal to that
    // window rather than floating free of the app.
    const win = BrowserWindow.fromWebContents(event.sender)
    const opts = { properties: ['openFile' as const], filters }
    const result = win
      ? await dialog.showOpenDialog(win, opts)
      : await dialog.showOpenDialog(opts)
    return result.canceled || !result.filePaths.length ? null : result.filePaths[0]
  })

  ipcMain.handle(channel('save-file'),
    async (event, filters?: FileFilter[], defaultPath?: string) => {
      const win = BrowserWindow.fromWebContents(event.sender)
      const opts = { filters, defaultPath }
      const result = win
        ? await dialog.showSaveDialog(win, opts)
        : await dialog.showSaveDialog(opts)
      return result.canceled || !result.filePath ? null : result.filePath
    })
}
