/**
 * report_problem.spec.ts — Help → Report a Problem…
 *
 * Covers the whole path a user takes: menu → gate → dialog → preload IPC →
 * errorReport.ts's collect/submit → the outcome shown back. Two launches,
 * because the two outcomes that matter are configured by the environment:
 *
 *   * NO reporting service — the offline/fork case. The report must still be
 *     WRITTEN and its path shown, rather than the dialog failing and losing
 *     what the user typed.
 *   * A reporting service that ANSWERS — the send is hand-written against
 *     Sentry's ingest protocol (see sentryEnvelope.ts), so a local server
 *     standing in for Sentry is the only way to prove the auth header and the
 *     envelope framing actually leave the app intact. The unit tests check the
 *     strings; this checks that Electron's net stack delivers them.
 *
 * Drives the MenuBar.tsx HTML dropdown (the only menu on Linux CI; the native
 * Electron menu is macOS-only chrome around the same handlers), mirroring
 * update_gpu_dialogs.spec.ts.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { createServer, Server } from 'http'
import { AddressInfo } from 'net'
import { existsSync, readFileSync } from 'fs'
import { join } from 'path'

const shots = join(__dirname, '..', 'report_problem_shots')
const mainEntry = join(__dirname, '..', 'out', 'main', 'index.js')

async function openReportDialog(page: Page) {
  await page.getByTestId('menu-help').click()
  await expect(page.getByTestId('menu-help-items')).toBeVisible()
  await page.getByTestId('menu-item-report-a-problem-').click()
  await expect(page.getByTestId('report-problem-dialog')).toBeVisible()
}

test.describe('with no reporting service configured', () => {
  let app: ElectronApplication
  let page: Page

  test.beforeAll(async () => {
    app = await electron.launch({
      args: [mainEntry],
      env: { ...process.env, SPYDE_NO_DASK: '1', SPYDE_SENTRY_DSN: '' },
    })
    page = await app.firstWindow()
    await page.waitForLoadState('domcontentloaded')
    await page.waitForSelector('[data-testid="mdi-area"]')
  })
  test.afterAll(async () => { await app?.close() })

  test('the dialog shows the diagnostics before anything is sent', async () => {
    await openReportDialog(page)

    // Send stays disabled until there is something to report — a blank report
    // is worse than none.
    await expect(page.getByTestId('report-send')).toBeDisabled()

    await page.getByTestId('report-toggle-details').click()
    const details = page.getByTestId('report-details')
    await expect(details).toBeVisible()

    // The machine facts are the point of the report; assert the ones a
    // maintainer triages on actually arrived rather than that the box rendered.
    const diagnostics = JSON.parse((await details.textContent()) ?? '{}')
    expect(diagnostics.app?.name).toBe('SpyDE')
    expect(diagnostics.app?.version).toMatch(/^\d+\.\d+\.\d+/)
    expect(['darwin', 'win32', 'linux']).toContain(diagnostics.os?.platform)
    expect(diagnostics.runtime?.electron).toBeTruthy()
    expect(Array.isArray(diagnostics.problems)).toBe(true)

    await page.screenshot({ path: join(shots, '01-details.png') })

    await page.getByTestId('report-cancel').click()
    await expect(page.getByTestId('report-problem-dialog')).toBeHidden()
  })

  test('the report is still written to disk', async () => {
    await openReportDialog(page)

    await page.getByTestId('report-message').fill(
      'Auto-update failed on Windows 11: the installer said SpyDE cannot be closed.',
    )
    await page.getByTestId('report-contact').fill('eric@example.org')
    await page.screenshot({ path: join(shots, '02-filled.png') })

    await page.getByTestId('report-send').click()
    await expect(page.getByTestId('report-outcome')).toBeVisible()
    await page.screenshot({ path: join(shots, '03-saved.png') })

    // The saved path is shown, and the file it names actually exists and
    // carries both what the user wrote and the diagnostics.
    const bundlePath = (await page.getByTestId('report-bundle-path').textContent())?.trim()
    expect(bundlePath).toMatch(/spyde-report-.*\.json$/)
    expect(existsSync(bundlePath!)).toBe(true)

    const saved = JSON.parse(readFileSync(bundlePath!, 'utf8'))
    expect(saved.message).toContain('SpyDE cannot be closed')
    expect(saved.contact).toBe('eric@example.org')
    expect(saved.diagnostics.os.platform).toBeTruthy()
    expect(saved.eventId).toMatch(/^[0-9a-f]{32}$/)

    await page.getByTestId('report-close').click()
    await expect(page.getByTestId('report-problem-dialog')).toBeHidden()
  })
})

test.describe('with a reporting service that answers', () => {
  let app: ElectronApplication
  let page: Page
  let server: Server
  const received: Array<{ auth: string; contentType: string; body: string }> = []

  test.beforeAll(async () => {
    // Stands in for Sentry's ingest endpoint: same URL shape, same 200 reply.
    server = createServer((req, res) => {
      let body = ''
      req.on('data', (chunk) => { body += chunk })
      req.on('end', () => {
        received.push({
          auth: String(req.headers['x-sentry-auth'] ?? ''),
          contentType: String(req.headers['content-type'] ?? ''),
          body,
        })
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end('{"id":"server-side-id"}')
      })
    })
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    const { port } = server.address() as AddressInfo

    app = await electron.launch({
      args: [mainEntry],
      env: {
        ...process.env,
        SPYDE_NO_DASK: '1',
        SPYDE_SENTRY_DSN: `http://publickey@127.0.0.1:${port}/42`,
      },
    })
    page = await app.firstWindow()
    await page.waitForLoadState('domcontentloaded')
    await page.waitForSelector('[data-testid="mdi-area"]')
  })
  test.afterAll(async () => {
    await app?.close()
    await new Promise<void>((resolve) => server.close(() => resolve()))
  })

  test('the report reaches the service as a well-formed envelope', async () => {
    await openReportDialog(page)

    // The dialog must say where this is going before the user commits to it.
    await expect(page.getByText(/straight to the SpyDE maintainers/)).toBeVisible()

    await page.getByTestId('report-message').fill('The navigator freezes when I scrub a .zspy.')
    await page.getByTestId('report-send').click()
    await expect(page.getByTestId('report-outcome')).toBeVisible()
    await page.screenshot({ path: join(shots, '04-sent.png') })

    await expect(page.getByText(/Report sent/)).toBeVisible()
    const eventId = (await page.getByTestId('report-event-id').textContent())?.trim()
    expect(eventId).toMatch(/^[0-9a-f]{32}$/)

    expect(received).toHaveLength(1)
    const [request] = received
    expect(request.contentType).toBe('application/x-sentry-envelope')
    expect(request.auth).toMatch(/sentry_version=7/)
    expect(request.auth).toMatch(/sentry_key=publickey/)

    // Three newline-delimited JSON lines: envelope header, item header, event.
    const [envelopeHeader, itemHeader, eventLine] = request.body.split('\n')
    expect(JSON.parse(envelopeHeader).event_id).toBe(eventId)
    expect(JSON.parse(envelopeHeader).dsn).toBe('http://publickey@127.0.0.1:'
      + (server.address() as AddressInfo).port + '/42')
    expect(JSON.parse(itemHeader).length).toBe(Buffer.byteLength(eventLine, 'utf8'))

    const event = JSON.parse(eventLine)
    expect(event.message.formatted).toContain('navigator freezes')
    expect(event.extra.report).toContain('.zspy')
    expect(event.extra.diagnostics.os.platform).toBeTruthy()
    expect(event.tags.os).toBeTruthy()
    // No email was typed, so no user block should be invented.
    expect(event.user).toBeUndefined()

    await page.getByTestId('report-close').click()
  })
})
