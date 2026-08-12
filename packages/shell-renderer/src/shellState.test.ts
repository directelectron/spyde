/**
 * shellState.test.ts — the chrome reducer.
 *
 * The interesting assertions are the composition ones: an app's state extends
 * ShellState and its reducer delegates, so the shell must (a) preserve fields it
 * knows nothing about and (b) return the state untouched for actions it does not
 * own — otherwise the delegation silently eats the app's own actions.
 */
import { test, describe } from 'node:test'
import assert from 'node:assert/strict'

import {
  shellReducer, shellInitialState, toShellAction, LOG_MAX, STREAM_MAX,
  type ShellState, type LogEntry,
} from './shellState.ts'

/** A state shaped like an app's: the shell's fields plus one of its own. */
interface AppState extends ShellState { mine: number }
const appState: AppState = { ...shellInitialState, mine: 42 }

const entry = (msg: string): LogEntry =>
  ({ level: 'INFO', name: 'x', area: 'a', msg, time: 0 })

describe('composition', () => {
  test('an unknown action returns the SAME state object', () => {
    // Identity, not just equality: a new object every dispatch would re-render
    // the whole app on any action the shell does not own.
    const next = shellReducer(appState, { type: 'NOPE' } as never)
    assert.equal(next, appState)
  })

  test("app fields survive an action the shell DOES own", () => {
    const next = shellReducer(appState, { type: 'STATUS', text: 'hi' })
    assert.equal(next.mine, 42)
    assert.equal(next.status, 'hi')
  })
})

describe('logs', () => {
  test('records accumulate in order', () => {
    let s = shellReducer(appState, { type: 'LOG', entries: [entry('a')] })
    s = shellReducer(s, { type: 'LOG', entries: [entry('b')] })
    assert.deepEqual(s.logEntries.map(e => e.msg), ['a', 'b'])
  })

  test('an empty batch is a no-op with no new object', () => {
    const s = shellReducer(appState, { type: 'LOG', entries: [] })
    assert.equal(s, appState)
  })

  test('the ring is bounded and keeps the NEWEST', () => {
    const many = Array.from({ length: LOG_MAX + 50 }, (_, i) => entry(String(i)))
    const s = shellReducer(appState, { type: 'LOG', entries: many })
    assert.equal(s.logEntries.length, LOG_MAX)
    assert.equal(s.logEntries.at(-1)!.msg, String(LOG_MAX + 49))
  })

  test('backfill REPLACES rather than appends', () => {
    // It is a re-send of history for a freshly-opened panel; appending would
    // double every record already shown.
    let s = shellReducer(appState, { type: 'LOG', entries: [entry('old')] })
    s = shellReducer(s, { type: 'LOG_BACKFILL', entries: [entry('x'), entry('y')] })
    assert.deepEqual(s.logEntries.map(e => e.msg), ['x', 'y'])
  })

  test('stream lines are bounded too', () => {
    let s: AppState = appState
    for (let i = 0; i < STREAM_MAX + 20; i++) {
      s = shellReducer(s, { type: 'STREAM', text: String(i), kind: 'stdout' })
    }
    assert.ok(s.streamLines.length <= STREAM_MAX + 1)
    assert.equal(s.streamLines.at(-1)!.text, String(STREAM_MAX + 19))
  })
})

describe('backend death and env setup', () => {
  test('backend death clears the setup overlay so the two cannot stack', () => {
    let s = shellReducer(appState, { type: 'ENV_SETUP_START' })
    assert.ok(s.envSetup)
    s = shellReducer(s, { type: 'BACKEND_EXITED', code: 1, reason: 'uv sync failed' })
    assert.equal(s.envSetup, null)
    assert.equal(s.ready, false)
    assert.deepEqual(s.backendExited, { code: 1, reason: 'uv sync failed' })
  })

  test('progress keeps the last meaningful phase/step when a line parses to nothing', () => {
    let s = shellReducer(appState, { type: 'ENV_SETUP_START' })
    s = shellReducer(s, {
      type: 'ENV_SETUP_PROGRESS', phase: 'downloading', step: 'torch', percent: 10, raw: 'a',
    })
    s = shellReducer(s, { type: 'ENV_SETUP_PROGRESS', percent: null, raw: 'noise' })
    assert.equal(s.envSetup!.phase, 'downloading')
    assert.equal(s.envSetup!.step, 'torch')
    assert.deepEqual(s.envSetup!.lines, ['a', 'noise'])
  })

  test('progress without a preceding start still works', () => {
    // env_setup progress can be the FIRST message a slow first launch sends.
    const s = shellReducer(appState, {
      type: 'ENV_SETUP_PROGRESS', percent: 5, raw: 'x',
    })
    assert.ok(s.envSetup)
  })

  test('done clears the overlay', () => {
    let s = shellReducer(appState, { type: 'ENV_SETUP_START' })
    s = shellReducer(s, { type: 'ENV_SETUP_DONE' })
    assert.equal(s.envSetup, null)
  })
})

describe('computing overlays and action state', () => {
  test('a repeated computing message is a no-op with no new object', () => {
    const on = shellReducer(appState, { type: 'WINDOW_COMPUTING', windowId: 1, computing: true })
    const again = shellReducer(on, { type: 'WINDOW_COMPUTING', windowId: 1, computing: true })
    assert.equal(again, on, 're-emit re-rendered every window')
  })

  test('computing toggles per window independently', () => {
    let s = shellReducer(appState, { type: 'WINDOW_COMPUTING', windowId: 1, computing: true })
    s = shellReducer(s, { type: 'WINDOW_COMPUTING', windowId: 2, computing: true })
    s = shellReducer(s, { type: 'WINDOW_COMPUTING', windowId: 1, computing: false })
    assert.deepEqual([...s.computingWindows], [2])
  })

  test('active actions add and remove per window', () => {
    let s = shellReducer(appState, { type: 'ACTION_ACTIVE', windowId: 1, name: 'fft', active: true })
    s = shellReducer(s, { type: 'ACTION_ACTIVE', windowId: 1, name: 'vi', active: true })
    s = shellReducer(s, { type: 'ACTION_ACTIVE', windowId: 1, name: 'fft', active: false })
    assert.deepEqual([...s.activeActions.get(1)!], ['vi'])
  })

  test('sub-items replace by name rather than duplicating', () => {
    let s = shellReducer(appState, {
      type: 'SUB_ITEM', windowId: 1, action: 'vi', name: 'r1', color: 'red', active: true,
    })
    s = shellReducer(s, {
      type: 'SUB_ITEM', windowId: 1, action: 'vi', name: 'r1', color: 'blue', active: true,
    })
    const list = s.subItems.get(1)!.get('vi')!
    assert.equal(list.length, 1)
    assert.equal(list[0].color, 'blue')
  })

  test('an inactive sub-item is removed', () => {
    let s = shellReducer(appState, {
      type: 'SUB_ITEM', windowId: 1, action: 'vi', name: 'r1', color: 'red', active: true,
    })
    s = shellReducer(s, {
      type: 'SUB_ITEM', windowId: 1, action: 'vi', name: 'r1', color: 'red', active: false,
    })
    assert.deepEqual(s.subItems.get(1)!.get('vi'), [])
  })
})

describe('toShellAction', () => {
  test('maps the chrome messages', () => {
    assert.deepEqual(toShellAction({ type: 'status', text: 'hi' }),
      { type: 'STATUS', text: 'hi' })
    assert.deepEqual(toShellAction({ type: 'backend_exited', code: 2 }),
      { type: 'BACKEND_EXITED', code: 2, reason: undefined })
    assert.equal(toShellAction({ type: 'env_setup', event: 'start' })!.type, 'ENV_SETUP_START')
    assert.equal(toShellAction({ type: 'env_setup', event: 'done' })!.type, 'ENV_SETUP_DONE')
    assert.equal(
      toShellAction({ type: 'env_setup', event: 'progress', raw: 'x' })!.type,
      'ENV_SETUP_PROGRESS')
  })

  test('returns null for anything the app owns', () => {
    // The contract that lets an app write `if (a) { dispatch(a); return }` and
    // then handle the rest — a wrong non-null here would SWALLOW a domain
    // message.
    assert.equal(toShellAction({ type: 'frame_stats' }), null)
    assert.equal(toShellAction({ type: 'figure' }), null)
    assert.equal(toShellAction({ type: 'log' }), null)
  })

  test('log is deliberately NOT mapped, so hosts can batch', () => {
    assert.equal(toShellAction({ type: 'log', msg: 'x' }), null)
  })
})
