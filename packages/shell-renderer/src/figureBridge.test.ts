/**
 * figureBridge.test.ts — the retention rules, pinned.
 *
 * Every case here corresponds to a bug that shipped: a blank presented slide, a
 * multi-panel figure with one panel drawn, a stash emptied by its own replay.
 * They are cheap to assert and expensive to rediscover.
 *
 * Run by `npm run test:unit` (node:test, native TS type-stripping).
 */
import { test, describe } from 'node:test'
import assert from 'node:assert/strict'

import { createFigureBridge } from './figureBridge.ts'

/** A stand-in iframe that records what was posted into it. */
function fakeIframe(testid = 'frame') {
  const posted: Array<{ message: any; transfer: any }> = []
  return {
    posted,
    el: {
      contentWindow: {
        postMessage(message: any, _origin: string, transfer?: any) {
          posted.push({ message, transfer })
        },
      },
      getAttribute: () => testid,
      getBoundingClientRect: () => ({ width: 10, height: 10 }),
    } as unknown as HTMLIFrameElement,
  }
}

describe('state forwarding and retention', () => {
  test('a state posted before any iframe mounts is retained and replayed', () => {
    const bridge = createFigureBridge()
    // Nothing registered yet — this post goes nowhere, silently. The retention
    // is the only thing that saves it.
    bridge.applyState('f1', 'title', 'hello')

    const frame = fakeIframe()
    bridge.registerIframe('f1', frame.el)
    bridge.replay('f1')

    assert.deepEqual(frame.posted.map(p => p.message),
      [{ type: 'awi_state', key: 'title', value: 'hello' }])
  })

  test('only the latest value per key is retained', () => {
    const bridge = createFigureBridge()
    bridge.applyState('f1', 'k', 1)
    bridge.applyState('f1', 'k', 2)
    const frame = fakeIframe()
    bridge.registerIframe('f1', frame.el)
    bridge.replay('f1')
    assert.deepEqual(frame.posted.map(p => p.message.value), [2])
  })

  test('state is posted live to a mounted frame', () => {
    const bridge = createFigureBridge()
    const frame = fakeIframe()
    bridge.registerIframe('f1', frame.el)
    bridge.applyState('f1', 'k', 'v')
    assert.equal(frame.posted.length, 1)
  })
})

describe('binary frames', () => {
  test('panels are stashed separately, keyed by geom', () => {
    // The multi-panel bug: `key` is the pixel FIELD and is identical across
    // panels, so stashing by key alone left one frame however many panels the
    // figure had — and a presented copy drew one panel and blanks.
    const bridge = createFigureBridge()
    bridge.applyBinary('f1', 'image_b64', { geom: 'panel_a' }, new Uint8Array([1]))
    bridge.applyBinary('f1', 'image_b64', { geom: 'panel_b' }, new Uint8Array([2]))

    const slot = bridge.binaryStates.current.get('f1')!
    assert.deepEqual([...slot.keys()].sort(),
      ['panel_a::image_b64', 'panel_b::image_b64'])
  })

  test('a frame with no geom falls back to the pixel key', () => {
    const bridge = createFigureBridge()
    bridge.applyBinary('f1', 'image_b64', {}, new Uint8Array([1]))
    assert.deepEqual([...bridge.binaryStates.current.get('f1')!.keys()], ['image_b64'])
  })

  test('the live post transfers the buffer', () => {
    const bridge = createFigureBridge()
    const frame = fakeIframe()
    bridge.registerIframe('f1', frame.el)
    const bytes = new Uint8Array([1, 2, 3])
    bridge.applyBinary('f1', 'image_b64', { geom: 'g' }, bytes)
    assert.deepEqual(frame.posted[0].transfer, [bytes.buffer])
  })

  test('the stash survives its own replay, and every replay sends a fresh copy', () => {
    // Transfer DETACHES the buffer it sends. Replaying the stashed array itself
    // would empty the stash on first use, so the second mount of a figure — the
    // presented copy — would get nothing.
    const bridge = createFigureBridge()
    bridge.applyBinary('f1', 'image_b64', { geom: 'g' }, new Uint8Array([7, 8]))

    const first = fakeIframe('first')
    bridge.registerIframe('f1', first.el)
    bridge.replay('f1')

    const second = fakeIframe('second')
    bridge.replay('f1', second.el)

    for (const frame of [first, second]) {
      const msg = frame.posted.at(-1)!.message
      assert.equal(msg.type, 'awi_state_binary')
      assert.deepEqual([...msg.buffer], [7, 8], 'replayed into a detached buffer')
    }
  })

  test('replay serves the TARGET, not whichever mount holds the registry slot', () => {
    // The blank-presented-deck bug. Two mounts share a figId; the map holds one.
    // A freshly-loaded frame passes itself and must be the one served.
    const bridge = createFigureBridge()
    bridge.applyState('f1', 'k', 'v')

    const winner = fakeIframe('winner')
    const loader = fakeIframe('loader')
    bridge.registerIframe('f1', winner.el)     // last registration wins the map
    bridge.replay('f1', loader.el)             // ...but THIS frame just loaded

    assert.equal(loader.posted.length, 1)
    assert.equal(winner.posted.length, 0, 'state went to the sibling mount')
  })
})

describe('registry and eviction', () => {
  test('registering null deregisters', () => {
    const bridge = createFigureBridge()
    const frame = fakeIframe()
    bridge.registerIframe('f1', frame.el)
    bridge.registerIframe('f1', null)
    assert.equal(bridge.iframes.current.has('f1'), false)
  })

  test('evict drops every trace of a figure', () => {
    // Without this, a long report-editing session grows the maps forever: each
    // re-render of a cell mints a new figId and the old one's pixels stay.
    const bridge = createFigureBridge()
    const frame = fakeIframe()
    bridge.registerIframe('f1', frame.el)
    bridge.applyState('f1', 'k', 'v')
    bridge.applyBinary('f1', 'image_b64', { geom: 'g' }, new Uint8Array([1]))

    bridge.evict('f1')

    assert.equal(bridge.states.current.has('f1'), false)
    assert.equal(bridge.binaryStates.current.has('f1'), false)
    assert.equal(bridge.iframes.current.has('f1'), false)
  })

  test('replaying an unmounted figure is a no-op, not a throw', () => {
    const bridge = createFigureBridge()
    bridge.applyState('f1', 'k', 'v')
    bridge.replay('f1')
  })

  test('dump reports retained counts per figure', () => {
    const bridge = createFigureBridge()
    const frame = fakeIframe()
    bridge.registerIframe('f1', frame.el)
    bridge.applyState('f1', 'k', 'v')
    bridge.applyBinary('f1', 'image_b64', { geom: 'g' }, new Uint8Array([1]))

    const row = bridge.dump().find(r => r.figId === 'f1')!
    assert.equal(row.jsonKeys, 1)
    assert.equal(row.binaryKeys, 1)
    assert.equal(row.binaryKeyNames, 'g::image_b64')
  })

  test('dump includes figures with retained state but no mounted frame', () => {
    // The diagnostic's whole job: telling "never arrived" from "arrived, not
    // mounted". A figure missing from the dump would be indistinguishable.
    const bridge = createFigureBridge()
    bridge.applyState('ghost', 'k', 'v')
    const row = bridge.dump().find(r => r.figId === 'ghost')!
    assert.equal(row.registeredIn, 'NONE')
  })
})
