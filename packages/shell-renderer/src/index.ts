/**
 * @de/shell-renderer — the React-side kernel shared by SpyDE, de-groundcrew and
 * de-autopilot.
 *
 * What is here is the machinery every shell app needs to talk to a Python
 * backend and put figures on screen: the core message protocol, the figure
 * bridge, and the iframe host. Layouts, sidebars and domain UI stay in the apps
 * — the whole point of the split is that SpyDE's MDI workspace and Ground Crew's
 * fixed panes are different answers over the SAME registry.
 */
export {
  createFigureBridge, useFigureBridge,
} from './figureBridge.react'
export type { FigureBridge, BinaryFrame, RefLike } from './figureBridge'

export { FigureFrame } from './FigureFrame'
export type { FigureFrameProps } from './FigureFrame'

export {
  shellReducer, shellInitialState, toShellAction, LOG_MAX, STREAM_MAX,
} from './shellState'
export type {
  ShellState, ShellAction, LogEntry, SubItem, EnvPhase, EnvSetupState,
} from './shellState'

export { asShellMessage } from './protocol'
export type {
  MsgBase, ShellMessage,
  ReadyMessage, StatusMessage, ErrorMessage, ProgressMessage,
  BackendExitedMessage, EnvSetupMessage,
  FigureMessage, StateUpdateMessage, StateUpdateBinaryMessage,
  WindowClosedMessage, WindowTitleMessage, WindowComputingMessage,
  LogMessage, LogBackfillMessage, LogLevelMessage,
} from './protocol'
