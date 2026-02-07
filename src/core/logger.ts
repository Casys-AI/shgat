/**
 * Logger Adapter for SHGAT
 *
 * Provides a simple console-based logger by default.
 * Can be replaced with custom logger via setLogger().
 *
 * @module shgat/logger
 */

export interface Logger {
  debug(msg: string, ...args: unknown[]): void;
  info(msg: string, ...args: unknown[]): void;
  warn(msg: string, ...args: unknown[]): void;
  error(msg: string, ...args: unknown[]): void;
}

/** Default console logger */
const consoleLogger: Logger = {
  debug: (msg, ...args) => console.debug(`[SHGAT] ${msg}`, ...args),
  info: (msg, ...args) => console.info(`[SHGAT] ${msg}`, ...args),
  warn: (msg, ...args) => console.warn(`[SHGAT] ${msg}`, ...args),
  error: (msg, ...args) => console.error(`[SHGAT] ${msg}`, ...args),
};

let currentLogger: Logger = consoleLogger;

/** Get the current logger instance */
export function getLogger(): Logger {
  return currentLogger;
}

/** Set a custom logger */
export function setLogger(logger: Logger): void {
  currentLogger = logger;
}

/** Reset to default console logger */
export function resetLogger(): void {
  currentLogger = consoleLogger;
}
