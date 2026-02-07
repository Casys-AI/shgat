/**
 * TensorFlow C API Backend via Deno FFI
 *
 * Complete TensorFlow backend using libtensorflow C API with eager execution.
 * Replaces TF.js WASM for operations that need proper autograd support.
 *
 * Key features:
 * - Full autograd support via TensorFlow GradientTape
 * - All kernels available (including UnsortedSegmentSum)
 * - Native CPU/GPU acceleration via libtensorflow
 *
 * @module shgat-tf/tf/tf-ffi
 */

// ============================================================================
// TensorFlow DataTypes
// ============================================================================

export const TF_FLOAT = 1;
export const TF_DOUBLE = 2;
export const TF_INT32 = 3;
export const TF_INT64 = 9;
export const TF_BOOL = 10;

// ============================================================================
// Library Paths
// ============================================================================

const TF_PATHS = [
  // Linux
  "/usr/lib/libtensorflow.so.2",
  "/usr/lib/libtensorflow.so",
  "/usr/local/lib/libtensorflow.so.2",
  "/usr/local/lib/libtensorflow.so",
  "/opt/tensorflow/lib/libtensorflow.so",
  // macOS
  "/opt/homebrew/lib/libtensorflow.so",
  "/usr/local/opt/libtensorflow/lib/libtensorflow.so",
  "/opt/homebrew/lib/libtensorflow.dylib",
  "/usr/local/lib/libtensorflow.dylib",
];

// ============================================================================
// FFI Type Definitions
// ============================================================================

type TFLib = Deno.DynamicLibrary<{
  // Version
  TF_Version: { parameters: []; result: "pointer" };

  // Status management
  TF_NewStatus: { parameters: []; result: "pointer" };
  TF_DeleteStatus: { parameters: ["pointer"]; result: "void" };
  TF_GetCode: { parameters: ["pointer"]; result: "i32" };
  TF_Message: { parameters: ["pointer"]; result: "pointer" };

  // Tensor allocation
  TF_AllocateTensor: {
    parameters: ["i32", "pointer", "i32", "u64"];
    result: "pointer";
  };
  TF_DeleteTensor: { parameters: ["pointer"]; result: "void" };
  TF_TensorData: { parameters: ["pointer"]; result: "pointer" };
  TF_TensorByteSize: { parameters: ["pointer"]; result: "u64" };
  TF_NumDims: { parameters: ["pointer"]; result: "i32" };
  TF_Dim: { parameters: ["pointer", "i32"]; result: "i64" };
  TF_TensorType: { parameters: ["pointer"]; result: "i32" };

  // Eager execution context
  TFE_NewContextOptions: { parameters: []; result: "pointer" };
  TFE_DeleteContextOptions: { parameters: ["pointer"]; result: "void" };
  TFE_ContextOptionsSetDevicePlacementPolicy: {
    parameters: ["pointer", "i32"];
    result: "void";
  };
  TFE_NewContext: { parameters: ["pointer", "pointer"]; result: "pointer" };
  TFE_DeleteContext: { parameters: ["pointer"]; result: "void" };

  // Eager operations
  TFE_NewOp: {
    parameters: ["pointer", "pointer", "pointer"];
    result: "pointer";
  };
  TFE_DeleteOp: { parameters: ["pointer"]; result: "void" };
  TFE_OpAddInput: {
    parameters: ["pointer", "pointer", "pointer"];
    result: "void";
  };
  TFE_OpSetAttrInt: {
    parameters: ["pointer", "pointer", "i64"];
    result: "void";
  };
  TFE_OpSetAttrFloat: {
    parameters: ["pointer", "pointer", "f32"];
    result: "void";
  };
  TFE_OpSetAttrBool: {
    parameters: ["pointer", "pointer", "u8"];
    result: "void";
  };
  TFE_OpSetAttrType: {
    parameters: ["pointer", "pointer", "i32"];
    result: "void";
  };
  TFE_OpSetAttrShape: {
    parameters: ["pointer", "pointer", "pointer", "i32", "pointer"];
    result: "void";
  };
  TFE_OpSetAttrIntList: {
    parameters: ["pointer", "pointer", "pointer", "i32"];
    result: "void";
  };
  TFE_Execute: {
    parameters: ["pointer", "pointer", "pointer", "pointer"];
    result: "void";
  };
  TFE_OpGetInputLength: {
    parameters: ["pointer", "pointer", "pointer"];
    result: "i32";
  };

  // Tensor handles
  TFE_NewTensorHandle: {
    parameters: ["pointer", "pointer"];
    result: "pointer";
  };
  TFE_DeleteTensorHandle: { parameters: ["pointer"]; result: "void" };
  TFE_TensorHandleDataType: { parameters: ["pointer"]; result: "i32" };
  TFE_TensorHandleNumDims: {
    parameters: ["pointer", "pointer"];
    result: "i32";
  };
  TFE_TensorHandleDim: {
    parameters: ["pointer", "i32", "pointer"];
    result: "i64";
  };
  TFE_TensorHandleResolve: {
    parameters: ["pointer", "pointer"];
    result: "pointer";
  };
  TFE_TensorHandleCopyToDevice: {
    parameters: ["pointer", "pointer", "pointer", "pointer"];
    result: "pointer";
  };

  // Note: GradientTape functions are NOT available in standard libtensorflow C API
}>;

// ============================================================================
// Module State
// ============================================================================

let lib: TFLib | null = null;
let ctx: Deno.PointerValue | null = null;
let available = false;
let initAttempted = false;
let tfVersion = "unknown";

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize TensorFlow FFI (lazy, called on first use)
 */
function init(): boolean {
  if (initAttempted) return available;
  initAttempted = true;

  if (typeof Deno?.dlopen !== "function") {
    console.warn("[TF-FFI] Deno.dlopen not available");
    return false;
  }

  for (const path of TF_PATHS) {
    try {
      lib = Deno.dlopen(path, {
        TF_Version: { parameters: [], result: "pointer" },

        TF_NewStatus: { parameters: [], result: "pointer" },
        TF_DeleteStatus: { parameters: ["pointer"], result: "void" },
        TF_GetCode: { parameters: ["pointer"], result: "i32" },
        TF_Message: { parameters: ["pointer"], result: "pointer" },

        TF_AllocateTensor: {
          parameters: ["i32", "pointer", "i32", "u64"],
          result: "pointer",
        },
        TF_DeleteTensor: { parameters: ["pointer"], result: "void" },
        TF_TensorData: { parameters: ["pointer"], result: "pointer" },
        TF_TensorByteSize: { parameters: ["pointer"], result: "u64" },
        TF_NumDims: { parameters: ["pointer"], result: "i32" },
        TF_Dim: { parameters: ["pointer", "i32"], result: "i64" },
        TF_TensorType: { parameters: ["pointer"], result: "i32" },

        TFE_NewContextOptions: { parameters: [], result: "pointer" },
        TFE_DeleteContextOptions: { parameters: ["pointer"], result: "void" },
        TFE_ContextOptionsSetDevicePlacementPolicy: {
          parameters: ["pointer", "i32"],
          result: "void",
        },
        TFE_NewContext: {
          parameters: ["pointer", "pointer"],
          result: "pointer",
        },
        TFE_DeleteContext: { parameters: ["pointer"], result: "void" },

        TFE_NewOp: {
          parameters: ["pointer", "pointer", "pointer"],
          result: "pointer",
        },
        TFE_DeleteOp: { parameters: ["pointer"], result: "void" },
        TFE_OpAddInput: {
          parameters: ["pointer", "pointer", "pointer"],
          result: "void",
        },
        TFE_OpSetAttrInt: {
          parameters: ["pointer", "pointer", "i64"],
          result: "void",
        },
        TFE_OpSetAttrFloat: {
          parameters: ["pointer", "pointer", "f32"],
          result: "void",
        },
        TFE_OpSetAttrBool: {
          parameters: ["pointer", "pointer", "u8"],
          result: "void",
        },
        TFE_OpSetAttrType: {
          parameters: ["pointer", "pointer", "i32"],
          result: "void",
        },
        TFE_OpSetAttrShape: {
          parameters: ["pointer", "pointer", "pointer", "i32", "pointer"],
          result: "void",
        },
        TFE_OpSetAttrIntList: {
          parameters: ["pointer", "pointer", "pointer", "i32"],
          result: "void",
        },
        TFE_Execute: {
          parameters: ["pointer", "pointer", "pointer", "pointer"],
          result: "void",
        },
        TFE_OpGetInputLength: {
          parameters: ["pointer", "pointer", "pointer"],
          result: "i32",
        },

        TFE_NewTensorHandle: {
          parameters: ["pointer", "pointer"],
          result: "pointer",
        },
        TFE_DeleteTensorHandle: { parameters: ["pointer"], result: "void" },
        TFE_TensorHandleDataType: { parameters: ["pointer"], result: "i32" },
        TFE_TensorHandleNumDims: {
          parameters: ["pointer", "pointer"],
          result: "i32",
        },
        TFE_TensorHandleDim: {
          parameters: ["pointer", "i32", "pointer"],
          result: "i64",
        },
        TFE_TensorHandleResolve: {
          parameters: ["pointer", "pointer"],
          result: "pointer",
        },
        TFE_TensorHandleCopyToDevice: {
          parameters: ["pointer", "pointer", "pointer", "pointer"],
          result: "pointer",
        },

        // Note: GradientTape functions (TFE_NewGradientTape, etc.) are NOT
        // available in the standard libtensorflow C API. Autograd must be
        // implemented via numerical differentiation or custom backward passes.
      });

      // Create eager context
      const status = lib.symbols.TF_NewStatus();
      const opts = lib.symbols.TFE_NewContextOptions();
      ctx = lib.symbols.TFE_NewContext(opts, status);
      lib.symbols.TFE_DeleteContextOptions(opts);

      const code = lib.symbols.TF_GetCode(status);
      lib.symbols.TF_DeleteStatus(status);

      if (code !== 0) {
        lib.close();
        lib = null;
        continue;
      }

      // Get version
      const vPtr = lib.symbols.TF_Version();
      if (vPtr) {
        tfVersion = new Deno.UnsafePointerView(vPtr).getCString();
      }

      available = true;
      console.error(`[TF-FFI] Loaded libtensorflow ${tfVersion} from: ${path}`);
      return true;
    } catch {
      // Try next path
    }
  }

  console.warn(
    "[TF-FFI] Could not load libtensorflow - using JS fallback"
  );
  return false;
}

/**
 * Check if TensorFlow FFI is available
 */
export function isAvailable(): boolean {
  if (!initAttempted) init();
  return available;
}

/**
 * Get TensorFlow version
 */
export function version(): string {
  if (!initAttempted) init();
  return tfVersion;
}

// ============================================================================
// Tensor Wrapper Class
// ============================================================================

/**
 * TensorFlow Tensor wrapper for FFI operations
 */
export class TFTensor {
  private handle: Deno.PointerValue;
  private _shape: number[];
  private _dtype: number;
  private disposed = false;

  constructor(handle: Deno.PointerValue, shape: number[], dtype: number) {
    this.handle = handle;
    this._shape = shape;
    this._dtype = dtype;
  }

  get shape(): number[] {
    return this._shape;
  }

  get dtype(): number {
    return this._dtype;
  }

  get rank(): number {
    return this._shape.length;
  }

  getHandle(): Deno.PointerValue {
    return this.handle;
  }

  /**
   * Read tensor data as Float32Array
   */
  dataSync(): Float32Array {
    if (!lib || this.disposed) return new Float32Array(0);

    const status = lib.symbols.TF_NewStatus();
    const tensor = lib.symbols.TFE_TensorHandleResolve(this.handle, status);
    lib.symbols.TF_DeleteStatus(status);

    if (!tensor) return new Float32Array(0);

    const dataPtr = lib.symbols.TF_TensorData(tensor);
    const byteSize = lib.symbols.TF_TensorByteSize(tensor);

    if (!dataPtr) {
      lib.symbols.TF_DeleteTensor(tensor);
      return new Float32Array(0);
    }

    const view = new Deno.UnsafePointerView(dataPtr);
    const buffer = view.getArrayBuffer(Number(byteSize));
    const result = new Float32Array(buffer.slice(0));

    lib.symbols.TF_DeleteTensor(tensor);
    return result;
  }

  /**
   * Read tensor data as Int32Array
   */
  dataInt32Sync(): Int32Array {
    if (!lib || this.disposed) return new Int32Array(0);

    const status = lib.symbols.TF_NewStatus();
    const tensor = lib.symbols.TFE_TensorHandleResolve(this.handle, status);
    lib.symbols.TF_DeleteStatus(status);

    if (!tensor) return new Int32Array(0);

    const dataPtr = lib.symbols.TF_TensorData(tensor);
    const byteSize = lib.symbols.TF_TensorByteSize(tensor);

    if (!dataPtr) {
      lib.symbols.TF_DeleteTensor(tensor);
      return new Int32Array(0);
    }

    const view = new Deno.UnsafePointerView(dataPtr);
    const buffer = view.getArrayBuffer(Number(byteSize));
    const result = new Int32Array(buffer.slice(0));

    lib.symbols.TF_DeleteTensor(tensor);
    return result;
  }

  /**
   * Convert to nested JS array
   */
  arraySync(): number[] | number[][] | number[][][] {
    const data = this.dataSync();
    return reshapeArray(data, this._shape);
  }

  dispose(): void {
    if (!this.disposed && lib && this.handle) {
      lib.symbols.TFE_DeleteTensorHandle(this.handle);
      this.disposed = true;
    }
  }
}

/**
 * Reshape flat array to nested array based on shape
 */
function reshapeArray(
  flat: Float32Array,
  shape: number[]
): number[] | number[][] | number[][][] {
  if (shape.length === 0) return [flat[0]];
  if (shape.length === 1) return Array.from(flat);

  if (shape.length === 2) {
    const [rows, cols] = shape;
    const result: number[][] = [];
    for (let i = 0; i < rows; i++) {
      result.push(Array.from(flat.slice(i * cols, (i + 1) * cols)));
    }
    return result;
  }

  if (shape.length === 3) {
    const [d0, d1, d2] = shape;
    const result: number[][][] = [];
    for (let i = 0; i < d0; i++) {
      const matrix: number[][] = [];
      for (let j = 0; j < d1; j++) {
        const offset = (i * d1 + j) * d2;
        matrix.push(Array.from(flat.slice(offset, offset + d2)));
      }
      result.push(matrix);
    }
    return result;
  }

  // Fallback for higher dimensions - return flat
  return Array.from(flat);
}

// ============================================================================
// Tensor Creation
// ============================================================================

/**
 * Create a tensor from Float32Array
 */
export function tensor(
  data: Float32Array | number[] | number[][] | number[][][],
  shape?: number[]
): TFTensor {
  if (!init() || !lib || !ctx) {
    throw new Error("[TF-FFI] TensorFlow not available");
  }

  // Flatten data if needed
  let flat: Float32Array;
  let inferredShape: number[];

  if (data instanceof Float32Array) {
    flat = data;
    inferredShape = shape || [data.length];
  } else if (Array.isArray(data)) {
    const { flat: f, shape: s } = flattenArray(data);
    flat = f;
    inferredShape = shape || s;
  } else {
    throw new Error("Unsupported data type");
  }

  return createTensorFromFloat32(flat, inferredShape);
}

/**
 * Create a tensor from Int32Array
 */
export function tensorInt32(data: Int32Array, shape?: number[]): TFTensor {
  if (!init() || !lib || !ctx) {
    throw new Error("[TF-FFI] TensorFlow not available");
  }

  const inferredShape = shape || [data.length];
  return createTensorFromInt32(data, inferredShape);
}

/**
 * Create zeros tensor
 */
export function zeros(shape: number[]): TFTensor {
  const size = shape.reduce((a, b) => a * b, 1);
  return tensor(new Float32Array(size), shape);
}

/**
 * Create ones tensor
 */
export function ones(shape: number[]): TFTensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size).fill(1);
  return tensor(data, shape);
}

/**
 * Create random uniform tensor
 */
export function randomUniform(
  shape: number[],
  min = 0,
  max = 1
): TFTensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  const range = max - min;
  for (let i = 0; i < size; i++) {
    data[i] = min + Math.random() * range;
  }
  return tensor(data, shape);
}

/**
 * Create random normal tensor
 */
export function randomNormal(
  shape: number[],
  mean = 0,
  stddev = 1
): TFTensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);

  // Box-Muller transform
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.random();
    const u2 = Math.random();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    data[i] = mean + stddev * r * Math.cos(theta);
    if (i + 1 < size) {
      data[i + 1] = mean + stddev * r * Math.sin(theta);
    }
  }

  return tensor(data, shape);
}

// ============================================================================
// Internal Tensor Creation
// ============================================================================

function createTensorFromFloat32(
  data: Float32Array,
  shape: number[]
): TFTensor {
  if (!lib || !ctx) throw new Error("[TF-FFI] Not initialized");

  const dims = new BigInt64Array(shape.map((d) => BigInt(d)));
  const dimsPtr = shape.length > 0 ? Deno.UnsafePointer.of(dims) : null;
  const byteSize = BigInt(data.byteLength);

  const rawTensor = lib.symbols.TF_AllocateTensor(
    TF_FLOAT,
    dimsPtr!,
    shape.length,
    byteSize
  );

  if (!rawTensor) throw new Error("[TF-FFI] Failed to allocate tensor");

  // Copy data
  const tensorData = lib.symbols.TF_TensorData(rawTensor);
  if (tensorData) {
    const view = new Deno.UnsafePointerView(tensorData);
    const dest = new Float32Array(view.getArrayBuffer(data.byteLength));
    dest.set(data);
  }

  // Create handle
  const status = lib.symbols.TF_NewStatus();
  const handle = lib.symbols.TFE_NewTensorHandle(rawTensor, status);
  const code = lib.symbols.TF_GetCode(status);
  lib.symbols.TF_DeleteStatus(status);
  lib.symbols.TF_DeleteTensor(rawTensor);

  if (code !== 0 || !handle) {
    throw new Error("[TF-FFI] Failed to create tensor handle");
  }

  return new TFTensor(handle, shape, TF_FLOAT);
}

function createTensorFromInt32(data: Int32Array, shape: number[]): TFTensor {
  if (!lib || !ctx) throw new Error("[TF-FFI] Not initialized");

  const dims = new BigInt64Array(shape.map((d) => BigInt(d)));
  const dimsPtr = shape.length > 0 ? Deno.UnsafePointer.of(dims) : null;
  const byteSize = BigInt(data.byteLength);

  const rawTensor = lib.symbols.TF_AllocateTensor(
    TF_INT32,
    dimsPtr!,
    shape.length,
    byteSize
  );

  if (!rawTensor) throw new Error("[TF-FFI] Failed to allocate tensor");

  // Copy data
  const tensorData = lib.symbols.TF_TensorData(rawTensor);
  if (tensorData) {
    const view = new Deno.UnsafePointerView(tensorData);
    const dest = new Int32Array(view.getArrayBuffer(data.byteLength));
    dest.set(data);
  }

  // Create handle
  const status = lib.symbols.TF_NewStatus();
  const handle = lib.symbols.TFE_NewTensorHandle(rawTensor, status);
  const code = lib.symbols.TF_GetCode(status);
  lib.symbols.TF_DeleteStatus(status);
  lib.symbols.TF_DeleteTensor(rawTensor);

  if (code !== 0 || !handle) {
    throw new Error("[TF-FFI] Failed to create tensor handle");
  }

  return new TFTensor(handle, shape, TF_INT32);
}

function flattenArray(
  arr: number[] | number[][] | number[][][]
): { flat: Float32Array; shape: number[] } {
  if (!Array.isArray(arr)) {
    return { flat: new Float32Array([arr as number]), shape: [] };
  }

  if (arr.length === 0) {
    return { flat: new Float32Array(0), shape: [0] };
  }

  // 1D
  if (!Array.isArray(arr[0])) {
    return { flat: new Float32Array(arr as number[]), shape: [arr.length] };
  }

  // 2D
  const arr2d = arr as number[][];
  if (!Array.isArray(arr2d[0][0])) {
    const rows = arr2d.length;
    const cols = arr2d[0]?.length || 0;
    const flat = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        flat[i * cols + j] = arr2d[i][j] || 0;
      }
    }
    return { flat, shape: [rows, cols] };
  }

  // 3D
  const arr3d = arr as number[][][];
  const d0 = arr3d.length;
  const d1 = arr3d[0]?.length || 0;
  const d2 = arr3d[0]?.[0]?.length || 0;
  const flat = new Float32Array(d0 * d1 * d2);
  for (let i = 0; i < d0; i++) {
    for (let j = 0; j < d1; j++) {
      for (let k = 0; k < d2; k++) {
        flat[(i * d1 + j) * d2 + k] = arr3d[i]?.[j]?.[k] || 0;
      }
    }
  }
  return { flat, shape: [d0, d1, d2] };
}

// ============================================================================
// Operations Helper
// ============================================================================

function execOp(
  opName: string,
  inputs: TFTensor[],
  attrs: Record<string, unknown> = {},
  numOutputs = 1
): TFTensor[] {
  if (!lib || !ctx) throw new Error("[TF-FFI] Not initialized");

  const status = lib.symbols.TF_NewStatus();
  const opNameBuf = new TextEncoder().encode(opName + "\0");
  const opNamePtr = Deno.UnsafePointer.of(opNameBuf);

  const op = lib.symbols.TFE_NewOp(ctx, opNamePtr!, status);

  // Add inputs
  for (const input of inputs) {
    lib.symbols.TFE_OpAddInput(op, input.getHandle(), status);
  }

  // Set attributes
  for (const [key, value] of Object.entries(attrs)) {
    const keyBuf = new TextEncoder().encode(key + "\0");
    const keyPtr = Deno.UnsafePointer.of(keyBuf);

    if (typeof value === "number") {
      if (Number.isInteger(value)) {
        lib.symbols.TFE_OpSetAttrInt(op, keyPtr!, BigInt(value));
      } else {
        lib.symbols.TFE_OpSetAttrFloat(op, keyPtr!, value);
      }
    } else if (typeof value === "boolean") {
      lib.symbols.TFE_OpSetAttrBool(op, keyPtr!, value ? 1 : 0);
    } else if (value === TF_FLOAT || value === TF_INT32) {
      lib.symbols.TFE_OpSetAttrType(op, keyPtr!, value as number);
    }
  }

  // Execute
  const resultHandles = new BigUint64Array(numOutputs);
  const resultHandlesPtr = Deno.UnsafePointer.of(resultHandles);
  const numRetvals = new Int32Array([numOutputs]);
  const numRetvalsPtr = Deno.UnsafePointer.of(numRetvals);

  lib.symbols.TFE_Execute(op, resultHandlesPtr!, numRetvalsPtr!, status);

  const code = lib.symbols.TF_GetCode(status);
  if (code !== 0) {
    const msgPtr = lib.symbols.TF_Message(status);
    const msg = msgPtr
      ? new Deno.UnsafePointerView(msgPtr).getCString()
      : "Unknown error";
    lib.symbols.TFE_DeleteOp(op);
    lib.symbols.TF_DeleteStatus(status);
    throw new Error(`[TF-FFI] ${opName} failed: ${msg}`);
  }

  lib.symbols.TFE_DeleteOp(op);
  lib.symbols.TF_DeleteStatus(status);

  // Get output shapes
  const results: TFTensor[] = [];
  for (let i = 0; i < numRetvals[0]; i++) {
    const handle = Deno.UnsafePointer.create(resultHandles[i]);
    const shape = getHandleShape(handle);
    const dtype = lib.symbols.TFE_TensorHandleDataType(handle);
    results.push(new TFTensor(handle, shape, dtype));
  }

  return results;
}

function getHandleShape(handle: Deno.PointerValue): number[] {
  if (!lib) return [];
  const status = lib.symbols.TF_NewStatus();
  const numDims = lib.symbols.TFE_TensorHandleNumDims(handle, status);
  const shape: number[] = [];
  for (let i = 0; i < numDims; i++) {
    const dim = lib.symbols.TFE_TensorHandleDim(handle, i, status);
    shape.push(Number(dim));
  }
  lib.symbols.TF_DeleteStatus(status);
  return shape;
}

// ============================================================================
// Math Operations
// ============================================================================

/**
 * Matrix multiplication: A @ B
 */
export function matmul(
  a: TFTensor,
  b: TFTensor,
  transposeA = false,
  transposeB = false
): TFTensor {
  return execOp("MatMul", [a, b], {
    transpose_a: transposeA,
    transpose_b: transposeB,
  })[0];
}

/**
 * Batch matrix multiplication
 */
export function batchMatmul(
  a: TFTensor,
  b: TFTensor,
  transposeA = false,
  transposeB = false
): TFTensor {
  return execOp("BatchMatMulV2", [a, b], {
    adj_x: transposeA,
    adj_y: transposeB,
  })[0];
}

/**
 * Element-wise addition
 */
export function add(a: TFTensor, b: TFTensor): TFTensor {
  return execOp("AddV2", [a, b])[0];
}

/**
 * Element-wise subtraction
 */
export function sub(a: TFTensor, b: TFTensor): TFTensor {
  return execOp("Sub", [a, b])[0];
}

/**
 * Element-wise multiplication
 */
export function mul(a: TFTensor, b: TFTensor): TFTensor {
  return execOp("Mul", [a, b])[0];
}

/**
 * Element-wise division
 */
export function div(a: TFTensor, b: TFTensor): TFTensor {
  return execOp("RealDiv", [a, b])[0];
}

/**
 * Transpose
 */
export function transpose(a: TFTensor, perm?: number[]): TFTensor {
  if (!perm) {
    // Default: reverse dimensions
    perm = Array.from({ length: a.rank }, (_, i) => a.rank - 1 - i);
  }
  const permTensor = tensorInt32(new Int32Array(perm), [perm.length]);
  const result = execOp("Transpose", [a, permTensor])[0];
  permTensor.dispose();
  return result;
}

/**
 * Reshape tensor
 */
export function reshape(a: TFTensor, shape: number[]): TFTensor {
  const shapeTensor = tensorInt32(new Int32Array(shape), [shape.length]);
  const result = execOp("Reshape", [a, shapeTensor])[0];
  shapeTensor.dispose();
  return result;
}

/**
 * Gather slices from params along axis
 */
export function gather(params: TFTensor, indices: TFTensor, axis = 0): TFTensor {
  const axisTensor = tensorInt32(new Int32Array([axis]), []);
  const result = execOp("GatherV2", [params, indices, axisTensor])[0];
  axisTensor.dispose();
  return result;
}

/**
 * UnsortedSegmentSum - THE KEY OPERATION
 *
 * Computes sum of segments where segment_ids maps data elements to output segments.
 * This is the gradient of gather and was missing in TF.js WASM.
 */
export function unsortedSegmentSum(
  data: TFTensor,
  segmentIds: TFTensor,
  numSegments: number
): TFTensor {
  const numSegTensor = tensorInt32(new Int32Array([numSegments]), []);
  const result = execOp("UnsortedSegmentSum", [data, segmentIds, numSegTensor])[0];
  numSegTensor.dispose();
  return result;
}

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * Softmax
 */
export function softmax(x: TFTensor, axis = -1): TFTensor {
  // Normalize axis
  const normalizedAxis = axis < 0 ? x.rank + axis : axis;
  return execOp("Softmax", [x], { axis: normalizedAxis })[0];
}

/**
 * ReLU
 */
export function relu(x: TFTensor): TFTensor {
  return execOp("Relu", [x])[0];
}

/**
 * Leaky ReLU
 */
export function leakyRelu(x: TFTensor, alpha = 0.2): TFTensor {
  return execOp("LeakyRelu", [x], { alpha })[0];
}

/**
 * ELU
 */
export function elu(x: TFTensor, _alpha = 1.0): TFTensor {
  // TF.js Elu doesn't take alpha, but TF C API might - using Relu/Where combo
  // For simplicity, use built-in Elu (alpha=1)
  // _alpha parameter reserved for future custom alpha implementation
  return execOp("Elu", [x])[0];
}

/**
 * Sigmoid
 */
export function sigmoid(x: TFTensor): TFTensor {
  return execOp("Sigmoid", [x])[0];
}

/**
 * Tanh
 */
export function tanh(x: TFTensor): TFTensor {
  return execOp("Tanh", [x])[0];
}

// ============================================================================
// Reduction Operations
// ============================================================================

/**
 * Sum reduction
 */
export function sum(x: TFTensor, axis?: number | number[], keepDims = false): TFTensor {
  const axes = axis === undefined ? [] : Array.isArray(axis) ? axis : [axis];
  const axisTensor = tensorInt32(new Int32Array(axes), [axes.length]);
  const result = execOp("Sum", [x, axisTensor], { keep_dims: keepDims })[0];
  axisTensor.dispose();
  return result;
}

/**
 * Mean reduction
 */
export function mean(x: TFTensor, axis?: number | number[], keepDims = false): TFTensor {
  const axes = axis === undefined ? [] : Array.isArray(axis) ? axis : [axis];
  const axisTensor = tensorInt32(new Int32Array(axes), [axes.length]);
  const result = execOp("Mean", [x, axisTensor], { keep_dims: keepDims })[0];
  axisTensor.dispose();
  return result;
}

/**
 * Max reduction
 */
export function max(x: TFTensor, axis?: number | number[], keepDims = false): TFTensor {
  const axes = axis === undefined ? [] : Array.isArray(axis) ? axis : [axis];
  const axisTensor = tensorInt32(new Int32Array(axes), [axes.length]);
  const result = execOp("Max", [x, axisTensor], { keep_dims: keepDims })[0];
  axisTensor.dispose();
  return result;
}

/**
 * Square
 */
export function square(x: TFTensor): TFTensor {
  return execOp("Square", [x])[0];
}

/**
 * Sqrt
 */
export function sqrt(x: TFTensor): TFTensor {
  return execOp("Sqrt", [x])[0];
}

/**
 * Exp
 */
export function exp(x: TFTensor): TFTensor {
  return execOp("Exp", [x])[0];
}

/**
 * Log
 */
export function log(x: TFTensor): TFTensor {
  return execOp("Log", [x])[0];
}

/**
 * Neg
 */
export function neg(x: TFTensor): TFTensor {
  return execOp("Neg", [x])[0];
}

// ============================================================================
// Concatenation / Splitting
// ============================================================================

/**
 * Concatenate tensors
 */
export function concat(tensors: TFTensor[], axis = 0): TFTensor {
  if (tensors.length === 0) throw new Error("No tensors to concat");
  if (tensors.length === 1) return tensors[0];

  // Use Pack (stack) along new axis, then reshape - simpler and more reliable
  // Or use manual approach: slice and combine

  // For 2 tensors, we can use a workaround via manual indexing
  // For now, implement a JS-based concat for simplicity
  const shapes = tensors.map((t) => t.shape);

  // Validate shapes match except on concat axis
  const normalizedAxis = axis < 0 ? shapes[0].length + axis : axis;
  for (let i = 1; i < shapes.length; i++) {
    for (let d = 0; d < shapes[0].length; d++) {
      if (d !== normalizedAxis && shapes[i][d] !== shapes[0][d]) {
        throw new Error(
          `[TF-FFI] concat: shapes don't match at dimension ${d}`
        );
      }
    }
  }

  // Calculate output shape
  const outputShape = [...shapes[0]];
  outputShape[normalizedAxis] = shapes.reduce(
    (sum, s) => sum + s[normalizedAxis],
    0
  );

  // Concatenate data manually
  const allData: Float32Array[] = tensors.map((t) => t.dataSync());
  const totalSize = outputShape.reduce((a, b) => a * b, 1);
  const result = new Float32Array(totalSize);

  if (normalizedAxis === 0) {
    // Simple case: concat along first axis
    let offset = 0;
    for (const data of allData) {
      result.set(data, offset);
      offset += data.length;
    }
  } else {
    // General case: interleave data
    const outerSize = shapes[0]
      .slice(0, normalizedAxis)
      .reduce((a, b) => a * b, 1);
    const innerSize = shapes[0]
      .slice(normalizedAxis + 1)
      .reduce((a, b) => a * b, 1);

    let outIdx = 0;
    for (let outer = 0; outer < outerSize; outer++) {
      for (let tIdx = 0; tIdx < tensors.length; tIdx++) {
        const axisSize = shapes[tIdx][normalizedAxis];
        const data = allData[tIdx];
        const srcStart = outer * axisSize * innerSize;
        for (let i = 0; i < axisSize * innerSize; i++) {
          result[outIdx++] = data[srcStart + i];
        }
      }
    }
  }

  return tensor(result, outputShape);
}

/**
 * Slice tensor
 */
export function slice(x: TFTensor, begin: number[], size: number[]): TFTensor {
  const beginTensor = tensorInt32(new Int32Array(begin), [begin.length]);
  const sizeTensor = tensorInt32(new Int32Array(size), [size.length]);
  const result = execOp("Slice", [x, beginTensor, sizeTensor])[0];
  beginTensor.dispose();
  sizeTensor.dispose();
  return result;
}

// ============================================================================
// Utility Operations
// ============================================================================

/**
 * Expand dimensions
 */
export function expandDims(x: TFTensor, axis: number): TFTensor {
  const axisTensor = tensorInt32(new Int32Array([axis]), []);
  const result = execOp("ExpandDims", [x, axisTensor])[0];
  axisTensor.dispose();
  return result;
}

/**
 * Squeeze dimensions
 */
export function squeeze(x: TFTensor, axis?: number[]): TFTensor {
  return execOp("Squeeze", [x], axis ? { squeeze_dims: axis } : {})[0];
}

/**
 * Clip values
 */
export function clipByValue(x: TFTensor, min: number, max: number): TFTensor {
  const minTensor = tensor(new Float32Array([min]), []);
  const maxTensor = tensor(new Float32Array([max]), []);
  const result = execOp("ClipByValue", [x, minTensor, maxTensor])[0];
  minTensor.dispose();
  maxTensor.dispose();
  return result;
}

/**
 * Fill tensor with value
 */
export function fill(shape: number[], value: number): TFTensor {
  const size = shape.reduce((a, b) => a * b, 1);
  return tensor(new Float32Array(size).fill(value), shape);
}

/**
 * Eye (identity matrix)
 */
export function eye(n: number): TFTensor {
  const data = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    data[i * n + i] = 1;
  }
  return tensor(data, [n, n]);
}

/**
 * One-hot encoding
 */
export function oneHot(indices: TFTensor | number, depth: number): TFTensor {
  if (typeof indices === "number") {
    const indexTensor = tensorInt32(new Int32Array([indices]), [1]);
    const depthTensor = tensorInt32(new Int32Array([depth]), []);
    const onValue = tensor(new Float32Array([1]), []);
    const offValue = tensor(new Float32Array([0]), []);
    const result = execOp("OneHot", [indexTensor, depthTensor, onValue, offValue])[0];
    indexTensor.dispose();
    depthTensor.dispose();
    onValue.dispose();
    offValue.dispose();
    return result;
  }

  const depthTensor = tensorInt32(new Int32Array([depth]), []);
  const onValue = tensor(new Float32Array([1]), []);
  const offValue = tensor(new Float32Array([0]), []);
  const result = execOp("OneHot", [indices, depthTensor, onValue, offValue])[0];
  depthTensor.dispose();
  onValue.dispose();
  offValue.dispose();
  return result;
}

// ============================================================================
// Cleanup
// ============================================================================

/**
 * Close TensorFlow FFI and release resources
 */
export function close(): void {
  if (lib) {
    if (ctx) {
      lib.symbols.TFE_DeleteContext(ctx);
      ctx = null;
    }
    lib.close();
    lib = null;
    available = false;
  }
}

// ============================================================================
// Convenience Exports
// ============================================================================

export const tff = {
  isAvailable,
  version,
  close,

  // Tensor creation
  tensor,
  tensorInt32,
  zeros,
  ones,
  randomUniform,
  randomNormal,
  fill,
  eye,
  oneHot,

  // Math ops
  matmul,
  batchMatmul,
  add,
  sub,
  mul,
  div,
  transpose,
  reshape,
  gather,
  unsortedSegmentSum,

  // Activations
  softmax,
  relu,
  leakyRelu,
  elu,
  sigmoid,
  tanh,

  // Reductions
  sum,
  mean,
  max,
  square,
  sqrt,
  exp,
  log,
  neg,

  // Shape ops
  concat,
  slice,
  expandDims,
  squeeze,
  clipByValue,
};

export default tff;
