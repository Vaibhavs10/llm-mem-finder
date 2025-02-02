const quantizationOptions = {
  "1-bit": 1,
  "2-bit": 2,
  "3-bit": 3,
  "4-bit": 4,
  "5-bit": 5,
  "6-bit": 6,
  "8-bit": 8,
  fp16: 16,
  fp32: 32,
} as const;

type QuantizationOption = keyof typeof quantizationOptions;

/**
 * Calculates the expected memory usage of an LLM in GB.
 *
 * @param parametersInBillions - Number of parameters in the model (in billions).
 * @param quantization - Quantization level (e.g., "4-bit", "fp16").
 * @param contextWindow - Size of the context window in tokens.
 * @param osOverheadGb - OS overhead in GB (default is 2 GB).
 * @returns The total memory usage in GB.
 */
function calculateMemoryUsage(
  parametersInBillions: number,
  quantization: QuantizationOption,
  contextWindow: number,
  osOverheadGb: number = 2
): number {
  // Convert parameters from billions to actual count
  const parameters = parametersInBillions * 1e9;

  // Calculate memory for parameters
  const bitsPerParameter = quantizationOptions[quantization];
  const bytesPerParameter = bitsPerParameter / 8;
  const parameterMemoryBytes = parameters * bytesPerParameter;

  // Calculate memory for context window
  const contextMemoryBytes = contextWindow * 0.5 * 1e6; // 0.5 bytes per token

  // Total memory in bytes
  const totalMemoryBytes = parameterMemoryBytes + contextMemoryBytes + osOverheadGb * 1e9;

  // Convert to GB
  const totalMemoryGb = totalMemoryBytes / 1e9;

  return totalMemoryGb;
}

// Example usage:
const parametersInBillions = 7; // 7 billion parameters
const quantization: QuantizationOption = "4-bit"; // Quantization level
const contextWindow = 2048; // Context window in tokens
const osOverheadGb = 2; // OS overhead in GB

const memoryUsage = calculateMemoryUsage(
  parametersInBillions,
  quantization,
  contextWindow,
  osOverheadGb
);

console.log(`Expected Memory Usage: ${memoryUsage.toFixed(2)} GB`);
