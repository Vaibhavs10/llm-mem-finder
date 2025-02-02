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

interface ModelInfo {
    parameters: number;
    quantization?: string;
}

/**
 * Fetches model details from Hugging Face API.
 * 
 * @param repoId - The Hugging Face model repository ID
 * @returns Promise containing model parameters and quantization info
 */
async function getModelDetails(repoId: string): Promise<ModelInfo> {
    // 1. Fetch model info from Hugging Face API
    const response = await fetch(`https://api-inference.huggingface.co/models/${repoId}`);
    const data = await response.json();

    // 2. Extract parameter count
    const parameters = data.modelCard.parameters;

    // 3. Determine quantization
    let quantization: string | undefined;
    const modelFiles = data.modelCard.modeldownloads;
    if (modelFiles) {
        // Look for files indicating quantization
        const fp16File = modelFiles.find(f => f.suffix === 'fp16');
        const int8File = modelFiles.find(f => f.suffix === 'int8');
        const int4File = modelFiles.find(f => f.suffix === 'int4');
        
        if (int4File) {
            quantization = '4-bit';
        } else if (int8File) {
            quantization = '8-bit';
        } else if (fp16File) {
            quantization = 'fp16';
        }
    }

    return {
        parameters: parameters,
        quantization: quantization
    };
}

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

/**
 * Calculates memory usage for a Hugging Face model.
 * 
 * @param repoId - The Hugging Face model repository ID
 * @param contextWindow - Size of the context window in tokens
 * @param osOverheadGb - OS overhead in GB (default is 2 GB)
 * @returns Promise containing the calculated memory usage in GB
 */
async function calculateHuggingFaceModelMemory(
    repoId: string,
    contextWindow: number,
    osOverheadGb: number = 2
): Promise<number> {
    const modelInfo = await getModelDetails(repoId);
    
    // Convert parameters to billions
    const parametersInBillions = modelInfo.parameters / 1e9;
    
    // Use default fp32 if no quantization specified
    const quantization = (modelInfo.quantization || 'fp32') as QuantizationOption;
    
    return calculateMemoryUsage(parametersInBillions, quantization, contextWindow, osOverheadGb);
}

// Example usage:
// const memoryUsage = await calculateHuggingFaceModelMemory('facebook/opt-350m', 2048);
// console.log(`Expected Memory Usage: ${memoryUsage.toFixed(2)} GB`); 