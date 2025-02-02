import { HfInference } from '@huggingface/inference';

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

// Initialize Hugging Face client (uses local credentials if available)
const hf = new HfInference();

/**
 * Fetches model details from Hugging Face API.
 * 
 * @param repoId - The Hugging Face model repository ID
 * @returns Promise containing model parameters and quantization info
 */
async function getModelDetails(repoId: string): Promise<ModelInfo> {
    try {
        // Get model info using the HF client
        const [owner, model] = repoId.split('/');
        const modelInfo = await fetch(`https://huggingface.co/api/models/${repoId}`).then(r => r.json());
        
        // Try to get parameters from model card metadata
        let parameters: number;
        if (modelInfo.model_card?.parameters) {
            parameters = Number(modelInfo.model_card.parameters);
        } else {
            // If parameters not in metadata, estimate from model name
            const match = repoId.toLowerCase().match(/(\d+)b/);
            if (match) {
                parameters = parseInt(match[1]) * 1e9;
            } else {
                const matchM = repoId.toLowerCase().match(/(\d+)m/);
                if (matchM) {
                    parameters = parseInt(matchM[1]) * 1e6;
                } else {
                    throw new Error('Could not determine model parameters');
                }
            }
        }

        // Default to fp32 as most models are distributed in this format
        const quantization = 'fp32';

        return {
            parameters,
            quantization
        };
    } catch (error) {
        console.error('Error in getModelDetails:', error);
        throw error;
    }
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
export function calculateMemoryUsage(
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
export async function calculateHuggingFaceModelMemory(
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

// Test function to demonstrate usage
async function runTests() {
    console.log('Running memory calculation tests...\n');

    // Test 1: Direct memory calculation
    const directTest = calculateMemoryUsage(7, "4-bit", 2048);
    console.log('Test 1: Direct Memory Calculation');
    console.log('Model: 7B parameters, 4-bit quantization, 2048 context window');
    console.log(`Expected Memory Usage: ${directTest.toFixed(2)} GB\n`);

    // Test 2: Hugging Face model calculation
    try {
        console.log('Test 2: Hugging Face Model Calculation');
        console.log('Model: meta-llama/Llama-2-7b-hf, 2048 context window');
        const hfTest = await calculateHuggingFaceModelMemory('meta-llama/Llama-2-7b-hf', 2048);
        console.log(`Expected Memory Usage: ${hfTest.toFixed(2)} GB\n`);
    } catch (error) {
        console.error('Error fetching Hugging Face model details:', error);
    }
}

// Run the tests
runTests().catch(console.error); 