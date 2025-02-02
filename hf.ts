interface ModelInfo {
    parameters: number;
    quantization?: string;
}

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
