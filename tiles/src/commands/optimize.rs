use bon::Builder;
use dspy_rs::{
    COPRO, ChatAdapter, Evaluator, Example, LM, MetaSignature, Module, Optimizable, Optimizer,
    Predict, Prediction, Predictor, Signature, configure, example,
};
use indexmap::IndexMap;
use serde::Deserialize;
use std::fs;
use tilekit::modelfile::Modelfile;

#[derive(Debug, Deserialize)]
struct TrainingExample {
    input: String,
    output: String,
}

#[Signature]
pub struct SystemPromptSignature {
    /// Act as a task-specific assistant based on the instructions provided.
    #[input]
    pub user_input: String,
    #[output]
    pub ai_response: String,
}

#[Signature]
pub struct SyntheticDataSignature {
    /// You are a data generator. Given a SYSTEM prompt, generate a JSON array of 5 diverse and representative training examples.
    /// Each example must be a JSON object with EXACTLY two fields: "input" (the user's query) and "output" (the expected AI response).
    #[input]
    pub system_prompt: String,
    #[output]
    /// A JSON array like: [{"input": "...", "output": "..."}, ...]
    pub synthetic_data: String,
}

#[derive(Builder)]
pub struct PromptOptimizerModule {
    #[builder(default = Predict::new(SystemPromptSignature::new()))]
    pub predictor: Predict,
}

impl Module for PromptOptimizerModule {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
        self.predictor.forward(inputs).await
    }
}

impl Optimizable for PromptOptimizerModule {
    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable> {
        let mut params: IndexMap<String, &mut dyn Optimizable> = IndexMap::new();
        params.insert("predictor".to_string(), &mut self.predictor);
        params
    }
}

impl Evaluator for PromptOptimizerModule {
    async fn metric(&self, _example: &Example, prediction: &Prediction) -> f32 {
        let ai_response_field = prediction.get("ai_response", None);
        let ai_response = ai_response_field.as_str().unwrap_or("");

        let mut score = 0.0;

        // Reward non-empty responses
        if !ai_response.is_empty() {
            score += 0.2;
        }

        // Reward reasonable length (avoid very short or extremely verbose ones)
        let len = ai_response.len();
        if len > 50 && len < 1000 {
            score += 0.3;
        }

        // Reward structure (presence of newlines or bullet points often indicate better prompts/responses)
        if ai_response.contains('\n') || ai_response.contains('-') || ai_response.contains('*') {
            score += 0.2;
        }

        // Reward persona-like language
        let lower = ai_response.to_lowercase();
        if lower.contains("you are") || lower.contains("act as") || lower.contains("assistant") {
            score += 0.3;
        }

        score
    }
}

pub async fn optimize(modelfile_path: String, data_path: Option<String>, model: String) {
    println!("Optimizing Modelfile: {}", modelfile_path);

    // 1. Read Modelfile
    let content = match fs::read_to_string(&modelfile_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading Modelfile: {}", e);
            return;
        }
    };

    let mut modelfile: Modelfile = match content.parse() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing Modelfile: {}", e);
            return;
        }
    };

    let system_prompt = modelfile.system.clone().unwrap_or_default();
    if system_prompt.trim().is_empty() {
        eprintln!(
            "Error: The Modelfile has an empty SYSTEM prompt. Optimization requires a starting prompt to understand the task objective."
        );
        return;
    }
    println!("Current SYSTEM prompt: \"{}\"", system_prompt);

    // 2. Configure DSRs
    let lm = match LM::builder().model(model).build().await {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "Error configuring LM: {}. Make sure appropriate API keys are set.",
                e
            );
            return;
        }
    };

    configure(lm, ChatAdapter);

    // 3. Load or Generate Training Data
    let examples = if let Some(path) = data_path {
        match fs::read_to_string(path) {
            Ok(c) => {
                let data: Vec<TrainingExample> = match serde_json::from_str(&c) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Error parsing data file: {}", e);
                        vec![]
                    }
                };
                data.into_iter()
                    .map(|e| {
                        example! {
                            "user_input": "input" => e.input,
                            "ai_response": "output" => e.output,
                        }
                    })
                    .collect()
            }
            Err(e) => {
                eprintln!("Error reading data file: {}", e);
                vec![]
            }
        }
    } else {
        println!("No training data provided. Generating synthetic examples...");
        match generate_synthetic_examples(&system_prompt).await {
            Ok(exs) => exs,
            Err(e) => {
                eprintln!("Failed to generate synthetic examples: {}", e);
                vec![]
            }
        }
    };

    if examples.is_empty() {
        println!("No training examples available. Cannot optimize effectively.");
        return;
    }

    // 4. Run COPRO Optimizer
    println!(
        "Running COPRO optimizer with {} examples...",
        examples.len()
    );

    let mut sig = SystemPromptSignature::new();
    sig.update_instruction(system_prompt).unwrap_or_default();

    let mut module = PromptOptimizerModule::builder()
        .predictor(Predict::new(sig))
        .build();

    let optimizer = COPRO::builder().breadth(5).depth(2).build();

    match optimizer.compile(&mut module, examples).await {
        Ok(_) => println!("Optimization process completed successfully."),
        Err(e) => {
            eprintln!("Optimization failed: {}", e);
            return;
        }
    };

    let optimized_instructions = module.predictor.get_signature().instruction();
    println!("Optimization complete!");
    println!("New SYSTEM prompt: \n{}", optimized_instructions);

    // 5. Update Modelfile
    modelfile.update_system(&optimized_instructions);

    match fs::write(&modelfile_path, modelfile.to_string()) {
        Ok(_) => println!("Successfully updated {}", modelfile_path),
        Err(e) => eprintln!("Error writing Modelfile: {}", e),
    }
}

async fn generate_synthetic_examples(system_prompt: &str) -> anyhow::Result<Vec<Example>> {
    let predictor = Predict::new(SyntheticDataSignature::new());
    let input = example! {
        "system_prompt": "input" => system_prompt.to_string(),
    };

    let prediction = predictor.forward(input).await?;
    let field = prediction.get("synthetic_data", None);
    let synthetic_json = field.as_str().unwrap_or("");

    // Clean up potential markdown formatting
    let clean_json = synthetic_json
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let data: Vec<TrainingExample> = serde_json::from_str(clean_json)?;

    Ok(data
        .into_iter()
        .map(|e| {
            example! {
                "user_input": "input" => e.input,
                "ai_response": "output" => e.output,
            }
        })
        .collect())
}
