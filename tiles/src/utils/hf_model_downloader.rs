use std::{env, path::PathBuf};

use hf_hub::api::{
    Siblings,
    tokio::{Api, ApiBuilder, ApiError},
};

/// Manages model snapshot downloading from HuggingFace

/// Download the entire model (including snapshot) for the given model name
pub async fn pull_model(model_name: &str) -> Result<(), String> {
    snapshot_download(model_name).await
}

pub async fn snapshot_download(modelname: &str) -> Result<(), String> {
    let allow_patterns = vec![
        ".json",
        ".txt",
        ".safetensors",
        ".md",
        ".gitattributes",
        "LICENSE",
    ];
    let api_build_result = ApiBuilder::new()
        .with_progress(true)
        .with_cache_dir(PathBuf::from(get_model_cache()))
        .build();

    match api_build_result {
        Ok(api) => {
            let repo = api.model(modelname.to_owned());
            match repo.info().await {
                Ok(repo_info) => {
                    let filtered_siblings = repo_info
                        .siblings
                        .iter()
                        .filter(|sibling| {
                            allow_patterns
                                .iter()
                                .any(|pat| sibling.rfilename.contains(pat))
                        })
                        .collect::<Vec<&Siblings>>();

                    for sibling in filtered_siblings {
                        if repo.get(&sibling.rfilename).await.is_err() {
                            return Err(format!(
                                "{:?} failed to download, retry again",
                                &sibling.rfilename,
                            ));
                        }
                    }
                }
                Err(err) => return Err(format_hf_api_error(err)),
            };
        }
        Err(err) => return Err(format_hf_api_error(err)),
    }

    Ok(())
}

fn format_hf_api_error(api_error: ApiError) -> String {
    match api_error {
        ApiError::RequestError(err) => err.to_string(),
        ApiError::TooManyRetries(err) => err.to_string(),
        _err => "Something unexpected happened, check your internet connection".to_owned(),
    }
}

fn parse_model_name(model_name: &str) -> (String, Option<String>) {
    match model_name.split_once('@') {
        Some((model_name, commit_hash)) if commit_hash.len() > 0 => {
            (String::from(model_name), Some(String::from(commit_hash)))
        }
        _ => (String::from(model_name), None),
    }
}

fn get_model_cache() -> String {
    let default_cache = format!(
        "{}/.cache/huggingface",
        env::home_dir().unwrap().to_str().unwrap()
    );
    let cache_root = if let Ok(home) = env::var("HF_HOME") {
        home.to_owned()
    } else {
        default_cache
    };

    format!("{}/hub", cache_root)
}

fn hf_to_cache_dir(model_name: &str) -> String {
    if model_name.starts_with("models--") {
        return model_name.to_owned();
    }

    match model_name.split_once("/") {
        Some((org, model)) if model.len() > 0 => {
            format!("models--{}--{}", org, model)
        }
        _ => format!("models--{}", model_name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pull_model() {
        let _ = pull_model("driaforall/Tiny-Agent-a-0.5B");
        // let _ = pull_model("driaforall/Tiny-Agent-a-0.5B");
    }
}
