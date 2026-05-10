from pathlib import Path


def load_prompt_or_image(prompt_source, prompt_idx, prompt, image_path):
    if prompt_source == "prompt":
        assert prompt_idx == 0, "prompt_idx must be 0 when --prompt is provided directly"
        return prompt, image_path

    if prompt_source == "I2V_Wan_Web":
        assert prompt == image_path, "Prompt root and image root must match for I2V_Wan_Web"
        sample_id = str(prompt_idx).zfill(3)
        prompt_path = Path(prompt) / sample_id / "prompt.txt"
        image_path = Path(image_path) / sample_id / "image.jpg"
        return prompt_path.read_text().strip(), str(image_path)

    if prompt_source in {"T2V_Hunyuan10_Web", "T2V_Xingyang_Motion"}:
        prompt_path = Path(prompt)
        assert prompt_path.suffix == ".txt", "Prompt must be a txt file"
        prompts = prompt_path.read_text().splitlines()
        return prompts[prompt_idx], None

    raise ValueError(f"Invalid prompt source: {prompt_source}")
