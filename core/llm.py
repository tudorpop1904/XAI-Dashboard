"""
llm.py — Ollama-based LLM integration.

Builds explanation prompts from SHAP/LIME run data and streams
the LLM response token-by-token for Streamlit's st.write_stream.
"""

import os
import ollama


OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_client = ollama.Client(host=OLLAMA_HOST)


def build_explanation_prompt(
    dataset_name,
    target_column,
    task_type,
    prediction,
    explanation_runs,
    model_name="Random Forest",
    xai_methods=None,
):
    """
    Build a prompt summarising all SHAP + LIME runs and asking the LLM
    for a plain-English, student-friendly interpretation.
    """
    if xai_methods is None:
        xai_methods = ["SHAP", "LIME"]

    parts = []

    parts.append(
        "You are a friendly and encouraging Career Guidance Counselor. "
        "Your goal is to explain to a student or a child why a specific career "
        "path was suggested based on their skills, grades, and interests. "
        "Also, for research purposes, and aiding a better human interpretation of "
        "graphical results, refer to the SHAP and LIME values, and explain them in a "
        "clear and concise manner. "
        "Your language should be inspiring, clear, and easy for a student to understand. "
        "Avoid heavy technical jargon and focus on how their strengths (features) "
        "map to the career path.\n"
    )

    parts.append(
        f"Dataset: {dataset_name}\n"
        f"Target column (what we are predicting): {target_column}\n"
        f"Task type: {task_type}\n"
        f"Predictor model used: {model_name}\n"
        f"Model prediction: {prediction}\n"
        f"XAI methods used: {', '.join(xai_methods)}\n"
    )

    method_descs = []
    if "SHAP" in xai_methods:
        method_descs.append(
            "  • SHAP — assigns each feature a contribution score (positive = "
            "pushes prediction up, negative = pushes it down). Usually stable."
        )
    if "LIME" in xai_methods:
        method_descs.append(
            "  • LIME — similar idea but uses a local surrogate model, so values "
            "may vary between runs."
        )

    num_runs = len(explanation_runs)
    parts.append(
        f"Below are the results of {num_runs} paired explanation run(s) using "
        f"the following explainability method(s):\n" + "\n".join(method_descs) + "\n"
    )

    for run in explanation_runs:
        run_num = run["run_number"]
        parts.append(f"--- Run {run_num} ---")

        if "shap_df" in run and run["shap_df"] is not None:
            parts.append(f"SHAP base value: {run['shap_base_value']:.6f}")
            shap_top = run["shap_df"].head(10)
            shap_lines = [
                f"  {row['feature']}: shap={row['shap_value']:.6f}"
                for _, row in shap_top.iterrows()
            ]
            parts.append("Top SHAP contributions:\n" + "\n".join(shap_lines))

        if "lime_df" in run and run["lime_df"] is not None:
            lime_top = run["lime_df"].head(10)
            lime_lines = [
                f"  {row['feature']}: weight={row['lime_weight']:.6f}"
                for _, row in lime_top.iterrows()
            ]
            parts.append("Top LIME contributions:\n" + "\n".join(lime_lines))

        parts.append("")

    parts.append(
        f"Based on ALL {num_runs} run(s) above, write a clear, human-readable summary that:\n"
        "1. States the prediction in plain language.\n"
        "2. Lists the top 3-5 features that most influenced the prediction and "
        "explains the direction of their influence (pushed the result higher or lower).\n"
        "3. Notes which features were consistently important across runs "
        "(stable) and which varied (unstable).\n"
        "4. Highlights any noteworthy agreement or disagreement between the XAI methods used.\n"
        "5. Ends with a short, one-sentence takeaway.\n\n"
        "Keep it under 300 words. Do NOT use bullet points with technical values — "
        "write flowing paragraphs instead."
    )

    return "\n".join(parts)


def stream_llm_explanation(prompt):
    """
    Stream the LLM response token-by-token via Ollama.
    Yields successive text chunks for st.write_stream().
    """
    stream = _client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token


def check_ollama_available():
    """
    Check whether Ollama is reachable and the required model is pulled.

    Returns
    -------
    ok : bool
    message : str
    """
    try:
        models = _client.list()
        model_names = [m.model for m in models.models]

        matches = [
            n for n in model_names if n.startswith(OLLAMA_MODEL.split(":")[0])
        ]

        if matches:
            return True, f"Ollama is running. Model `{matches[0]}` found and ready."
        else:
            return False, (
                f"Ollama is running, but model `{OLLAMA_MODEL}` is not found.\n"
                f"Please run `docker exec -it xai-ollama ollama pull {OLLAMA_MODEL}` "
                "to download it (~2.3 GB)."
            )

    except Exception as e:
        return False, (
            f"Could not connect to Ollama: {e}\n"
            "If using Docker, ensure the `ollama` container is running "
            "(`docker-compose up -d`)."
        )


def build_viz_counselor_prompt(
    expression: str,
    predicted_answer: str,
    pred_class: int,
    user_correction: str | None,
    skill_attempts: int,
    skill_marked_correct: int,
    val_accuracy: float | None,
    xai_methods: str,
):
    """
    Prompt for the Visualization XAI track (Grad-CAM + saliency on handwriting).
    """
    ratio = (
        skill_marked_correct / skill_attempts
        if skill_attempts > 0
        else None
    )
    corr = user_correction.strip() if user_correction else None
    parts = [
        "You are a supportive math tutor and learning coach (like a friendly "
        "Photomath-style helper). A student used an AI that reads a handwritten "
        "expression from a photo and predicts which expression it is, then shows "
        "the numeric answer for that expression.\n",
        "Explainability: the app also shows **Grad-CAM** and **saliency maps** — "
        "heatmaps over the image showing which pixels most influenced the model's "
        "decision. Use that idea qualitatively (regions of the digit/operator strokes) "
        "without claiming you saw the exact image pixels.\n",
    ]
    if val_accuracy is not None:
        parts.append(
            f"Model validation accuracy (held-out synthetic set): {val_accuracy:.1%}\n"
        )
    parts.extend(
        [
            f"Predicted expression: {expression}\n",
            f"Predicted numeric answer: {predicted_answer}\n",
            f"Predicted class index: {pred_class}\n",
            f"XAI overlays shown to the student: {xai_methods}\n",
        ]
    )
    if corr:
        parts.append(
            f"The student said the AI was wrong and gave their answer / correction: {corr}\n"
            "Acknowledge this, reconcile politely with the model output, and suggest "
            "how to verify by hand.\n"
        )
    parts.append(
        f"Self-reported practice so far: {skill_marked_correct} marked-correct out of "
        f"{skill_attempts} attempts.\n"
    )
    if ratio is not None:
        parts.append(
            f"Approx. self-reported accuracy: {ratio:.0%}. "
            "Map this cautiously to a **skill level label** (e.g. Beginner / "
            "Developing / Proficient) with one sentence of justification. "
            "If attempts are very few, say the estimate is uncertain.\n"
        )
    parts.append(
        "Write under 250 words in short paragraphs. Be encouraging. Mention how "
        "Grad-CAM vs saliency differ at a high level (class-discriminative regions "
        "vs input-gradient sensitivity). Do not invent SHAP/LIME values."
    )
    return "".join(parts)
