# """Tests for core.llm prompt builder functions (pure string logic, no Ollama)."""

# from core.llm import build_explanation_prompt, build_viz_counselor_prompt


# class TestBuildExplanationPrompt:
#     """Tests for the tabular FA counselor prompt builder."""

#     def _make_runs(self, n=1):
#         """Helper: create fake explanation run dicts."""
#         import pandas as pd

#         runs = []
#         for i in range(1, n + 1):
#             runs.append(
#                 {
#                     "run_number": i,
#                     "shap_df": pd.DataFrame(
#                         {"feature": ["math_score", "creativity"], "shap_value": [0.35, -0.12]}
#                     ),
#                     "shap_base_value": 0.33,
#                     "lime_df": pd.DataFrame(
#                         {"feature": ["math_score", "creativity"], "lime_weight": [0.40, -0.08]}
#                     ),
#                 }
#             )
#         return runs

#     def test_contains_dataset_name(self):
#         prompt = build_explanation_prompt(
#             dataset_name="Career Survey",
#             target_column="career",
#             task_type="classification",
#             prediction="Engineer",
#             explanation_runs=self._make_runs(),
#         )
#         assert "Career Survey" in prompt

#     def test_contains_prediction(self):
#         prompt = build_explanation_prompt(
#             dataset_name="ds",
#             target_column="career",
#             task_type="classification",
#             prediction="Designer",
#             explanation_runs=self._make_runs(),
#         )
#         assert "Designer" in prompt

#     def test_contains_xai_method_names(self):
#         prompt = build_explanation_prompt(
#             dataset_name="ds",
#             target_column="career",
#             task_type="classification",
#             prediction="Manager",
#             explanation_runs=self._make_runs(),
#             xai_methods=["SHAP", "LIME"],
#         )
#         assert "SHAP" in prompt
#         assert "LIME" in prompt

#     def test_multiple_runs_listed(self):
#         runs = self._make_runs(n=3)
#         prompt = build_explanation_prompt(
#             dataset_name="ds",
#             target_column="career",
#             task_type="classification",
#             prediction="Engineer",
#             explanation_runs=runs,
#         )
#         assert "Run 1" in prompt
#         assert "Run 2" in prompt
#         assert "Run 3" in prompt

#     def test_run_count_mentioned(self):
#         runs = self._make_runs(n=5)
#         prompt = build_explanation_prompt(
#             dataset_name="ds",
#             target_column="career",
#             task_type="classification",
#             prediction="Engineer",
#             explanation_runs=runs,
#         )
#         assert "5" in prompt

#     def test_custom_model_name(self):
#         prompt = build_explanation_prompt(
#             dataset_name="ds",
#             target_column="career",
#             task_type="classification",
#             prediction="Engineer",
#             explanation_runs=self._make_runs(),
#             model_name="Gradient Boosting",
#         )
#         assert "Gradient Boosting" in prompt


# class TestBuildVizCounselorPrompt:
#     """Tests for the visualization XAI track prompt builder."""

#     def test_contains_expression(self):
#         prompt = build_viz_counselor_prompt(
#             expression="3 + 5",
#             predicted_answer="8",
#             pred_class=0,
#             user_correction=None,
#             skill_attempts=5,
#             skill_marked_correct=3,
#             val_accuracy=0.92,
#             xai_methods="Grad-CAM, Saliency",
#         )
#         assert "3 + 5" in prompt
#         assert "8" in prompt

#     def test_contains_xai_method_names(self):
#         prompt = build_viz_counselor_prompt(
#             expression="1 + 1",
#             predicted_answer="2",
#             pred_class=0,
#             user_correction=None,
#             skill_attempts=1,
#             skill_marked_correct=1,
#             val_accuracy=0.95,
#             xai_methods="Grad-CAM, Saliency",
#         )
#         assert "Grad-CAM" in prompt
#         assert "saliency" in prompt.lower()

#     def test_user_correction_included_when_provided(self):
#         prompt = build_viz_counselor_prompt(
#             expression="7 - 3",
#             predicted_answer="5",
#             pred_class=1,
#             user_correction="The answer is 4",
#             skill_attempts=2,
#             skill_marked_correct=1,
#             val_accuracy=0.90,
#             xai_methods="Grad-CAM",
#         )
#         assert "The answer is 4" in prompt
#         assert "wrong" in prompt.lower() or "correction" in prompt.lower()

#     def test_no_correction_section_when_none(self):
#         prompt = build_viz_counselor_prompt(
#             expression="2 + 2",
#             predicted_answer="4",
#             pred_class=0,
#             user_correction=None,
#             skill_attempts=1,
#             skill_marked_correct=1,
#             val_accuracy=0.95,
#             xai_methods="Grad-CAM",
#         )
#         assert "correction" not in prompt.lower()

#     def test_accuracy_formatted(self):
#         prompt = build_viz_counselor_prompt(
#             expression="1 + 1",
#             predicted_answer="2",
#             pred_class=0,
#             user_correction=None,
#             skill_attempts=10,
#             skill_marked_correct=7,
#             val_accuracy=0.923,
#             xai_methods="Grad-CAM",
#         )
#         assert "92" in prompt  # 92.3% or 92%

#     def test_zero_attempts_handled(self):
#         """Edge case: no attempts yet, ratio should not cause division by zero."""
#         prompt = build_viz_counselor_prompt(
#             expression="5 + 5",
#             predicted_answer="10",
#             pred_class=0,
#             user_correction=None,
#             skill_attempts=0,
#             skill_marked_correct=0,
#             val_accuracy=None,
#             xai_methods="Saliency",
#         )
#         assert isinstance(prompt, str)
#         assert len(prompt) > 0
