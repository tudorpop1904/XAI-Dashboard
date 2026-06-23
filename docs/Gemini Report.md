# Forensic Detection of Synthetic Imagery: Generative Architectures, Edge-Based Vision-Language Models, and Visual Attribution Frameworks

> To establish a mathematically rigorous and interpretably robust diagnostic framework for distinguishing authentic photographs from synthetic media, one must analyze the generative pipelines of modern text-to-image (T2I) systems. Modern synthetic image engines typically rely on Latent Diffusion Models (LDMs) or Diffusion Transformers (DiTs) to translate high-dimensional textual conditioning into coherent visual matrices. This transformation involves complex operations across distinct neural domains, leaving physical and statistical traces that forensic tools can exploit.

## 1. The Mechanics of Text-to-Image Generation and Latent Synthesis

> Standard pixel-space diffusion is computationally prohibitive due to the high dimensionality of raw image matrices. Latent Diffusion Models circumvent this constraint by utilizing a pre-trained Variational Autoencoder (VAE) to project an image $X \in \mathbb{R}^{H \times W \times 3}$ into a low-dimensional latent representation $z = \mathcal{E}(X) \in \mathbb{R}^{h \times w \times c}$ via an encoder network $\mathcal{E}$. The generative diffusion process is then executed entirely within this compressed latent manifold. Once the denoising trajectory is complete, a corresponding decoder network $\mathcal{D}$ reconstructs the latent features back into pixel space, yielding the final image $X' = \mathcal{D}(z)$.

 +--------------------+      +-------------------------+      +-------------------+
  | Text Prompt Input  | ---> |  CLIP / T5 Text Encoder | ---> | Text Embedding(e) |
  +--------------------+      +-------------------------+      +-------------------+
                                                                         |
                                                                         v
  +--------------------+      +-------------------------+      +-------------------+
  | Latent Noise (z_T) | ---> |  Denoising Block (DiT)  | <--- | Cross-Attention   |
  +--------------------+      +-------------------------+      +-------------------+
                                           | (Iterative Denoising)
                                           v
                              +-------------------------+
                              | Resolved Latent (z_0)   |
                              +-------------------------+
                                           |
                                           v
                              +-------------------------+      +-------------------+
                              |       VAE Decoder       | ---> | Generated Image X'|
                              +-------------------------+      +-------------------+

> This compression process introduces a fundamental information bottleneck. The standard VAE encoder inevitably discards high-frequency details, such as subtle textures and fine edges, during the downsampling phase. This loss of detailed structural data forces the decoder to reconstruct fine-grained features using learned priors, which often manifests as localized blurriness or structural inconsistencies in the final output.

> Furthermore, standard VAE decoders frequently introduce aliasing and translation-invariance anomalies due to a lack of shift-equivariance in their upsampling operations. Even minor perturbations or sub-pixel shifts in the latent noise can lead to dramatically different and physically inconsistent visual structures, leaving distinct forensic signatures in the reconstructed image.

> Historically, the core of the denoising engine—responsible for predicting and removing noise at each step $t$ of the reverse trajectory—has been dominated by U-Net architectures utilizing convolutional layers. More recently, architectural scaling has shifted toward Diffusion Transformers (DiTs), which replace the convolutional backbone with self-attention and cross-attention blocks.

> Unlike U-Net models that pass text conditioning representations through a static cross-attention mechanism, DiTs treat both the text embeddings and the noised image latents as sequences of tokens within a unified sequence. This joint processing facilitates a bidirectional flow of semantic information between modalities. To handle multi-scale feature learning, hierarchical variants such as the U-shaped Transformer (UViT) and the Hierarchical Diffusion Transformer (HDiT) have emerged.

> The UViT architecture maintains a consistent spatial resolution across the network, relying on skip connections to bridge multi-scale features. UViT treats all conditioning parameters (timesteps, class attributes, and text tokens) as input tokens within the unified sequence rather than using adaptive layer normalization (adaLN) or cross-attention. Conversely, the HDiT architecture incorporates explicit downsampling and upsampling blocks, employing localized attention mechanisms (such as Neighborhood Attention) in high-resolution outer layers and global attention in lower-resolution bottlenecks to optimize computational efficiency on consumer hardware.

> The mathematical formulation governing the latent denoising loss at step $t$ is expressed as:

$$\mathcal{L}_{LDM} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0, I), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, \tau(P))\|_2^2 \right]$$

> where $z_t$ is the noised latent at timestep $t$, $\epsilon$ is the target Gaussian noise, $\epsilon_\theta$ is the noise-prediction network, and $\tau(P)$ represents the conditioning textual embedding derived from the input prompt $P$.

> The translation of semantic concepts from a text prompt to localized image tokens within a DiT occurs through specialized cross-modal circuits. When processing a prompt, spatial-relation information (such as "under," "above," or "to the left of") and single-object attributes (such as color, texture, or scale) are transmitted to the image tokens via a distinct two-stage circuit. This circuit is governed by specialized cross-attention heads that dynamically read and bind attributes to their corresponding spatial coordinates.

> If these heads fail to coordinate properly—often due to quantization constraints or prompt complexity—the model yields spatial and attribute-binding errors. These errors appear as physical anomalies, such as objects floating, shadows projecting at physically impossible angles, or text rendering with scrambled characters.

> Analyses of the diffusion process reveal that image generation occurs autoregressively within the frequency domain. Low-frequency components, which dictate global layout, color distribution, and coarse shapes, are resolved early in the denoising trajectory. High-frequency components, which govern texture, fine detail, and sharp boundaries, only emerge in the final denoising steps.

> Because high-frequency details are synthesized in noise-dominated regimes during these final steps, the generation of high-frequency details is prone to instability. This instability often results in grid-like spectral artifacts or high-frequency discrepancies that do not align with the natural statistical distributions of real photographs. These frequency-domain inconsistencies provide a highly reliable statistical signal for forensic detection systems.

## 2. Forensic Verification Benchmarks and Vision-Language Architectures

> Forensic detection has evolved from static binary classification toward interpretable, reasoning-driven validation pipelines. While traditional classifiers excel at isolating low-level texture patterns and frequency discrepancies, they function as uninterpretable black boxes and struggle to generalize to unseen generative models. Modern forensic architectures integrate Vision-Language Models (VLMs) to combine visual perception with deep linguistic reasoning.

> Generative anomalies are broadly categorized into low-level structural artifacts and high-level semantic inconsistencies. Low-level artifacts include grid patterns from convolutional upsampling, edge-based discrepancies, and unnatural high-frequency noise distributions.

> Conversely, semantic inconsistencies violate real-world knowledge, common sense, and physical laws. Examples include anatomical anomalies (such as asymmetrical pupils or extra digits), geometrical rendering errors in text, distorted reflections, and perspective projection failures.

| Model Name | Primary Architecture | Resolution Limits | Characteristic Forensic Anomalies |
|------------|----------------------|-------------------|-----------------------------------|
| Stable Diffusion 1.5 | Convolutional U-Net + VAE | $512 \times 512$ pixels | Severe high-frequency upsampling grid artifacts, poor text rendering. |
| Stable Diffusion 3.5 | MMDiT + VAE | High / Scalable | Localized aliasing, subtle anatomical asymmetry, minor perspective errors. |
| Flux.1 (Dev / Schnell) | Hybrid MMDiT / Single-DiT | High / Any Resolution | Subtle skin-texture over-smoothing, minor text boundary imperfections. |
| Recraft V3 | DiT + VAE | High / Vector Oriented | Geometric alignment failures, structural edge artifacts in high-contrast transitions. |
| Imagen 3 / 4 | Cascade Diffusion | High / Variable | Mild background blur inconsistencies, shadow-direction violations. |
| GigaGAN | GAN Generator + VAE | Low to Medium | Periodic frequency artifacts, alignment distortions in dense textures. |
| Visual Autoregressive (VAR) | Autoregressive Scales | Grid-based Scaled | Step-wise resolution transitions, boundary artifacts along hierarchical scale edges. |

> To resolve the transparency limitations of black-box classifiers, systems such as AIGI-Holmes and the Faster-Than-Lies framework integrate VLMs (specifically the Qwen2-VL family) with dedicated visual classification engines. In these architectures, a lightweight convolutional model (e.g., Faster-Than-Lies) serves as the primary technical classification engine, while the VLM acts as a linguistic explainer. The VLM processes both the image and the localized technical anomalies to generate natural language explanations that align with human judgment.

> To guide the linguistic generation of the VLM, these frameworks construct visual artifact localization heatmaps using autoencoder-based reconstruction error maps. By projecting the reconstruction discrepancies as localized spatial heatmaps directly into the VLM's visual input, the model is guided to "see" and explain the specific regional anomalies.

> Standard datasets like COCO-2017 are used as real baselines, while synthetic corpora are generated across diverse SOTA models (e.g., Stable Diffusion 3.5, Flux.1 Dev, Imagen 3, Recraft V3) to train the models on semantic fake detection and fine-grained visual forensics.

> Evaluating the visual and linguistic reasoning capabilities of these models is essential for measuring their alignment with human forensic performance. The benchmark metrics below compare human inter-annotator agreement against state-of-the-art multimodal large language models (MLLMs) and custom edge-compatible frameworks across standard explainable AI-generated image detection datasets.

| Metric / Benchmark | Dataset Source | Evaluated System / Baseline | Score / Accuracy | Forensic Performance Implications |
| Inter-Annotator Agreement | STSF / RewardBench | Human Experts | 98.30% | Establishes the target ceiling for semantic inconsistency detection. |
| Explainable VLM Score | XAIGID-RewardBench | Best Performing MLLM | 88.76% | Demonstrates a remaining reasoning gap in complex scene analysis. |
| Edge Accuracy (Perturbed) | Extended CiFAKE | Faster-Than-Lies (29.8M) | 96.50% | High technical detection robustness under adversarial settings. |
| Edge VLM Accuracy | CiFAKE Low-Resolution | Qwen2-VL-7B (Local) | 90.00% | High visual reasoning accuracy on resource-constrained devices. |
| Inference Latency | 8-Core CPU Edge | Faster-Than-Lies | 175 ms | Demonstrates the feasibility of real-time local deployment. |
| Semantic GenEval Score | GenEval Benchmark | SVG-T2I Pipeline | 0.75 | Baseline metric evaluating compositional layout compliance. |

## 3. Mathematical Formulations of Visual Explainable AI Methods

> To establish transparency, the diagnostic dashboard must integrate post-hoc Explainable AI (XAI) methods. These methods map the relationship between input visual features and the model's output predictions.

### a. Saliency Maps (Vanilla Gradients)

> Vanilla gradient-based saliency maps visualize the sensitivity of a model's prediction to individual pixel perturbations. Let $S_c(I)$ be the unnormalized class score (logit) predicted by the network for class $c$ given an input image $I \in \mathbb{R}^{H \times W \times 3}$. The saliency map $M_{\text{Saliency}} \in \mathbb{R}^{H \times W}$ is computed by taking the absolute value of the gradient of the class score with respect to the input image pixels:

$$M_{\text{Saliency}}(x, y) = \max_{c' \in \{R, G, B\}} \left| \frac{\partial S_c(I)}{\partial I(x, y, c')} \right|$$

> While vanilla saliency provides highly detailed pixel-level attributions, it suffers from several limitations. In saturation regions—where the activation function's output changes minimally in response to input modifications—the resulting gradients approach zero, failing to accurately reflect feature importance.

> Furthermore, raw gradients are highly susceptible to noise, often highlighting irrelevant background pixels. To mitigate this, advanced variations such as SmoothGrad and Integrated Gradients average the gradients over multiple perturbed inputs or along a straight path from a blank baseline to the target image.


### b. Grad-CAM (Gradient-Weighted Class Activation Mapping)

> Grad-CAM addresses the noise limitations of pixel-level saliency by leveraging the coarse, high-level semantic features captured in the final convolutional or attention layers of a network. By computing gradients with respect to intermediate feature maps rather than input pixels, Grad-CAM isolates the specific regions driving the classification decision.

> Let $A^k \in \mathbb{R}^{U \times V}$ represent the activation map of the $k$-th channel in the final convolutional layer. The importance weight $\alpha_k^c$ of this channel for a target class $c$ is computed using the global average pooling of the gradients of the class score $y^c$ with respect to the feature map $A^k$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A^k_{i, j}}$$

> where $Z = U \times V$ represents the spatial dimensions of the feature map. The final Grad-CAM heatmap $L^c_{\text{Grad-CAM}} \in \mathbb{R}^{U \times V}$ is computed as a weighted linear combination of the feature maps, passed through a Rectified Linear Unit (ReLU) to isolate only the features that positively influence the target class:

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

> Within multimodal language architectures, Grad-CAM can be applied to both the visual encoder and the autoregressive language decoder. Applying Grad-CAM to the visual encoder (e.g., a CLIP Vision Transformer) highlights the specific visual regions that the model relies on when projecting image tokens.

> When applied to the LLM decoder, Grad-CAM maps the flow of information across layers, revealing how visual tokens interact with text tokens to generate specific descriptive outputs. This combined approach enables fine-grained tracking of visual-semantic reasoning. 

+-------------+     Forward Pass     +-------------+     GAP of Gradients     +---------------+
  | Feature Map | -------------------> | Class Score | -----------------------> | Channel Weight|
  |    (A^k)    |                      |    (y^c)    |                          |   (\alpha_k^c)|
  +-------------+                      +-------------+                          +---------------+
         |                                                                              |
         +---------------------------->   Weighted Sum   <------------------------------+
                                                |
                                                v
                                       +-----------------+
                                       | ReLU Activation |
                                       +-----------------+
                                                |
                                                v
                                       +-----------------+
                                       | Grad-CAM Map L^c|
                                       +-----------------+

### c. Visual Sobol Indices (Global Variance-Based Sensitivity Analysis)

> To evaluate the global sensitivity of a model's prediction without relying on local, gradient-dependent approximations, researchers employ Sobol Sensitivity Analysis. Built upon the Hoeffding-Sobol (ANOVA) decomposition, this method decomposes the total variance of a model's output into fractions that can be directly attributed to individual input features or their higher-order interactions.

> Let the model's prediction function be $Y = f(X)$, where $X = (X_1, \dots, X_p)$ represents independent input variables (e.g., perturbed regions or masked patches of an input image). The Hoeffding-Sobol decomposition expresses the model as an orthogonal sum of increasing dimensions:

$$f(X) = f_0 + \sum_{i=1}^p f_i(X_i) + \sum_{1 \le i < j \le p} f_{i,j}(X_i, X_j) + \dots + f_{1\dots p}(X_1, \dots, X_p)$$

> where $f_0 = \mathbb{E}[Y]$ represents the global mean. By integrating this orthogonality, the total variance $Var(Y)$ is decomposed as:

$$\text{Var}(Y) = \sum_{i=1}^p V_i + \sum_{1 \le i < j \le p} V_{i,j} + \dots + V_{1\dots p}$$

> where $V_i = \text{Var}_{X_i}(\mathbb{E}_{X_{\sim i}}[Y \mid X_i])$ represents the first-order variance contribution of input $X_i$, and $X_{\sim i}$ denotes all variables except $X_i$.

> The first-order Sobol index $S_i$, which measures the direct, isolated effect of input region $X_i$ on the model's output variance, is expressed as:

$$S_i = \frac{\text{Var}_{X_i}\left[ \mathbb{E}_{X_{\sim i}}[f(X) \mid X_i] \right]}{\text{Var}(Y)}$$

> To capture the total contribution of input region $X_i$, including both its direct effects and all higher-order interactions with other spatial regions, we compute the Total Sobol Index $S_{Ti}$:

$$S_{Ti} = 1 - \frac{\text{Var}_{X_{\sim i}}\left[ \mathbb{E}_{X_i}[f(X) \mid X_{\sim i}] \right]}{\text{Var}(Y)}$$

> To compute these total indices on high-dimensional images, the Sobol Attribution Method leverages Quasi-Monte Carlo (QMC) sequences to sample perturbation masks $M$. These masks are applied to the input image through a continuous perturbation function $\Phi$ (such as Gaussian blur or inpainting), generating perturbed inputs that are fed into the model.

> To optimize estimation efficiency and ensure numerical stability, the framework employs the Jansen estimator. Given two independent sampling design matrices $A$ and $B$ containing $n$ samples, the Jansen estimator for the total variance contribution $V_{tot, i}$ is formulated as:

$$\hat{V}_{tot, i} = \frac{1}{2n - 1} \sum_{k=1}^n \left( f(A^{(k)}) - f(A^{(k)}_{B(i)}) \right)^2$$

> where $A^{(k)}$ is the $k$-th row of the baseline design matrix, and $A^{(k)}_{B(i)}$ represents a re-sampled matrix where all columns are drawn from $A$ except for the $i$-th column, which is sourced from $B$.

> By projecting these total indices back onto the image plane, the dashboard generates highly reliable, mathematically rigorous attribution maps that capture non-linear feature interactions. These indices can also be generalized to the space-scale domain using wavelet transforms (the Wavelet Scale Attribution Method, or WCAM), enabling simultaneously localized spatial and frequency sensitivity analysis.

### d. Visual Pointwise Mutual Information (Visual PMI)

> Pointwise Mutual Information (PMI) is an information-theoretic metric that quantifies the statistical association between specific realized events. Within visual attribution, PMI measures how strongly a specific visual feature or image region $x$ co-occurs with a target class prediction or textual output token $y$, compared to their independent probability of occurrence.

$$\text{PMI}(x, y) = \log \frac{p(x, y)}{p(x)p(y)}$$

> Since standard PMI can approach $-\infty$ for unobserved or highly rare feature pairs, the visual framework utilizes Positive Pointwise Mutual Information (PPMI), which clips negative association values at zero:

$$\text{PPMI}(x, y) = \max\left(0, \log \frac{p(x, y)}{p(x)p(y)}\right)$$

> In vision-language models, the text prompt introduces context that can bias standard mutual information. To isolate the direct relationship between specific descriptive words $y_*$ and image pixels $x$, we employ Conditional Pointwise Mutual Information (CPMI), which factors out the surrounding textual context $c$:

$$\text{PMI}(x; y_* \mid c) = h(y_* \mid c) - h(y_* \mid x, c)$$

> where $h(\cdot)$ represents the pointwise entropic uncertainty.

> By evaluating this conditional metric across all spatial coordinates, visual PMI highlights the specific fine-grained visual details (e.g., eyes, text contours, structural edges) associated with a target token, while effectively filtering out background context. These pointwise estimations are highly effective at identifying semantic fakes, as they directly isolate inconsistencies between descriptive words and their corresponding visual representations.

## 4. Engineering a Local Diagnostic Dashboard with Quantized VLMs

> Deploying a local diagnostic dashboard that combines these five attribution methods with a quantized VLM requires resolving a fundamental engineering contradiction: gradient-based explainability vs. non-differentiable quantized runtimes.

### a. The Quantization Bottleneck and Implementation Paradox

> When a model is quantized to 4-bit (such as a $q4\_K\_M$ GGUF model) and executed locally within Ollama or llama.cpp, the underlying engine optimizes memory efficiency by discarding the backward pass graph and intermediate activation tensors. Consequently, calling standard gradient functions is impossible, as the necessary automatic differentiation graphs do not exist in the optimized runtime. This creates an architectural split in the dashboard's implementation:

                                  DASHBOARD PIPELINE
                                          |
                     +--------------------+--------------------+
                     |                                         |
             Gradient-Based                             Perturbation-Based
        (Grad-CAM, Saliency Maps)                   (Occlusion, Visual Sobol)
                     |                                         |
         [Requires Unquantized                       [Requires Forward Only,
            Surrogate Model]                            Fully Model-Agnostic]
                     |                                         |
                     v                                         v
         +-----------------------+                 +-----------------------+
         | Run on PyTorch FP16/  |                 | Run Directly on GGUF  |
         | FP32 Active Backend   |                 | Quantized Local Model |
         +-----------------------+                 +-----------------------+

> To resolve this limitation, three distinct architectural integration paths are proposed:

1. The Dual-Model Proxy Architecture (Surrogate Integration): The VLM (Qwen2.5-VL 7B Q4_K_M) runs locally under Ollama and is queried strictly via forward-pass inference to output the classification label and generate descriptive text. Concurrently, a lightweight, unquantized convolutional neural network (such as a ResNet50 or the Faster-Than-Lies classifier) is maintained in a PyTorch backend. The gradient-based methods (Grad-CAM and Saliency Maps) are computed using PyTorch's backward hooks on this unquantized surrogate model, projecting the resulting heatmaps onto the dashboard.
2. The Model-Agnostic Perturbation Paradigm: Methods such as Occlusion Sensitivity and Visual Sobol Indices are fully model-agnostic and do not require gradients. They require only forward-pass evaluation scores. The dashboard generates spatial masks, applies them to the input image, and passes these perturbed variants through the quantized Ollama model, using the variations in the output classification probability to compute the Sobol indices and Occlusion maps. This fully preserves local execution without requiring surrogate models.
3. Forward-Pass Visual PMI Approximation: Visual PMI is approximated by evaluating the conditional probability of output tokens across systematically masked iterations of the input image. By measuring the drop in the target token's output probability when specific regions are masked, the system estimates pointwise dependency directly from forward-pass logits, sidestepping the need for backpropagation.

### b. Technical Comparison of Visual Attribution Methods

> To guide the architectural design of the diagnostic dashboard, the table below provides a comparative analysis of the five explainability methods, outlining their computational characteristics, data requirements, and specific utility in detecting synthetic imagery.

| Explainability Method | Mathematical Basis | Runtime Access Type | Computational Cost | Edge Deployment Feasibility | Primary Utility in Synthetic Image Forensic Dashboards | 
|-----------------------|--------------------|---------------------|--------------------|--------------------------------|--------------------------------------------------------|
| Occlusion Sensitivity | Systematic sliding-window perturbation of image patches; evaluates local drop in output score. | Black-Box (Forward pass only). | Low to Medium; scales with window stride and image size. | High; trivial to implement with simple sliding loop. | Excellent for identifying coarse, localized artifacts (e.g., distorted hands, warped text blocks). |
| Saliency Maps | Raw gradient of the target class logit with respect to input pixels. | White-Box (Backward gradient pass). | Extremely Low; requires only a single backward pass. | Low on quantized models; requires unquantized surrogate backend. | Highlights fine-grained structural anomalies, high-frequency edge inconsistencies, and noise patterns. |
| Grad-CAM | Weighted linear combination of final-layer convolutional or attention feature maps. | White-Box (Layer activations and backward gradients). | Low; single forward-backward pass. | Low on quantized models; requires surrogate or dual-model framework. | Extremely effective for mapping global semantic context and identifying which high-level concepts drove the VLM's classification. |
| Visual Sobol Indices | Variance decomposition using global sensitivity analysis and QMC perturbation sequences.Black-Box (Evaluates forward pass variations of sampled masks). | Extremely High; requires $N(d+2)$ forward evaluations (typically $N \ge 32$). | Medium; restricted by forward-pass latency on constrained CPUs. | Rigorous quantification of non-linear feature interactions; isolates complex, distributed fakes (e.g., lighting and shadow perspective errors). |
| Visual PMI | Ratio of joint probability to marginal probabilities of image regions and target tokens. | Can be formulated as Black-Box (perturbed forward passes) or White-Box (attention weight gradients). | Medium to High; scales with sampling resolution or token length. | High if using attention-map approximations; Medium if utilizing sampling loops. | Best for validating attribute binding and semantic fakes (e.g., confirming if a rendered object matches its textual description). |

## 5. Architectural Recommendations for Dashboard Synthesis

> For the implementation of the Bachelor's Thesis, the engineering pipeline should split its processing into two concurrent paths.

> For reasoning and linguistic explanations, the quantized local VLM (Qwen2.5-VL 7B Q4_K_M) should be hosted via an optimized engine like Ollama or llama.cpp. This model is queried via standard forward inference to produce text-based forensic justifications and identify semantic discrepancies.

> For low-level visual attribution, a dual pipeline is recommended:

  +------------------+     Original Image     +----------------------------------+
  |   Input Image    | ---------------------> | Autoencoder Reconstruction Path  |
  +------------------+                        +----------------------------------+
           |                                                   |
           |                                                   v
           |                                  +----------------------------------+
           |                                  |  Reconstruction Error Heatmap    |
           |                                  +----------------------------------+
           |                                                   |
           v                                                   v
  +------------------+                        +----------------------------------+
  | Local Quantized  |                        |  VLM Visual Prompts (Guided by   |
  |  VLM (Ollama)    | <--------------------- |    Reconstruction Error Map)     |
  +------------------+                        +----------------------------------+
           |
           +---------------------------------> Generated Linguistic Explanation

1. To generate Saliency and Grad-CAM maps without encountering non-differentiable quantized layers, implement a lightweight convolutional surrogate model (such as a ResNet50 trained on synthetic imagery datasets like CiFAKE) in a PyTorch backend. This surrogate model processes the input image in parallel to generate gradient-dependent heatmaps.
2. To generate Visual Sobol and Occlusion maps directly from the quantized VLM without relying on surrogate networks, implement a perturbation loop. This loop applies sliding-window occlusions and Quasi-Monte Carlo masks to the input image, passes the perturbed variants through the quantized model's forward pass, and computes global sensitivity indices based on the variations in the output logits.
3. Utilize autoencoder-based reconstruction error maps to generate baseline spatial anomaly heatmaps. Feeding these reconstruction error maps alongside the raw image as a visual prompt to the quantized Qwen2.5-VL model provides explicit spatial guidance, helping the model "see" and explain localized artifacts with high precision.