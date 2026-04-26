# CS6423 Retrieval Upgrade Checklist

This file tracks what has already been implemented for the CS6423 retrieval upgrade and what still remains.

## Completed

### Retrieval Stack

- [x] Structure-aware chunking exists in `src/preprocessing/chunking.py` and is wired through `src/config.py` and `src/index_builder.py`.
- [x] Hybrid ranking from `ranking.md` is implemented.
- [x] FAISS semantic retrieval contributes an explicit retrieval signal.
- [x] BM25 lexical retrieval contributes an explicit retrieval signal.
- [x] Textbook index keyword retrieval contributes an explicit retrieval signal.
- [x] `EnsembleRanker` fuses only active retrieval sources.
- [x] LightRAG-style graph retrieval has been added as a fourth retrieval source.
- [x] Indexing now builds graph artifacts alongside chunk, FAISS, BM25, and page-to-chunk artifacts.
- [x] Query-time retrieval now loads graph artifacts alongside the existing indexes.
- [x] Configuration now supports enabling, disabling, and weighting the `graph` retriever.
- [x] Tests now cover graph construction, persistence, retrieval, and hybrid ranking participation.
- [x] Ingestion now supports checkpointed resume for chunking and embedding batches.
- [x] CLI indexing logs now show chunking and embedding progress, including batch progress and remaining work.
- [x] CLI ingestion telemetry now estimates elapsed time, estimated total time, and remaining time after the first completed chunking or embedding batch.

- [x] FAISS semantic retrieval
- [x] BM25 lexical retrieval
- [x] Textbook index keyword retrieval
- [x] LightRAG-style graph retrieval over entity and relation neighborhoods

### Graph Retrieval Upgrade

This section covers the LightRAG-style graph layer that was added on top of the existing hybrid retrieval stack.

#### Existing Files Updated

- [x] `src/config.py`
  - Added graph-related config support.
  - Added `graph` support in retriever weighting and activation logic.
- [x] `src/index_builder.py`
  - Extended indexing to build and persist graph artifacts.
  - Added checkpointed resume for chunking and embedding batches.
  - Added clearer CLI progress logging for sections and embedding batches.
  - Added elapsed-time and ETA telemetry for chunking sections and embedding batches.
- [x] `src/retriever.py`
  - Extended artifact loading and retriever construction for graph retrieval.
- [x] `src/main.py`
  - Index mode now builds graph artifacts.
  - Chat mode now loads and uses the graph retriever when enabled.
- [x] `src/api_server.py`
  - Uses the same graph-aware retrieval pipeline as `src/main.py`.
- [x] `src/ranking/ranker.py`
  - Validation works cleanly with `graph` as an active retriever.
- [x] `tests/test_hybrid_retrieval.py`
  - Added hybrid retrieval coverage including `graph`.
- [x] `tests/test_api.py`
  - Updated config, artifact loading, and retriever expectations for `graph`.
- [x] `tests/test_end_to_end.py`
  - Updated end-to-end coverage so graph retrieval participates in the pipeline.
- [x] `tests/test_benchmarks.py`
  - Updated benchmark retrieval setup to use graph-aware retriever construction.

#### New Files Added

- [x] `src/graph/__init__.py`
- [x] `src/graph/extraction.py`
- [x] `src/graph/store.py`
- [x] `src/graph/retrieval.py`
- [x] `tests/test_graph_indexing.py`
- [x] `tests/test_graph_retrieval.py`
- [x] `tests/test_index_builder_checkpointing.py`

#### Graph Artifact Contract

- [x] Persist normalized entity nodes.
- [x] Persist relation edges.
- [x] Persist source chunk IDs for extracted facts.
- [x] Persist aliases or lexical forms for matching.
- [x] Store the graph in a human-reviewable JSON format.

#### Graph Configuration Support

- [x] Update `RAGConfig` so `graph` is a supported retriever name.
- [x] Give `graph` a nonzero default weight in `ranker_weights`.
- [x] Extend retriever enablement and active-weight resolution so `graph` behaves like the existing retrievers.
- [x] Add graph artifact path configuration support.

#### Graph Extraction

- [x] Implement deterministic entity extraction over chunks.
- [x] Implement deterministic relation extraction over chunks.
- [x] Tie extracted graph facts back to `chunk_id`.
- [x] Keep the first version simple and inspectable.

#### Graph Persistence During Indexing

- [x] Extend `build_index()` to create graph artifacts after chunk creation.
- [x] Save graph artifacts into the same artifact directory and prefix scheme as the other indexes.
- [x] Ensure persisted graph artifacts survive reloads and remain human-inspectable.

#### Graph Loading

- [x] Extend `load_artifacts()` to load graph artifacts with the rest of the retrieval state.
- [x] Require graph artifacts when graph retrieval is enabled.
- [x] Keep graph artifacts optional when graph retrieval is disabled.
- [x] Fail explicitly if graph retrieval is enabled but graph artifacts are missing.

#### Graph Retrieval

- [x] Add a `GraphRetriever` that follows the same interface as the other retrievers.
- [x] Extract or normalize query entities at query time.
- [x] Match query entities against graph nodes and aliases.
- [x] Traverse relation neighborhoods.
- [x] Accumulate chunk scores from direct and neighboring graph evidence.
- [x] Keep scoring deterministic and bounded for fusion.

#### Hybrid Pipeline Integration

- [x] Update retriever construction so `graph` can be enabled like the other retrieval sources.
- [x] Pass graph raw scores into `EnsembleRanker` from `src/main.py`.
- [x] Pass graph raw scores into `EnsembleRanker` from `src/api_server.py`.
- [x] Avoid graph-specific ranking logic outside the ensemble.

#### Ranking Compatibility

- [x] Confirm `src/ranking/ranker.py` accepts the new source name cleanly.
- [x] Keep score fusion within the existing ranker path rather than adding ad hoc graph-only logic.

#### Deterministic Test Coverage

- [x] Verify graph entities and relations are extracted and linked to the correct `chunk_id`s.
- [x] Verify persisted graph artifacts reload without losing chunk linkage.
- [x] Verify graph retrieval produces chunk score dictionaries.
- [x] Verify hybrid ranking accepts four-source raw score dictionaries.
- [x] Verify graph retrieval can contribute to final ranking outcomes.
- [x] Verify ingestion can resume from a chunking checkpoint after interruption.
- [x] Verify embedding batches can resume from the last completed checkpointed batch.

### Ingestion Reliability

This section covers the completed checkpointing, resume, and CLI telemetry work for indexing.

#### Checkpointing And Resume

- [x] Add checkpoint files for interrupted ingestion work.
- [x] Resume chunking from the last completed section instead of restarting from the beginning.
- [x] Resume embedding from the last completed batch instead of recomputing finished batches.
- [x] Remove checkpoint files after a successful build finishes.

#### CLI Progress And Telemetry

- [x] Show user-facing progress for section chunking and embedding batches during indexing.
- [x] Estimate elapsed time and approximate remaining time after the first completed chunking or embedding batch.

### Imitation Learning Tuning App

This section covers the completed Streamlit-based imitation-learning tuning app for retrieval-weight adjustment from human-selected chunks.

#### Scope And Data Contract

- [x] Define the exact user workflow for the app:
  1. enter a query
  2. run retrieval with current weights
  3. inspect the top 10 chunks
  4. mark one or more chunks as preferred
  5. submit feedback
  6. update the ranking model or weights
  7. show the revised weights and keep collecting examples
- [x] Define the minimum retrieval sources the app will tune:
  - `faiss`
  - `bm25`
  - `index_keywords`
  - `graph`
- [x] Define the feedback format for one training example:
  - query text
  - top 10 candidate chunk IDs
  - raw per-retriever scores for those chunks
  - selected positive chunk IDs
  - optional rejected chunk IDs
  - timestamp or session metadata
- [x] Decide on the first learning objective:
  - start with a simple weight-learning approach over retriever score features
  - do not introduce a heavy neural model in the first version
- [x] Define where tuning data and learned parameters will be stored on disk in a human-readable format.

Committed implementation details:
- workflow is fixed to a 7-step top-10 labeling loop
- the tuned retriever set is fixed to `faiss`, `bm25`, `index_keywords`, and `graph`
- one demonstration stores query text, top-10 chunk candidates, per-retriever raw scores, fused score, selected chunk IDs, optional rejected chunk IDs, and session/timestamp metadata
- the first model is fixed to a simple linear retriever-weight model with deterministic online pairwise preference updates
- storage is fixed under `{artifacts_dir}/{index_prefix}_tuning/`
  - `demonstrations.jsonl` for raw demonstrations
  - `learned_weights.json` for the current learned model state

#### Retrieval-Tuning Backend

- [x] Add a small backend module for the tuning workflow instead of putting all logic in Streamlit.
- [x] Expose a function that runs retrieval for a query and returns:
  - top 10 chunk IDs
  - chunk text
  - chunk metadata
  - per-retriever raw scores
  - current fused score
- [x] Reuse the existing artifact-loading and retriever-construction pipeline rather than duplicating retrieval logic.
- [x] Add a serialization format for saved demonstrations and learned weights.
- [x] Add a loader that can initialize the tuning session from the current config and previously saved learned weights.

Implemented backend components:
- `src/tuning/backend.py` provides:
  - `load_tuning_session(...)`
  - `run_tuning_query(...)`
  - `append_training_example(...)`
  - `load_training_examples(...)`
  - `save_tuning_state(...)`
  - `load_tuning_state(...)`
- retrieval for the tuning workflow reuses:
  - `load_artifacts(...)`
  - `build_retrievers(...)`
  - `EnsembleRanker`
- demonstrations are serialized as JSON Lines in `demonstrations.jsonl`
- learned model state is serialized as human-readable JSON in `learned_weights.json`

#### Learnable Ranking Model

- [x] Define the first model as a weighted linear combination over normalized retriever scores.
- [x] Treat each chunk as a feature vector:
  - FAISS score
  - BM25 score
  - index keyword score
  - graph score
- [x] Define the update rule from user feedback:
  - selected chunks should score above unselected chunks from the same top 10 set
  - begin with a simple online update rule or pairwise preference update
- [x] Keep the first optimizer deterministic and lightweight.
- [x] Ensure learned weights remain bounded, interpretable, and easy to inspect.
- [x] Normalize or reproject weights after each update so the model remains stable.
- [x] Support saving and reloading learned weights across app restarts.

Implemented model components:
- `src/tuning/model.py` provides:
  - normalized per-example retriever feature vectors
  - deterministic pairwise online weight updates
  - bounded nonnegative weight reprojection with sum-to-one normalization
- `src/tuning/backend.py` now provides:
  - `training_example_from_query_result(...)`
  - `apply_training_example(...)`
- the current learnable model is:
  - linear weighted fusion over `faiss`, `bm25`, `index_keywords`, and `graph`
  - updated from selected-vs-unselected chunk preferences within the current top-k set
  - persisted through the existing `learned_weights.json` state path

#### Streamlit UI

- [x] Add a standalone Streamlit app entrypoint, separate from the CLI and API server.
- [x] Add controls for:
  - entering a query
  - loading current retriever weights
  - resetting learned weights
  - running retrieval
  - submitting selected chunks as feedback
- [x] Render the current top 10 retrieved chunks in a way that is easy to inspect.
- [x] Show, for each chunk:
  - rank
  - chunk ID
  - chunk text preview
  - page numbers or section metadata when available
  - per-retriever raw scores
  - current fused score
- [x] Add multi-select or checkbox controls so the user can mark one or more best chunks.
- [x] After feedback submission, show:
  - updated weights
  - a short training summary
  - optionally the re-ranked top 10 results under the new weights

Implemented UI components:
- `src/tuning/streamlit_app.py` provides a standalone Streamlit app entrypoint
- the app supports:
  - loading a tuning session from config and index prefix
  - viewing current learned weights
  - resetting learned weights to baseline
  - running retrieval for a query
  - inspecting the current top-k chunks with metadata and per-retriever scores
  - selecting one or more preferred chunks
  - submitting feedback to update the linear tuning model
  - viewing previous and updated weights
  - viewing the re-ranked results immediately after training feedback

#### Persistence

- [x] Save every user feedback action as a separate training record.
- [x] Save the current learned weights after every successful update.
- [x] Make the saved data easy to inspect and edit manually.
- [x] Separate raw demonstration logs from the current learned model state.
- [x] Prevent accidental corruption by writing files atomically or through a safe temp-file swap.

Implemented persistence details:
- raw demonstrations remain separate from learned model state:
  - `demonstrations.jsonl`
  - `learned_weights.json`
- a human-readable `manifest.json` now records:
  - storage formats
  - file names
  - demonstration counts
  - last update timestamp
- persistence now uses safe temp-file swap semantics for:
  - learned model state writes
  - manifest writes
  - demonstration log updates

#### Integration With Existing Ranking Configuration

- [x] Add a clear boundary between:
  - the default static config weights in `RAGConfig`
  - the learned weights produced by the tuning workflow
- [x] Define how the app chooses which weights to start from:
  - current config defaults
  - last saved learned weights
  - explicit reset to baseline
- [x] Make sure learned weights can be exported back into a config-friendly form.
- [x] Keep the existing retrieval pipeline unchanged for normal CLI or API use until the learned weights are explicitly adopted.

Implemented integration details:
- baseline config weights and learned tuning weights are now tracked separately in the tuning session
- the tuning app can start from either:
  - `baseline_config`
  - `last_saved_learned`
- the Streamlit UI now exposes startup weight-source selection and baseline reset as separate actions
- learned weights can now be exported into a config-friendly YAML file:
  - `ranker_weights.export.yaml`
- normal CLI and API retrieval paths remain unchanged unless exported weights are explicitly adopted elsewhere

#### Tests For The Tuning Workflow

- [x] Add unit tests for the training-example serialization format.
- [x] Add unit tests for the online weight update rule.
- [x] Add unit tests that confirm positive selections move preferred chunks upward.
- [x] Add tests that ensure learned weights can be saved and reloaded without drift.
- [x] Add tests for the retrieval backend used by the Streamlit app so the UI is not the only place where logic is exercised.

Implemented test coverage:
- contract-level serialization tests for:
  - `TrainingExample`
  - `TuningModelState`
- backend retrieval tests for:
  - `run_tuning_query(...)`
  - returned candidate structure, raw scores, and metadata
- model-update tests for:
  - bounded normalized online updates
  - preferred chunks moving upward after feedback
- persistence tests for:
  - learned-weight save/load round trips
  - repeated save/load without drift
  - manifest and export file behavior

#### UX And Safety Checks

- [x] Prevent training updates when the user submits no selected chunks.
- [x] Prevent duplicate chunk selections from corrupting a training example.
- [x] Handle missing artifacts with a clear UI error message.
- [x] Handle missing learned-weight files by falling back cleanly to baseline config weights.
- [x] Add small status messages in the UI so the user can see:
  - current artifact set
  - current weights
  - number of demonstrations collected
  - whether the latest update was saved successfully

Implemented UX and safety details:
- feedback creation now rejects:
  - empty preferred-chunk selections
  - duplicate selected chunk IDs
  - duplicate rejected chunk IDs
- missing retrieval artifacts now surface a clearer load error that tells the user to index first or verify the selected prefix
- missing learned-weight files now fall back cleanly to baseline config weights with an explicit session status message
- the Streamlit UI now shows session status details for:
  - artifact directory
  - demonstration log path
  - learned weights path
  - manifest path
  - export path
  - whether saved learned weights were loaded
  - demonstration count
  - last successful save or retrieval action

#### Validation And Iteration

- [x] Verify that repeated feedback over a small set of queries meaningfully changes ranking behavior.
- [x] Compare baseline vs learned weights on a held-out set of prompts when available.
- [x] Confirm that the first version remains simple, inspectable, and easy to debug.
- [x] Only after the simple version works, consider whether a more sophisticated imitation-learning objective is justified.

Implemented validation details:
- `src/tuning/validation.py` provides:
  - baseline-vs-learned held-out query comparison
  - per-query ranking-difference summaries
  - aggregate overlap and top-1 change summaries
- the Streamlit app now includes a validation section for held-out prompts
- tuning persistence is now explicitly verified by test coverage that:
  - trains weights
  - saves them
  - reloads a fresh tuning session
  - confirms the learned weights persist
- the current validation path keeps the first version simple and inspectable:
  - no heavy learner is introduced
  - baseline-vs-learned comparisons are transparent
  - further sophistication remains a later follow-up rather than part of the first shipped version

### Local Model Runtime Upgrade

This section covers the completed llama.cpp installation and runtime model-selection upgrade for stronger local models on GPU-capable machines.

#### Existing Files Updated

- [x] `src/config.py`
  - Added baseline and GPU-specific model configuration fields.
  - Added runtime model profile selection support with `auto`, `baseline`, and `gpu`.
  - Added runtime resolution helpers that choose stronger models when a valid GPU backend is available and fall back cleanly when GPU model files are missing.
- [x] `src/main.py`
  - CLI startup now resolves effective embedding and generation models from the runtime hardware profile.
  - Index and chat modes now use the resolved runtime model selection rather than assuming a single static default.
  - CLI startup now prints the selected runtime model profile and hardware-detection reason.
- [x] `src/api_server.py`
  - API startup now uses the same runtime model-resolution path as the CLI.
  - API startup now logs the selected runtime model profile and fallback behavior.
- [x] `src/generator.py`
  - LLM loading now checks for GPU availability before attempting GPU offload.
  - Generation now falls back cleanly to CPU loading if accelerated model initialization fails.
- [x] `src/embedder.py`
  - Embedding model loading now checks for GPU availability before enabling GPU offload.
  - Embedding workers now fall back cleanly to CPU loading if accelerated initialization fails.
- [x] `scripts/setup_env.sh`
  - Linux setup now detects NVIDIA GPUs for CUDA and AMD GPUs for Vulkan.
  - CPU-only Linux installs now default to an OpenBLAS-backed llama-cpp-python build instead of assuming GPU support.
- [x] `scripts/build_llama.sh`
  - llama.cpp source builds now select CUDA on NVIDIA, Vulkan on AMD, and BLAS on CPU-only Linux hosts.
- [x] `environment.yml`
  - Added `huggingface-hub` so recommended GGUF model downloads can be scripted from the project environment.
- [x] `Makefile`
  - Added an `install-models` helper target for downloading the recommended GGUF model sets.
- [x] `config/config.yaml`
  - Added runtime model profile config entries and stronger GPU-model defaults.
- [x] `README.md`
  - Documented the baseline versus GPU model profiles.
  - Documented the AMD/Vulkan llama.cpp install path and the scripted model download flow.

#### New Files Added

- [x] `src/runtime_models.py`
- [x] `scripts/install_recommended_models.sh`
- [x] `tests/test_runtime_models.py`

#### Runtime Model Selection

- [x] Detect whether a valid accelerated backend is available at runtime.
- [x] Prefer a stronger embedding model when GPU acceleration is available.
- [x] Prefer a stronger generation model when GPU acceleration is available.
- [x] Fall back automatically to the baseline model set when no valid GPU backend is detected.
- [x] Fall back automatically to the baseline model set when GPU profile model files are missing.
- [x] Support explicit runtime profile overrides with `auto`, `baseline`, and `gpu`.
- [x] Keep runtime model selection consistent between CLI and API entrypoints.

#### llama.cpp Installation Support

- [x] Detect NVIDIA-backed Linux hosts and build/install llama.cpp with CUDA support.
- [x] Detect AMD-backed Linux hosts and build/install llama.cpp with Vulkan support.
- [x] Keep CPU-only Linux installs on a supported BLAS-backed llama.cpp path.
- [x] Add a scripted path for downloading the recommended baseline and GPU GGUF model files.

#### Deterministic Test Coverage

- [x] Verify forced runtime device overrides are honored during hardware detection.
- [x] Verify GPU-capable hosts select the stronger GPU model profile when the model files are available.
- [x] Verify runtime selection falls back to the baseline profile when GPU model files are missing.
- [x] Verify CLI generation-model overrides still apply on top of runtime model resolution.

### Assignment Boundary Checklist

- [x] The graph improves retrieval rather than replacing answer generation.
- [x] The deliverable remains a graph-enhanced hybrid retriever inside TokenSmith’s existing architecture.
- [x] Existing chunking and ranking improvements were preserved and extended rather than rewritten.

## Remaining Work

### Validation And Benchmarking

- [ ] Re-run benchmark-oriented validation after graph integration.
- [ ] Compare retrieval quality before and after graph integration.
- [ ] Measure Recall@k, NDCG, and final aggregate score in a way consistent with `CS6423.pdf`.
- [ ] Track indexing time, query latency, and artifact size to quantify graph overhead.

### Environment-Limited Follow-Up

- [ ] Run the full `pytest` suite in an environment where `pytest` is installed.
- [ ] Run config/runtime validation in an environment with all Python dependencies installed, including `langchain_text_splitters`.
