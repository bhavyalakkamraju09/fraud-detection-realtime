# Clinical RAG Voice Agent

<div align="center">

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-orange?style=for-the-badge)](https://huggingface.co/spaces/bhavyalakkamraju09/clinical-rag-voice-agent)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/bhavyalakkamraju09/clinical-rag-voice-agent)
[![CI](https://github.com/bhavyalakkamraju09/clinical-rag-voice-agent/actions/workflows/test.yml/badge.svg)](https://github.com/bhavyalakkamraju09/clinical-rag-voice-agent/actions/workflows/test.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**HIPAA-compliant multi-turn voice QA over 10,000 synthetic clinical notes.**  
Hybrid BM25+FAISS retrieval · LangGraph orchestration · Presidio PHI redaction · ElevenLabs TTS

</div>

---

## Overview

A production-grade clinical information retrieval system that answers questions about patient records using voice or text input. Built with a LangGraph multi-agent pipeline, the system combines sparse and dense retrieval with cross-encoder reranking, applies Microsoft Presidio PHI de-identification to every response, and synthesizes answers as speech via ElevenLabs TTS.

All patient data is synthetic — generated with Faker — and no real clinical records are used.

---

## Architecture

```
Audio Input (Whisper STT)
        │
        ▼
   Text Query
        │
        ▼
┌─────────────────────────────────────────────────┐
│              LangGraph Agent                     │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  BM25    │  │  FAISS   │  │  RRF Fusion   │  │
│  │ Sparse   │──│  Dense   │──│    (k=60)     │  │
│  └──────────┘  └──────────┘  └───────┬───────┘  │
│                                       │          │
│                              ┌────────▼────────┐ │
│                              │ Cross-Encoder   │ │
│                              │  Re-Ranker      │ │
│                              └────────┬────────┘ │
│                                       │          │
│                              ┌────────▼────────┐ │
│                              │  Llama 3.1 8B   │ │
│                              │ (Groq / Ollama) │ │
│                              └────────┬────────┘ │
│                                       │          │
│                              ┌────────▼────────┐ │
│                              │ Presidio PHI    │ │
│                              │  Redaction      │ │
│                              └────────┬────────┘ │
└───────────────────────────────────────┼──────────┘
                                        │
                               ┌────────▼────────┐
                               │  ElevenLabs TTS │
                               └────────┬────────┘
                                        │
                                  Audio Response
```

---

## Dataset

| Property | Value |
|----------|-------|
| Synthetic clinical notes | **10,000** |
| Indexed chunks | **28,821** |
| Unique diagnoses | **25** |
| Note types | **5** (follow-up, new patient, procedure, discharge, urgent) |
| Medications | **40** |
| Lab panel types | **6** |
| Imaging report types | **10** |

---

## Evaluation

| Metric | Score | Target |
|--------|-------|--------|
| LLM-as-judge Faithfulness | 0.60 | > 0.85 |
| Answer Relevancy | 0.66 | > 0.80 |
| Context Recall | 0.28 | > 0.75 |
| End-to-end latency (Groq) | **6.1s** ✅ | < 8s |
| Avg latency (warm) | **3.7s** ✅ | < 6s |

Evaluation uses a custom LLM-as-judge framework over 50 diverse clinical QA pairs spanning medications, vital signs, lab results, imaging, and treatment plans. Improvement roadmap: LoRA domain fine-tuning + query expansion (implemented in `src/retrieval/query_expansion.py`).

---

## HIPAA Compliance

| Feature | Implementation |
|---------|---------------|
| PHI De-identification | Microsoft Presidio — all 18 Safe Harbor identifiers |
| Audit Trail | SHA-256 hashed query/response pairs in SQLite |
| Local Inference | Ollama — no PHI leaves the device |
| Guardrail | Lexical grounding filter on every response |
| TTS Safety | PHI redacted before speech synthesis |

---

## Quick Start

### Prerequisites

- Mac Apple Silicon (M1/M2/M3) or Linux
- Python 3.11
- [Ollama](https://ollama.com) (for local inference)
- Free API keys: [Groq](https://console.groq.com) · [ElevenLabs](https://elevenlabs.io)

### Setup

```bash
git clone https://github.com/bhavyalakkamraju09/clinical-rag-voice-agent
cd clinical-rag-voice-agent

conda create -n clinical-rag python=3.11 && conda activate clinical-rag
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Local LLM

```bash
ollama pull llama3.1:8b
ollama serve
```

### Configuration

```bash
cp .env.example .env
# GROQ_API_KEY       → console.groq.com  (free, 14,400 req/day)
# ELEVENLABS_API_KEY → elevenlabs.io     (free, 10K chars/month)
```

### Build & Run

```bash
make data      # generate 10,000 synthetic clinical notes
make index     # build FAISS + BM25 indexes (~8 min)
make run-api   # terminal 1 → FastAPI on :8000
make run-ui    # terminal 2 → Streamlit on :8501
```

---

## Project Structure

```
clinical-rag-voice-agent/
├── src/
│   ├── ingestion/         chunker · embedder · indexer
│   ├── retrieval/         bm25 · faiss · rrf_fusion · reranker · query_expansion
│   ├── agent/             state · nodes · graph · memory
│   ├── llm/               ollama_client · groq_client · prompts
│   ├── voice/             stt (Whisper) · tts (ElevenLabs)
│   ├── compliance/        phi_redactor · audit_logger · mlflow_tracker
│   └── api/               FastAPI routes · streaming endpoint
├── app/
│   └── streamlit_app.py
├── data/
│   └── synthetic/         generate_synthetic_notes.py
├── evaluation/
│   ├── create_golden_set.py
│   ├── run_evaluation.py
│   └── benchmark_retrieval.py
├── fine_tuning/
│   ├── train_lora_colab.py
│   └── prepare_dataset.py
├── .github/workflows/     CI/CD pipeline
├── docker-compose.yml
└── Makefile
```

---

## Design Decisions

**RRF over weighted fusion** — Parameter-free combination of BM25 sparse scores and FAISS cosine similarities. Eliminates the need to tune a mixing coefficient α.

**LangGraph over vanilla LangChain** — Typed state machine with conditional edges makes the guardrail retry loop, multi-turn memory, and future human-in-the-loop extensions first-class citizens.

**Two-stage retrieval** — Bi-encoder FAISS handles broad candidate generation efficiently; cross-encoder re-ranker scores (query, passage) pairs jointly for precision. Pre-filtering to top-10 keeps O(N) cross-encoder cost manageable.

**Presidio over regex** — NER-based PHI detection handles partial matches and contextual identifiers that regex patterns miss, covering all 18 HIPAA Safe Harbor types.

**Groq for inference** — llama-3.1-8b-instant at 14,400 free requests/day reduces latency from ~60s (CPU Ollama) to 6s without compromising HIPAA safety for demo purposes. Ollama remains the default for fully air-gapped deployments.

---

## Voice Pipeline

```python
from src.agent.graph import get_graph

agent = get_graph()
result = agent.invoke({
    "audio_path": "patient_question.wav",
    "turn_history": [],
    "guardrail_passed": False,
    "guardrail_attempts": 0,
})

print(result["phi_clean_answer"])   # PHI-redacted text response
# result["audio_path"]              # path to ElevenLabs .mp3
```

---

## LoRA Fine-Tuning

Domain adaptation for clinical abbreviations (HTN, SOB, Hx) using Colab T4 (free):

```bash
make finetune-prep                  # prepare training data locally
# Upload fine_tuning/ to Colab T4
# Run train_lora_colab.py (~2 hrs, r=16 LoRA on q_proj + v_proj)
python fine_tuning/merge_adapter.py # merge weights for Ollama import
```

---

## Docker

```bash
make docker-up
# FastAPI  → localhost:8000
# Streamlit → localhost:8501
# MLflow   → localhost:5000
```

---

## Free Tier

| Service | Allowance |
|---------|-----------|
| Ollama | Unlimited (local) |
| Groq | 14,400 req/day |
| ElevenLabs | 10,000 chars/month |
| HuggingFace Spaces | Free CPU |
| Colab T4 | ~4 hrs/day |

---

<div align="center">

Built by **Bhavya Lakkamraju** · MS Computer Science, Lawrence Technological University · 2026

[GitHub](https://github.com/bhavyalakkamraju09) · [LinkedIn](https://linkedin.com/in/bhavya-varma) · [HuggingFace](https://huggingface.co/bhavyalakkamraju09)

</div>
