# **Claude Code Tasking: Implement Deterministic Inference for MLX**

## **Project Overview**

Implement batch-invariant operations for MLX to enable deterministic LLM inference on Apple Silicon, based on Thinking Machine Labs' research.

---

## **Reference Documentation**

### **Primary Research & Implementation References**

* **TML Blog Post:** https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/  
* **TML GitHub (batch\_invariant\_ops):** https://github.com/thinking-machines-lab/batch\_invariant\_ops  
* **SGLang Implementation:** https://lmsys.org/blog/2025-09-22-sglang-deterministic/  
* **SGLang GitHub:** https://github.com/sgl-project/sglang

### **MLX Framework Documentation**

* **MLX GitHub:** https://github.com/ml-explore/mlx  
* **MLX Documentation:** https://ml-explore.github.io/mlx/build/html/index.html  
* **MLX-LM GitHub:** https://github.com/ml-explore/mlx-lm  
* **MLX Examples (LLM Inference):** https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html

### **Technical Deep Dives**

* **SGLang Deterministic PR (FlashInfer):** https://github.com/sgl-project/sglang/pull/10645  
* **SGLang Deterministic PR (FA3):** https://github.com/sgl-project/sglang/pull/10651  
* **SGLang Deterministic PR (Triton):** https://github.com/sgl-project/sglang/pull/10694  
* **Batch Invariant Test Suite:** https://github.com/sgl-project/sglang/blob/f1d789231896da438749b395f7bf007a5b0819c0/python/sglang/test/test\_deterministic.py

---

## **Phase 1: Environment Setup**

### **Setup Tasks**

Install MLX and dependencies

 pip install mlx mlx-lm numpy

1. 

Create project structure

 mlx\_deterministic/  
├── \_\_init\_\_.py  
├── ops/  
│   ├── \_\_init\_\_.py  
│   ├── rms\_norm.py  
│   ├── matmul.py  
│   └── attention.py  
├── tests/  
│   ├── \_\_init\_\_.py  
│   ├── test\_rms\_norm.py  
│   ├── test\_matmul.py  
│   ├── test\_attention.py  
│   └── test\_integration.py  
├── benchmarks/  
│   ├── \_\_init\_\_.py  
│   └── benchmark\_determinism.py  
└── README.md

2. 

Clone TML reference implementation for study

 git clone https://github.com/thinking-machines-lab/batch\_invariant\_ops

3. 

### **Verification**

* Confirm MLX imports work  
* Verify access to reference implementations  
* Create basic test harness skeleton

---

## **Phase 2: Batch-Invariant RMSNorm**

### **Implementation Requirements**

**File:** `mlx_deterministic/ops/rms_norm.py`

Key requirements from TML research:

* Fixed reduction chunk size regardless of input batch size  
* Deterministic variance computation  
* Must produce identical outputs for different batch sizes of same data

**Reference Implementation:**

* Study TML's PyTorch implementation  
* Review MLX's existing `mlx.nn.RMSNorm` implementation  
* Understand fixed reduction tree structure from TML blog

### **Implementation Steps**

1. Create `BatchInvariantRMSNorm` class inheriting from `mlx.nn.Module`

2. Implement fixed-chunk variance computation:

   * Define fixed `CHUNK_SIZE` (e.g., 64\)  
   * Pad input to multiple of chunk size  
   * Compute chunk-wise means  
   * Reduce across chunks with fixed pattern  
   * Remove padding effects from output

Implement standard RMSNorm formula with batch-invariant variance:

 output \= x \* rsqrt(variance \+ eps) \* weight

3.   
4. Handle edge cases:

   * Variable sequence lengths  
   * Padding and unpadding  
   * Numerical stability

### **Test Suite**

**File:** `mlx_deterministic/tests/test_rms_norm.py`

Required tests:

def test\_batch\_invariance\_basic():  
    """  
    Test: Process batch\_size=1 vs batch\_size=100, slice first element.  
    Expected: Outputs are bitwise identical.  
    """  
    pass

def test\_batch\_invariance\_varied\_sizes():  
    """  
    Test: Multiple batch sizes (1, 2, 4, 8, 16, 32, 64, 128).  
    Expected: All produce identical results for same input data.  
    """  
    pass

def test\_numerical\_correctness():  
    """  
    Test: Output values are numerically reasonable.  
    Expected: Mean ≈ 0, variance ≈ 1 after normalization.  
    """  
    pass

def test\_vs\_standard\_rmsnorm():  
    """  
    Test: Compare against MLX standard RMSNorm for single item.  
    Expected: Results are close (within floating point tolerance).  
    """  
    pass

def test\_gradient\_computation():  
    """  
    Test: Verify gradients can be computed.  
    Expected: Backward pass works without errors.  
    """  
    pass

### **Acceptance Criteria**

* All tests pass  
* Batch invariance verified across sizes \[1, 2, 4, 8, 16, 32, 64, 128\]  
* Maximum difference between any batch sizes is exactly 0.0  
* Performance is within 2x of standard MLX RMSNorm

---

## **Phase 3: Batch-Invariant Matrix Multiplication**

### **Implementation Requirements**

**File:** `mlx_deterministic/ops/matmul.py`

Key requirements from TML research:

* Fixed tile size for reduction dimension  
* Same reduction tree regardless of batch size  
* Support both batched and non-batched inputs

**Reference Implementations:**

* TML's PyTorch matmul kernels  
* SGLang's batch-invariant matmul integration

### **Implementation Steps**

1. Create `batch_invariant_matmul` function

2. Implement fixed-tile multiplication:

   * Define fixed `TILE_SIZE` (e.g., 128\)  
   * Pad K dimension to multiple of tile size  
   * Split into fixed-size tiles  
   * Compute partial products per tile  
   * Sum with fixed reduction pattern  
3. Handle input shapes:

   * 2D: `[M, K] @ [K, N] -> [M, N]`  
   * 3D batched: `[B, M, K] @ [K, N] -> [B, M, N]`  
   * 3D batched both: `[B, M, K] @ [B, K, N] -> [B, M, N]`  
4. Optimize for MLX:

   * Leverage MLX lazy evaluation  
   * Use efficient reshape/transpose operations  
   * Minimize unnecessary copies

### **Test Suite**

**File:** `mlx_deterministic/tests/test_matmul.py`

Required tests:

def test\_matmul\_batch\_invariance\_2d():  
    """  
    Test: 2D matmul with different batch sizes.  
    Expected: Identical results.  
    """  
    pass

def test\_matmul\_batch\_invariance\_3d():  
    """  
    Test: 3D batched matmul with varying batch dimensions.  
    Expected: Identical results.  
    """  
    pass

def test\_matmul\_correctness():  
    """  
    Test: Compare against MLX standard matmul for accuracy.  
    Expected: Results within numerical tolerance.  
    """  
    pass

def test\_matmul\_various\_shapes():  
    """  
    Test: Multiple matrix shapes and sizes.  
    Expected: All produce deterministic results.  
    """  
    pass

def test\_matmul\_non\_divisible\_dims():  
    """  
    Test: K dimension not divisible by tile size.  
    Expected: Padding/unpadding works correctly.  
    """  
    pass

def test\_matmul\_performance():  
    """  
    Test: Benchmark against standard matmul.  
    Expected: Within acceptable performance overhead.  
    """  
    pass

### **Acceptance Criteria**

* All tests pass  
* Batch invariance verified for shapes: `[(1,512,512), (8,512,512), (64,512,512)]`  
* Correctness within 1e-4 tolerance vs standard matmul  
* Performance overhead \< 3x standard MLX matmul

---

## **Phase 4: Batch-Invariant Attention**

### **Implementation Requirements**

**File:** `mlx_deterministic/ops/attention.py`

Key requirements from TML/SGLang research:

* Batch-invariant softmax reduction  
* Deterministic attention score computation  
* Fixed reduction patterns for all operations  
* Cache handling for incremental generation

**Reference Implementations:**

* SGLang's batch-invariant attention backends (FlashInfer, FA3, Triton)  
* TML's attention implementation  
* Focus on core attention computation, not long-sequence chunking

### **Implementation Steps**

1. Create `BatchInvariantAttention` class

2. Implement batch-invariant softmax:

   * Fixed reduction pattern for max computation  
   * Fixed reduction for sum computation  
   * Ensure numerical stability  
   * Use fixed chunk sizes for reduction operations  
3. Implement batch-invariant attention computation:

   * Fixed reduction strategy for attention scores  
   * Deterministic computation regardless of batch size  
   * Same reduction tree structure for all batch sizes  
4. Handle KV cache:

   * Deterministic cache update logic  
   * Support incremental generation  
5. Support standard attention features:

   * Causal masking  
   * Multi-head attention  
   * Rotary positional embeddings (if needed)

### **Test Suite**

**File:** `mlx_deterministic/tests/test_attention.py`

Required tests:

def test\_attention\_batch\_invariance\_simple():  
    """  
    Test: Single vs batched attention computation.  
    Expected: Identical outputs.  
    """  
    pass

def test\_attention\_batch\_invariance\_varied\_seq\_lengths():  
    """  
    Test: Different sequence lengths in batch.  
    Expected: Deterministic results.  
    """  
    pass

def test\_attention\_with\_cache():  
    """  
    Test: Incremental generation with KV cache.  
    Expected: Deterministic cache updates.  
    """  
    pass

def test\_attention\_with\_mask():  
    """  
    Test: Causal and padding masks.  
    Expected: Correct masking behavior, deterministic.  
    """  
    pass

def test\_attention\_multihead():  
    """  
    Test: Multiple attention heads.  
    Expected: All heads produce deterministic results.  
    """  
    pass

def test\_attention\_vs\_standard():  
    """  
    Test: Compare against MLX standard attention.  
    Expected: Numerically close results.  
    """  
    pass

### **Acceptance Criteria**

* All tests pass  
* Batch invariance verified for sequence lengths \[512, 1024, 2048, 4096\]  
* KV cache produces identical results across runs  
* Correctness within 1e-3 tolerance vs standard attention  
* Performance overhead \< 2x standard MLX attention

---

## **Phase 5: Model Integration**

### **Implementation Requirements**

**File:** `mlx_deterministic/__init__.py`

Create integration layer to enable deterministic mode for MLX-LM models.

### **Implementation Steps**

1. Create `enable_deterministic_mode(model)` function:

   * Traverse model layers  
   * Replace RMSNorm instances with BatchInvariantRMSNorm  
   * Wrap Linear layers to use batch\_invariant\_matmul  
   * Replace attention modules with BatchInvariantAttention  
2. Create `DeterministicConfig` class:

   * `chunk_size`: RMSNorm chunk size  
   * `tile_size`: Matmul tile size  
   * `reduction_size`: Attention reduction chunk size  
   * Allow configuration per operation

Implement model wrapping:

 from mlx\_lm import load  
from mlx\_deterministic import enable\_deterministic\_mode

model, tokenizer \= load("mlx-community/Qwen3-8B-4bit")  
model \= enable\_deterministic\_mode(model)

3.   
4. Create deterministic generation wrapper:

   * Ensure fixed random seed handling  
   * Batch size management  
   * Cache alignment

### **Test Suite**

**File:** `mlx_deterministic/tests/test_integration.py`

Required tests:

def test\_full\_model\_determinism():  
    """  
    Test: Load Qwen3-8B, generate 50 times with temp=0.  
    Expected: All 50 outputs are identical.  
    """  
    pass

def test\_determinism\_across\_batch\_sizes():  
    """  
    Test: Generate with batch\_size=\[1,4,8\] for same prompts.  
    Expected: Identical results after slicing.  
    """  
    pass

def test\_determinism\_with\_varied\_prompt\_lengths():  
    """  
    Test: Prompts of different lengths processed together.  
    Expected: Deterministic outputs.  
    """  
    pass

def test\_determinism\_with\_kv\_cache():  
    """  
    Test: Multi-turn conversation using cache.  
    Expected: Reproducible across runs.  
    """  
    pass

def test\_non\_greedy\_determinism():  
    """  
    Test: Generation with temperature \> 0, fixed seed.  
    Expected: Reproducible randomness.  
    """  
    pass

### **Acceptance Criteria**

* Successfully loads and wraps Qwen3-8B model  
* 50 consecutive generations produce 1 unique output (temp=0)  
* Works with multiple model architectures (Qwen, Llama, Mistral)  
* Generation speed within 2x of standard MLX-LM

---

## **Phase 6: Comprehensive Validation**

### **Validation Requirements**

Implement comprehensive test suite matching SGLang's validation approach.

**File:** `mlx_deterministic/benchmarks/benchmark_determinism.py`

### **Validation Tests**

1. **Single Test:**

   * Run same prompt 50 times with varying batch sizes  
   * Count unique outputs  
   * Expected: 1 unique output  
2. **Mixed Test:**

   * Mix short prompts, long prompts in same batches  
   * Test scenarios:  
     * Prompt 1: Short (10 tokens)  
     * Prompt 2: Short (15 tokens)  
     * Prompt 3: Long (1000 tokens)  
   * Expected: 1 unique output per prompt across 50 runs  
3. **Prefix Test:**

   * Create prompts from same text with different prefix lengths  
   * Prefix lengths: \[1, 511, 2048, 4097\]  
   * Batch randomly  
   * Expected: 1 unique output per prefix across 50 runs  
4. **Performance Benchmark:**

   * Measure tokens/second for different configurations  
   * Input/output combinations: \[(1024,1024), (4096,4096), (8192,8192)\]  
   * Compare deterministic vs standard mode  
   * Report overhead percentage

### **Implementation Steps**

1. Create benchmark harness  
2. Implement all validation test scenarios  
3. Add performance profiling  
4. Generate comparison reports  
5. Create visualization of results

### **Acceptance Criteria**

* Single test: 1/50 unique outputs  
* Mixed test: 1/50 unique for each prompt type  
* Prefix test: 1/50 unique for each prefix length  
* Performance overhead documented and \< 50% on average  
* All results reproducible across multiple runs

---

## **Phase 7: Documentation**

### **Documentation Requirements**

1. **README.md:**

   * Installation instructions  
   * Quick start guide  
   * Usage examples  
   * Performance characteristics  
   * Known limitations  
2. **API Documentation:**

   * Docstrings for all public functions  
   * Type hints throughout  
   * Usage examples in docstrings  
3. **Implementation Notes:**

   * Design decisions  
   * Differences from TML approach  
   * MLX-specific optimizations  
   * Future improvement areas  
4. **Benchmark Results:**

   * Performance comparison tables  
   * Determinism validation results  
   * Hardware specifications used

### **Deliverables**

* Complete README with examples  
* API documentation  
* Benchmark report  
* Implementation design doc

---

## **Testing Commands**

### **Unit Tests**

python \-m pytest mlx\_deterministic/tests/test\_rms\_norm.py \-v  
python \-m pytest mlx\_deterministic/tests/test\_matmul.py \-v  
python \-m pytest mlx\_deterministic/tests/test\_attention.py \-v  
python \-m pytest mlx\_deterministic/tests/test\_integration.py \-v

### **Integration Tests**

python \-m pytest mlx\_deterministic/tests/ \-v

### **Benchmarks**

python mlx\_deterministic/benchmarks/benchmark\_determinism.py

### **Full Validation**

\# Run comprehensive determinism validation  
python mlx\_deterministic/benchmarks/benchmark\_determinism.py \--full-validation

\# Run performance benchmarks  
python mlx\_deterministic/benchmarks/benchmark\_determinism.py \--performance

\# Generate report  
python mlx\_deterministic/benchmarks/benchmark\_determinism.py \--report

---

## **Success Metrics**

### **Functional Requirements**

* \[ \] RMSNorm: 100% batch invariant across all test cases  
* \[ \] Matmul: 100% batch invariant across all test cases  
* \[ \] Attention: 100% batch invariant across all test cases  
* \[ \] Full model: Single test produces 1/50 unique outputs  
* \[ \] Full model: Mixed test produces 1/50 unique per type  
* \[ \] Full model: Prefix test produces 1/50 unique per prefix

### **Performance Requirements**

* \[ \] RMSNorm overhead \< 100%  
* \[ \] Matmul overhead \< 200%  
* \[ \] Attention overhead \< 100%  
* \[ \] End-to-end overhead \< 50% (average across workloads)

### **Code Quality Requirements**

* \[ \] All functions have docstrings  
* \[ \] All functions have type hints  
* \[ \] Test coverage \> 90%  
* \[ \] All tests pass  
* \[ \] Code follows MLX style guidelines

### **Documentation Requirements**

* \[ \] README with installation and examples  
* \[ \] API documentation complete  
* \[ \] Benchmark results documented  
* \[ \] Design decisions documented

---

## **Implementation Priority Order**

Execute phases in order:

1. Phase 1: Environment Setup  
2. Phase 2: RMSNorm (validates approach)  
3. Phase 3: Matmul (core performance component)  
4. Phase 4: Attention (most complex)  
5. Phase 5: Integration (makes it usable)  
6. Phase 6: Validation (proves it works)  
7. Phase 7: Documentation (makes it shareable)

After each phase, verify all acceptance criteria before proceeding.

