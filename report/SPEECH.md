# Speech Script — Structure-Aware SpGEMM
### APPT 2026 · Student GPU Programming Challenge · online talk (all English)

*Target length: ~9–10 minutes. Pace yourself — pause at each "//" mark and at the end of every slide. It is better to speak a little slower than to rush.*

---

## Slide 1 — Title

Good afternoon, everyone. Thank you for having me. //

My talk is about **structure-aware SpGEMM on GPUs** — that is, general sparse matrix–matrix multiplication. //

For this challenge we compute two products over the SuiteSparse collection: the self-product, A times A, and the Gram product, A times A-transpose. Our contribution is a portable kernel, and — the main idea — a **zero-profiling dispatcher** that decides how to run each matrix. //

Let me start with the background.

---

## Slide 2 — Background

SpGEMM is the multiplication of two **sparse** matrices. It is a core building block in scientific computing, in graph analytics, and increasingly in AI systems. //

In this challenge there are two tasks. The first is the self-product, A times A. The second is the Gram product, A times A-transpose — this one is symmetric, because entry i-j is simply the inner product of row i and row j. //

Here is the key point about performance. On a GPU, the cost of SpGEMM is **not** the arithmetic. The cost is **irregularity**. We do not know the size of the output in advance; the row lengths vary enormously; and the memory access pattern depends on the data itself. //

And the challenge does not only reward raw speed — it rewards **robustness and generality** across the whole matrix suite. Please keep that word, *generality*, in mind — it is the heart of my talk.

---

## Slide 3 — The hard part

So, what is actually hard here? //

Real matrices are **structurally heterogeneous**. In one single suite you find banded operators from partial differential equations, power-law graphs, symmetric stiffness matrices, and unsymmetric circuit matrices. They could not be more different from one another. //

The consequence is this: a kernel that is tuned for one class **loses badly on another**. So the real difficulty is generality, not peak speed. //

And this is not just true for our own kernel. Even NVIDIA's own vendor library, cuSPARSE, is **not robust**. On certain structured stiffness matrices it falls into a pathological code path and becomes — in our measurements — up to **one hundred and ninety times slower** than our simple engine. //

So we reframed the question. The winning question is not "which single kernel is the fastest?" It is: **"which kernel should we use for which matrix?"**

---

## Slide 4 — Our engine

That reframing needs two pieces. The first piece is a solid, portable kernel of our own. //

Our engine assigns **one warp per output row**. Each row accumulates its products into an open-addressing hash table, using a robust atomic compare-and-swap insert. //

We size the hash table **per row**, from a cheap upper bound on that row's work. Small rows hash entirely in fast **shared memory**; large rows spill over into a **global-memory arena**. So we get shared-memory speed without ever overflowing on the big rows. //

One single templated engine handles **both tasks**: we just feed it A for the self-product, or the explicit transpose of A for the Gram product. We also built a symmetry-only variant that computes just the upper triangle. //

Two things I want to stress. First, the kernel is compiled strictly for **sm_80** — no Hopper-only features — so it runs identically on the A100. Second, it is **byte-exact against cuSPARSE on all two hundred and fifty-eight tasks**. Getting that correctness was real work — it exposed a subtle shared-memory overflow bug that we tracked down and fixed.

---

## Slide 5 — The dispatcher (key idea)

Now the second piece — and this is the core idea of the talk. //

Because no single implementation wins everywhere, we do not force one. At load time we compute a handful of **cheap structural descriptors** directly from the matrix — no timing, no trial runs at all. //

A tiny fixed decision rule then routes each matrix to the faster of the two: our own engine, or cuSPARSE. //

And here is the interesting part. The dominant feature in that decision is the **row-length imbalance** — formally, the coefficient of variation of the number of nonzeros per row. Its importance is zero-point-eight-one, far above everything else. In other words, the decision is driven by **exactly the irregularity that makes SpGEMM hard in the first place.** //

Conceptually, this is the same move as predicting an execution policy from a cheap static descriptor. It turns the problem "one kernel cannot win everywhere" into a concrete, **profiling-free** system.

---

## Slide 6 — Results

So, does it work? Here are the results over the whole benchmark — two hundred and fifty-eight tasks. //

Everything is measured as speedup over cuSPARSE, so higher is better. //

Always using cuSPARSE is our baseline, one-point-zero. Always using our own engine gives one-point-four times. But the **dispatcher reaches two-point-six-three times** over cuSPARSE. //

The last bar, the oracle, is the theoretical upper bound if we always picked perfectly in advance — three-point-four times. So our dispatcher captures **eighty-eight percent of that ideal gain** — and it does so with under one percent overhead on small matrices, and with **zero correctness mismatches** across all two hundred and fifty-eight tasks.

---

## Slide 7 — Robustness

A word on robustness, and on being honest. //

On the official small-matrix regime, our engine alone is already about **two-point-seven times faster** in aggregate, and it wins on roughly eighty percent of the matrices. //

But we do lose on a few very large, high-fill matrices — our simple hashing is not tuned for those. We are open about that. And it matters, because **that is exactly where the dispatcher sends the work to cuSPARSE instead.** //

That is the whole point. The system never depends on one kernel being universally best. It is **robust by construction** — it routes each matrix to whatever solves it fastest.

---

## Slide 8 — Conclusion

To conclude. //

Real-world SpGEMM is fundamentally a **generality** problem. //

Our answer has two parts: a correct, portable hash engine, and — the key contribution — a **profiling-free, structure-aware dispatcher**. Together they reach two-point-six-three times over cuSPARSE, driven by row-length imbalance. //

The principle is simple: **route each matrix to whatever solves it fastest.** //

Thank you very much. I am happy to take any questions.

---

## Handling Q&A — a few likely questions

- **"Why not use Tensor Cores?"** → SpGEMM is irregular sparse-times-sparse: the output structure is unknown, and the access pattern is scattered, so there is no dense tile to feed a Tensor Core. Like cuSPARSE and prior work such as nsparse and spECK, our accumulation runs on the CUDA cores. Tensor-core SpGEMM only helps when the matrix has dense block structure, which the general SuiteSparse suite does not.
- **"Is the dispatcher trained or learned?"** → No — it is a tiny fixed rule distilled from the structural descriptors. That keeps it zero-profiling and fully portable. A learned cost model is our natural next step.
- **"These are H100 numbers — will they hold on A100?"** → The kernel is sm_80-only, so it runs identically on A100. cuSPARSE currently gets its faster Hopper paths on the H100, so our relative speedups are, if anything, conservative — they should hold or improve on A100.
- **"How do you guarantee correctness?"** → Byte-exact structural nonzero count plus order-independent value checksums against both cuSPARSE and a SciPy reference, on all 258 tasks — zero mismatches.

---

# Pronunciation Guide (生词读音表)

*Format: word — IPA — simple respelling (stress in CAPS). Practice the ones marked ⚑ out loud a few times.*

### Core technical terms
| Word | IPA | Say it like |
|---|---|---|
| ⚑ SpGEMM | / ɛs piː dʒɛm / | "S-P-**gem**" (spell S-P, then "gem") |
| ⚑ sparse | / spɑːrs / | "sparss" |
| ⚑ SuiteSparse | / suːt spɑːrs / | "**SWEET**-sparss" |
| matrix / matrices | / ˈmeɪtrɪks / · / ˈmeɪtrɪsiːz / | "**MAY**-triks" / "**MAY**-tri-seez" |
| ⚑ Gram (product) | / ɡræm / | "gramm" |
| transpose | / trænzˈpoʊz / | "tranz-**POHZ**" |
| symmetric | / sɪˈmɛtrɪk / | "si-**MET**-rik" |
| ⚑ irregularity | / ɪˌrɛɡjəˈlærɪti / | "ih-reg-yoo-**LAIR**-i-tee" |
| ⚑ heterogeneous | / ˌhɛtərəˈdʒiːniəs / | "het-er-oh-**JEE**-nee-us" |
| ⚑ coefficient | / ˌkoʊɪˈfɪʃənt / | "koh-eh-**FISH**-ent" |
| dispatcher / dispatch | / dɪˈspætʃər / | "dis-**PATCH**-er" |
| ⚑ oracle | / ˈɔːrəkəl / | "**OR**-uh-kul" |
| aggregate (adj/n) | / ˈæɡrɪɡət / | "**AG**-ri-gut" |
| descriptor | / dɪˈskrɪptər / | "dis-**KRIP**-ter" |

### Kernel / hardware terms
| Word | IPA | Say it like |
|---|---|---|
| ⚑ cuSPARSE | / kjuː spɑːrs / | "cue-**sparss**" |
| ⚑ nsparse | / ɛn spɑːrs / | "N-sparss" |
| warp | / wɔːrp / | "worp" |
| ⚑ hash / hashing | / hæʃ / | "hash" |
| accumulate / accumulator | / əˈkjuːmjəleɪt / | "uh-**KYOO**-myoo-late" |
| arena | / əˈriːnə / | "uh-**REE**-nuh" |
| ⚑ atomic (compare-and-swap) | / əˈtɑːmɪk / | "uh-**TOM**-ik" |
| ⚑ nonzeros | / nɑnˈzɪroʊz / | "non-**ZEER**-ohz" |
| byte-exact | / baɪt ɪɡˈzækt / | "byte ig-**ZAKT**" |
| stiffness (matrix) | / ˈstɪfnəs / | "**STIFF**-ness" |
| ⚑ pathological | / ˌpæθəˈlɑːdʒɪkəl / | "path-uh-**LODJ**-i-kul" |
| vendor | / ˈvɛndər / | "**VEN**-der" |
| templated | / ˈtɛmpleɪtɪd / | "**TEM**-play-tid" |

### Easy to mis-stress (watch these)
| Word | Correct stress |
|---|---|
| **gen**erality → generality | jen-er-**AL**-i-tee |
| **rob**ust → robust | roh-**BUST** (stress 2nd) |
| an**al**ytics → analytics | an-uh-**LIT**-iks |
| par**tial** (differential) | "**PAR**-shul" |
| **NVIDIA** | "en-**VID**-ee-uh" |

### Numbers you will say (rehearse these exact phrasings)
- **2.63×** → "two point six three times"
- **190×** → "one hundred and ninety times"
- **258** → "two hundred and fifty-eight"
- **0.81** → "zero point eight one"
- **88%** → "eighty-eight percent"
- **sm_80** → "S-M eighty"
- **A100 / H100** → "A one hundred" / "H one hundred" (or "A-one-hundred")
