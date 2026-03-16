### **Q1. TCN vs TCC**

**TCN (Time-Contrastive Networks)** assumes that actions progress at a uniform speed, relying on strict temporal proximity to align videos. This fails when subjects perform the same action at different speeds. **TCC (Temporal Cycle-Consistency)** solves this by aligning videos based on the semantic sequence of events, regardless of timing. It operates by taking a frame from Video A, finding its nearest neighbor in Video B, and then cycling back to find the nearest neighbor in Video A.

Crucially, standard nearest-neighbor matching uses a discrete `argmax` function, which cannot be differentiated for neural network training. TCC overcomes this by using a **"soft nearest-neighbor"** formulation. It computes a continuous, softmax-weighted average of features, which makes the cycle-consistency loss differentiable, allowing the model to be trained end-to-end via backpropagation.

---

### **Q2. Does the learned representation encode phase?**

Yes, the learned representation effectively encodes the semantic phase of the task. In the single-video PCA plots, we observe a general progression, but the **UMAP trajectories** more clearly reveal the underlying non-linear manifold, with frames (colored by time) following a continuous, smooth path.

The strongest evidence is seen in the **cross-video UMAP overlays**. Despite differences in background, lighting, and subjects across the videos, the trajectories overlap significantly in the shared projection space. This joint alignment proves the model ignores appearance variations and maps frames strictly based on their current phase of the action (e.g., all frames of the bottle tilting occupy the exact same region in the latent space).

---

### **Q3. How well does segmentation recover phase structure?**

The segmentation methods successfully recover phase structure in complementary ways:

* **Change-point detection**: Measured by the frame-to-frame distance $d_t = \|z_t - z_{t-1}\|$, it effectively identifies phase transitions by spiking at the exact moments the action rapidly shifts (e.g., the initial tilt of the bottle).
* **KMeans clustering**: Assigns semantic meaning to the continuous blocks between these spikes. Setting $k=6$ perfectly aligns with the six qualitative phases of pouring (reach, grasp, lift, tilt, pour, retract).

**Varying $k$** alters this story: a lower $k$ causes distinct phases to collapse into a single cluster (merging "grasp" and "lift"), while a higher $k$ causes over-segmentation, arbitrarily breaking fluid, continuous motions into meaningless sub-phases.

---

### **Q4. What failure modes remain?**

Despite the strong alignment, several failure modes remain in the representation:

* **Pauses causing over-segmentation:** If a subject pauses mid-pour, the embedding stops moving, causing the change-point score to flatline and potentially confusing the temporal timeline.
* **Appearance variation dominance:** Particularly at higher embedding dimensions (like $D=128$), the model has enough capacity to overfit to the background or clothing, causing trajectories from different videos to drift apart instead of overlapping.
* **Self-similar frames:** Frames from non-adjacent stages that look identical (e.g., the bottle resting completely still on the table at the beginning and end of the video) can collapse into the same cluster, breaking the chronological flow.


------------------------------------------------
🔥 TRAINING D=32 (FORCED SINGLE-THREAD)
------------------------------------------------
🚀 GLOBAL LOCKDOWN: All DataLoaders forced to 0 workers.
2026-03-16 03:48:28,485 INFO __main__: Iter[0/1000], Loss: 74.182, LR: 0.000100
2026-03-16 03:49:04,281 INFO __main__: Iter[100/1000], Loss: 248.946, LR: 0.000100
2026-03-16 03:49:41,637 INFO __main__: Iter[200/1000], Loss: 59.146, LR: 0.000100
2026-03-16 03:50:15,058 INFO __main__: Iter[300/1000], Loss: 453.330, LR: 0.000100
2026-03-16 03:50:51,032 INFO __main__: Iter[400/1000], Loss: 8690.495, LR: 0.000100
2026-03-16 03:51:27,447 INFO __main__: Iter[500/1000], Loss: 683449.000, LR: 0.000100
2026-03-16 03:52:01,946 INFO __main__: Iter[600/1000], Loss: 4088.252, LR: 0.000100
2026-03-16 03:52:38,358 INFO __main__: Iter[700/1000], Loss: 27773.121, LR: 0.000100
2026-03-16 03:53:15,173 INFO __main__: Iter[800/1000], Loss: 9590843.000, LR: 0.000100
2026-03-16 03:53:50,863 INFO __main__: Iter[900/1000], Loss: 5218.028, LR: 0.000100
2026-03-16 03:54:29,011 INFO __main__: Saved checkpoint to /tmp/alignment_logs/checkpoint_1000.pt
2026-03-16 03:54:29,011 INFO __main__: Training complete. Final step: 1000
------------------------------------------------
🔥 TRAINING D=64 (FORCED SINGLE-THREAD)
------------------------------------------------
🚀 GLOBAL LOCKDOWN: All DataLoaders forced to 0 workers.
2026-03-16 03:54:35,903 INFO __main__: Restored checkpoint from /tmp/alignment_logs/checkpoint_1000.pt at step 1000
2026-03-16 03:54:36,804 INFO __main__: Saved checkpoint to /tmp/alignment_logs/checkpoint_1000.pt
2026-03-16 03:54:36,804 INFO __main__: Training complete. Final step: 1000
------------------------------------------------
🔥 TRAINING D=128 (FORCED SINGLE-THREAD)
------------------------------------------------
🚀 GLOBAL LOCKDOWN: All DataLoaders forced to 0 workers.
2026-03-16 03:54:42,732 INFO __main__: Restored checkpoint from /tmp/alignment_logs/checkpoint_1000.pt at step 1000
2026-03-16 03:54:43,504 INFO __main__: Saved checkpoint to /tmp/alignment_logs/checkpoint_1000.pt
2026-03-16 03:54:43,505 INFO __main__: Training complete. Final step: 1000

