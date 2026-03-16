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


