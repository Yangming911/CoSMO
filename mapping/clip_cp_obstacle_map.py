"""
CLIP-based obstacle map with Conformal Prediction threshold.

In fixed-threshold mode, each CLIP-updated cell whose obstacle-label
similarity exceeds the threshold is marked as an obstacle and dilated.
In online ACP mode, oracle detections provide nonconformity scores used to
update the threshold toward the configured target coverage.

Unexplored cells (feature_norm ≈ 0) are ignored — no prior is applied here.
The planner must separately handle unexplored regions.
"""

from collections import Counter, deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class CLIPCPObstacleMap:
    def __init__(
        self,
        n_cells: int,
        cell_size: float,
        threshold: float = 0.0,
        use_oacp: bool = False,
        target_coverage: float = 0.9,
        window_size: int = 200,
        gamma: float = 0.05,
        initial_alpha: float = 0.5,
        use_margin_score: bool = False,
        max_seeds_per_label: int = 0,   # 0 = unlimited; >0 = keep only top-K seeds per label
        temporal_persistence: int = 0,  # 0 = disabled; N>0 = cell must be seed for N consecutive frames
        argmax_confidence: float = 0.0,  # minimum obs similarity for argmax mode (0 = disabled)
        use_softmax_score: bool = False,   # Table B ablation: use -log softmax as nonconformity score
        softmax_recompute: bool = False,   # Table B C2: recompute buffer scores on expand_label()
    ) -> None:
        self.n_cells = n_cells
        self.cell_size = cell_size
        self.threshold = threshold          # τ in similarity space
        self.use_oacp = use_oacp
        self.argmax_confidence = argmax_confidence
        self.target_coverage = target_coverage

        # Margin-based score mode: s = (max_bg_sim - sim(feat, label) + 1) / 2
        # Replaces raw 1-sim score; threshold lives near 0.5 (argmax boundary)
        self.use_margin_score = use_margin_score
        self.max_seeds_per_label = int(max_seeds_per_label)
        self.temporal_persistence = int(temporal_persistence)

        # ACI state
        self._initial_alpha = initial_alpha  # stored for reset()
        self._initial_threshold = threshold  # stored for reset()
        self._alpha_t = initial_alpha        # risk level (miscoverage rate)
        self._gamma = gamma                  # ACI step size
        self._alpha_target = 1.0 - target_coverage  # target miscoverage rate

        # Set by set_text_features() after CLIP model is ready
        self._text_features: Optional[torch.Tensor] = None  # [N_obs, F]
        self._labels: List[str] = []
        self._radii: List[int] = []
        self._kernels: List[Optional[np.ndarray]] = []

        # Optional background/non-obstacle reference labels for argmax mode
        self._bg_text_features: Optional[torch.Tensor] = None  # [N_bg, F]
        self._bg_labels: List[str] = []  # names for each bg label

        # Per-cell state: 0 = free, label_idx+1 = predicted obstacle
        self._seed_map = np.zeros((n_cells, n_cells), dtype=np.uint8)
        # Temporal persistence: remember raw seeds from previous frame
        self._prev_seed_map = np.zeros((n_cells, n_cells), dtype=np.uint8)
        # Nonconformity score at each seed cell (lower = more confident obstacle)
        self._seed_score_map = np.ones((n_cells, n_cells), dtype=np.float32)
        # For bg cells: winning bg label index+1 (0=unobserved or obstacle winner)
        self._bg_winner_map = np.zeros((n_cells, n_cells), dtype=np.uint16)
        self._obstacle_mask = np.zeros((n_cells, n_cells), dtype=bool)

        # Softmax score mode (Table B ablation: closed-set CP)
        self.use_softmax_score = use_softmax_score
        self.softmax_recompute = softmax_recompute
        self._nc_threshold: float = 1.0 - threshold  # nonconformity threshold C_t

        # OACP calibration buffer (nonconformity scores)
        self._scores: deque = deque(maxlen=window_size)
        # C2 recompute buffer: stores (feat, label) for recomputing softmax scores on expand
        self._feat_label_buffer: Optional[deque] = (
            deque(maxlen=window_size) if softmax_recompute else None
        )

        # Per-call calibration log for experiment analysis
        self._calib_step: int = 0
        self._calibration_log: List[dict] = []

        # Worst-case OACP stats: track |C_j| distribution and label combinations
        self._cj_stats = {
            "n_updates": 0,
            "seed_cells": 0,        # cumulative cells with |C_j| >= 1
            "multi_cells": 0,       # cumulative cells with |C_j| >= 2
            "flipped_cells": 0,     # cumulative cells where max-radius != argmax-sim
            "set_counter": Counter(),  # tuple(sorted label indices) -> count
            # Detection stats (GT obstacle cell → what happened on the map?)
            "det_total": 0,             # observed GT obstacle cells
            "det_miss": 0,              # seed_map==0: GCLIP did not detect obstacle
            "det_hit": 0,               # seed_map label == GT label
            "det_wrong_cp_ok": 0,       # wrong label but GT ∈ C_j (CP recovered)
            "det_wrong_cp_miss": 0,     # wrong label and GT ∉ C_j
        }
        self._cj_print_every = 50   # print summary every N update() calls

    # ------------------------------------------------------------------
    def set_text_features(
        self,
        text_features: torch.Tensor,
        labels: List[str],
        radii: List[int],
        bg_text_features: Optional[torch.Tensor] = None,
        bg_labels: Optional[List[str]] = None,
    ) -> None:
        """Called once at init with precomputed obstacle label embeddings.

        Args:
            bg_text_features: [N_bg, F] background/non-obstacle label embeddings.
                When provided, uses argmax mode: a cell is an obstacle only if
                its best obstacle-label sim beats all background-label sims.
                When None, falls back to threshold mode (sim >= self.threshold).
        """
        self._text_features = text_features   # [N_obs, F], already normalised
        self._labels = labels
        self._radii = radii
        self._kernels = [self._make_disk(r) for r in radii]
        self._bg_text_features = bg_text_features  # [N_bg, F] or None
        self._bg_labels = bg_labels if bg_labels is not None else []

    @staticmethod
    def _make_disk(radius: int) -> Optional[np.ndarray]:
        if radius < 0:
            return None
        if radius == 0:
            return np.ones((1, 1), dtype=np.uint8)
        d = radius * 2 + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

    @property
    def argmax_mode(self) -> bool:
        return self._bg_text_features is not None

    # ------------------------------------------------------------------
    def save_initial_label_state(self) -> None:
        """Snapshot current label config so it can be restored on episode reset (open-vocab)."""
        self._saved_text_features = self._text_features.clone()
        self._saved_labels = list(self._labels)
        self._saved_radii = list(self._radii)
        self._saved_bg_text_features = (
            self._bg_text_features.clone() if self._bg_text_features is not None else None
        )
        self._saved_bg_labels = list(self._bg_labels)

    def restore_initial_label_state(self) -> None:
        """Restore label config to the initial (pre-expansion) state."""
        if not hasattr(self, "_saved_text_features"):
            return
        self.set_text_features(
            self._saved_text_features.clone(),
            list(self._saved_labels),
            list(self._saved_radii),
            bg_text_features=(
                self._saved_bg_text_features.clone()
                if self._saved_bg_text_features is not None
                else None
            ),
            bg_labels=list(self._saved_bg_labels) if self._saved_bg_labels else None,
        )

    def expand_label(
        self, label: str, radius: int, text_feature: torch.Tensor
    ) -> None:
        """Dynamically add a new obstacle label (open-vocabulary discovery)."""
        if label in self._labels:
            return
        self._labels.append(label)
        self._radii.append(radius)
        self._kernels.append(self._make_disk(radius))
        feat = text_feature.to(self._text_features.device, self._text_features.dtype)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        self._text_features = torch.cat([self._text_features, feat], dim=0)
        if label in self._bg_labels:
            idx = self._bg_labels.index(label)
            self._bg_labels.pop(idx)
            if self._bg_text_features is not None and self._bg_text_features.shape[0] > idx:
                self._bg_text_features = torch.cat(
                    [self._bg_text_features[:idx], self._bg_text_features[idx + 1 :]],
                    dim=0,
                )
        # C2 (softmax_recompute): recompute all buffered scores with new V_t
        if self.use_softmax_score and self.softmax_recompute and self._feat_label_buffer:
            self._recompute_softmax_buffer()

    def _recompute_softmax_buffer(self) -> None:
        """C2: recompute all buffered softmax scores using current V_t."""
        new_scores: deque = deque(maxlen=self._scores.maxlen)
        text_feats = self._text_features
        for feat, label in self._feat_label_buffer:
            if label not in self._labels:
                continue
            j = self._labels.index(label)
            feat_n = F.normalize(feat.unsqueeze(0), dim=1)
            tf = text_feats.to(feat.device, feat_n.dtype)
            obs_sims = (feat_n @ tf.T).squeeze(0).detach().cpu().numpy()
            # Softmax over all categories (obstacle + background)
            if self._bg_text_features is not None and self._bg_text_features.shape[0] > 0:
                bg_tf = self._bg_text_features.to(feat.device, feat_n.dtype)
                bg_sims = (feat_n @ bg_tf.T).squeeze(0).detach().cpu().numpy()
                full_sims = np.concatenate([obs_sims, bg_sims])
            else:
                full_sims = obs_sims
            max_s = np.max(full_sims)
            logsumexp_val = max_s + np.log(np.sum(np.exp(full_sims - max_s)))
            s = logsumexp_val - float(obs_sims[j])
            new_scores.append(s)
        self._scores = new_scores
        if len(self._scores) >= 5:
            C_new = float(np.quantile(list(self._scores), 1.0 - self._alpha_t))
            self._nc_threshold = C_new
            self.threshold = float(np.clip(1.0 - C_new, 0.0, 1.0))
        n_labels = len(self._labels)
        print(
            f"[SOFTMAX-RECOMP] Recomputed {len(self._scores)} buffered scores "
            f"with {n_labels} labels, nc_thresh={self._nc_threshold:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._seed_map.fill(0)
        self._seed_score_map.fill(1.0)
        self._obstacle_mask.fill(False)
        self._bg_winner_map.fill(0)
        self._prev_seed_map.fill(0)
        self._scores.clear()
        if self._feat_label_buffer is not None:
            self._feat_label_buffer.clear()
        self._alpha_t = self._initial_alpha    # restore configured initial value
        self.threshold = self._initial_threshold
        self._nc_threshold = 1.0 - self._initial_threshold
        self._calib_step = 0
        self._calibration_log = []
        self._cj_stats = {
            "n_updates": 0, "seed_cells": 0, "multi_cells": 0,
            "flipped_cells": 0, "set_counter": Counter(),
            "det_total": 0, "det_miss": 0, "det_hit": 0,
            "det_wrong_cp_ok": 0, "det_wrong_cp_miss": 0,
        }

    def get_calibration_log(self) -> List[dict]:
        """Return a copy of the calibration log and clear it."""
        log = list(self._calibration_log)
        self._calibration_log = []
        self._calib_step = 0
        return log

    # ------------------------------------------------------------------
    def calibrate(
        self,
        clip_feat: torch.Tensor,
        yolo_label: str,
    ) -> None:
        """Add an OACP calibration point from a YOLO detection.

        Args:
            clip_feat: CLIP feature vector at the projected cell [F].
            yolo_label: the obstacle label YOLO detected at that cell.
        """
        if not self.use_oacp or self._text_features is None:
            return
        if yolo_label not in self._labels:
            return
        norm = clip_feat.norm()
        if norm < 1e-6:
            return  # unexplored cell — skip, no calibration signal
        j = self._labels.index(yolo_label)
        text_feat = self._text_features[j].to(clip_feat.device)
        sim = float(F.normalize(clip_feat.unsqueeze(0), dim=1)
                    .mm(text_feat.unsqueeze(0).T)   # text_feat already normalised
                    .squeeze())
        score = 1.0 - sim   # nonconformity score: high when CLIP missed the obstacle
        self._scores.append(score)
        if len(self._scores) >= 10:
            # τ = 1 - (target_coverage quantile of nonconformity scores)
            self.threshold = float(1.0 - np.quantile(list(self._scores), self.target_coverage))

    # ------------------------------------------------------------------
    def calibrate_aci(
        self,
        clip_feat: torch.Tensor,
        true_label: str,
        cell_xy: tuple = None,
        obs_tau: float = None,
        gamma_override: float = None,
    ) -> None:
        """ACI update using GT label (real-time, no distance binning).

        Called when the robot is within δ_obs of a cell whose GT label is known.
        Adjusts α_t via gradient step so that coverage tracks α_target over time,
        then updates τ from the calibration score buffer.

        Args:
            clip_feat: CLIP feature vector at the cell [F].
            true_label: ground-truth semantic label of that cell.
            cell_xy: (cx, cy) cell coordinates for detection-aware stats.
        """
        if not self.use_oacp or self._text_features is None:
            return
        if true_label not in self._labels:
            return
        feat_norm = clip_feat.norm()
        if feat_norm < 1e-6:
            return  # unobserved cell

        j = self._labels.index(true_label)
        text_feats = self._text_features.to(clip_feat.device)
        feat_n = F.normalize(clip_feat.unsqueeze(0), dim=1)
        if text_feats.dtype != feat_n.dtype:
            text_feats = text_feats.to(feat_n.dtype)
        all_sims_np = (feat_n @ text_feats.T).squeeze(0).detach().cpu().numpy()  # [N_obs]
        sim = float(all_sims_np[j])

        if self.use_softmax_score:
            # Closed-set CP: softmax over ALL known categories (obstacle + background)
            if self._bg_text_features is not None and self._bg_text_features.shape[0] > 0:
                bg_tf = self._bg_text_features.to(feat_n.device, feat_n.dtype)
                bg_sims = (feat_n @ bg_tf.T).squeeze(0).detach().cpu().numpy()
                full_sims = np.concatenate([all_sims_np, bg_sims])
            else:
                full_sims = all_sims_np
            max_s = np.max(full_sims)
            logsumexp_val = max_s + np.log(np.sum(np.exp(full_sims - max_s)))
            s = logsumexp_val - sim
            # Prediction set size: only obstacle labels, but softmax over all
            softmax_nc_obs = logsumexp_val - all_sims_np
            n_above_tau = int((softmax_nc_obs <= self._nc_threshold).sum())
            err = float(s > self._nc_threshold)
            # Store (feat, label) for C2 recompute
            if self._feat_label_buffer is not None:
                self._feat_label_buffer.append((clip_feat.clone(), true_label))
        else:
            n_above_tau = int((all_sims_np >= self.threshold).sum())
            # Nonconformity score: s = 1 - sim (cosine distance)
            s = 1.0 - sim
            tau_for_err = obs_tau if obs_tau is not None else self.threshold
            C_t = 1.0 - tau_for_err
            err = float(s > C_t)

        tau_before = self.threshold

        # ---- Detection stats: GT obstacle cell → what happened on the map? ----
        # C_j = {l : sim(feat, l) >= tau}
        conf_cal = all_sims_np >= tau_before
        gt_in_cj = bool(conf_cal[j])

        if cell_xy is not None:
            cx, cy = cell_xy
            seed_val = int(self._seed_map[cx, cy])
            self._cj_stats["det_total"] += 1
            if seed_val == 0:
                self._cj_stats["det_miss"] += 1
            else:
                pred_label = self._labels[seed_val - 1] if seed_val <= len(self._labels) else "?"
                if pred_label == true_label:
                    self._cj_stats["det_hit"] += 1
                elif gt_in_cj:
                    self._cj_stats["det_wrong_cp_ok"] += 1
                else:
                    self._cj_stats["det_wrong_cp_miss"] += 1
            dt = self._cj_stats["det_total"]
            if dt % 20 == 0:
                dm = self._cj_stats["det_miss"]
                dh = self._cj_stats["det_hit"]
                dw = self._cj_stats["det_wrong_cp_ok"]
                dx = self._cj_stats["det_wrong_cp_miss"]
                print(
                    f"[OACP-DET] gt_obs={dt} "
                    f"miss={dm} ({100.0*dm/dt:.1f}%) "
                    f"hit={dh} ({100.0*dh/dt:.1f}%) "
                    f"wrong+cp_ok={dw} ({100.0*dw/dt:.1f}%) "
                    f"wrong+cp_miss={dx} ({100.0*dx/dt:.1f}%) "
                    f"tau={tau_before:.4f}",
                    flush=True,
                )
        # ------------------------------------------------------------------

        # ACI gradient step (use adaptive gamma if provided)
        _g = gamma_override if gamma_override is not None else self._gamma
        self._alpha_t = float(np.clip(
            self._alpha_t + _g * (self._alpha_target - err),
            1e-4, 1.0 - 1e-4,
        ))

        # Update τ: nonconformity threshold = (1-α_t) quantile of stored scores
        self._scores.append(s)
        if len(self._scores) >= 5:
            C_new = float(np.quantile(list(self._scores), 1.0 - self._alpha_t))
            if self.use_softmax_score:
                self._nc_threshold = C_new
                self.threshold = float(np.clip(1.0 - C_new, 0.0, 1.0))
            else:
                self.threshold = float(np.clip(1.0 - C_new, 0.0, 1.0))
                self._nc_threshold = C_new

        # Log calibration call: tau_before pairs with err at this step
        self._calibration_log.append({
            "step": self._calib_step,
            "tau": tau_before,
            "tau_after": self.threshold,
            "err": err,
            "alpha": self._alpha_t,
            "s": s,
            "|C|": n_above_tau,
            "n_labels": len(self._labels),
        })
        self._calib_step += 1

    def calibrate_aci_batch(self, entries):
        """Aggregated ACI update: one α step per sim-step batch of cells.

        Each entry is (clip_feat, true_label, cell_xy, obs_tau).
        Computes per-cell err using obs_tau, averages into a single
        fractional err, then does one α gradient step + τ update.
        """
        if not self.use_oacp or self._text_features is None:
            return
        if self._text_features.shape[0] == 0:
            return

        errs = []
        scores = []
        tau_before = self.threshold
        text_feats = self._text_features

        for clip_feat, true_label, cell_xy, obs_tau in entries:
            if true_label not in self._labels:
                continue
            if clip_feat.norm() < 1e-6:
                continue
            j = self._labels.index(true_label)
            feat_n = F.normalize(clip_feat.unsqueeze(0), dim=1)
            tf = text_feats.to(feat_n.device, feat_n.dtype)
            sim = float((feat_n @ tf.T).squeeze(0)[j])
            s = 1.0 - sim
            scores.append(s)
            C_t = 1.0 - obs_tau
            errs.append(float(s > C_t))

        if not errs:
            return

        err_avg = sum(errs) / len(errs)

        self._alpha_t = float(np.clip(
            self._alpha_t + self._gamma * (self._alpha_target - err_avg),
            1e-4, 1.0 - 1e-4,
        ))

        for s in scores:
            self._scores.append(s)
        if len(self._scores) >= 5:
            C_new = float(np.quantile(list(self._scores), 1.0 - self._alpha_t))
            self.threshold = float(np.clip(1.0 - C_new, 0.0, 1.0))

        self._calibration_log.append({
            "step": self._calib_step,
            "tau": tau_before,
            "tau_after": self.threshold,
            "err": err_avg,
            "alpha": self._alpha_t,
            "s": sum(scores) / len(scores),
        })
        self._calib_step += 1

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        updated_mask: torch.Tensor,   # bool mask [X, Y] of cells with new CLIP features
        feature_map: torch.Tensor,    # [X, Y, F]
    ) -> None:
        """Recompute obstacle seeds for all CLIP-updated cells."""
        if self._text_features is None or self._text_features.shape[0] == 0:
            return
        if not updated_mask.any():
            return

        changed_np = updated_mask.detach().cpu().numpy().astype(bool)
        feats = feature_map[updated_mask, :]          # [M, F]
        feat_norms = feats.norm(dim=1)               # [M]

        text_feats = self._text_features.to(feats.device)
        if text_feats.dtype != feats.dtype:
            text_feats = text_feats.to(feats.dtype)

        feats_norm = F.normalize(feats, dim=1)        # [M, F]
        obs_sims = (feats_norm @ text_feats.T).detach().cpu().numpy()   # [M, N_obs]
        norms_np = feat_norms.detach().cpu().numpy()

        if self._bg_text_features is not None:
            bg_feats = self._bg_text_features.to(feats.device)
            if bg_feats.dtype != feats.dtype:
                bg_feats = bg_feats.to(feats.dtype)
            bg_sims = (feats_norm @ bg_feats.T).detach().cpu().numpy()  # [M, N_bg]
            best_bg_sim = bg_sims.max(axis=1)                            # [M]
        else:
            best_bg_sim = None

        xs, ys = np.nonzero(changed_np)
        new_seed = self._seed_map.copy()
        new_bg_winner = self._bg_winner_map.copy()

        # Vectorized cell classification (replaces per-cell Python loop)
        valid = norms_np >= 1e-6                               # [M] bool
        best_j = np.argmax(obs_sims, axis=1)                  # [M]
        best_sim = obs_sims[np.arange(len(obs_sims)), best_j] # [M]

        new_seed[xs[~valid], ys[~valid]] = 0

        new_score = self._seed_score_map.copy()

        if valid.any():
            vxs, vys = xs[valid], ys[valid]
            vbest_j   = best_j[valid]
            vbest_sim = best_sim[valid]

            if self.use_margin_score and best_bg_sim is not None:
                # Detection-first OACP:
                #   Step 1 — GCLIP argmax detection gate (obs_sim > bg_sim)
                #   Step 2 — CP confidence set C_j = {l : sim >= tau} for label uncertainty
                #   Step 3 — worst-case radius from C_j; fallback to argmax label if C_j empty
                v_obs_sims   = obs_sims[valid]                                # [M', N_obs]
                v_max_bg     = best_bg_sim[valid]                             # [M']
                argmax_sim_j = np.argmax(v_obs_sims, axis=1)                  # [M']
                argmax_best  = v_obs_sims[np.arange(len(v_obs_sims)), argmax_sim_j]
                is_det       = argmax_best > v_max_bg                         # [M'] detection gate

                # Confidence set (only for detected cells)
                if self.use_softmax_score:
                    # Softmax over ALL categories (obstacle + background)
                    if self._bg_text_features is not None and self._bg_text_features.shape[0] > 0:
                        full_sims = np.concatenate([v_obs_sims, bg_sims[valid]], axis=1)
                    else:
                        full_sims = v_obs_sims
                    max_row = np.max(full_sims, axis=1, keepdims=True)
                    lse = max_row + np.log(np.sum(np.exp(full_sims - max_row), axis=1, keepdims=True))
                    softmax_nc = lse - v_obs_sims  # [M', N_obs] — only obstacle labels
                    conf_mask = (softmax_nc <= self._nc_threshold) & is_det[:, None]
                else:
                    conf_mask = (v_obs_sims >= self.threshold) & is_det[:, None]
                radii_arr    = np.asarray(self._radii, dtype=np.int32)        # [N_obs]
                masked_radii = np.where(conf_mask, radii_arr[None, :], -1)    # [M', N_obs]
                chosen_j     = np.argmax(masked_radii, axis=1)                # [M']
                has_conf     = conf_mask.any(axis=1)                          # [M']

                # Fallback: detected but C_j empty → use argmax label
                fallback = is_det & ~has_conf
                chosen_j[fallback] = argmax_sim_j[fallback]

                is_obs   = is_det
                chosen_s = 1.0 - v_obs_sims[np.arange(len(v_obs_sims)), chosen_j]

                new_seed[vxs[is_obs],  vys[is_obs]]  = chosen_j[is_obs] + 1
                new_score[vxs[is_obs],  vys[is_obs]]  = chosen_s[is_obs]
                new_score[vxs[~is_obs], vys[~is_obs]] = 1.0
                new_seed[vxs[~is_obs], vys[~is_obs]] = 0
                new_bg_winner[vxs[is_obs],  vys[is_obs]]  = 0
                new_bg_winner[vxs[~is_obs], vys[~is_obs]] = (
                    np.argmax(bg_sims[valid][~is_obs], axis=1) + 1
                )

                # ---- Confidence set stats ----
                cj_sizes = conf_mask.sum(axis=1)                     # [M']
                seed_mask = is_det
                multi_mask = is_det & (cj_sizes >= 2)
                flipped = is_det & has_conf & (chosen_j != argmax_sim_j)
                self._cj_stats["seed_cells"]    += int(seed_mask.sum())
                self._cj_stats["multi_cells"]   += int(multi_mask.sum())
                self._cj_stats["flipped_cells"] += int(flipped.sum())
                if multi_mask.any():
                    conf_multi = conf_mask[multi_mask]               # [M_multi, N_obs]
                    for row in conf_multi:
                        key = tuple(int(i) for i in np.nonzero(row)[0])
                        self._cj_stats["set_counter"][key] += 1
                self._cj_stats["n_updates"] += 1
                if self._cj_stats["n_updates"] % self._cj_print_every == 0:
                    seed = self._cj_stats["seed_cells"]
                    mult = self._cj_stats["multi_cells"]
                    flip = self._cj_stats["flipped_cells"]
                    pct_m = 100.0 * mult / max(seed, 1)
                    pct_f = 100.0 * flip / max(seed, 1)
                    top5 = self._cj_stats["set_counter"].most_common(5)
                    top5_named = [
                        (
                            "+".join(self._labels[i] for i in key),
                            cnt,
                        )
                        for key, cnt in top5
                    ]
                    print(
                        f"[OACP-CP-STATS] upd={self._cj_stats['n_updates']} "
                        f"seeds={seed} |C|>=2={mult} ({pct_m:.1f}%) "
                        f"flipped={flip} ({pct_f:.1f}%) top5={top5_named}",
                        flush=True,
                    )
                # -----------------------------------------------------------
            elif best_bg_sim is not None:
                # Argmax mode: obstacle wins only if it beats all background labels
                # AND exceeds minimum confidence (argmax_confidence > 0)
                is_obs = vbest_sim > best_bg_sim[valid]
                if self.argmax_confidence > 0:
                    is_obs = is_obs & (vbest_sim >= self.argmax_confidence)
                new_seed[vxs[is_obs],  vys[is_obs]]  = vbest_j[is_obs] + 1
                new_seed[vxs[~is_obs], vys[~is_obs]] = 0
                new_bg_winner[vxs[is_obs],  vys[is_obs]]  = 0
                new_bg_winner[vxs[~is_obs], vys[~is_obs]] = (
                    np.argmax(bg_sims[valid][~is_obs], axis=1) + 1
                )
            else:
                # Threshold mode: obstacle wins if sim >= τ
                is_obs = vbest_sim >= self.threshold
                new_seed[vxs[is_obs],  vys[is_obs]]  = vbest_j[is_obs] + 1
                new_seed[vxs[~is_obs], vys[~is_obs]] = 0

        # Temporal persistence: only keep seeds that were also seeds last frame
        if self.temporal_persistence > 0:
            stable = (new_seed > 0) & (self._prev_seed_map > 0)
            new_seed[~stable] = 0
            new_score[~stable] = 1.0
        self._prev_seed_map = new_seed.copy()  # remember raw seeds before TopK for next frame

        if not np.array_equal(new_seed, self._seed_map):
            self._seed_map = new_seed
            self._seed_score_map = new_score
            self._recompute_obstacle_mask()
        self._bg_winner_map = new_bg_winner

    # ------------------------------------------------------------------
    def _recompute_obstacle_mask(self) -> None:
        combined = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        for j, kernel in enumerate(self._kernels):
            if kernel is None:
                continue
            seed = (self._seed_map == j + 1)
            if not seed.any():
                continue
            # TopK: keep only the max_seeds_per_label cells with lowest score (most confident)
            if self.max_seeds_per_label > 0:
                seed_xs, seed_ys = np.nonzero(seed)
                n_seeds = len(seed_xs)
                if n_seeds > self.max_seeds_per_label:
                    scores = self._seed_score_map[seed_xs, seed_ys]
                    topk_idx = np.argpartition(scores, self.max_seeds_per_label)[:self.max_seeds_per_label]
                    seed_pruned = np.zeros_like(seed)
                    seed_pruned[seed_xs[topk_idx], seed_ys[topk_idx]] = True
                    seed = seed_pruned
            combined |= cv2.dilate(seed.astype(np.uint8), kernel, iterations=1).astype(bool)
        self._obstacle_mask = combined

    # ------------------------------------------------------------------
    def get_obstacle_mask(self) -> np.ndarray:
        return self._obstacle_mask

    def apply_to_navigable_map(self, base: np.ndarray) -> np.ndarray:
        return base & ~self._obstacle_mask
