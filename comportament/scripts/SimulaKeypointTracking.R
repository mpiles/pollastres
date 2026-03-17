###############################################################
# SIMULATE RAW KEYPOINT TRACKING DATA FOR MULTIPLE CHICKENS
# -------------------------------------------------------------
# This script mimics the raw output of a pose-estimation /
# keypoint tracking algorithm (e.g. DeepLabCut, SLEAP) applied
# to a group of 50 chickens recorded over one hour at 10 fps.
#
# For each individual the script produces:
#   - A ground-truth behavioural label per frame, generated
#     via a first-order Markov chain with minimum block
#     durations to ensure biologically realistic bout lengths.
#   - Raw (x, y) coordinates for three keypoints:
#       tail  – base of the tail
#       neck  – body centre / neck junction
#       head  – tip of the head
#     The neck follows a behaviour-dependent random walk
#     (faster during locomotion, near-stationary during rest).
#     Head and tail positions are derived from the neck with
#     behaviour-specific offsets (head drops for pecking,
#     pulls back for preening).
#   - Independent Gaussian noise added to every keypoint to
#     simulate the detection uncertainty of a real tracker.
#
# Crucially, NO smoothing and NO feature engineering are
# applied here.  The output is the equivalent of the raw CSV
# that a tracking algorithm would write to disk.
#
# Output file written to the working directory:
#   chicken_tracking_raw.csv
#     Columns: individual_id, frame, state,
#              tail_x, tail_y, neck_x, neck_y, head_x, head_y
#     Rows:    n_individuals × n_frames  (default: 50 × 36,000)
#
# The companion script ClassifBehaviour.R reads this file and
# performs all subsequent analysis: temporal smoothing, feature
# engineering, normalisation, behaviour classification, time-
# budget quantification, and visualisation.
###############################################################

library(dplyr)  # data manipulation (mutate, bind_rows, %>%)

set.seed(123)   # fix the random seed for full reproducibility

# ==============================================================
# PARAMETERS
# ==============================================================

fps              <- 10            # frames per second of the recorded video
video_seconds    <- 3600          # total recording duration in seconds (1 hour)
n_frames         <- fps * video_seconds  # total frames per individual (36,000)
n_individuals    <- 50            # number of chickens to simulate

# Per-frame displacement standard deviation (coordinate units) for each
# behaviour.  Controls how fast the body moves in the random walk.
# Larger values → chicken covers more ground per frame.
step_sd <- c(
  inactive   = 0.001,   # nearly stationary
  locomotion = 0.015,   # active walking across the pen
  foraging   = 0.005,   # slow purposeful wandering
  pecking    = 0.002,   # stays in place while pecking at the ground
  preening   = 0.001    # stationary while grooming
)

# Standard deviation of the independent Gaussian noise added to each
# keypoint coordinate to simulate keypoint-detector uncertainty.
# A typical pose estimator introduces a few pixels of jitter per keypoint.
tracking_noise_sd <- 0.008   # coordinate units (≈ several pixels in a real video)

# Anatomical constants (coordinate units, i.e. fractions of the scene width)
body_length    <- 0.30   # approximate tail-to-neck distance
head_offset_x  <- 0.15   # default forward extension of the head beyond the neck
head_offset_y  <- 0.02   # default upward tilt of the head above the neck line

# ==============================================================
# 1. BEHAVIOURAL STATES
# ==============================================================

states <- c("inactive", "locomotion", "foraging", "pecking", "preening")

# Minimum number of consecutive frames (bout duration) for each behaviour.
# Prevents unrealistically brief bouts and ensures temporal structure.
min_duration <- c(
  inactive   = 30,   # ≥ 3 s at 10 fps
  locomotion = 20,   # ≥ 2 s
  foraging   = 25,   # ≥ 2.5 s
  pecking    =  5,   # ≥ 0.5 s
  preening   = 15    # ≥ 1.5 s
)

# ==============================================================
# 2. MARKOV TRANSITION MATRIX
# ==============================================================

# Rows = current state, Columns = next state.
# Every row sums to 1.0 and encodes the probability of moving
# from the current behaviour (row) to each possible next behaviour (column).
transition_matrix <- matrix(
  c(
    0.60, 0.20, 0.10, 0.05, 0.05,   # from inactive
    0.20, 0.20, 0.40, 0.10, 0.10,   # from locomotion
    0.30, 0.10, 0.40, 0.10, 0.10,   # from foraging
    0.20, 0.20, 0.20, 0.30, 0.10,   # from pecking
    0.30, 0.10, 0.20, 0.10, 0.30    # from preening
  ),
  nrow = 5, byrow = TRUE,
  dimnames = list(states, states)
)

# Interpretation:
#   inactive stays inactive 60 % of the time, but may switch to locomotion
#     (20 %), foraging (10 %), pecking (5 %), or preening (5 %).
#   locomotion tends to transition into foraging (40 %) rather than
#     continuing to walk (20 %).
#   etc.

# ==============================================================
# 3. HELPER FUNCTION: simulate_state_sequence
# ==============================================================

# Generates a vector of n_frames behaviour labels for one individual
# using a first-order Markov chain with minimum block durations.
#
# Arguments:
#   n_frames          – total number of frames to generate
#   states            – character vector of behaviour names
#   min_duration      – named integer vector: minimum frames per behaviour
#   transition_matrix – 5×5 row-stochastic transition probability matrix
#
# Returns: character vector of length n_frames

simulate_state_sequence <- function(n_frames, states,
                                    min_duration, transition_matrix) {
  seq_out    <- character(0)        # accumulates the label sequence
  curr_state <- sample(states, 1)   # random initial behaviour

  while (length(seq_out) < n_frames) {

    # Minimum bout duration for the current behaviour
    dur_min <- min_duration[curr_state]

    # Actual bout length: minimum + exponentially distributed extra frames.
    # The exponential distribution gives realistic right-skewed bout lengths
    # (most bouts are short; a few are much longer than the minimum).
    bout_len <- round(dur_min + rexp(1, rate = 1 / (dur_min * 2)))

    # Append the current label for every frame of this bout
    seq_out <- c(seq_out, rep(curr_state, bout_len))

    # Break early if we have reached the required length
    if (length(seq_out) >= n_frames) break

    # Sample the next behaviour from the Markov transition probabilities
    # of the current state (Markov property: only the current state matters)
    row_idx    <- which(states == curr_state)
    curr_state <- sample(states, 1, prob = transition_matrix[row_idx, ])
  }

  seq_out[seq_len(n_frames)]   # trim to exact length
}

# ==============================================================
# 4. HELPER FUNCTION: simulate_individual
# ==============================================================

# Generates raw keypoint tracking data for one chicken across all frames.
# The output mimics the per-frame CSV that a pose estimator would export:
# only positions and the ground-truth label are included; no smoothing
# or derived features of any kind are computed here.
#
# Arguments:
#   ind_id  – integer: unique identifier for this individual
#   (all other arguments are the global parameters defined above)
#
# Returns: data.frame with n_frames rows and columns:
#   individual_id, frame, state, tail_x, tail_y,
#   neck_x, neck_y, head_x, head_y

simulate_individual <- function(ind_id, n_frames, states,
                                min_duration, transition_matrix,
                                step_sd, body_length,
                                head_offset_x, head_offset_y,
                                tracking_noise_sd) {

  # --- 4a. Ground-truth behavioural sequence --------------------------------
  # One label per frame, generated via the Markov chain helper above.
  state_seq <- simulate_state_sequence(n_frames, states,
                                       min_duration, transition_matrix)

  # --- 4b. Neck (body centre) trajectory ------------------------------------
  # The neck follows a behaviour-dependent random walk:
  #   dx[t], dy[t] ~ Normal(0, step_sd[state_seq[t]])
  # step_sd[state_seq] maps the behaviour label at each frame to the
  # corresponding displacement SD, producing larger steps during locomotion
  # and near-zero steps during inactive or preening bouts.
  # cumsum() converts frame-to-frame displacements into absolute positions.
  # No smoothing is applied: the trajectory is deliberately noisy, just as
  # a real tracker output would be.
  neck_x <- cumsum(rnorm(n_frames, mean = 0, sd = step_sd[state_seq]))
  neck_y <- cumsum(rnorm(n_frames, mean = 0, sd = step_sd[state_seq]))

  # --- 4c. Tail position ----------------------------------------------------
  # The tail is a fixed distance (body_length) behind the neck in the
  # positive-X direction, with small independent noise on both axes.
  # This noise represents natural tail sway as well as tracker uncertainty.
  tail_x <- neck_x - body_length + rnorm(n_frames, 0, 0.01)
  tail_y <- neck_y               + rnorm(n_frames, 0, 0.01)

  # --- 4d. Head position ----------------------------------------------------
  # Default head posture: slightly in front of and above the neck.
  head_x <- neck_x + head_offset_x + rnorm(n_frames, 0, 0.015)
  head_y <- neck_y + head_offset_y + rnorm(n_frames, 0, 0.015)

  # Behaviour-specific head offsets simulate characteristic postures:
  #   Pecking  → head drops below the neck level (negative Y offset)
  #   Preening → head pulls backward toward the body (negative X offset)
  peck_idx  <- which(state_seq == "pecking")
  preen_idx <- which(state_seq == "preening")

  if (length(peck_idx) > 0)
    head_y[peck_idx]  <- neck_y[peck_idx]  - abs(rnorm(length(peck_idx),  0.05, 0.01))
  if (length(preen_idx) > 0)
    head_x[preen_idx] <- neck_x[preen_idx] - abs(rnorm(length(preen_idx), 0.10, 0.01))

  # --- 4e. Keypoint detection noise -----------------------------------------
  # Add independent Gaussian noise to every keypoint on every frame.
  # This simulates the pixel-level uncertainty of a real keypoint detector:
  # even if the chicken is perfectly still, the tracker will report slightly
  # different coordinates from frame to frame.
  neck_x <- neck_x + rnorm(n_frames, 0, tracking_noise_sd)
  neck_y <- neck_y + rnorm(n_frames, 0, tracking_noise_sd)
  tail_x <- tail_x + rnorm(n_frames, 0, tracking_noise_sd)
  tail_y <- tail_y + rnorm(n_frames, 0, tracking_noise_sd)
  head_x <- head_x + rnorm(n_frames, 0, tracking_noise_sd)
  head_y <- head_y + rnorm(n_frames, 0, tracking_noise_sd)

  # --- 4f. Assemble output --------------------------------------------------
  # Return a tidy data frame: one row per frame, columns matching the
  # format a real tracker would export to CSV.
  data.frame(
    individual_id = ind_id,           # unique chicken identifier
    frame         = seq_len(n_frames), # frame index within the recording
    state         = state_seq,         # ground-truth behavioural label
    tail_x, tail_y,                    # tail keypoint coordinates
    neck_x, neck_y,                    # neck keypoint coordinates
    head_x, head_y                     # head keypoint coordinates
  )
}

# ==============================================================
# 5. SIMULATE ALL INDIVIDUALS
# ==============================================================

message("Simulating ", n_individuals, " individuals ",
        "(", n_frames, " frames each) ...")

# lapply iterates over individual IDs 1 … n_individuals, calling
# simulate_individual() for each one.  do.call(rbind, ...) stacks
# the resulting data frames into a single long-format data frame.
# Each individual's random seed is derived from the global set.seed(123),
# so results are fully reproducible.
tracking_data <- do.call(rbind, lapply(seq_len(n_individuals), function(i) {
  if (i %% 10 == 0)
    message("  Individual ", i, " / ", n_individuals, " done")
  simulate_individual(
    ind_id            = i,
    n_frames          = n_frames,
    states            = states,
    min_duration      = min_duration,
    transition_matrix = transition_matrix,
    step_sd           = step_sd,
    body_length       = body_length,
    head_offset_x     = head_offset_x,
    head_offset_y     = head_offset_y,
    tracking_noise_sd = tracking_noise_sd
  )
}))

message("Simulation complete.  Total rows: ", nrow(tracking_data))

# ==============================================================
# 6. SAVE OUTPUT
# ==============================================================

# Write the raw multi-animal tracking data to CSV.
# This file is the only output of SimulaKeypointTracking.R.
# It contains purely positional data (+ ground-truth labels) with
# no smoothing, no derived features, and no normalisation.
write.csv(tracking_data, "chicken_tracking_raw.csv", row.names = FALSE)

message("Output saved to: chicken_tracking_raw.csv")
message("  Individuals : ", n_individuals)
message("  Frames each : ", n_frames,
        "  (", round(video_seconds / 3600, 2), " hour(s) at ", fps, " fps)")
message("  Total rows  : ", nrow(tracking_data))
message("  Columns     : ",
        paste(names(tracking_data), collapse = ", "))
