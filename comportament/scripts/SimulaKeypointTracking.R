###############################################################
# SIMULATE KEYPOINT TRACKING FOR CHICKEN BEHAVIOUR ANALYSIS
# -------------------------------------------------------------
# This script simulates realistic chicken movement data for
# three keypoints: neck (body centre), head, and tail.
# A one-hour video at 10 fps is modelled (36,000 frames).
#
# The simulation proceeds as follows:
#   1. A sequence of behavioural states is generated using a
#      first-order Markov chain with minimum block durations,
#      ensuring each behaviour lasts a realistic amount of time.
#   2. The positions of the three keypoints are simulated as
#      smooth random trajectories, with behaviour-specific
#      offsets added to the head (e.g. downward for pecking,
#      backward for preening).
#   3. A set of motion features is derived from the raw
#      positions: body/head vectors, velocities, accelerations,
#      and rolling-window statistics.
#
# Output files written to the working directory:
#   - chicken_simulated_data_full.csv : raw positions +
#     behaviours + all derived features (36,000 rows)
#
# The companion script ClassifBehaviour.R reads this CSV,
# builds a feature matrix, trains a Random Forest classifier,
# and quantifies the time budget per behaviour.
###############################################################

library(dplyr)  # data manipulation (mutate, select, %>%)
library(zoo)    # rolling-window functions (rollmean, rollapply)

set.seed(123)   # fix random seed so results are reproducible

# ---- Tuning constants -------------------------------------------------------
window_short <- 20   # short rolling-window size in frames (~2 seconds at 10 fps)
window_long  <- 50   # long  rolling-window size in frames (~5 seconds at 10 fps)
epsilon      <- 1e-6 # small value added to denominators to avoid division by zero
# -----------------------------------------------------------------------------

###############################################################
# 1. VIDEO PARAMETERS
###############################################################

fps <- 10                      # frames per second of the video
video_seconds <- 3600          # total video duration in seconds (1 hour)
n_frames <- fps * video_seconds # total number of frames (36,000)
time <- 1:n_frames              # integer frame index vector

# We simulate a one-hour video at 10 fps, which gives 36,000 frames.

###############################################################
# 2. BEHAVIOURAL STATES
###############################################################

states <- c("inactive", "locomotion", "foraging", "pecking", "preening")

# Minimum duration (in frames) for each behavioural block.
# This prevents unrealistically brief bouts of behaviour.
min_duration <- c(inactive = 30, locomotion = 20, foraging = 25,
                  pecking = 5, preening = 15)

# This ensures that each behaviour lasts at least the minimum number of frames,
# making the simulated sequence more biologically realistic.

###############################################################
# 3. MARKOV TRANSITION MATRIX
###############################################################

# Rows = current state, Columns = next state.
# Each row sums to 1 and represents the probability of transitioning
# from the current behaviour (row) to each possible next behaviour (column).
transition_matrix <- matrix(
  c(
    0.6, 0.2, 0.1, 0.05, 0.05,  # from inactive
    0.2, 0.2, 0.4, 0.1,  0.1,   # from locomotion
    0.3, 0.1, 0.4, 0.1,  0.1,   # from foraging
    0.2, 0.2, 0.2, 0.3,  0.1,   # from pecking
    0.3, 0.1, 0.2, 0.1,  0.3    # from preening
  ),
  nrow = 5, byrow = TRUE,
  dimnames = list(states, states)
)

# These probabilities encode realistic behaviour transitions.
# For example, inactive often stays inactive (0.6), but can start moving.

###############################################################
# 4. SIMULATE BEHAVIOURAL SEQUENCE WITH MINIMUM BLOCK DURATIONS
###############################################################

state_sequence <- c()
current_state <- sample(states, 1)  # start with a randomly chosen behaviour

while (length(state_sequence) < n_frames) {

  # Minimum duration (in frames) for the current behaviour
  dur_min <- min_duration[current_state]

  # Sample block length: min_duration plus an exponentially distributed extra
  # duration so that block lengths have a realistic right-skewed distribution
  block_duration <- round(dur_min + rexp(1, rate = 1 / (dur_min * 2)))

  # Append the current behaviour label for every frame in this block
  state_sequence <- c(state_sequence, rep(current_state, block_duration))

  # Stop building the sequence once we have enough frames
  if (length(state_sequence) >= n_frames) break

  # Sample the next behaviour according to the transition probabilities
  # of the current state (Markov property: next state depends only on current)
  row_idx    <- which(states == current_state)
  next_state <- sample(states, 1, prob = transition_matrix[row_idx, ])

  # Move to the next behaviour block
  current_state <- next_state
}

# Truncate the sequence to the exact number of frames required
state_sequence <- state_sequence[1:n_frames]

# At this point we have a realistic behavioural timeline: one label per frame.

###############################################################
# 5. SIMULATE NECK (BODY) TRAJECTORY – SMOOTH MOVEMENT
###############################################################

# The neck keypoint represents the body centre.
# We generate a smooth path using cumulative sums of small random steps
# (a random walk) and apply spline smoothing for realistic, continuous motion.

# Base random walk with small step sizes (mostly stationary)
dx <- rnorm(n_frames, 0, 0.01)
dy <- rnorm(n_frames, 0, 0.01)
neck_x <- cumsum(dx)   # cumulative sum gives absolute X position over time
neck_y <- cumsum(dy)   # cumulative sum gives absolute Y position over time

# For frames labelled as locomotion, replace positions with a larger random walk
# to simulate the chicken walking across the pen
locomotion_idx <- which(state_sequence == "locomotion")
neck_x[locomotion_idx] <- cumsum(rnorm(length(locomotion_idx), 0, 0.05))
neck_y[locomotion_idx] <- cumsum(rnorm(length(locomotion_idx), 0, 0.05))

# Apply spline smoothing (spar = 0.6) to produce continuously smooth trajectories
# smooth.spline fits a cubic smoothing spline and returns interpolated y values
neck_x <- smooth.spline(1:n_frames, neck_x, spar = 0.6)$y
neck_y <- smooth.spline(1:n_frames, neck_y, spar = 0.6)$y

###############################################################
# 6. SIMULATE TAIL POSITION
###############################################################

# The tail is positioned a fixed distance (body_length) behind the neck
# along the body axis, with a small amount of random noise to simulate
# natural lateral sway of the tail.

body_length <- 0.3  # approximate body length in normalized coordinate units
tail_x <- neck_x - body_length + rnorm(n_frames, 0, 0.005)  # behind neck in X
tail_y <- neck_y                + rnorm(n_frames, 0, 0.005)  # aligned in Y ± noise

###############################################################
# 7. SIMULATE HEAD POSITION
###############################################################

# The head is placed in front of (ahead of) the neck by default.
# Behaviour-specific offsets are added to capture characteristic movements,
# and the resulting trajectory is smoothed with splines.

head_x <- neck_x + 0.15 + rnorm(n_frames, 0, 0.01)  # head is 0.15 units ahead in X
head_y <- neck_y + 0.02 + rnorm(n_frames, 0, 0.01)  # slight upward offset in Y

# Pecking: head moves downward (negative Y offset)
peck_idx <- which(state_sequence == "pecking")
head_y[peck_idx] <- neck_y[peck_idx] - abs(rnorm(length(peck_idx), 0.05, 0.01))

# Preening: head moves backward toward the body (negative X offset)
preen_idx <- which(state_sequence == "preening")
head_x[preen_idx] <- neck_x[preen_idx] - abs(rnorm(length(preen_idx), 0.1, 0.01))

# Smooth head positions so that behaviour-specific offsets blend naturally
head_x <- smooth.spline(1:n_frames, head_x, spar = 0.6)$y
head_y <- smooth.spline(1:n_frames, head_y, spar = 0.6)$y

###############################################################
# 8. CREATE DATA FRAME
###############################################################

# Combine all simulated positions and the behavioural label into one data frame.
# Each row corresponds to one video frame.
data <- data.frame(
  frame  = time,           # frame index (1 to n_frames)
  state  = state_sequence, # behavioural label for this frame
  tail_x, tail_y,          # tail keypoint coordinates
  neck_x, neck_y,          # neck (body centre) keypoint coordinates
  head_x, head_y           # head keypoint coordinates
)

###############################################################
# 9. TEMPORAL SMOOTHING (ADDITIONAL)
###############################################################

# Apply a rolling mean (window = 5 frames) to further reduce positional jitter
# that remains after spline smoothing. fill = "extend" pads boundaries by
# repeating the nearest valid value so that no NAs are introduced.
smooth_roll <- function(x, k = 5) { rollmean(x, k, fill = "extend") }

data <- data %>%
  mutate(
    neck_x = smooth_roll(neck_x),  # smooth neck X
    neck_y = smooth_roll(neck_y),  # smooth neck Y
    head_x = smooth_roll(head_x),  # smooth head X
    head_y = smooth_roll(head_y)   # smooth head Y
  )

###############################################################
# 10. BODY AND HEAD VECTORS
###############################################################

# Compute 2-D direction vectors that describe body orientation and head position
# relative to the body. These vectors are the basis for later features.
data <- data %>%
  mutate(
    body_dx = neck_x - tail_x,   # body vector X-component (tail → neck)
    body_dy = neck_y - tail_y,   # body vector Y-component (tail → neck)
    head_dx = head_x - neck_x,   # head vector X-component (neck → head)
    head_dy = head_y - neck_y    # head vector Y-component (neck → head)
  )

###############################################################
# 11. BODY LENGTH NORMALISATION
###############################################################

# Compute the Euclidean length of both vectors.
# body_length will be used to normalize the parallel and perpendicular
# projections in the next section so that the features are scale-invariant.
data <- data %>%
  mutate(
    body_length = sqrt(body_dx^2 + body_dy^2),  # magnitude of body vector
    head_length = sqrt(head_dx^2 + head_dy^2)   # magnitude of head vector
  )

###############################################################
# 12. PARALLEL AND PERPENDICULAR HEAD COMPONENTS
###############################################################

# The goal here is to describe how the head moves relative to the body.
# We decompose the head vector into two orthogonal components:
#   1) Parallel to the body axis (forward / backward movement)
#   2) Perpendicular to the body axis (sideways movement)

# ----------------------------
# Parallel component:
# ----------------------------
# Formula: dot product of head vector and body unit vector
# = (head_dx * body_dx + head_dy * body_dy) / body_length
# Positive values mean the head extends forward along the body direction;
# negative values mean it is pulled backward (toward the tail).
data$parallel <- (data$head_dx * data$body_dx + data$head_dy * data$body_dy) /
                   data$body_length

# ----------------------------
# Perpendicular component:
# ----------------------------
# Formula: 2-D "cross product" of head vector and body unit vector
# = (head_dx * (-body_dy) + head_dy * body_dx) / body_length
# Positive values indicate the head is to one side of the body axis;
# negative values indicate the opposite side.
# This captures sideways head movements such as turning or exploration.
data$perpendicular <- (data$head_dx * (-data$body_dy) + data$head_dy * data$body_dx) /
                        data$body_length

###############################################################
# 13. VELOCITY AND ACCELERATION
###############################################################

# Velocity of the body (neck keypoint)
# diff(neck_x) = change in X between consecutive frames
# c(0, ...) prepends a zero to maintain the same vector length as n_frames
# sqrt(dx^2 + dy^2) = Euclidean distance moved per frame (speed in coordinate units)
data$vel_body <- sqrt(diff(c(0, data$neck_x))^2 + diff(c(0, data$neck_y))^2)

# Velocity of the head (head keypoint)
# Same computation applied to the head position
data$vel_head <- sqrt(diff(c(0, data$head_x))^2 + diff(c(0, data$head_y))^2)

# Acceleration of the head
# diff(vel_head) = change in head speed between consecutive frames
# High positive or negative values indicate rapid head movements (e.g. pecking)
data$acc_head <- c(0, diff(data$vel_head))

###############################################################
# 14. TEMPORAL CONTEXT FEATURES (ROLLING WINDOWS)
###############################################################

# Total body displacement over a short rolling window (~2 seconds = window_short frames)
# rollapply applies sum() over a sliding window of window_short frames
# fill = NA: boundary frames where a full window is unavailable receive NA
# (NAs are removed later when building the feature matrix in ClassifBehaviour.R)
data$dist_body_20 <- rollapply(data$vel_body, window_short, sum, fill = NA)  # last ~2 s

# Total body displacement over a longer rolling window (~5 seconds = window_long frames)
# Captures slower but sustained movement patterns (e.g. foraging)
data$dist_body_50 <- rollapply(data$vel_body, window_long, sum, fill = NA)  # last ~5 s

# Variance of the head-body angle over the short rolling window (~2 seconds)
# atan2(y, x) returns the angle of a 2-D vector in radians
# Subtracting the body angle from the head angle gives the relative head angle
# Rolling variance of this angle measures how actively the head is moving:
#   - High variance → frequent head reorientation (e.g. pecking)
#   - Low variance  → head mostly stable relative to body (e.g. inactive)
angle          <- atan2(data$head_dy, data$head_dx) - atan2(data$body_dy, data$body_dx)
data$angle_var <- rollapply(angle, window_short, var, fill = NA)

# Ratio of head speed to body speed
# High ratio (>1): head moves much faster than the body (pecking, preening)
# Low ratio (~1): head moves roughly at the same speed as the body (locomotion)
# epsilon is added to the denominator to avoid division by zero in stationary frames
data$ratio_head_body <- data$vel_head / (data$vel_body + epsilon)

###############################################################
# SAVE OUTPUT
###############################################################

# Write the complete simulated dataset (raw positions + behavioural labels +
# all derived features) to a CSV file.
# This file is read by ClassifBehaviour.R to build the classifier.
write.csv(data, "chicken_simulated_data_full.csv", row.names = FALSE)

message("SimulaKeypointTracking.R completed successfully.")
message("Output saved to: chicken_simulated_data_full.csv")
