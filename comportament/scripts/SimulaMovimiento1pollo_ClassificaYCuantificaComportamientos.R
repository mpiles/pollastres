###############################################################
# REALISTIC CHICKEN POSE SIMULATION WITH BEHAVIOUR BLOCKS
# -------------------------------------------------------------
# This script simulates chicken movements with realistic
# trajectories for the neck (body), head, and tail.
# Features are computed for behaviour classification.
# Random Forest classifier predicts behaviour for each frame.
###############################################################

library(dplyr)    # for data manipulation
library(zoo)      # for rolling windows and smoothing
library(randomForest) # for behaviour classification

set.seed(123)     # for reproducibility

###############################################################
# 1. VIDEO PARAMETERS
###############################################################

fps <- 10                     # frames per second
video_seconds <- 3600          # duration of video in seconds
n_frames <- fps * video_seconds # total number of frames
time <- 1:n_frames             # frame index

# We simulate a one-hour video at 10 fps, which gives 36,000 frames.

###############################################################
# 2. BEHAVIOURAL STATES
###############################################################

states <- c("inactive","locomotion","foraging","pecking","preening")
# Minimum duration of each behaviour block in frames
min_duration <- c(inactive=30, locomotion=20, foraging=25, pecking=5, preening=15)

# This ensures that each behaviour lasts at least the minimum number of frames
# making the simulated sequence realistic.

###############################################################
# 3. MARKOV TRANSITION MATRIX
###############################################################

# Rows = current state, Columns = next state
# Values represent probabilities of changing from current to next behaviour.
transition_matrix <- matrix(
  c(
    0.6, 0.2, 0.1, 0.05, 0.05,  # inactive
    0.2, 0.2, 0.4, 0.1, 0.1,    # locomotion
    0.3, 0.1, 0.4, 0.1, 0.1,    # foraging
    0.2, 0.2, 0.2, 0.3, 0.1,    # pecking
    0.3, 0.1, 0.2, 0.1, 0.3     # preening
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
current_state <- sample(states,1) # start with a random behaviour

while(length(state_sequence) < n_frames){
  
  # Determine minimum duration for this behaviour
  dur_min <- min_duration[current_state]
  
  # Random block length: min_duration + small random variability
  block_duration <- round(dur_min + rexp(1, rate=1/(dur_min*2)))
  
  # Append repeated states for the block
  state_sequence <- c(state_sequence, rep(current_state, block_duration))
  
  # Stop if exceeding total frames
  if(length(state_sequence) >= n_frames) break
  
  # Pick next behaviour according to Markov transition probabilities
  row_idx <- which(states == current_state)
  next_state <- sample(states,1,prob=transition_matrix[row_idx,])
  
  # Update current behaviour
  current_state <- next_state
}

# Truncate to exact number of frames
state_sequence <- state_sequence[1:n_frames]

# At this point, we have a realistic behavioural timeline.

###############################################################
# 5. SIMULATE NECK (BODY) TRAJECTORY – SMOOTH MOVEMENT
###############################################################

# Neck is considered the body center. 
# We generate a smooth path using cumulative sum of small random steps
# and apply a spline interpolation for realistic motion.

# Base random walk
dx <- rnorm(n_frames, 0, 0.01)
dy <- rnorm(n_frames, 0, 0.01)
neck_x <- cumsum(dx)
neck_y <- cumsum(dy)

# For locomotion frames, increase step size to simulate walking
locomotion_idx <- which(state_sequence=="locomotion")
neck_x[locomotion_idx] <- cumsum(rnorm(length(locomotion_idx), 0, 0.05))
neck_y[locomotion_idx] <- cumsum(rnorm(length(locomotion_idx), 0, 0.05))

# Apply spline smoothing to produce continuous smooth trajectories
neck_x <- smooth.spline(1:n_frames, neck_x, spar=0.6)$y
neck_y <- smooth.spline(1:n_frames, neck_y, spar=0.6)$y

###############################################################
# 6. SIMULATE TAIL POSITION
###############################################################

# Tail is positioned a fixed distance behind the neck along the body axis
# Slight random noise is added to simulate natural sway.

body_length <- 0.3
tail_x <- neck_x - body_length + rnorm(n_frames,0,0.005)
tail_y <- neck_y + rnorm(n_frames,0,0.005)

###############################################################
# 7. SIMULATE HEAD POSITION
###############################################################

# Head is in front of neck by default.
# We add behaviour-specific offsets to simulate characteristic movements
# and smooth the trajectory with splines for natural appearance.

head_x <- neck_x + 0.15 + rnorm(n_frames,0,0.01)
head_y <- neck_y + 0.02 + rnorm(n_frames,0,0.01)

# Pecking: head goes downward
peck_idx <- which(state_sequence=="pecking")
head_y[peck_idx] <- neck_y[peck_idx] - abs(rnorm(length(peck_idx), 0.05,0.01))

# Preening: head goes backward
preen_idx <- which(state_sequence=="preening")
head_x[preen_idx] <- neck_x[preen_idx] - abs(rnorm(length(preen_idx),0.1,0.01))

# Smooth head positions
head_x <- smooth.spline(1:n_frames, head_x, spar=0.6)$y
head_y <- smooth.spline(1:n_frames, head_y, spar=0.6)$y

###############################################################
# 8. CREATE DATA FRAME
###############################################################

data <- data.frame(
  frame = time,
  state = state_sequence,
  tail_x, tail_y,
  neck_x, neck_y,
  head_x, head_y
)

###############################################################
# 9. TEMPORAL SMOOTHING (ADDITIONAL)
###############################################################

# Additional rolling mean smoothing to reduce jitter
smooth_roll <- function(x, k=5){ rollmean(x, k, fill="extend") }
data <- data %>%
  mutate(
    neck_x = smooth_roll(neck_x),
    neck_y = smooth_roll(neck_y),
    head_x = smooth_roll(head_x),
    head_y = smooth_roll(head_y)
  )

###############################################################
# 10. BODY AND HEAD VECTORS
###############################################################

data <- data %>%
  mutate(
    body_dx = neck_x - tail_x,   # body vector x-component
    body_dy = neck_y - tail_y,   # body vector y-component
    head_dx = head_x - neck_x,   # head vector x-component
    head_dy = head_y - neck_y    # head vector y-component
  )

###############################################################
# 11. BODY LENGTH NORMALISATION
###############################################################

data <- data %>%
  mutate(
    body_length = sqrt(body_dx^2 + body_dy^2),
    head_length = sqrt(head_dx^2 + head_dy^2)
  )

###############################################################
# 12. PARALLEL AND PERPENDICULAR HEAD COMPONENTS
###############################################################

# The goal here is to describe how the head moves relative to the body.
# We decompose the head movement into two directions:
# 1) Parallel to the body (forward/backward)
# 2) Perpendicular to the body (sideways)

# ----------------------------
# Parallel component:
# ----------------------------
# Formula: projection of the head vector onto the body vector
# (dot product of head vector and body vector) / body length
# - head_dx, head_dy: head vector (neck -> head)
# - body_dx, body_dy: body vector (tail -> neck)
# - body_length: magnitude of the body vector
# Result: scalar value that measures how much the head moves along the body axis
# - Positive: head moves forward along body direction
# - Negative: head moves backward (toward the tail)
data$parallel <- (data$head_dx*data$body_dx + data$head_dy*data$body_dy) / data$body_length

# ----------------------------
# Perpendicular component:
# ----------------------------
# Formula: projection of the head vector onto the perpendicular axis to the body
# (cross product-like computation) / body length
# Measures sideways movement of the head relative to the body
# - Positive: head moves to one side
# - Negative: head moves to the other side
# This helps detect sideways or diagonal movements (e.g., head turning, exploration)
data$perpendicular <- (data$head_dx*(-data$body_dy) + data$head_dy*data$body_dx) / data$body_length

###############################################################
# 13. VELOCITY AND ACCELERATION
###############################################################

# Velocity of the body (neck)
# -------------------------------
# diff(data$neck_x) calculates the difference in X position between consecutive frames
# diff(data$neck_y) does the same for Y position
# We add a 0 at the beginning (c(0, ...)) to keep vector lengths consistent
# sqrt(dx^2 + dy^2) gives the Euclidean distance moved between frames
# Result: vel_body = speed of the neck (body) at each frame
data$vel_body <- sqrt(diff(c(0,data$neck_x))^2 + diff(c(0,data$neck_y))^2)

# Velocity of the head
# ------------------------
# Same idea but applied to the head position (neck -> head)
# Captures how fast the head moves in each frame
data$vel_head <- sqrt(diff(c(0,data$head_x))^2 + diff(c(0,data$head_y))^2)

# Acceleration of the head
# -----------------------------
# diff(data$vel_head) calculates how much the head speed changes from frame to frame
# Adding c(0, ...) ensures the vector has the same length as number of frames
# High acceleration values indicate quick head movements (like pecking)
data$acc_head <- c(0,diff(data$vel_head))


###############################################################
# 14. TEMPORAL CONTEXT FEATURES (ROLLING WINDOWS)
###############################################################

# Total body movement over short window (~2 seconds)
# rollapply calculates a rolling sum over the last 20 frames
# This gives a measure of how much the body has moved recently.
# Useful to distinguish between stillness and locomotion.
data$dist_body_20 <- rollapply(data$vel_body, 20, sum, fill=NA) # last ~2s

# Total body movement over a longer window (~5 seconds)
# Similar to above but over 50 frames. This captures longer-term trends.
# Helps detect slow behaviours like foraging, where the body moves slowly but continuously.
data$dist_body_50 <- rollapply(data$vel_body, 50, sum, fill=NA) # last ~5s

# Head-body angle variance over short window
# atan2 computes the angle of a vector (head or body) in radians
# By subtracting body angle from head angle, we get the head’s angle relative to the body
# rollapply with var calculates how variable this angle has been in the last 20 frames
# High variance: rapid head movement (pecking)
# Low variance: head mostly still relative to body (preening, inactive)
angle <- atan2(data$head_dy, data$head_dx) - atan2(data$body_dy, data$body_dx)
data$angle_var <- rollapply(angle, 20, var, fill=NA)

# Ratio of head speed to body speed
# Compares how much the head moves relative to the body
# - High ratio: head moves faster than body (pecking, preening)
# - Low ratio: head moves along with body (locomotion, inactivity)
# Adding 1e-6 to denominator avoids division by zero
data$ratio_head_body <- data$vel_head / (data$vel_body + 1e-6)


#++++++++++++++++++++++++++++++
#  Save the full simulated data (positions + behaviours)
write.csv(data, "chicken_simulated_data_full.csv", row.names = FALSE)
#++++++++++++++++++++++++++++++
#+
###############################################################
# 15. FEATURE MATRIX
###############################################################

features <- data %>%
  select(
    vel_body, dist_body_20, dist_body_50,
    parallel, perpendicular, angle_var,
    acc_head, head_length, vel_head, ratio_head_body
  )

###############################################################
# 16. NORMALIZE FEATURES
###############################################################

features_scaled <- scale(features) # z-score normalization
dataset <- data.frame(features_scaled, state=data$state)
dataset <- na.omit(dataset)

#+++++++++++++++++++++++++++++
# Save the feature dataset used for classification
write.csv(dataset, "chicken_features_for_classification.csv", row.names = FALSE)
#+++++++++++++++++++++++++++++


###############################################################
# 17. TRAIN RANDOM FOREST CLASSIFIER (CORRECTED)
###############################################################

# Convert the response variable 'state' to a factor
# Explanation:
# - Random Forest in R automatically chooses regression if the response is numeric
# - If 'state' is a character vector, RF can misinterpret it
# - Converting to factor ensures that RF performs classification instead of regression
dataset$state <- as.factor(dataset$state)

# Train the Random Forest classifier
# - state ~ . : predict 'state' using all other columns as features
# - ntree = 300 : build 300 decision trees for better stability
model <- randomForest(state ~ ., data = dataset, ntree = 300)

###############################################################
# 18. PREDICT BEHAVIOURS
###############################################################

# Predict behaviour for each frame using the trained model
predicted <- predict(model, dataset)

###############################################################
# 19. TIME SPENT IN EACH BEHAVIOUR
###############################################################

# Count frames per predicted behaviour
frame_counts <- table(predicted)

# Convert to minutes (frames / fps / 60)
minutes_per_behaviour <- frame_counts / fps / 60

# Display time budget per behaviour
print(minutes_per_behaviour)

###############################################################
# 20. VISUALIZE RESULTS
###############################################################

# Save the plot as a PNG file
png(filename = "behaviour_time_budget.png", width = 800, height = 600)
barplot(minutes_per_behaviour,
        col = "steelblue",
        ylab = "Minutes",
        main = "Estimated behavioural time budget")
dev.off() # close the graphics device
