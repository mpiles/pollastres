###############################################################
# BEHAVIOUR CLASSIFICATION AND TIME-BUDGET QUANTIFICATION
# (MULTI-ANIMAL PIPELINE)
# -------------------------------------------------------------
# This script carries out all analysis steps downstream of
# the raw keypoint tracking data produced by
# SimulaKeypointTracking.R.
#
# It processes data from n_individuals chickens simultaneously,
# keeping each individual's time series separate throughout to
# prevent information leakage across animals.
#
# Pipeline overview:
#   1.  Read the raw multi-animal tracking CSV.
#   2.  Temporal smoothing of keypoint positions per individual
#       (rolling mean) to suppress high-frequency detection noise.
#   3.  Compute body and head direction vectors (per frame).
#   4.  Compute vector lengths for body-length normalisation.
#   5.  Decompose head motion into parallel and perpendicular
#       components relative to the body axis.
#   6.  Compute frame-to-frame velocities and head acceleration
#       per individual (requires consecutive frames within the
#       same animal's time series).
#   7.  Compute rolling-window motion statistics per individual.
#   8.  Assemble the feature matrix; apply z-score normalisation.
#   9.  Save the normalised feature dataset to CSV.
#  10.  Train a Random Forest classifier (state ~ features).
#  11.  Predict behaviour labels for every frame.
#  12.  Compute classification quality metrics:
#         per-class  – Precision, Recall (Sensitivity), Specificity, F1-score
#         overall    – Accuracy, Macro-averaged F1
#       Save metrics to CSV and plot a confusion-matrix heatmap.
#  13.  Compute time budget per individual and in aggregate.
#  14.  Save per-individual time budget to CSV.
#
# Generates four visualisations:
#   confusion_matrix.png        – heatmap of the confusion matrix
#                                 (row-normalised proportions)
#   behaviour_time_budget.png   – mean +/- SD minutes per behaviour
#                                 (aggregated across all individuals)
#   keypoint_trajectories.png   – neck trajectory of one individual
#                                 coloured by predicted behaviour
#   keypoint_snapshots.png      – 4 × 4 grid of skeleton snapshots
#                                 (tail → neck → head) at evenly
#                                 spaced frames for one individual
#
# Input file (must exist in the working directory):
#   chicken_tracking_raw.csv   (produced by SimulaKeypointTracking.R)
#
# Output files written to the working directory:
#   chicken_features_for_classification.csv
#   classification_metrics.csv
#   time_budget_per_individual.csv
#   confusion_matrix.png
#   behaviour_time_budget.png
#   keypoint_trajectories.png
#   keypoint_snapshots.png
###############################################################

library(dplyr)        # data manipulation (mutate, group_by, summarise, %>%)
library(zoo)          # rolling-window functions (rollmean, rollapply)
library(randomForest) # Random Forest classifier

# ==============================================================
# PARAMETERS
# ==============================================================

fps          <- 10    # frames per second (must match SimulaKeypointTracking.R)
smooth_k     <-  5    # rolling-mean window width for temporal smoothing (frames)
window_short <- 20    # short rolling window (~2 s at 10 fps)
window_long  <- 50    # long  rolling window (~5 s at 10 fps)
epsilon      <- 1e-6  # small constant added to denominators to prevent division by zero

# Individual to display in the trajectory and snapshot plots.
# Change to any valid individual_id present in the data.
show_individual <- 1

# Colour palette used consistently across all three plots.
# Each behaviour is assigned a distinct, easily distinguishable colour.
beh_col <- c(
  inactive   = "gray70",
  locomotion = "dodgerblue",
  foraging   = "forestgreen",
  pecking    = "firebrick",
  preening   = "darkorchid"
)

# ==============================================================
# 1. READ RAW TRACKING DATA
# ==============================================================

# Load the CSV produced by SimulaKeypointTracking.R.
# Expected columns: individual_id, frame, state,
#                   tail_x, tail_y, neck_x, neck_y, head_x, head_y
# Each row is one video frame for one individual.
raw <- read.csv("chicken_tracking_raw.csv", stringsAsFactors = FALSE)

message("Loaded ", nrow(raw), " rows for ",
        length(unique(raw$individual_id)), " individuals.")

# ==============================================================
# 2. TEMPORAL SMOOTHING PER INDIVIDUAL
# ==============================================================

# Apply a rolling mean (window = smooth_k frames) to the neck and head
# positions of every individual independently.
# Purpose: suppress the high-frequency detection noise present in raw
#          tracker output while preserving the underlying movement signal.
# The tail is near the body and less noisy; it is not smoothed here.
#
# group_by(individual_id) is essential: it ensures that rollmean() is
# never computed across the boundary between two different animals.
# fill = "extend" pads the edges of each time series by repeating the
# nearest valid value, so no NAs are introduced by the smoothing step.
data <- raw %>%
  group_by(individual_id) %>%
  mutate(
    neck_x = rollmean(neck_x, k = smooth_k, fill = "extend"),
    neck_y = rollmean(neck_y, k = smooth_k, fill = "extend"),
    head_x = rollmean(head_x, k = smooth_k, fill = "extend"),
    head_y = rollmean(head_y, k = smooth_k, fill = "extend")
  ) %>%
  ungroup()

# ==============================================================
# 3. BODY AND HEAD DIRECTION VECTORS
# ==============================================================

# Compute 2-D direction vectors from the smoothed keypoint positions.
# These vectors describe body orientation and the direction the head
# is pointing, and are the foundation for the subsequent features.
# Each vector is computed independently per frame (no grouping needed).
data <- data %>%
  mutate(
    body_dx = neck_x - tail_x,   # body vector: points from tail → neck (body axis)
    body_dy = neck_y - tail_y,   # body vector Y-component
    head_dx = head_x - neck_x,   # head vector: points from neck → head
    head_dy = head_y - neck_y    # head vector Y-component
  )

# ==============================================================
# 4. VECTOR LENGTHS
# ==============================================================

# Euclidean length of each vector:
#   body_length: observed distance from tail to neck (≈ chicken body length)
#   head_length: observed distance from neck to head
# body_length is used below to normalise the parallel and perpendicular
# features, making them independent of the absolute size of the individual
# and of any scale differences between recordings.
data <- data %>%
  mutate(
    body_length = sqrt(body_dx^2 + body_dy^2),   # magnitude of body vector
    head_length = sqrt(head_dx^2 + head_dy^2)    # magnitude of head vector
  )

# ==============================================================
# 5. PARALLEL AND PERPENDICULAR HEAD COMPONENTS
# ==============================================================

# Decompose the head vector into two components relative to the body axis.
# This describes head posture in a body-centred reference frame, making
# the features invariant to the direction the chicken is facing.
#
#   parallel      = projection of head vector onto the body axis
#                 = dot(head, body_unit)
#                 = (head_dx·body_dx + head_dy·body_dy) / body_length
#                 Positive → head extends forward; negative → head pulls back.
#
#   perpendicular = projection of head vector onto the axis 90° from body
#                 = 2-D cross product / body_length
#                 = (head_dx·(−body_dy) + head_dy·body_dx) / body_length
#                 Captures sideways head movements (turning, exploration).
data <- data %>%
  mutate(
    parallel      = (head_dx *  body_dx + head_dy *  body_dy) / body_length,
    perpendicular = (head_dx * -body_dy + head_dy *  body_dx) / body_length
  )

# ==============================================================
# 6. VELOCITY AND ACCELERATION PER INDIVIDUAL
# ==============================================================

# Frame-to-frame speed = Euclidean distance moved between consecutive frames.
# diff(neck_x) computes the displacement in X between frame t-1 and frame t.
# c(0, diff(...)) prepends 0 so the vector length matches the group size;
# this assigns zero speed to the very first frame of each individual's series.
#
# group_by(individual_id) guarantees that diff() is computed within each
# animal's time series and never across the boundary between two animals.
#
# acc_head = change in head speed between consecutive frames.
# Large positive or negative values signal rapid head movements (e.g. pecking).
data <- data %>%
  group_by(individual_id) %>%
  mutate(
    vel_body = sqrt(c(0, diff(neck_x))^2 + c(0, diff(neck_y))^2),  # body speed
    vel_head = sqrt(c(0, diff(head_x))^2 + c(0, diff(head_y))^2),  # head speed
    acc_head = c(0, diff(vel_head))                                  # head acceleration
  ) %>%
  ungroup()

# ==============================================================
# 7. ROLLING-WINDOW FEATURES PER INDIVIDUAL
# ==============================================================

# All rolling operations must be confined to each individual's time series
# (group_by) so that data from one animal never influences another animal's
# features.  fill = NA leaves boundary frames (where the full window does
# not yet exist) as NA; these are removed by na.omit() in Section 8.
data <- data %>%
  group_by(individual_id) %>%
  mutate(

    # Total body displacement over a short window (~2 seconds = window_short frames).
    # A high value indicates a burst of locomotion; a low value indicates rest.
    dist_body_20 = rollapply(vel_body, window_short, sum, fill = NA),

    # Total body displacement over a longer window (~5 seconds = window_long frames).
    # Captures sustained locomotion patterns (e.g. extended foraging bouts).
    dist_body_50 = rollapply(vel_body, window_long, sum, fill = NA),

    # Variance of the head-to-body angle over the short window.
    # atan2(y, x) returns the angle of a 2-D vector in radians.
    # Subtracting the body direction from the head direction gives the
    # relative head orientation within the body reference frame.
    # High rolling variance → active repetitive head movements (e.g. pecking).
    # Low rolling variance  → head mostly stable relative to body (e.g. inactive).
    angle_var = rollapply(
      atan2(head_dy, head_dx) - atan2(body_dy, body_dx),
      window_short, var, fill = NA
    ),

    # Ratio of head speed to body speed.
    # Head moves much faster than body → active head behaviour (pecking, preening).
    # Head moves at similar speed to body → passive locomotion or inactivity.
    # epsilon is added to the denominator to prevent division by zero.
    # Note: when both velocities are near zero (perfectly still animal),
    # the numerator is also near zero, so the ratio remains close to 0.
    ratio_head_body = vel_head / (vel_body + epsilon)

  ) %>%
  ungroup()

# ==============================================================
# 8. FEATURE MATRIX AND NORMALISATION
# ==============================================================

# The ten motion features used as predictors for the Random Forest classifier
feature_cols <- c(
  "vel_body",         # body (neck) speed per frame
  "dist_body_20",     # total body displacement over last ~2 s
  "dist_body_50",     # total body displacement over last ~5 s
  "parallel",         # head component along the body axis
  "perpendicular",    # head component perpendicular to the body axis
  "angle_var",        # rolling variance of the head-body angle (~2 s window)
  "acc_head",         # frame-to-frame head acceleration
  "head_length",      # Euclidean distance from neck to head
  "vel_head",         # head speed per frame
  "ratio_head_body"   # ratio of head speed to body speed
)

# Retain metadata and smoothed positions alongside the features so they
# are available for classification and visualisation without a join.
keep_cols <- c(
  "individual_id", "frame", "state",
  "tail_x", "tail_y", "neck_x", "neck_y", "head_x", "head_y",
  feature_cols
)

feats <- data[, keep_cols]

# Remove rows with NA values.
# NAs arise at the start of each individual's series because the rolling
# windows (window_long = 50 frames) require at least 50 preceding frames.
# Only ~50 rows per individual are affected out of 36,000.
feats_complete <- na.omit(feats)

# Z-score normalisation: for each feature column, subtract its mean and
# divide by its standard deviation across the entire dataset.
# Only the 10 feature columns are normalised; positions and metadata are
# preserved in their original coordinate units for visualisation.
# Normalisation ensures that no feature dominates the classifier simply
# because it has a larger numeric range than the others.
feats_scaled <- feats_complete
feats_scaled[, feature_cols] <- scale(feats_complete[, feature_cols])

# ==============================================================
# 9. SAVE FEATURE DATASET
# ==============================================================

# Save the normalised feature matrix with individual ID, frame index,
# and ground-truth state label so it can be reused or inspected later.
write.csv(
  feats_scaled[, c("individual_id", "frame", "state", feature_cols)],
  "chicken_features_for_classification.csv",
  row.names = FALSE
)
message("Feature dataset saved: chicken_features_for_classification.csv")

# ==============================================================
# 10. TRAIN RANDOM FOREST CLASSIFIER
# ==============================================================

# Convert 'state' to a factor; this tells randomForest() to perform
# classification rather than regression.
feats_scaled$state <- as.factor(feats_scaled$state)

# Build the model formula: predict state from all 10 feature columns.
# The formula is constructed programmatically from feature_cols so that
# adding or removing features only requires editing the feature_cols vector.
rf_formula <- as.formula(paste("state ~", paste(feature_cols, collapse = " + ")))

# Train the Random Forest:
#   ntree = 300 : grow 300 independent decision trees.  More trees give
#                 more stable predictions and a better OOB error estimate.
# NOTE: With 50 individuals × ~36,000 frames this dataset contains roughly
#       1.8 million rows.  Training may take several minutes.  To speed up
#       testing, reduce n_individuals in SimulaKeypointTracking.R.
message("Training Random Forest (ntree = 300) – this may take a few minutes ...")
model <- randomForest(rf_formula, data = feats_scaled, ntree = 300)
message("Training complete.")
print(model)   # displays the OOB error estimate and the confusion matrix

# ==============================================================
# 11. PREDICT BEHAVIOURS
# ==============================================================

# Apply the trained model to every row of the dataset.
# predict() returns a factor vector with one predicted label per frame.
# The predictions are stored back in feats_scaled so that all columns
# (positions, features, predictions) remain in a single data frame,
# making it straightforward to use them in the visualisation sections.
feats_scaled$predicted <- predict(model, feats_scaled)

# ==============================================================
# 12. CLASSIFICATION QUALITY METRICS
# ==============================================================
#
# We compare the ground-truth label (feats_scaled$state) with the
# Random Forest prediction (feats_scaled$predicted) to evaluate how
# well the classifier has learned each behaviour.
#
# All metrics are derived from the confusion matrix, a square table
# where rows = true class and columns = predicted class.
# For a five-class problem, per-class metrics are computed using a
# one-vs-rest decomposition:
#
#   TP_k = frames correctly predicted as behaviour k
#   FP_k = frames of another behaviour wrongly predicted as k
#   FN_k = frames of behaviour k wrongly predicted as something else
#   TN_k = frames of neither k (true) nor k (predicted)
#
# From these counts we derive:
#   Precision_k   = TP_k / (TP_k + FP_k)   – of all frames predicted k,
#                                              how many really are k?
#   Recall_k      = TP_k / (TP_k + FN_k)   – of all true k frames, how
#   (Sensitivity)                              many were correctly found?
#   Specificity_k = TN_k / (TN_k + FP_k)   – of all true non-k frames,
#                                              how many were left as non-k?
#   F1_k          = 2 * Precision_k * Recall_k /
#                       (Precision_k + Recall_k)
#
# Overall metrics:
#   Accuracy  = sum of diagonal / total frames
#   Macro F1  = unweighted mean of per-class F1 scores

# ------------------------------------------------------------------
# 12a. Build the confusion matrix
# ------------------------------------------------------------------

# table() counts every (truth, prediction) combination.
# Both vectors are converted to character first so that factor levels
# that happen to be absent in one column do not create empty rows/cols.
conf_mat <- table(
  Truth     = as.character(feats_scaled$state),
  Predicted = as.character(feats_scaled$predicted)
)

message("\nConfusion matrix (rows = truth, columns = predicted):")
print(conf_mat)

# ------------------------------------------------------------------
# 12b. Per-class metrics
# ------------------------------------------------------------------

class_names <- rownames(conf_mat)   # behaviour labels in alphabetical order
n_classes   <- length(class_names)
n_total     <- sum(conf_mat)         # total number of frames evaluated

# Pre-allocate result vectors
precision   <- numeric(n_classes)
recall      <- numeric(n_classes)
specificity <- numeric(n_classes)
f1          <- numeric(n_classes)
names(precision) <- names(recall) <- names(specificity) <- names(f1) <- class_names

for (k in class_names) {
  TP <- conf_mat[k, k]
  FP <- sum(conf_mat[, k]) - TP   # column sum minus diagonal
  FN <- sum(conf_mat[k, ]) - TP   # row sum minus diagonal
  TN <- n_total - TP - FP - FN

  # Guard against division by zero (class absent in predictions or truth)
  precision[k]   <- if ((TP + FP) > 0) TP / (TP + FP)   else NA_real_
  recall[k]      <- if ((TP + FN) > 0) TP / (TP + FN)   else NA_real_
  specificity[k] <- if ((TN + FP) > 0) TN / (TN + FP)   else NA_real_
  f1[k]          <- if (!is.na(precision[k]) & !is.na(recall[k]) &
                         (precision[k] + recall[k]) > 0)
                      2 * precision[k] * recall[k] / (precision[k] + recall[k])
                    else NA_real_
}

# ------------------------------------------------------------------
# 12c. Overall metrics
# ------------------------------------------------------------------

accuracy <- sum(diag(conf_mat)) / n_total
macro_f1 <- mean(f1, na.rm = TRUE)

message(sprintf("\nOverall accuracy : %.4f", accuracy))
message(sprintf("Macro-avg F1     : %.4f", macro_f1))

# ------------------------------------------------------------------
# 12d. Assemble and print per-class metrics table
# ------------------------------------------------------------------

metrics_df <- data.frame(
  behaviour   = class_names,
  precision   = round(precision,   4),
  recall      = round(recall,      4),   # = sensitivity
  specificity = round(specificity, 4),
  f1_score    = round(f1,          4),
  row.names   = NULL
)

message("\nPer-class classification metrics:")
print(metrics_df)

# ------------------------------------------------------------------
# 12e. Save metrics to CSV
# ------------------------------------------------------------------

# Append summary rows following the sklearn classification_report convention:
#   "macro avg"  row: mean precision, recall, specificity and macro F1
#   "accuracy"   row: overall accuracy placed in the f1_score column,
#                     other per-class columns left NA (accuracy is not a
#                     per-class metric)
macro_row <- data.frame(
  behaviour   = "macro avg",
  precision   = round(mean(precision,   na.rm = TRUE), 4),
  recall      = round(mean(recall,      na.rm = TRUE), 4),
  specificity = round(mean(specificity, na.rm = TRUE), 4),
  f1_score    = round(macro_f1, 4)
)
accuracy_row <- data.frame(
  behaviour   = "accuracy",
  precision   = NA_real_,
  recall      = NA_real_,
  specificity = NA_real_,
  f1_score    = round(accuracy, 4)
)

write.csv(
  rbind(metrics_df, macro_row, accuracy_row),
  "classification_metrics.csv",
  row.names = FALSE
)
message("Classification metrics saved: classification_metrics.csv")

# ------------------------------------------------------------------
# VISUALISATION 1: CONFUSION MATRIX HEATMAP
# ------------------------------------------------------------------
#
# The confusion matrix is displayed as a colour-coded heatmap where
# each cell contains the row-normalised proportion of frames:
#   proportion[i, j] = conf_mat[i, j] / sum(conf_mat[i, ])
# Row normalisation shows recall per class along the diagonal and
# the distribution of errors off the diagonal, independently of
# class size differences.
# Darker blue = higher proportion; the ideal matrix is dark blue
# only along the diagonal and white everywhere else.

# Row-normalise: divide each row by its row total
conf_norm <- conf_mat / rowSums(conf_mat)

png("confusion_matrix.png", width = 700, height = 620)

# colour ramp: white (0) → steel-blue (1)
cm_cols <- colorRampPalette(c("white", "steelblue"))(100)

# image() draws the matrix with x = predicted (columns) and y = truth (rows).
# Columns of conf_norm are indexed by x; rows by y.
# We reverse the y axis (ylim, y values descending) so that the first class
# appears at the top, matching the conventional confusion-matrix orientation.
n_cl <- nrow(conf_norm)
image(
  x    = seq_len(n_cl),
  y    = seq_len(n_cl),
  z    = t(conf_norm[n_cl:1, ]),   # transpose + flip rows for image() orientation
  col  = cm_cols,
  zlim = c(0, 1),
  xaxt = "n", yaxt = "n",
  xlab = "Predicted behaviour",
  ylab = "True behaviour",
  main = "Confusion matrix (row-normalised proportions)"
)

# Axis labels
axis(1, at = seq_len(n_cl), labels = colnames(conf_norm), las = 2, cex.axis = 0.9)
axis(2, at = seq_len(n_cl), labels = rev(rownames(conf_norm)), las = 1, cex.axis = 0.9)

# Cell text: proportion value (2 decimal places)
for (i in seq_len(n_cl)) {          # i indexes predicted (x, columns of conf_norm)
  for (j in seq_len(n_cl)) {        # j indexes truth    (y, rows of conf_norm, reversed)
    val <- conf_norm[n_cl + 1 - j, i]   # map (x=i, y=j) back to original matrix cell
    text(
      x      = i,
      y      = j,
      labels = sprintf("%.2f", val),
      cex    = 0.85,
      # Use white text on dark cells and black text on light cells for readability
      col    = if (val > 0.55) "white" else "black"
    )
  }
}

# Colour-scale legend on the right side
par(new = TRUE)
image(
  x    = c(n_cl + 0.6, n_cl + 0.9),
  y    = seq(0.5, n_cl + 0.5, length.out = 100),
  z    = matrix(seq(0, 1, length.out = 100), nrow = 1),
  col  = cm_cols,
  add  = FALSE,
  xaxt = "n", yaxt = "n",
  xlim = c(0.5, n_cl + 1),
  ylim = c(0.5, n_cl + 0.5),
  xlab = "", ylab = "", main = "",
  bty  = "n"
)
axis(4, at = c(0.5, n_cl / 2, n_cl + 0.5),
     labels = c("0.00", "0.50", "1.00"), las = 1, cex.axis = 0.8)

dev.off()
message("Saved: confusion_matrix.png")

# ==============================================================
# 13. TIME BUDGET PER INDIVIDUAL AND IN AGGREGATE
# ==============================================================

# Count predicted frames per behaviour per individual.
# Dividing by fps gives seconds; dividing by 60 gives minutes.
time_budget <- feats_scaled %>%
  group_by(individual_id, predicted) %>%
  summarise(frames = n(), .groups = "drop") %>%
  mutate(minutes = frames / fps / 60)

# ==============================================================
# 14. SAVE PER-INDIVIDUAL TIME BUDGET
# ==============================================================

write.csv(time_budget, "time_budget_per_individual.csv", row.names = FALSE)
message("Per-individual time budget saved: time_budget_per_individual.csv")

# Compute mean and SD of minutes per behaviour across all individuals.
# This serves as the aggregate summary displayed in the bar chart.
agg_budget <- time_budget %>%
  group_by(predicted) %>%
  summarise(
    mean_min = mean(minutes),
    sd_min   = sd(minutes),
    .groups  = "drop"
  ) %>%
  arrange(desc(mean_min))

message("\nBehavioural time budget (mean ± SD minutes per individual):")
print(as.data.frame(agg_budget))

# ==============================================================
# VISUALISATION 2: BEHAVIOUR TIME BUDGET
# ==============================================================
#
# Bar chart showing the mean minutes per behaviour across all
# individuals, with ± 1 SD error bars to convey individual
# variation in time allocation.
# Bars are filled using the shared behaviour colour palette.

png("behaviour_time_budget.png", width = 800, height = 600)

# Order bars by decreasing mean so the most common behaviours appear first.
bar_order  <- order(agg_budget$mean_min, decreasing = TRUE)
bar_names  <- as.character(agg_budget$predicted[bar_order])
bar_means  <- agg_budget$mean_min[bar_order]
bar_sds    <- agg_budget$sd_min[bar_order]
bar_colors <- beh_col[bar_names]

# barplot() returns the midpoint x-position of each bar,
# which is needed to place the error-bar arrows correctly.
bar_mid <- barplot(
  setNames(bar_means, bar_names),
  col     = bar_colors,
  ylim    = c(0, max(bar_means + bar_sds, na.rm = TRUE) * 1.18),
  ylab    = "Mean minutes per individual",
  xlab    = "Behaviour",
  main    = paste0("Behavioural time budget\n",
                   "(mean +/- SD across ",
                   length(unique(time_budget$individual_id)),
                   " individuals)"),
  cex.names = 0.95,
  las       = 1
)

# Draw ± 1 SD error bars using arrows() with two heads (code = 3).
# angle = 90 makes the arrow heads horizontal (T-bar style).
arrows(
  x0     = bar_mid,
  y0     = bar_means - bar_sds,
  x1     = bar_mid,
  y1     = bar_means + bar_sds,
  angle  = 90, code = 3, length = 0.08, lwd = 1.5
)

dev.off()
message("Saved: behaviour_time_budget.png")

# ==============================================================
# VISUALISATION 3: KEYPOINT TRAJECTORIES
# ==============================================================
#
# The neck (body centre) trajectory of one individual is drawn
# as a coloured line, where the colour of each segment reflects
# the predicted behaviour at that frame.
# This reveals the spatial arrangement of behaviours within
# the recording area.

# Subset the dataset to the reference individual and sort by frame.
ind_data <- feats_scaled[feats_scaled$individual_id == show_individual, ]
ind_data <- ind_data[order(ind_data$frame), ]

png("keypoint_trajectories.png", width = 900, height = 700)

# Open an empty plot with the correct axis ranges.
# asp = 1 preserves the true shape of the path (no distortion).
plot(
  ind_data$neck_x, ind_data$neck_y,
  type = "n",
  xlab = "X (coordinate units)",
  ylab = "Y (coordinate units)",
  main = paste0("Neck trajectory coloured by predicted behaviour\n",
                "(Individual ", show_individual, ")"),
  asp  = 1
)

# Draw the trajectory as coloured line segments.
# rle() identifies runs of consecutive identical predicted labels,
# so we call lines() once per run rather than once per frame.
# Overlapping the start of each new run (pos <- run_end, not run_end + 1)
# ensures that segment endpoints join seamlessly at behaviour transitions.
pred_runs <- rle(as.character(ind_data$predicted))
pos <- 1   # pointer to the first row of the current run

for (r in seq_along(pred_runs$values)) {
  run_end <- pos + pred_runs$lengths[r] - 1   # last row of this run
  lines(
    ind_data$neck_x[pos:run_end],
    ind_data$neck_y[pos:run_end],
    col = beh_col[pred_runs$values[r]],
    lwd = 0.8
  )
  pos <- run_end   # overlap by 1 frame so segments join without gaps
}

legend(
  "topright",
  legend = names(beh_col),
  col    = beh_col,
  lwd    = 2,
  title  = "Predicted behaviour",
  cex    = 0.9,
  bty    = "n"
)

dev.off()
message("Saved: keypoint_trajectories.png")

# ==============================================================
# VISUALISATION 4: SKELETON SNAPSHOTS
# ==============================================================
#
# A 4 × 4 grid of skeleton panels, each showing the three-
# keypoint skeleton (tail → neck → head) for one individual at
# a regularly spaced moment in time.
# The skeleton line is coloured by the predicted behaviour;
# keypoint dots use fixed colours (orange = tail, blue = neck,
# red = head) to distinguish the three landmarks.
# Each panel is centred on the neck position with a fixed
# coordinate window so that all skeletons appear at the same
# scale and are directly comparable.

n_snapshots     <- 16     # total panels arranged in a 4 × 4 grid
snapshot_margin <- 0.40   # half-width (and half-height) of the view window
                          # around the neck, in coordinate units

# Select n_snapshots row indices that are evenly spaced across the
# reference individual's data.
snapshot_idx <- round(seq(1, nrow(ind_data), length.out = n_snapshots))

png("keypoint_snapshots.png", width = 1200, height = 1300)

# Set up the 4 × 4 multi-panel layout.
# oma provides an outer margin at the top (for the title) and at the
# bottom (for the shared keypoint legend).
par(
  mfrow = c(4, 4),
  mar   = c(1.0, 1.0, 2.8, 0.4),
  oma   = c(4.0, 0,   3.5, 0)
)

for (idx in snapshot_idx) {
  row <- ind_data[idx, ]

  # Centre the coordinate window on the neck position
  cx <- row$neck_x
  cy <- row$neck_y

  # Open an empty plot with a fixed window around the neck.
  # xaxt / yaxt = "n" suppress tick marks for a cleaner grid appearance.
  # asp = 1 keeps the skeleton geometrically undistorted.
  plot(
    NA,
    xlim     = c(cx - snapshot_margin, cx + snapshot_margin),
    ylim     = c(cy - snapshot_margin, cy + snapshot_margin),
    xlab     = "", ylab     = "",
    xaxt     = "n", yaxt    = "n",
    main     = paste0("f = ", row$frame,
                      "\n", row$predicted),
    cex.main = 0.85,
    asp      = 1
  )

  # Draw the skeleton as a two-segment polyline: tail → neck → head.
  # The line colour encodes the predicted behaviour at this frame.
  lines(
    c(row$tail_x, row$neck_x, row$head_x),
    c(row$tail_y, row$neck_y, row$head_y),
    lwd = 2.5,
    col = beh_col[as.character(row$predicted)]
  )

  # Mark each keypoint with a filled circle using a fixed colour:
  #   darkorange → tail   (rear of the body)
  #   steelblue  → neck   (body centre / junction)
  #   firebrick  → head   (front, most mobile keypoint)
  points(
    c(row$tail_x, row$neck_x, row$head_x),
    c(row$tail_y, row$neck_y, row$head_y),
    pch = 19, cex = 1.8,
    col = c("darkorange", "steelblue", "firebrick")
  )
}

# Outer title spanning all panels
mtext(
  paste0("Skeleton snapshots  –  Individual ", show_individual,
         "  (", n_snapshots, " evenly spaced frames)"),
  outer = TRUE, cex = 1.1, line = 1.8
)

# Shared keypoint legend in the outer bottom margin.
# par(fig = c(0,1,0,1), new = TRUE) resets the graphics device to the
# full plot region so that legend() can be placed in the outer margin.
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
legend(
  x       = "bottom",
  legend  = c("Tail", "Neck", "Head"),
  pch     = 19,
  col     = c("darkorange", "steelblue", "firebrick"),
  horiz   = TRUE,
  bty     = "n",
  cex     = 1.05,
  inset   = 0.01,
  title   = "Keypoints"
)

dev.off()
message("Saved: keypoint_snapshots.png")

# ==============================================================
# SUMMARY
# ==============================================================

message("\nClassifBehaviour.R completed successfully.")
message("Output files:")
message("  chicken_features_for_classification.csv")
message("  classification_metrics.csv")
message("  time_budget_per_individual.csv")
message("  confusion_matrix.png")
message("  behaviour_time_budget.png")
message("  keypoint_trajectories.png")
message("  keypoint_snapshots.png")
