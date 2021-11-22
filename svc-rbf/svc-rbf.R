# NOTE:
# The benchmark was run with {cuda.ml} installed using the following options
#
# ```
# install.packages("cuda.ml", type="source", INSTALL_opts="--byte-compile")
# ```

library(cuda.ml)
library(rbenchmark)
library(reticulate)

cuml <- import("cuml")

runners <- cuml$benchmark$runners
algorithms <- cuml$benchmark$algorithms
datagen <- cuml$benchmark$datagen

dataset_name <- "classification"
dataset_n_samples <- 10000L
dataset_n_features <- 10L

cat("\nBEGIN: cuML vs. scikit-learn benchmark:\n")
runners$run_variations(
  algos = list(algorithms$algorithm_by_name("SVC-RBF")),
  dataset_name = dataset_name,
  bench_rows = list(dataset_n_samples),
  bench_dims = list(dataset_n_features),
  n_reps = 10L
)
cat("\nEND: cuML vs. scikit-learn benchmark:\n")

data <- datagen$gen_data(
  dataset_name = dataset_name,
  dataset_format = "numpy",
  n_samples = dataset_n_samples,
  n_features = dataset_n_features
)

cuda_ml_input_x <- force(as.matrix(data[[1]]))
cuda_ml_input_y <- force(as.integer(data[[2]]))

cat("\n\nBEGIN: {cuda.ml} benchmark\n")
print(
  benchmark(
    {
      # NOTE: using the R-equivalent of the parameters from
      # https://github.com/rapidsai/cuml/blob/branch-21.10/python/cuml/benchmark/algorithms.py#L337
      # here to ensure benchmarks done using {cuda.ml} will be comparible to those
      # done using {reticulate}+cuML
      cuda_ml_svm(x = cuda_ml_input_x, y = cuda_ml_input_y, kernel = "rbf")
    },
    replications = 10
  )
)
cat("\nEND: {cuda.ml} benchmark\n")

# Sample output from running this benchmark on a 'p3.2xlarge' Amazon EC2 instance
# using the Docker image at https://hub.docker.com/r/yitaoli/rapids-cuml-benchmark:
#
# ```
# BEGIN: cuML vs. scikit-learn benchmark:
# Running:
#  SVC-RBF
# Running SVC-RBF...
# Finished all benchmark runs
#       algo  input   cu_time  cpu_time  cuml_acc   cpu_acc    speedup  n_samples  n_features
# 0  SVC-RBF  numpy  0.037717  0.424695  0.992799  0.992799  11.259953      10000          10
#
# END: cuML vs. scikit-learn benchmark:
#
#
# BEGIN: {cuda.ml} benchmark
#                                                                              test
# 1 {\n    cuda_ml_svm(x = cuda_ml_input_x, y = cuda_ml_input_y, kernel = "rbf")\n}
#   replications elapsed relative user.self sys.self user.child sys.child
# 1           10   0.746        1     0.449    0.296          0         0
#
# END: {cuda.ml} benchmark
# ```
#
# Please notice the following:
# - The 'cu_time' output from the above is the average time elapsed per run
#   using the GPU-accelerated cuML kMeans implementation, which was around
#   0.037717 s/run.
# - The 'cpu_time' output from the above is the average time elapsed per run
#   using the scikit-learn implementation (i.e., CPU-only), which was around
#   0.424695 s/run.
# - The 'elapsed' output from the above within the '{cuda.ml} benchmark' section
#   is the total elapsed time for 10 runs using {cuda.ml}, so the time elapsed
#   for {cuda.ml} was around 0.0746 s/run.
