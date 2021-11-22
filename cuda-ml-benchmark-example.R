library(reticulate)

cuml <- import("cuml")

runners <- cuml$benchmark$runners
algorithms <- cuml$benchmark$algorithms

runners$run_variations(
  algos = list(algorithms$algorithm_by_name("KMeans")),
  dataset_name = "blobs",
  bench_rows = list(10000L),
  bench_dims = list(1000L)
)
