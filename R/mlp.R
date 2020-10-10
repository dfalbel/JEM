mlp <- torch::nn_module(
  initialize = function(n_classes) {
    self$l1 = torch::nn_linear(28 * 28, 1000)
    self$l2 = torch::nn_linear(1000, 1000)
    self$l3 = torch::nn_linear(1000, 10)
  },
  forward = function(x) {
    x %>%
      #torch::torch_flatten(start_dim = 2) %>%
      self$l1() %>%
      torch::nnf_relu() %>%
      self$l2() %>%
      torch::nnf_relu() %>%
      self$l3()
  }
)
