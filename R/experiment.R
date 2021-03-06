
logsumexp <- function(x) {
  torch::torch_logsumexp(x, dim = 2)
}

sgld_sample <- function(model, data, config) {

  x <- data$x
  y_ <- data$y
  
  eta <- config$eta
  alpha <- config$alpha
  sigma <- config$sigma
  
  x$requires_grad_(TRUE)
  for (i in seq_len(eta)) {
    
    y <- model(x, y_)
    
    jacobian <- torch::autograd_grad(
      outputs = y,
      inputs = x,
      grad_outputs = torch::torch_ones_like(y)
    )
    
    noise <- torch::torch_randn(size = x$shape, device = x$device) * sigma
    x <- x + jacobian[[1]] * alpha + noise
  }
  x <- x$detach()
  x
}

train <- function(model, buffer, train_dl, optimizer, loss_fn, config) {

  model$train()
  device <- config$device
  losses <- c()
  pb <- bar(length = length(train_dl))
  
  for (batch in torch::enumerate(train_dl)) {
    
    data <- batch[[1]]$to(device = device)
    targets <- batch[[2]]$to(device = device)
    
    # Classifier loss
    logits <- model$classifier(data)
    loss_clf <- loss_fn(logits, targets)
    
    # EBM loss
    # p(x)_0
    sampl <- buffer$get_batch(config$batch_size, config$rho)
    sampl$x <- sgld_sample(model, sampl, config)
    buffer$add(sampl)
    
    # Computes energy loss
    if (!config$conditional) {
      p_sample <- model(sampl$x)
      p_data <- model(data)
    } else {
      p_sample <- model(sampl$x, sampl$y)
      p_data <- model(data, targets)
    }
    
    loss_gen <- -torch::torch_mean(p_data - p_sample)
    L <- loss_clf + loss_gen
    
    optimizer$zero_grad()
    L$backward()
    optimizer$step()
    
    losses <- c(losses, L$item())
    pb$tick(tokens = list(loss = mean(losses)))
  }
  
  losses
}

valid <- function(model, valid_dl, loss_fn, config) {
  
  model$eval()
  correct <- 0
  n <- 0
  losses <- c()
  
  torch::with_no_grad({
    for (batch in torch::enumerate(valid_dl)) {
      
      data <- batch[[1]]$to(device = config$device)
      targets <- batch[[2]]$to(device = config$device)
      
      logits <- model$classifier(data)
      predicted <- torch::torch_max(logits, dim = 2)[[2]]
      
      correct <- correct + predicted$eq(targets)$sum()$item()
      
      loss <- loss_fn(logits, targets)
      losses <- c(losses, loss$item())
     
      n <- n + targets$shape[1] 
    }
  })
  
  acc <- correct/n
  list(acc = acc, correct = correct, n = n, loss = mean(losses))
} 

ebm_model <- torch::nn_module(
  "ebm_model",
  initialize = function(type, n_classes, input_dim) {
    if (type == "cnn")
      self$classifier <- cnn(n_classes = n_classes, input_dim)
    else if (type == "mlp")
      self$classifier <- mlp(n_classes = n_classes, input_dim)
  }, 
  forward = function(x, y = NULL) {
    logits <- self$classifier(x)
    if (is.null(y))
      logsumexp(logits)
    else {
      torch::torch_gather(logits, dim = 2L, index = y$unsqueeze(2))
    }
  }
)

#' Run experiment
#' 
#' Trains annd validates the JEM model using the specified config.
#' 
#' @param config a config list with the information below. If empty, will use
#'   the config.yml in the working directory.
#'   
#' @section Configuration
#' 
#' The configuration object has the following options:
#' 
#' ```
#' dataset: "mnist"
#' model: "mlp"
#' n_epochs: 20
#' lr: 0.001
#' rho: 0.05
#' eta: 20
#' sigma: 0.01
#' alpha: 1
#' buffer_size: 10000
#' device: "cuda"
#' batch_size: 100
#' ```
#'   
#' @return
#' Returns all experiment information. Losses, model, buffer and configuration.
#'
#' @export
run_experiment <- function(config = config::get()) {
  
  print_config(config)
  
  ds <- get_datasets(config$dataset)
  
  train_dl <- torch::dataloader(
    dataset = ds$train,
    batch_size = config$batch_size,
    shuffle = TRUE
  )
  
  valid_dl <- torch::dataloader(
    dataset = ds$valid,
    batch_size = config$batch_size,
    shuffle = FALSE
  )
  
  input_shape <- ds$train[1][[1]]$shape

  if (!config$conditional)
    buffer <- replay_buffer(
      buffer_size = config$buffer_size, 
      dim = input_shape,
      device = config$device
    )
  else
    buffer <- conditional_replay_buffer(
      n_class = length(ds$train$classes),
      buffer_size = config$buffer_size,
      dim = input_shape,
      device = config$device
    )
  
  model <- ebm_model(config$model, ds$n_classes, input_shape)
  model$to(device = config$device)

  optimizer <- torch::optim_adam(
    params = model$parameters,
    lr = config$lr
  )

  loss_fn <- torch::nn_cross_entropy_loss()
  
  valid_metrics <- list()
  for (epoch in seq_len(config$n_epochs)) {
    train_losses <- train(model, buffer, train_dl, optimizer, loss_fn, config)
    metrics <- valid(model, valid_dl, loss_fn, config)
    valid_metrics[[epoch]] <- metrics
    cat(sprintf("[Epoch %d] Train{Loss: %3f} Valid{Loss: %3f, Acc: %3f}\n", 
                epoch, mean(train_losses), metrics$loss, metrics$acc))
  }
  
  list(
    model = model,
    buffer = buffer,
    train_losses = train_losses,
    valid_metrics = valid_metrics,
    config = config
  )
}

#' Generate samples from a model
#' 
#' @param experiment the experiment data returned by the [run_experiment()] function.
#' @param n the number of samples to generate
#' @param eta the number of iterations in the SGLD sampler.
#'
#' @export
generate_samples <- function(experiment, n, eta = NULL) {
  
  b <- experiment$buffer$get_batch(n, reinit_freq = 1) # all randomly initialized
  
  if (!is.null(eta))
    experiment$config$eta <- eta
  
  samps <- sgld_sample(experiment$model, b, config = experiment$config)
  samps <- torch::torch_clamp(samps, -1, 1)$
    add(1)$
    div(2)$
    to(device = "cpu")
  
  samps
}


#' Plot samples
#' 
#' Plot generated samples
#' 
#' @param samps a 4D tensor of samples (as generated by the [generate_samples()])
#'   function.
#' @param nrow number of rows in the generated figure.
#'
#' @export
plot_samples <- function(samps, nrow = 8) {
  
  grid <- torchvision::vision_make_grid(samps, num_rows = nrow)
  
  if (grid$shape[1] == 1)
    grid <- torch::torch_cat(list(grid, grid, grid), 1)
  
  grid <- grid$transpose(1, 3) # channels last
  grid %>% 
    as.array() %>% 
    as.raster() %>% 
    plot()
}
